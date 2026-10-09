#!/usr/bin/env python3
"""fakinflu — worker body3d : extraction 3D du mouvement (SAM 3D Body via ComfyUI). Plan motion, Phase 1.

Remplace le /handler.py de runpod/worker-comfyui ; /start.sh lance ComfyUI (127.0.0.1:8188).
Handler GENERATEUR : progression et apercus lisibles en direct via /stream
(return_aggregate_stream = False).

Entree (contrat transfer.py) :
  {"run_id", "r2": bool,
   "inputs":  {"video": {b64|url, name, bytes, sha256}},
   "outputs": {<role>: {put_url, key, max_bytes} | {max_bytes}},
   "outputs_json": {put_url, key} | null,
   "params":  {"preset": {...preset d'extraction resolu (app/src/schemas ExtractPreset)...}}}

Sorties (roles) : mhr_params (.npz), camera (.json), qa (.json), masks (.npz), motion_glb, motion_bvh,
                  overlay (.mp4), control_mesh (.mp4), control_scail_pose (.mp4)
Chaque sortie suit la regle de routage de transfer.py (direct <= 1 Mio, sinon R2, sinon morceaux).
"""
import glob, json, os, queue, shutil, subprocess, tempfile, threading, time, traceback, urllib.error, urllib.request, uuid

import runpod

from transfer import Outputs, fetch_inputs, preview_event, version_event
from workflow import build_workflow, OUTPUT_FILES, OUTPUT_NAMES

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
JOB_TIMEOUT = int(os.environ.get("BODY3D_JOB_TIMEOUT_S", "1700"))


def _http(method, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def wait_comfy(max_s=600):
    t = time.time()
    while time.time() - t < max_s:
        try:
            _http("GET", "/system_stats", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError(f"ComfyUI injoignable sur {COMFY} apres {max_s}s")


def _ws_progress(client_id, q, stop):
    """Relaye les messages de progression ComfyUI (websocket) vers la file q."""
    try:
        import websocket  # websocket-client, present dans worker-comfyui
    except ImportError:
        return
    url = COMFY.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
    try:
        ws = websocket.create_connection(url, timeout=5)
    except Exception:
        return
    ws.settimeout(1)
    while not stop.is_set():
        try:
            msg = ws.recv()
        except Exception:
            continue
        if not isinstance(msg, str):
            continue
        try:
            m = json.loads(msg)
        except ValueError:
            continue
        if m.get("type") in ("progress", "executing"):
            q.put(m)
    ws.close()


def run_workflow(wf):
    """Generateur : soumet le graphe, emet la progression, renvoie (outputs, secondes) via StopIteration."""
    client_id = uuid.uuid4().hex
    q, stop = queue.Queue(), threading.Event()
    th = threading.Thread(target=_ws_progress, args=(client_id, q, stop), daemon=True)
    th.start()
    try:
        try:
            r = _http("POST", "/prompt", {"prompt": wf, "client_id": client_id})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI a refuse le graphe (HTTP {e.code}) : {e.read().decode()[:1500]}")
        pid = r["prompt_id"]
        t0, last_poll = time.time(), 0.0
        names = {k: v.get("_meta", {}).get("title") or v["class_type"] for k, v in wf.items()}
        while True:
            if time.time() - t0 > JOB_TIMEOUT:
                _http("POST", "/interrupt", {})
                raise RuntimeError(f"timeout ComfyUI ({JOB_TIMEOUT}s)")
            try:
                m = q.get(timeout=1)
                d = m.get("data", {})
                if m["type"] == "executing" and d.get("node"):
                    yield {"type": "progress", "step": names.get(d["node"], d["node"])}
                elif m["type"] == "progress" and d.get("max"):
                    yield {"type": "progress", "step": names.get(d.get("node"), "calcul"),
                           "pct": round(100 * d["value"] / d["max"]), "frame": d["value"]}
            except queue.Empty:
                pass
            if time.time() - last_poll < 2:
                continue
            last_poll = time.time()
            h = _http("GET", f"/history/{pid}")
            if pid not in h:
                continue
            st = h[pid].get("status") or {}
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                detail = msgs[-1][1] if msgs else st
                if isinstance(detail, dict):
                    detail = {k: detail.get(k) for k in ("node_id", "node_type", "exception_message")}
                raise RuntimeError(f"erreur d'execution ComfyUI : {json.dumps(detail, ensure_ascii=False)[:800]}")
            if st.get("completed"):
                return time.time() - t0
    finally:
        stop.set()


def collect(prefix):
    """{role: chemin} des fichiers produits sous output/<prefix>/ (noms fixes, cf. workflow.OUTPUT_FILES)."""
    base = os.path.join(COMFY_DIR, "output", prefix)
    found = {}
    for role, pattern in OUTPUT_FILES.items():
        hits = sorted(glob.glob(os.path.join(base, pattern)))
        if hits:
            found[role] = hits[-1]
    return found


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                          "-of", "json", path], capture_output=True, text=True, timeout=120)
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den or 1), "frames": int(s.get("nb_read_packets") or 0)}


def thumb(video, t=0.5):
    """Image JPEG de l'overlay pour l'apercu de l'UI (bien sous 1 Mio)."""
    dst = video + ".jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", video, "-frames:v", "1",
                    "-vf", "scale=480:-2", "-q:v", "5", dst], timeout=60, check=True)
    with open(dst, "rb") as f:
        return f.read()


def handler(job):
    inp = job.get("input") or {}
    params = inp.get("params") or {}
    work = tempfile.mkdtemp(prefix="body3d_")
    in_path = None
    prefix = f"fakinflu/{inp.get('run_id') or job.get('id', 'run')}"
    t0 = time.time()
    yield version_event("body3d")  # toujours en premier : le client verifie la version de l'image
    try:
        outputs = Outputs(inp)
        paths, events = fetch_inputs(inp, os.path.join(work, "in"), outputs)
        yield from events
        if "video" not in paths:
            raise ValueError("entree 'video' manquante")
        meta = ffprobe(paths["video"])
        yield {"type": "progress", "step": f"video {meta['width']}x{meta['height']} {meta['fps']:.2f} fps, {meta['frames']} images", "pct": 2}

        in_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(paths['video'])}"
        in_path = os.path.join(COMFY_DIR, "input", in_name)
        shutil.copy(paths["video"], in_path)
        wait_comfy()
        wf = build_workflow(params.get("preset") or {}, in_name, prefix, meta)
        seconds = yield from run_workflow(wf)

        produced = collect(prefix)
        if "overlay" in produced:
            yield preview_event(thumb(produced["overlay"]), "image/jpeg")
        missing = [r for r in ("mhr_params", "qa", "camera") if r not in produced]
        if missing:
            raise RuntimeError(f"sorties manquantes : {missing} (produites : {sorted(produced)})")
        for role, path in produced.items():
            yield from outputs.emit(role, path, OUTPUT_NAMES.get(role, os.path.basename(path)))
        outputs.write_summary()
        yield {"type": "done", "seed": None,
               "timings": {"comfy_s": round(seconds, 1), "total_s": round(time.time() - t0, 1)}}
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(os.path.join(COMFY_DIR, "output", prefix), ignore_errors=True)
        if in_path and os.path.exists(in_path):
            os.remove(in_path)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
