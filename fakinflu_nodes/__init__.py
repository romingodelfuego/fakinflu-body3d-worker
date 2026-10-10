"""fakinflu — noeuds ComfyUI de l'etage d'extraction 3D (plan motion 1.2).

Les noeuds natifs SAM 3D Body gardent les parametres MHR bruts EN MEMOIRE (type MHR_POSE_DATA)
et n'exportent que GLB / BVH / rendus. Ce noeud ecrit dans <output>/<prefix>/ :

  mhr_params.npz   parametres MHR par image (personne 0) : source de verite pour la suite
  camera.json      focale, champ de vue, taille d'image, camera fixe ou mobile
  qa.json          score par image : detection, saut entre images, membres hors champ
  masks.npz        masques de la personne (suivi SAM3), bits compresses, si disponibles

v1 : une seule personne (la premiere de chaque image).
"""
import json
import os

import numpy as np

import folder_paths

# Champs MHR conserves tels quels (tableaux par image, empiles sur l'axe 0)
PARAM_KEYS = [
    "mhr_model_params", "shape_params", "expr_params", "body_pose_params", "hand_pose_params",
    "scale_params", "global_rot", "pred_cam_t", "focal_length",
    "pred_keypoints_2d", "pred_keypoints_3d", "pred_joint_coords", "pred_global_rots", "bbox",
]

# Seuils QA (ajustables via le preset : qa.jump_m, qa.out_of_frame)
JUMP_M = 0.25          # deplacement median des articulations entre 2 images (metres, repere camera)
OUT_OF_FRAME = 0.20    # part des points 2D hors image


def _np(v):
    if v is None:
        return None
    if hasattr(v, "detach"):
        v = v.detach().float().cpu().numpy()
    return np.asarray(v)


def _camera_motion(frames_bgr, masks):
    """Mouvement median de l'arriere-plan (px/image) par flot optique hors du masque de la personne."""
    try:
        import cv2
    except ImportError:
        return None
    moves = []
    prev = None
    for i, img in enumerate(frames_bgr):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            bg = np.ones_like(g, dtype=np.uint8)
            if masks is not None and masks[i] is not None:
                m = cv2.dilate(masks[i].astype(np.uint8), np.ones((25, 25), np.uint8))
                bg = (1 - m).astype(np.uint8)
            pts = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8, mask=bg)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None)
                ok = st.ravel() == 1
                if ok.sum() >= 12:
                    H, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok])
                    if H is not None:
                        moves.append(float(np.hypot(H[0, 2], H[1, 2])))
        prev = g
    if not moves:
        return None
    return float(np.median(moves))


class FakinfluExportMHR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mhr_pose_data": ("MHR_POSE_DATA",),
                "filename_prefix": ("STRING", {"default": "fakinflu/motion"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
                "jump_m": ("FLOAT", {"default": JUMP_M, "min": 0.01, "max": 5.0, "step": 0.01}),
                "out_of_frame": ("FLOAT", {"default": OUT_OF_FRAME, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fov_deg": ("FLOAT", {"forceInput": True}),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "fakinflu"

    def export(self, mhr_pose_data, filename_prefix, fps, jump_m, out_of_frame, fov_deg=None, images=None):
        out_dir = os.path.join(folder_paths.get_output_directory(), filename_prefix)
        os.makedirs(out_dir, exist_ok=True)
        frames = mhr_pose_data["frames"]
        H, W = mhr_pose_data["image_size"]
        n = len(frames)
        detected = np.array([len(f) > 0 for f in frames], dtype=bool)

        # ---- parametres MHR (personne 0), images sans detection remplies de NaN
        arrays = {}
        for key in PARAM_KEYS:
            ref = next((_np(f[0].get(key)) for f in frames if f and f[0].get(key) is not None), None)
            if ref is None:
                continue
            stack = np.full((n,) + ref.shape, np.nan, dtype=np.float32)
            for i, f in enumerate(frames):
                if f and f[0].get(key) is not None:
                    stack[i] = _np(f[0][key])
            arrays[key] = stack
        np.savez_compressed(os.path.join(out_dir, "mhr_params.npz"), detected=detected, fps=np.float32(fps),
                            image_size=np.array([H, W], dtype=np.int32), **arrays)

        # ---- masques (suivi SAM3) : bits compresses. SAM 3.1 livre (H, W, 1) ou (1, H, W) selon les
        # versions : on retire les dimensions de taille 1 ; une forme inattendue ignore le masque
        # (avertissement) au lieu de faire echouer l'export.
        masks = []
        for i, f in enumerate(frames):
            m = _np(f[0].get("mask")) if f else None
            if m is not None:
                m = np.squeeze(m)
                if m.shape != (H, W):
                    print(f"[fakinflu] masque image {i} ignore : forme {m.shape} != {(H, W)}")
                    m = None
                else:
                    m = m > 0.5
            masks.append(m)
        if any(m is not None for m in masks):
            full = np.zeros((n, H, W), dtype=bool)
            for i, m in enumerate(masks):
                if m is not None:
                    full[i] = m
            np.savez_compressed(os.path.join(out_dir, "masks.npz"),
                                bits=np.packbits(full, axis=-1), shape=np.array(full.shape, dtype=np.int32))

        # ---- QA par image
        kp3 = arrays.get("pred_keypoints_3d")
        cam_t = arrays.get("pred_cam_t")
        kp2 = arrays.get("pred_keypoints_2d")
        qa_frames = []
        for i in range(n):
            q = {"frame": i, "detected": bool(detected[i]), "jump_m": None, "out_of_frame": None, "flags": []}
            if not detected[i]:
                q["flags"].append("no_detection")
            if kp3 is not None and cam_t is not None and i > 0 and detected[i] and detected[i - 1]:
                a = kp3[i] + cam_t[i][None, :]
                b = kp3[i - 1] + cam_t[i - 1][None, :]
                q["jump_m"] = round(float(np.nanmedian(np.linalg.norm(a - b, axis=-1))), 4)
                if q["jump_m"] > jump_m:
                    q["flags"].append("jump")
            if kp2 is not None and detected[i]:
                p = kp2[i]
                outside = (p[:, 0] < 0) | (p[:, 0] >= W) | (p[:, 1] < 0) | (p[:, 1] >= H)
                q["out_of_frame"] = round(float(outside.mean()), 3)
                if q["out_of_frame"] > out_of_frame:
                    q["flags"].append("out_of_frame")
            qa_frames.append(q)
        bad = [q["frame"] for q in qa_frames if q["flags"]]
        qa = {"frames": qa_frames, "summary": {
            "n_frames": n, "detected": int(detected.sum()), "flagged": len(bad),
            "flagged_frames": bad, "thresholds": {"jump_m": jump_m, "out_of_frame": out_of_frame}}}
        with open(os.path.join(out_dir, "qa.json"), "w") as fh:
            json.dump(qa, fh)

        # ---- camera : focale, FOV, fixe ou mobile
        focal = arrays.get("focal_length")
        motion_px = None
        if images is not None:
            step = max(1, n // 60)  # ~60 images suffisent ; sous-echantillonner AVANT de convertir (sinon copie de toute la video)
            imgs = (images[::step].detach().float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            motion_px = _camera_motion(list(imgs), masks[::step])
        cam = {
            "image_size": [int(H), int(W)], "fps": fps,
            "fov_deg": None if fov_deg is None else float(fov_deg),
            "fov_source": "moge" if fov_deg is not None else "default",
            "focal_length_px": None if focal is None else float(np.nanmedian(focal)),
            "background_motion_px": motion_px,
            # caméra fixe : moins de 1,5 px de mouvement median du fond entre images echantillonnees
            "static": None if motion_px is None else bool(motion_px < 1.5),
        }
        with open(os.path.join(out_dir, "camera.json"), "w") as fh:
            json.dump(cam, fh)
        return {"ui": {"text": [json.dumps(qa["summary"])]}}


class FakinfluRenderVideo:
    """SAM3DBody_Render par paquets d'images, encode en H.264 au fil de l'eau.

    Le noeud natif empile tout le rendu en float32 (liste + torch.stack : 2 copies de la video),
    puis CreateVideo/SaveVideo le gardent en cache : ~11 Go par rendu pour 1100 images en 632x1280,
    trois rendus -> OOM du conteneur. Ici la memoire reste O(paquet) : chaque paquet est rendu par
    le noeud natif (meme resultat), converti en uint8 et pousse dans ffmpeg.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mhr_pose_data": ("MHR_POSE_DATA",),
                "filename": ("STRING", {"default": "fakinflu/render.mp4"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                # render_style deja imbrique comme le recoit SAM3DBody_Render.execute (JSON)
                "render_style": ("STRING", {"default": '{"render_style": "mesh"}'}),
                "chunk": ("INT", {"default": 32, "min": 1, "max": 512}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
            },
            "optional": {"background": ("IMAGE",)},
        }

    RETURN_TYPES = ()
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "fakinflu"

    def render(self, mhr_pose_data, filename, fps, width, height, render_style, chunk, crf, background=None):
        import subprocess
        import torch
        import comfy.utils
        from comfy_extras.nodes_sam3d_body import SAM3DBody_Render

        style = json.loads(render_style)
        frames = mhr_pose_data["frames"]
        n = len(frames)
        path = os.path.join(folder_paths.get_output_directory(), filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # nom temporaire hors des motifs de sortie : le handler n'envoie que des fichiers termines
        tmp = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".tmp")
        log = tmp + ".log"
        pbar = comfy.utils.ProgressBar(n)
        hook = comfy.utils.PROGRESS_BAR_HOOK
        proc, errf = None, None

        def ffmpeg_error():
            errf.flush()
            with open(log, "rb") as fh:
                return RuntimeError(f"ffmpeg ({filename}) : {fh.read().decode(errors='replace')[-800:]}")

        try:
            for a in range(0, n, chunk):
                part = dict(mhr_pose_data)
                part["frames"] = frames[a:a + chunk]  # seule cle par image du pose_data
                bg = None
                if background is not None:
                    i = min(a, background.shape[0] - 1)
                    bg = background[i:i + len(part["frames"])]
                comfy.utils.set_progress_bar_global_hook(None)  # la barre du noeud natif repartirait de 0 a chaque paquet
                try:
                    img = SAM3DBody_Render.execute(pose_data=part, background=bg, width=width, height=height,
                                                   render_style=style).args[0]
                finally:
                    comfy.utils.set_progress_bar_global_hook(hook)
                u8 = (img.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
                del img
                if proc is None:
                    h, w = u8.shape[1:3]
                    errf = open(log, "wb")
                    proc = subprocess.Popen(
                        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                         "-r", f"{float(fps):.6f}", "-i", "-", "-an",
                         # libx264 en yuv420p exige des dimensions paires
                         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", "-f", "mp4", tmp],
                        stdin=subprocess.PIPE, stderr=errf)
                try:
                    proc.stdin.write(u8.tobytes())
                except BrokenPipeError:
                    proc.wait()
                    raise ffmpeg_error()
                pbar.update(len(u8))
            if proc is None:
                raise ValueError("aucune image a rendre")
            proc.stdin.close()
            if proc.wait() != 0:
                raise ffmpeg_error()
            os.replace(tmp, path)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            if errf is not None:
                errf.close()
            for p in (tmp, log):
                if os.path.exists(p):
                    os.remove(p)
        # ui non vide : sinon ComfyUI n'emet pas « executed », que le handler attend pour envoyer le fichier
        return {"ui": {"text": [filename]}}


NODE_CLASS_MAPPINGS = {"FakinfluExportMHR": FakinfluExportMHR, "FakinfluRenderVideo": FakinfluRenderVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"FakinfluExportMHR": "fakinflu — Export MHR params + QA",
                              "FakinfluRenderVideo": "fakinflu — Render 3D Body -> MP4 (streamed)"}
