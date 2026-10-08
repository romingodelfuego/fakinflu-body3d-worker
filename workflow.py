"""fakinflu — graphe ComfyUI (format API) de l'extraction 3D, construit depuis le preset (plan 1.3).

Derive du template officiel Comfy-Org « utility_sam3d_body » et valide par ComfyUI 0.34.0 lui-meme
(`python3 workflow.py --validate URL`). Les entrees DynamicCombo s'ecrivent en cles pointees :
"render_style": "mesh", "render_style.shader": "default", ...

  video -> composantes (images, fps)
        -> SAM 3.1 suivi video (texte « person »)          -> track_data (masques + boites)
        -> MoGe-2 sur quelques images -> FOV vertical       (si camera.fov = moge)
        -> SAM 3D Body Predict (+ mains) -> Smooth           -> MHR_POSE_DATA
        -> FakinfluExportMHR : mhr_params.npz, qa.json, camera.json, masks.npz
        -> BuildPoseFile -> SaveGLB : motion .glb / .bvh
        -> Render mesh / scail (+ overlay sur la video) -> CreateVideo -> SaveVideo
"""
import json, sys

# role -> (motif sous output/<prefix>/, nom final cote client)
OUTPUT_FILES = {
    "mhr_params": "mhr_params.npz",
    "camera": "camera.json",
    "qa": "qa.json",
    "masks": "masks.npz",
    "motion_glb": "motion_*.glb",
    "motion_bvh": "motion_*.bvh",
    "overlay": "overlay_*.mp4",
    "control_mesh": "control_mesh_*.mp4",
    "control_scail_pose": "control_scail_pose_*.mp4",
}
OUTPUT_NAMES = {
    "mhr_params": "mhr_params.npz", "camera": "camera.json", "qa": "qa.json", "masks": "masks.npz",
    "motion_glb": "motion.glb", "motion_bvh": "motion.bvh", "overlay": "overlay.mp4",
    "control_mesh": "control_mesh.mp4", "control_scail_pose": "control_pose.mp4",
}

SAM3D_FILES = {"bf16": "sam_3d_body_dinov3_bf16.safetensors", "int8": "sam_3d_body_dinov3_int8_convrot.safetensors"}
MOGE_FRAMES = 8  # le FOV d'une camera fixe n'a pas besoin de toute la video


def _render_size(meta, resolution):
    """Cote court = resolution (multiple de 8), l'autre cote derive par le noeud (0)."""
    w, h = meta.get("width") or 0, meta.get("height") or 0
    r = int(resolution) // 8 * 8
    if not (w and h) or min(w, h) <= r:
        return 0, 0
    return (0, r) if h <= w else (r, 0)


def _save_video(wf, nid, images, fps, prefix):
    wf[nid] = {"class_type": "CreateVideo", "inputs": {"images": images, "fps": fps}}
    wf[nid + "s"] = {"class_type": "SaveVideo", "_meta": {"title": f"enregistrement {prefix.rsplit('/', 1)[-1]}"},
                     "inputs": {"video": [nid, 0], "filename_prefix": prefix,
                                "format": "mp4", "format.codec": "h264",
                                "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 18.0}}


def build_workflow(preset, video_name, prefix, meta=None):
    p = preset or {}
    meta = meta or {}
    target, tracking = p.get("target", {}), p.get("tracking", {})
    predict, camera = p.get("predict", {}), p.get("camera", {})
    smoothing, render = p.get("smoothing", {}), p.get("render", {})
    export, qa = p.get("export", {"glb": True, "bvh": True}), p.get("qa", {})
    if target.get("mode", "auto_largest") != "auto_largest":
        raise ValueError(f"target.mode={target.get('mode')} pas encore supporte en v1 (auto_largest seulement)")

    images, fps = ["video", 0], ["video", 2]
    wf = {
        "load": {"class_type": "LoadVideo", "_meta": {"title": "lecture video"}, "inputs": {"file": video_name}},
        "video": {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}},
        # --- suivi SAM 3.1 de la personne (masques + boites par image)
        "sam3": {"class_type": "CheckpointLoaderSimple", "_meta": {"title": "chargement SAM 3.1"},
                 "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
        "sam3_text": {"class_type": "CLIPTextEncode", "inputs": {"text": target.get("text", "person"), "clip": ["sam3", 1]}},
        "track": {"class_type": "SAM3_VideoTrack", "_meta": {"title": "suivi SAM 3.1"},
                  "inputs": {"images": images, "model": ["sam3", 0], "conditioning": ["sam3_text", 0],
                             "detection_threshold": float(tracking.get("threshold", 0.5)),
                             "max_objects": 1,  # v1 : une seule personne
                             # re-detection a chaque image si on accepte la reacquisition, sinon toutes les 30
                             "detect_interval": 1 if tracking.get("reacquire_on_loss", True) else 30}},
        # --- SAM 3D Body
        "body_model": {"class_type": "SAM3DBody_Loader", "_meta": {"title": "chargement SAM 3D Body"},
                       "inputs": {"model_file": SAM3D_FILES[predict.get("precision", "bf16")]}},
        "predict": {"class_type": "SAM3DBody_Predict", "_meta": {"title": "prediction SAM 3D Body"},
                    "inputs": {"sam3d_body_model": ["body_model", 0], "image": images, "track_data": ["track", 0],
                               "run_hand_refinement": bool(predict.get("hands", True)), "batch_size": 64}},
    }

    # --- champ de vue : MoGe sur les premieres images, ou valeur fixe
    if camera.get("fov", "moge") == "moge":
        wf["moge_model"] = {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge_2_vitl_normal_fp16.safetensors"}}
        wf["moge_frames"] = {"class_type": "ImageFromBatch", "inputs": {"image": images, "batch_index": 0, "length": MOGE_FRAMES}}
        wf["moge"] = {"class_type": "MoGeInference", "_meta": {"title": "champ de vue (MoGe)"},
                      "inputs": {"moge_model": ["moge_model", 0], "image": ["moge_frames", 0], "resolution_level": 9,
                                 "fov_x_degrees": 0.0, "batch_size": 4, "force_projection": True, "apply_mask": True}}
        wf["fov"] = {"class_type": "MoGeGeometryToFOV", "inputs": {"moge_geometry": ["moge", 0], "axis": "vertical", "unit": "degrees"}}
        fov = ["fov", 0]
    else:
        fov = float(camera.get("fixed_fov_deg") or 0.0)  # 0 = repli ~53° du noeud
    wf["predict"]["inputs"]["fov"] = fov

    pose = ["predict", 0]
    if smoothing.get("enabled", True) and float(smoothing.get("strength", 0)) > 0:
        wf["smooth"] = {"class_type": "SAM3DBody_Smooth", "_meta": {"title": "lissage"},
                        "inputs": {"mhr_pose_data": pose, "strength": float(smoothing["strength"]),
                                   "method": smoothing.get("method", "savgol"), "window": int(smoothing.get("window", 7)),
                                   "rotation_threshold_degrees": 30.0}}
        pose = ["smooth", 0]

    export_inputs = {"mhr_pose_data": pose, "filename_prefix": prefix, "fps": fps, "images": images,
                     "jump_m": float(qa.get("jump_m", 0.25)), "out_of_frame": float(qa.get("out_of_frame", 0.2))}
    if isinstance(fov, list):
        export_inputs["fov_deg"] = fov
    wf["export"] = {"class_type": "FakinfluExportMHR", "_meta": {"title": "export MHR + QA"}, "inputs": export_inputs}

    # --- fichiers 3D (visualisation : Blender, inspecteur)
    if export.get("glb", True):
        wf["glb"] = {"class_type": "BuildPoseFile", "_meta": {"title": "export GLB"},
                     "inputs": {"pose_data": pose, "sam3d_body_model": ["body_model", 0], "fps": fps,
                                "camera_translation": "absolute", "track_index": 0,
                                "format": "glb", "format.mesh_style": "body_mesh",
                                "format.mesh_style.bone_vis": "off", "format.mesh_style.shader": "default",
                                "format.bone_smooth_window": 0}}
        wf["glb_save"] = {"class_type": "SaveGLB", "inputs": {"mesh": ["glb", 0], "filename_prefix": f"{prefix}/motion"}}
    if export.get("bvh", True):
        wf["bvh"] = {"class_type": "BuildPoseFile", "_meta": {"title": "export BVH"},
                     "inputs": {"pose_data": pose, "sam3d_body_model": ["body_model", 0], "fps": fps,
                                "camera_translation": "absolute", "track_index": 0,
                                "format": "bvh", "format.units": "cm"}}
        wf["bvh_save"] = {"class_type": "SaveGLB", "inputs": {"mesh": ["bvh", 0], "filename_prefix": f"{prefix}/motion"}}

    # --- rendus : controle SCAIL-2 (camera source) + overlay de QA
    width, height = _render_size(meta, render.get("resolution", 704))
    mesh_style = {"render_style": "mesh", "render_style.shader": "default", "render_style.opacity": 1.0,
                  "render_style.person_palette_falloff": 0.6, "render_style.region": "full_body"}
    if "mesh" in render.get("control", []):
        wf["r_mesh"] = {"class_type": "SAM3DBody_Render", "_meta": {"title": "rendu controle maillage"},
                        "inputs": {"pose_data": pose, "width": width, "height": height, **mesh_style}}
        _save_video(wf, "v_mesh", ["r_mesh", 0], fps, f"{prefix}/control_mesh")
    if "scail_pose" in render.get("control", []):
        wf["r_scail"] = {"class_type": "SAM3DBody_Render", "_meta": {"title": "rendu controle SCAIL"},
                         "inputs": {"pose_data": pose, "width": width, "height": height,
                                    "render_style": "scail", "render_style.radius_m": 0.022,
                                    "render_style.hand_style": "dwpose", "render_style.face_style": "disabled",
                                    "render_style.person_palette_falloff": 0.6}}
        _save_video(wf, "v_scail", ["r_scail", 0], fps, f"{prefix}/control_scail_pose")
    if render.get("overlay", True):
        wf["r_overlay"] = {"class_type": "SAM3DBody_Render", "_meta": {"title": "rendu overlay (QA)"},
                           "inputs": {"pose_data": pose, "background": images, "width": width, "height": height,
                                      **mesh_style, "render_style.opacity": 0.6}}
        _save_video(wf, "v_overlay", ["r_overlay", 0], fps, f"{prefix}/overlay")
    return wf


def _validate(url, preset_path, video_name):
    """Soumet le graphe a un ComfyUI (local, 0 credit) : il le valide avant toute execution."""
    import urllib.error, urllib.request
    preset = json.load(open(preset_path))
    wf = build_workflow(preset, video_name, "fakinflu/validate", {"width": 720, "height": 1280})
    req = urllib.request.Request(url + "/prompt", data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        print(e.read().decode()[:4000])
        sys.exit(1)
    # accepte -> on vide la file tout de suite (les modeles locaux sont des fichiers vides)
    for path in ("/interrupt", "/queue"):
        body = {} if path == "/interrupt" else {"clear": True}
        urllib.request.urlopen(urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=10)
    print(f"OK : graphe accepte par ComfyUI ({len(wf)} noeuds), prompt_id={r.get('prompt_id')}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--validate":
        _validate(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "validate.mp4")
    else:
        print(json.dumps(build_workflow(json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}, "input.mp4", "fakinflu/run"), indent=1))
