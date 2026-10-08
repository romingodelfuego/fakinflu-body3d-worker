# fakinflu-body3d-worker

Worker RunPod Serverless de l'**extraction 3D du mouvement** (plan « Motion », Phase 1) :
vidéo → SAM 3.1 (suivi de la personne) → MoGe-2 (champ de vue) → SAM 3D Body (+ mains) → lissage.

- Base `runpod/worker-comfyui:5.10.0-base` = ComfyUI **0.34.0** : tous les nœuds sont natifs
  (`SAM3DBody_*`, `BuildPoseFile`, `SAM3_VideoTrack`, `MoGe*`). Le Dockerfile échoue au build s'ils manquent.
- `fakinflu_nodes/` : un seul nœud maison, `FakinfluExportMHR`, car les nœuds natifs gardent les
  paramètres MHR bruts en mémoire. Il écrit `mhr_params.npz`, `qa.json`, `camera.json` (FOV, focale,
  **caméra fixe ou mobile** par flot optique hors du masque) et `masks.npz`.
- `workflow.py` : graphe ComfyUI (format API) construit depuis le preset `presets/extract/*.json`,
  dérivé du template officiel `utility_sam3d_body`. Vérifié contre ComfyUI 0.34.0 : graphe accepté,
  et chaque clé `DynamicCombo` (`render_style.*`, `format.*`) bien transmise au nœud.
- `handler.py` : handler **générateur** (progression nœud par nœud et aperçu JPEG de l'overlay via `/stream`),
  sorties routées par `transfer.py` (copie identique de `r2smoke/serverless/transfer.py`).
- Poids bakés (~5,2 Go, commits Hugging Face épinglés, sha256 vérifiés) : `download_models.sh`.

| Rôle | Fichier client (`data/work/<id>/motion/`) |
|---|---|
| `mhr_params` | `mhr_params.npz` — paramètres MHR par image (personne 0), NaN si non détectée |
| `camera` / `qa` | `camera.json`, `qa.json` |
| `masks` | `masks.npz` — `bits` (packbits) + `shape` |
| `motion_glb` / `motion_bvh` | `motion.glb`, `motion.bvh` (visualisation, Blender) |
| `overlay` | `overlay.mp4` — maillage sur la vidéo (QA humaine) |
| `control_mesh` / `control_scail_pose` | `control_mesh.mp4`, `control_pose.mp4` — contrôle SCAIL-2 |

Côté client : `make ingest FILE=…|URL=…` puis `make motion-extract ID=<id> PRESET=fast_motion [DRY=1]`.

## Licences
- **SAM 3D Body, SAM 3.1** : SAM License (Meta) — à valider avant tout usage commercial.
- MoGe : MIT. Code de ce dépôt : voir le dépôt fakinflu.
