# fakinflu — worker RunPod : body3d (extraction 3D du mouvement, SAM 3D Body via ComfyUI). Plan motion, Phase 1.
# Build par RunPod (integration GitHub) a chaque GitHub Release : make worker-release W=body3d
#
# Base : runpod/worker-comfyui 5.10.0-base = ComfyUI 0.34.0 (5.11.0 n existe pas sur Docker Hub) (torch cu128, ffmpeg, PyAV). Les noeuds
# SAM 3D Body (Loader/Predict/Smooth/Render, BuildPoseFile), SAM 3.1 (VideoTrack) et MoGe y sont
# NATIFS depuis 0.34.0 : aucun custom node tiers. Seul ajout : fakinflu_nodes (export MHR + QA).
ARG WORKER_TAG=5.10.0-base
FROM runpod/worker-comfyui:${WORKER_TAG}

# Garde-fou de parite de version : echoue ICI si ComfyUI n'a pas les noeuds attendus.
RUN grep -q '"SAM3DBody_Predict"' /comfyui/comfy_extras/nodes_sam3d_body.py \
 && grep -q '"BuildPoseFile"' /comfyui/comfy_extras/nodes_sam3d_body.py \
 && grep -q '"SAM3_VideoTrack"' /comfyui/comfy_extras/nodes_sam3.py \
 && grep -q '"MoGeGeometryToFOV"' /comfyui/comfy_extras/nodes_moge.py

# OpenCV (detection camera fixe/mobile dans fakinflu_nodes) ; aria2 pour des telechargements rapides
RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "opencv-python-headless==4.10.0.84"

# --- Poids BAKES (~5,2 Go ; WITH_INT8=1 ajoute SAM 3D Body int8, 2 Go) -------
ARG WITH_INT8=0
COPY download_models.sh /tmp/download_models.sh
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh core
RUN if [ "${WITH_INT8}" = "1" ]; then MODELS_DIR=/comfyui/models bash /tmp/download_models.sh int8; fi \
 && rm -f /tmp/download_models.sh

COPY fakinflu_nodes /comfyui/custom_nodes/fakinflu_nodes

# --- Handler fakinflu (remplace /handler.py de worker-comfyui) ---------------
COPY transfer.py workflow.py /fakinflu/
COPY handler.py /handler.py
COPY test_input.json /test_input.json
ENV PYTHONPATH=/fakinflu \
    COMFY_LOG_LEVEL=INFO

CMD ["/start.sh"]
