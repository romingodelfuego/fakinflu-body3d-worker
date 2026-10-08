#!/usr/bin/env bash
# fakinflu — poids de l'etage d'extraction 3D (plan motion, Phase 1), BAKES dans l'image.
# Depots Comfy-Org (reconditionnements ComfyUI), EPINGLES sur un commit, chaque fichier verifie
# par son sha256 (= oid LFS lu via l'API HF le 2026-10-08).
#   SAM 3D Body (dinov3)  : SAM License (Meta)            -> detection/
#   SAM 3.1 multiplex     : SAM License (Meta)            -> checkpoints/   (suivi video)
#   MoGe-2 ViT-L normal   : MIT                           -> geometry_estimation/ (champ de vue)
#
#   MODELS_DIR=/comfyui/models ./download_models.sh core   # ~5,2 Go (bf16)
#   MODELS_DIR=/comfyui/models ./download_models.sh int8   # option : SAM 3D Body int8 (2 Go)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/comfyui/models}"

# DL <depot> <commit> <sous-dossier> <fichier> <sha256>
DL() {
  local repo="$1" rev="$2" sub="$3" name="$4" sha="$5" dest="$MODELS_DIR/$3/$4"
  local url="https://huggingface.co/${repo}/resolve/${rev}/${sub}/${name}"
  mkdir -p "$MODELS_DIR/$sub"
  echo ">> $sub/$name"
  local t0=$SECONDS
  if command -v aria2c >/dev/null; then
    aria2c -q -c -x 16 -s 16 -k 64M --max-tries=10 --retry-wait=5 --timeout=30 \
           --lowest-speed-limit=1M --file-allocation=none --console-log-level=warn \
           -d "$MODELS_DIR/$sub" -o "$name" "$url"
  else
    wget -q -c --tries=10 --timeout=30 --waitretry=5 -O "$dest" "$url"
  fi
  echo "   telecharge en $((SECONDS - t0)) s"
  echo "$sha  $dest" | sha256sum -c --quiet - || { echo "!! sha256 invalide : $dest"; rm -f "$dest"; exit 1; }
}

SAM3D=7c974aea58e062252a6cb5b6f9b7f639d545b729
SAM31=7bb8374780a725b4353ed31f3a9395c9742b5621
MOGE=1484985258ba7ef45c314144fcf0e90924ffbf44

case "${1:-}" in
  core)
    DL Comfy-Org/sam-3d-body "$SAM3D" detection sam_3d_body_dinov3_bf16.safetensors \
       59fa45200c504c5b56625004d7d3385daf48c616613e88099e43bf83b3e249cf
    DL Comfy-Org/sam3.1 "$SAM31" checkpoints sam3.1_multiplex_fp16.safetensors \
       9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03
    DL Comfy-Org/MoGe "$MOGE" geometry_estimation moge_2_vitl_normal_fp16.safetensors \
       cb1a692d03235671e959e81360d7b4d9f44aefadb1f852d6ca6aa17799d5e31f ;;
  int8)
    DL Comfy-Org/sam-3d-body "$SAM3D" detection sam_3d_body_dinov3_int8_convrot.safetensors \
       ba8d4c34a81bc072b3f53db5241074d370b26e3b72ae793630be5344efd28934 ;;
  *) echo "usage: $0 core|int8" >&2; exit 2 ;;
esac
