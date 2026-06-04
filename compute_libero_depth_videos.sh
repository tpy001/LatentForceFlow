#!/usr/bin/env bash
set -euo pipefail

# Compute Depth Anything videos for the two LIBERO camera views in a LeRobot dataset.
#
# Usage:
#   ./compute_libero_depth_videos.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0 \
#   REPO_ID=physical-intelligence/libero \
#   OUTPUT_DIR=depth_videos/libero \
#   EPISODES="0 1 2" \
#   LIMIT_FRAMES=100 \
#   ./compute_libero_depth_videos.sh --overwrite --save-npy

PYTHON_BIN="${PYTHON_BIN:-env/.venv_006/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
REPO_ID="${REPO_ID:-physical-intelligence/libero}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/libero_depth_videos}"
MODEL="${MODEL:-depth-anything/Depth-Anything-V2-Base-hf}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
FPS="${FPS:-}"
EPISODES="${EPISODES:-}"
MAX_EPISODES="${MAX_EPISODES:-}"
LIMIT_FRAMES="${LIMIT_FRAMES:-}"
COLORMAP="${COLORMAP:-inferno}"
CPU_THREADS="${CPU_THREADS:-4}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
OPENCV_THREADS="${OPENCV_THREADS:-2}"

cmd=(
  "${PYTHON_BIN}" scripts/compute_lerobot_depth_video.py
  --repo-id "${REPO_ID}"
  --output-dir "${OUTPUT_DIR}"
  --image-keys
    image
    wrist_image
  --stream-names
    image
    wrist_image
  --model "${MODEL}"
  --device "${DEVICE}"
  --video-backend "${VIDEO_BACKEND}"
  --colormap "${COLORMAP}"
  --torch-threads "${CPU_THREADS}"
  --torch-interop-threads "${TORCH_INTEROP_THREADS}"
  --opencv-threads "${OPENCV_THREADS}"
)

if [[ -n "${DTYPE}" ]]; then
  cmd+=(--dtype "${DTYPE}")
fi

if [[ -n "${FPS}" ]]; then
  cmd+=(--fps "${FPS}")
fi

if [[ -n "${MAX_EPISODES}" ]]; then
  cmd+=(--max-episodes "${MAX_EPISODES}")
fi

if [[ -n "${LIMIT_FRAMES}" ]]; then
  cmd+=(--limit-frames "${LIMIT_FRAMES}")
fi

if [[ -n "${EPISODES}" ]]; then
  # shellcheck disable=SC2206
  episode_args=(${EPISODES})
  cmd+=(--episodes "${episode_args[@]}")
fi

cmd+=("$@")

echo "Running:"
printf ' %q' "${cmd[@]}"
echo

OMP_NUM_THREADS="${CPU_THREADS}" \
MKL_NUM_THREADS="${CPU_THREADS}" \
OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${cmd[@]}"
