#!/usr/bin/env bash
set -euo pipefail

# Compute future optical-flow videos for physical-intelligence/libero.
# LIBERO image fields in this dataset are: image, wrist_image.

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ID="${REPO_ID:-physical-intelligence/libero}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/libero_flow_step10}"
FUTURE_STEP="${FUTURE_STEP:-10}"
DEVICE="${DEVICE:-cuda}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" scripts/compute_lerobot_future_flow_video.py \
  --repo-id "${REPO_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --image-keys image wrist_image \
  --stream-names image wrist_image \
  --future-step "${FUTURE_STEP}" \
  --device "${DEVICE}" \
  --video-backend "${VIDEO_BACKEND}" \
  --overwrite \
  "$@"
