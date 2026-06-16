#!/usr/bin/env bash
set -euo pipefail
CONFIG_NAME=pi05_piper_gripper2_lora
EXP_NAME=${EXP_NAME:-piper_0616}

CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

CUDA_VISIBLE_DEVICES=6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py  "${CONFIG_NAME}" --exp-name="${EXP_NAME}"  --fsdp-devices 2   --overwrite 

