#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/ext_workspace/tpy/code/ustc_openpi_clean/env/.venv_007}
CONFIG_NAME=pi05_piper_gripper2_lora
EXP_NAME=${EXP_NAME:-piper_gripper2_lora_15k}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} .venv/bin/python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95} \
.venv/bin/python scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --overwrite
