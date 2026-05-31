export UV_PROJECT_ENVIRONMENT=/ext_workspace/tpy/code/ustc_openpi_clean/env/.venv_007

CUDA_VISIBLE_DEVICES=5 uv run scripts/compute_norm_stats.py --config-name pi0_latent_flow_noise


CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_latent_flow_depth_teachers_libero --exp-name=pi05_latent_flow_depth_teachers_libero --overwrite

# 多卡训练
CUDA_VISIBLE_DEVICES=4,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_seer_0409 --exp-name=pi0_seer_0409  --fsdp-devices 2   --overwrite 


# 训练 probe
CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/train_future_query_probe.py \
  --config-name pi0_latent_flow_noise \
  --exp-name probe_from_30k_2gpu \
  --pretrained-params checkpoints/pi0_latent_force_flow_noise/pi0_latent_flow_noise_0428/29999/params \
  --batch-size 16 \
  --probe-layer 12 \
  --overwrite

