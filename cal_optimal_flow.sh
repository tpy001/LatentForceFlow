export UV_PROJECT_ENVIRONMENT=/ext_workspace/tpy/code/ustc_openpi_clean/env/.venv_007

uv run python scripts/compute_lerobot_future_flow_video.py \
  --repo-id tpy/forge_all_0413 \
  --output-dir ./forge_all_0413 \
  --future-step 32 \
  --overwrite

