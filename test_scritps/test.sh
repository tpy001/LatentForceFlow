CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_jaka_tavla_0409 --policy.dir=/home/tpy/LatentForceFlow/checkpoints/pi0_jaka_tavla_0409/pi0_jaka_tavla_0409/29999

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_jaka_tavla_0409_depth --policy.dir=checkpoints/pi0_jaka_tavla_0409_depth/pi0_jaka_tavla_0409_depth_no_force_pred/29999


CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_jaka_forceWAM_depth --policy.dir=checkpoints/pi0_jaka_forceWAM_depth/pi0_jaka_forceWAM_depth/29999

CUDA_VISIBLE_DEVICES=4 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_latent_flow_depth_teachers_libero --policy.dir=checkpoints/pi05_latent_flow_depth_teachers_libero/debug2/29999

CUDA_VISIBLE_DEVICES=4 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_libero --policy.dir=checkpoints/pi0_libero_0526/29999

CUDA_VISIBLE_DEVICES=4 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/pi05_libero_new/29999