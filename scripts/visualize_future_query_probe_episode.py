"""Visualize future-query probe predictions on one LeRobot episode."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any

import einops
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from flax import nnx
from openpi_client import image_tools

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ImportError:
    lerobot_dataset = None

import infer_future_query_probe as _infer
import train_future_query_probe as _probe_train
from openpi.models import gemma as _gemma
from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_latent_flow_noise"
    repo_id: str = "llly/all_0409_stage_flow"
    probe_checkpoint_dir: str = "checkpoints/future_query_probe/pi0_latent_flow_noise/probe_from_30k_2gpu_/"
    output_dir: str = "outputs/future_probe_episode_vis"
    episode: int = 0
    predict_stride: int = 32
    limit_frames: int | None = None
    video_key: str = "observation.images.head_camera"
    stream_name: str = "base_0_rgb"
    fps: float | None = None
    video_size: int = 224
    arrow_step: int = 10
    arrow_scale: float = 0.8
    arrow_thickness: int = 1
    arrow_min_magnitude: float = 1.5
    arrow_color: tuple[int, int, int] = (0, 255, 255)
    seed: int = 0
    num_workers: int = 0
    pretrained_params: str | None = None
    probe_layer: int | None = None


def _require_cv2():
    if cv2 is None:
        raise ImportError("This script requires opencv-python.")
    return cv2


def _require_lerobot_dataset():
    if lerobot_dataset is None:
        raise ImportError("This script requires lerobot.")
    return lerobot_dataset


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_numpy_image(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}.")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.min() < 0:
            image = (image + 1.0) / 2.0
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image.astype(np.uint8)


def _batch_sample(sample: dict) -> dict:
    def convert_leaf(value):
        array = _to_numpy(value)
        return array[None, ...]

    return jax.tree.map(convert_leaf, sample)


def _episode_ranges(dataset) -> list[tuple[int, int]]:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is None:
        raise ValueError("LeRobotDataset does not expose episode_data_index.")
    starts = np.asarray(episode_data_index["from"], dtype=np.int64)
    ends = np.asarray(episode_data_index["to"], dtype=np.int64)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _videojam_magnitude_scale(height: int, width: int, sigma: float = 0.08) -> float:
    return float(sigma * np.sqrt(height * height + width * width))


def _flow_rgb_to_dxdy(flow_rgb: np.ndarray) -> np.ndarray:
    """Invert the default white-background HSV flow rendering approximately."""
    cv = _require_cv2()
    flow_uint8 = _to_numpy_image(flow_rgb)
    hsv = cv.cvtColor(flow_uint8, cv.COLOR_RGB2HSV)
    angle = hsv[..., 0].astype(np.float32) / 179.0 * (2.0 * np.pi)
    mag_norm = hsv[..., 1].astype(np.float32) / 255.0
    magnitude = mag_norm * _videojam_magnitude_scale(flow_uint8.shape[0], flow_uint8.shape[1])
    return np.stack([magnitude * np.cos(angle), magnitude * np.sin(angle)], axis=-1).astype(np.float32)


def _draw_flow_arrows(
    rgb: np.ndarray,
    flow_rgb: np.ndarray,
    *,
    step: int,
    scale: float,
    thickness: int,
    min_magnitude: float,
    color: tuple[int, int, int],
) -> np.ndarray:
    cv = _require_cv2()
    frame = rgb.copy()
    flow = _flow_rgb_to_dxdy(flow_rgb)
    height, width = frame.shape[:2]
    if flow.shape[:2] != (height, width):
        flow = cv.resize(flow, (width, height), interpolation=cv.INTER_LINEAR)

    half = max(step // 2, 1)
    for y in range(half, height, step):
        for x in range(half, width, step):
            dx, dy = flow[y, x]
            if float(np.hypot(dx, dy)) < min_magnitude:
                continue
            end = (int(round(x + dx * scale)), int(round(y + dy * scale)))
            cv.arrowedLine(frame, (x, y), end, color, thickness, line_type=cv.LINE_AA, tipLength=0.25)
    return frame


def _write_video(path: pathlib.Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames, axis=0), fps=fps, codec="libx264", macro_block_size=1)


def _load_probe_settings(args: Args) -> tuple[pathlib.Path, int, dict]:
    ckpt_dir = pathlib.Path(args.probe_checkpoint_dir).resolve()
    step = _infer._latest_checkpoint_step(ckpt_dir)
    ckpt_args = _infer._load_probe_checkpoint_args(ckpt_dir, step)
    return ckpt_dir, step, ckpt_args


def _init_model_and_probe(args: Args, config: _config.TrainConfig, ckpt_dir: pathlib.Path, ckpt_args: dict):
    probe_layer = int(
        args.probe_layer
        if args.probe_layer is not None
        else ckpt_args.get("probe_layer")
        if ckpt_args.get("probe_layer") is not None
        else config.model.distill_layer_indices[-1]
    )
    pretrained_params = args.pretrained_params or ckpt_args.get("pretrained_params")
    patch_size = int(ckpt_args.get("patch_size", 16))
    decoder_dim = ckpt_args.get("decoder_dim")
    decoder_depth = int(ckpt_args.get("decoder_depth", 2))
    decoder_heads = int(ckpt_args.get("decoder_heads", 8))

    mesh = sharding.make_mesh(num_fsdp_devices=1)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    rng = jax.random.key(args.seed)
    rng, model_rng, probe_rng, infer_rng = jax.random.split(rng, 4)
    model_def, model_params = _probe_train._init_frozen_model(
        config,
        model_rng,
        pretrained_params=pretrained_params,
    )
    model_params = jax.device_put(model_params, replicated_sharding)

    student_width = int(_gemma.get_config(config.model.action_expert_variant).width)
    probe = _probe_train.FutureQueryProbe(
        student_width=student_width,
        flow_token_count=int(config.model.flow_token_count),
        action_horizon=int(config.model.action_horizon),
        effort_dim=int(config.model.effort_dim if config.model.effort_dim is not None else config.model.effort_dim_in),
        decoder_dim=int(decoder_dim or student_width),
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        image_size=224,
        patch_size=patch_size,
        rngs=nnx.Rngs(probe_rng),
    )
    probe_def, probe_params = nnx.split(probe)
    tx = optax.adamw(1e-4)
    probe_state = _probe_train.ProbeTrainState(step=0, params=probe_params, opt_state=tx.init(probe_params), tx=tx)
    probe_state = _probe_train._restore_probe_checkpoint(ckpt_dir, probe_state)
    probe_state = jax.device_put(probe_state, replicated_sharding)
    return mesh, model_def, model_params, probe_def, probe_state, infer_rng, probe_layer


def _make_predict_fn(model_def, probe_def, infer_rng, probe_layer: int):
    @jax.jit
    def predict(frozen_params, probe_state, observation):
        return _infer._predict_and_target(
            model_def,
            frozen_params,
            probe_def,
            probe_state,
            observation,
            infer_rng,
            probe_layer=probe_layer,
        )

    return predict


def _plot_force_comparison(
    output_path: pathlib.Path,
    true_curve: np.ndarray,
    pred_sum: np.ndarray,
    counts: np.ndarray,
    *,
    fps: float,
) -> None:
    valid = counts > 0
    pred_curve = np.full_like(pred_sum, np.nan, dtype=np.float32)
    pred_curve[valid] = pred_sum[valid] / counts[valid, None]

    dim = true_curve.shape[-1]
    x = np.arange(true_curve.shape[0]) / fps
    fig, axes = plt.subplots(dim, 1, figsize=(12, max(2.0 * dim, 5.0)), sharex=True)
    if dim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(x, true_curve[:, i], label="true", linewidth=1.5)
        ax.plot(x, pred_curve[:, i], label="pred", linewidth=1.2, alpha=0.9)
        ax.set_ylabel(f"F{i}")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("episode time (s)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _prediction_starts(frame_count: int, horizon: int, stride: int) -> list[int]:
    last_start = frame_count - horizon - 1
    if last_start < 0:
        return []
    starts = list(range(0, last_start + 1, stride))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.predict_stride <= 0:
        raise ValueError("--predict-stride must be positive.")
    if args.video_size <= 0:
        raise ValueError("--video-size must be positive.")

    lr_dataset = _require_lerobot_dataset()
    raw_dataset = lr_dataset.LeRobotDataset(args.repo_id)
    metadata = lr_dataset.LeRobotDatasetMetadata(args.repo_id)
    ranges = _episode_ranges(raw_dataset)
    if not 0 <= args.episode < len(ranges):
        raise ValueError(f"Episode {args.episode} is out of range [0, {len(ranges) - 1}].")
    start, end = ranges[args.episode]
    frame_count = end - start
    if args.limit_frames is not None:
        frame_count = min(frame_count, args.limit_frames)
    fps = float(args.fps or metadata.fps)

    ckpt_dir, step, ckpt_args = _load_probe_settings(args)
    base_config = _config.get_config(args.config_name)
    config = dataclasses.replace(base_config, batch_size=1, seed=args.seed, num_workers=args.num_workers)
    if not isinstance(config.model, pi0_config.Pi0LatentFlowConfig):
        raise ValueError(f"Config {args.config_name!r} is not a Pi0LatentFlow config.")

    data_config = config.data.create(config.assets_dirs, config.model)
    model_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    model_dataset = _data_loader.transform_dataset(model_dataset, data_config)

    mesh, model_def, model_params, probe_def, probe_state, infer_rng, probe_layer = _init_model_and_probe(
        args,
        config,
        ckpt_dir,
        ckpt_args,
    )
    predict = _make_predict_fn(model_def, probe_def, infer_rng, probe_layer)

    output_dir = pathlib.Path(args.output_dir) / f"episode_{args.episode:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_frames: list[np.ndarray] = []
    arrow_frames: list[np.ndarray] = []
    latest_flow_rgb: np.ndarray | None = None

    effort_dim = int(config.model.effort_dim if config.model.effort_dim is not None else config.model.effort_dim_in)
    true_curve = np.full((frame_count, effort_dim), np.nan, dtype=np.float64)
    pred_sum = np.zeros((frame_count, effort_dim), dtype=np.float64)
    counts = np.zeros((frame_count,), dtype=np.float64)
    prediction_starts = _prediction_starts(frame_count, int(config.model.action_horizon), args.predict_stride)

    for local_idx in range(frame_count):
        global_idx = start + local_idx
        raw_sample = raw_dataset[global_idx]
        if args.video_key not in raw_sample:
            raise KeyError(f"Video key {args.video_key!r} not found in dataset sample.")
        rgb = _to_numpy_image(raw_sample[args.video_key])
        rgb = image_tools.resize_with_pad(rgb, args.video_size, args.video_size)
        if "observation.effort" in raw_sample:
            raw_effort = _to_numpy(raw_sample["observation.effort"]).astype(np.float64)
            true_curve[local_idx] = raw_effort[:effort_dim]

        if local_idx in prediction_starts:
            sample = model_dataset[global_idx]
            observation = _model.Observation.from_dict(_batch_sample(sample))
            with sharding.set_mesh(mesh):
                pred_force, true_force, pred_flow, _ = predict(model_params, probe_state, observation)
            pred_force = np.asarray(jax.device_get(pred_force[0]))
            true_force = np.asarray(jax.device_get(true_force[0]))
            pred_force = _infer._unnormalize_effort(config, pred_force, skip_norm=False)
            true_force = _infer._unnormalize_effort(config, true_force, skip_norm=False)
            latest_flow_rgb = np.asarray(jax.device_get(pred_flow[0]))

            for horizon_idx in range(pred_force.shape[0]):
                target_local = local_idx + horizon_idx + 1
                if target_local >= frame_count:
                    break
                pred_sum[target_local] += pred_force[horizon_idx]
                counts[target_local] += 1.0

        original_frames.append(rgb)
        if latest_flow_rgb is None:
            arrow_frames.append(rgb)
        else:
            arrow_frames.append(
                _draw_flow_arrows(
                    rgb,
                    latest_flow_rgb,
                    step=args.arrow_step,
                    scale=args.arrow_scale,
                    thickness=args.arrow_thickness,
                    min_magnitude=args.arrow_min_magnitude,
                    color=args.arrow_color,
                )
            )

    original_path = output_dir / "rgb_video.mp4"
    arrow_path = output_dir / "rgb_with_pred_flow_arrows.mp4"
    figure_path = output_dir / "force_prediction_vs_true.png"
    _write_video(original_path, original_frames, fps)
    _write_video(arrow_path, arrow_frames, fps)
    _plot_force_comparison(figure_path, true_curve, pred_sum, counts, fps=fps)

    np.savez(
        output_dir / "force_prediction_vs_true.npz",
        true_curve=true_curve,
        pred_sum=pred_sum,
        counts=counts,
        prediction_starts=np.asarray(prediction_starts, dtype=np.int64),
        fps=np.asarray(fps, dtype=np.float32),
    )
    metadata_payload = {
        "repo_id": args.repo_id,
        "episode": args.episode,
        "checkpoint_step": step,
        "probe_layer": probe_layer,
        "predict_stride": args.predict_stride,
        "action_horizon": int(config.model.action_horizon),
        "frame_count": frame_count,
        "fps": fps,
        "video_key": args.video_key,
        "outputs": {
            "rgb_video": str(original_path),
            "rgb_with_pred_flow_arrows": str(arrow_path),
            "force_figure": str(figure_path),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    logging.info("Saved RGB video: %s", original_path)
    logging.info("Saved flow-arrow video: %s", arrow_path)
    logging.info("Saved force figure: %s", figure_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
