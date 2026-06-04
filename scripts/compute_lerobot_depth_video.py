"""Compute Depth Anything depth maps for a LeRobot dataset and render videos."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

import numpy as np

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ImportError:
    lerobot_dataset = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
except ImportError:
    torch = None
    F = None

try:
    from transformers import AutoImageProcessor
    from transformers import AutoModelForDepthEstimation
except ImportError:
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None

try:
    import tqdm
except ImportError:
    tqdm = None


DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"
DEFAULT_IMAGE_KEY_CANDIDATES = (
    ("observation.images.head_camera", "base_0_rgb"),
    ("observation.images.front", "base_0_rgb"),
    ("observation.images.wrist_left_camera", "left_wrist_0_rgb"),
    ("observation.images.left_wrist", "left_wrist_0_rgb"),
    ("observation.images.wrist_camera", "left_wrist_0_rgb"),
    ("observation.images.fixed_camera", "base_0_rgb"),
)


def require_cv2():
    if cv2 is None:
        raise ImportError("This script requires opencv-python. Run it in the project environment with cv2 installed.")
    return cv2


def require_lerobot_dataset():
    if lerobot_dataset is None:
        raise ImportError("This script requires lerobot. Run it in the project environment with LeRobot installed.")
    return lerobot_dataset


def require_pil_image():
    if Image is None:
        raise ImportError("This script requires pillow. Run it in the project environment with PIL installed.")
    return Image


def require_torch():
    if torch is None or F is None:
        raise ImportError("This script requires torch. Run it in the project environment with torch installed.")
    return torch


def require_transformers():
    if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
        raise ImportError("This script requires transformers. Run it in the project environment with transformers installed.")
    return AutoImageProcessor, AutoModelForDepthEstimation


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm.tqdm(iterable, **kwargs)


def pad_to_even(image: np.ndarray) -> np.ndarray:
    cv = require_cv2()
    height, width = image.shape[:2]
    pad_bottom = height % 2
    pad_right = width % 2
    if pad_bottom == 0 and pad_right == 0:
        return image
    return cv.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv.BORDER_CONSTANT, value=(0, 0, 0))


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def default_stream_name(image_key: str) -> str:
    return safe_name(image_key.rsplit(".", maxsplit=1)[-1])


def infer_image_keys_and_streams(metadata, requested_image_keys: list[str] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if requested_image_keys:
        return tuple(requested_image_keys), tuple(default_stream_name(key) for key in requested_image_keys)

    features = getattr(metadata, "features", None)
    if features is None:
        features = {}
    inferred = []
    used_streams = set()
    for image_key, stream_name in DEFAULT_IMAGE_KEY_CANDIDATES:
        feature = features.get(image_key)
        if feature is None or feature.get("dtype") not in ("image", "video") or stream_name in used_streams:
            continue
        inferred.append((image_key, stream_name))
        used_streams.add(stream_name)

    if inferred:
        return tuple(image_key for image_key, _ in inferred), tuple(stream_name for _, stream_name in inferred)

    video_keys = [
        key for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") in ("image", "video")
    ]
    if video_keys:
        return tuple(video_keys), tuple(default_stream_name(key) for key in video_keys)

    raise ValueError("Could not infer image keys from dataset metadata. Please pass --image-keys explicitly.")


def parse_episode_selector(items: list[str] | None) -> set[int] | None:
    if not items:
        return None
    episodes = set()
    for item in items:
        match = re.fullmatch(r"episode_(\d+)", item)
        episodes.add(int(match.group(1)) if match else int(item))
    return episodes


def to_numpy_image(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}.")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = to_numpy_image(image)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image = image.astype(np.float32)
    if image.size == 0:
        raise ValueError("Image is empty.")
    if np.nanmin(image) < 0:
        image = image / 2.0 + 0.5
    elif np.nanmax(image) > 1.5:
        image = image / 255.0
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray((np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8))


def get_episode_ranges(dataset) -> list[tuple[int, int]]:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is None:
        raise ValueError("LeRobotDataset does not expose episode_data_index.")
    starts = np.asarray(episode_data_index["from"], dtype=np.int64)
    ends = np.asarray(episode_data_index["to"], dtype=np.int64)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def make_dataset(repo_id: str, video_backend: str):
    lr_dataset = require_lerobot_dataset()
    try:
        return lr_dataset.LeRobotDataset(repo_id, video_backend=video_backend)
    except TypeError:
        return lr_dataset.LeRobotDataset(repo_id)


def depth_path(output_dir: pathlib.Path, episode_index: int, stream_name: str, local_frame_index: int) -> pathlib.Path:
    return (
        output_dir
        / "depths"
        / f"episode_{episode_index:06d}"
        / stream_name
        / f"frame_{local_frame_index:06d}.npy"
    )


def normalize_depth_for_display(depth: np.ndarray, *, invert: bool) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth)
    if not valid.any():
        raise ValueError("Predicted depth contains no finite values.")

    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(depth[valid].min()), float(depth[valid].max())
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.uint8)

    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    return (normalized * 255.0).round().astype(np.uint8)


def depth_to_bgr_image(depth: np.ndarray, colormap: str, *, invert: bool) -> np.ndarray:
    cv = require_cv2()
    color_maps = {
        "turbo": cv.COLORMAP_TURBO,
        "magma": cv.COLORMAP_MAGMA,
        "inferno": cv.COLORMAP_INFERNO,
        "viridis": cv.COLORMAP_VIRIDIS,
        "jet": cv.COLORMAP_JET,
    }
    gray = normalize_depth_for_display(depth, invert=invert)
    return cv.applyColorMap(gray, color_maps[colormap])


class DepthAnythingPredictor:
    def __init__(self, model_id: str, device: str | None, dtype: str) -> None:
        torch_module = require_torch()
        processor_cls, model_cls = require_transformers()

        self.device = device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        if dtype == "default":
            self.dtype = None
        elif dtype == "float16":
            self.dtype = torch_module.float16
        elif dtype == "bfloat16":
            self.dtype = torch_module.bfloat16
        elif dtype == "float32":
            self.dtype = torch_module.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.processor = processor_cls.from_pretrained(model_id, use_fast=True)
        if self.dtype is None:
            self.model = model_cls.from_pretrained(model_id).to(self.device).eval()
        else:
            self.model = model_cls.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()

    def predict(self, image: np.ndarray) -> np.ndarray:
        pil_image_cls = require_pil_image()
        torch_module = require_torch()
        rgb = to_uint8_rgb(image)
        pil_image = pil_image_cls.fromarray(rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        if self.dtype is None:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        else:
            inputs = {
                key: value.to(self.device, dtype=self.dtype) if torch_module.is_floating_point(value) else value.to(self.device)
                for key, value in inputs.items()
            }
        with torch_module.inference_mode():
            prediction = self.model(**inputs).predicted_depth
            prediction = F.interpolate(
                prediction[:, None],
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )[:, 0]
        return prediction.squeeze(0).detach().cpu().numpy().astype(np.float32)


def load_or_compute_depths(
    dataset,
    predictor: DepthAnythingPredictor,
    output_dir: pathlib.Path,
    image_keys: tuple[str, ...],
    stream_names: tuple[str, ...],
    sample_index: int,
    episode_index: int,
    local_frame_index: int,
    *,
    overwrite: bool,
    save_npy: bool,
) -> dict[str, np.ndarray]:
    paths = {
        stream_name: depth_path(output_dir, episode_index, stream_name, local_frame_index)
        for stream_name in stream_names
    }
    if save_npy and not overwrite and all(path.exists() for path in paths.values()):
        return {stream_name: np.load(path).astype(np.float32) for stream_name, path in paths.items()}

    sample = dataset[sample_index]
    depths = {}
    for image_key, stream_name in zip(image_keys, stream_names, strict=True):
        if image_key not in sample:
            raise KeyError(f"Image key {image_key!r} not found in sample.")
        depth = predictor.predict(to_numpy_image(sample[image_key]))
        if save_npy:
            path = paths[stream_name]
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("wb") as file:
                np.save(file, depth.astype(np.float16))
            tmp_path.replace(path)
        depths[stream_name] = depth
    return depths


def render_episode(
    dataset,
    predictor: DepthAnythingPredictor,
    output_dir: pathlib.Path,
    image_keys: tuple[str, ...],
    stream_names: tuple[str, ...],
    episode_index: int,
    start: int,
    end: int,
    fps: float,
    colormap: str,
    *,
    invert: bool,
    overwrite: bool,
    save_npy: bool,
    limit_frames: int | None,
) -> int:
    if end <= start:
        return 0

    frame_count = end - start
    if limit_frames is not None:
        frame_count = min(frame_count, limit_frames)
    if frame_count <= 0:
        return 0

    cv = require_cv2()
    writers: dict[str, Any] = {}
    video_paths: dict[str, pathlib.Path] = {}
    try:
        frame_iter = progress(range(frame_count), desc=f"episode_{episode_index:06d}", unit="frame", leave=False)
        for local_frame_index in frame_iter:
            sample_index = start + local_frame_index
            depths = load_or_compute_depths(
                dataset=dataset,
                predictor=predictor,
                output_dir=output_dir,
                image_keys=image_keys,
                stream_names=stream_names,
                sample_index=sample_index,
                episode_index=episode_index,
                local_frame_index=local_frame_index,
                overwrite=overwrite,
                save_npy=save_npy,
            )

            for stream_name, depth in depths.items():
                frame = pad_to_even(depth_to_bgr_image(depth, colormap=colormap, invert=invert))
                if stream_name not in writers:
                    stream_video_dir = output_dir / "videos" / stream_name
                    stream_video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = stream_video_dir / f"episode_{episode_index:06d}.mp4"
                    if video_path.exists() and overwrite:
                        video_path.unlink()
                    height, width = frame.shape[:2]
                    writer = cv.VideoWriter(
                        str(video_path),
                        cv.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {video_path}")
                    writers[stream_name] = writer
                    video_paths[stream_name] = video_path
                writers[stream_name].write(frame)
    finally:
        for writer in writers.values():
            writer.release()

    for video_path in video_paths.values():
        print(f"Saved {video_path}")
    return frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output root directory.")
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=None,
        help="RGB feature keys to read. Defaults to auto-detecting video keys from dataset metadata.",
    )
    parser.add_argument("--stream-names", nargs="+", default=None, help="Output stream names. Defaults to image key suffixes.")
    parser.add_argument("--episodes", nargs="*", default=None, help="Episode ids or names, e.g. 0 episode_000001.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional maximum number of selected episodes.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional maximum frames per episode.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face depth model id.")
    parser.add_argument("--device", default=None, help="Torch device for Depth Anything. Defaults to cuda if available.")
    parser.add_argument(
        "--dtype",
        choices=("default", "float16", "bfloat16", "float32"),
        default="default",
        help="Model dtype. default uses the precision chosen by the model/framework.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS. Defaults to dataset metadata fps.")
    parser.add_argument(
        "--colormap",
        choices=("turbo", "magma", "inferno", "viridis", "jet"),
        default="inferno",
        help="OpenCV color map used to render depth videos.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert rendered colors.")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw H x W relative depth maps under OUTPUT_DIR/depths.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing depth files and videos.")
    parser.add_argument("--video-backend", default="torchcodec", help="LeRobot video backend.")
    parser.add_argument("--torch-threads", type=int, default=None, help="Limit PyTorch CPU worker threads.")
    parser.add_argument("--torch-interop-threads", type=int, default=None, help="Limit PyTorch inter-op CPU threads.")
    parser.add_argument("--opencv-threads", type=int, default=None, help="Limit OpenCV CPU worker threads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps is not None and args.fps <= 0:
        raise ValueError(f"Expected --fps > 0, got {args.fps}.")
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError(f"Expected --torch-threads > 0, got {args.torch_threads}.")
        if torch is not None:
            torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        if args.torch_interop_threads <= 0:
            raise ValueError(f"Expected --torch-interop-threads > 0, got {args.torch_interop_threads}.")
        if torch is not None:
            torch.set_num_interop_threads(args.torch_interop_threads)
    if args.opencv_threads is not None:
        if args.opencv_threads <= 0:
            raise ValueError(f"Expected --opencv-threads > 0, got {args.opencv_threads}.")
        if cv2 is not None:
            cv2.setNumThreads(args.opencv_threads)

    lr_dataset = require_lerobot_dataset()
    metadata = lr_dataset.LeRobotDatasetMetadata(args.repo_id)

    image_keys, inferred_stream_names = infer_image_keys_and_streams(metadata, args.image_keys)
    stream_names = tuple(args.stream_names or inferred_stream_names)
    if len(stream_names) != len(image_keys):
        raise ValueError("--stream-names must have the same length as --image-keys.")
    stream_names = tuple(safe_name(name) for name in stream_names)

    dataset = make_dataset(args.repo_id, args.video_backend)
    episode_ranges = get_episode_ranges(dataset)

    selected_episodes = parse_episode_selector(args.episodes)
    indexed_ranges = [
        (episode_index, start, end)
        for episode_index, (start, end) in enumerate(episode_ranges)
        if selected_episodes is None or episode_index in selected_episodes
    ]
    if args.max_episodes is not None:
        indexed_ranges = indexed_ranges[: args.max_episodes]
    if not indexed_ranges:
        raise ValueError("No episodes selected.")

    fps = float(args.fps or metadata.fps)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = DepthAnythingPredictor(model_id=args.model, device=args.device, dtype=args.dtype)

    total_frames = 0
    episode_iter = progress(indexed_ranges, desc="Episodes", unit="episode")
    for episode_index, start, end in episode_iter:
        video_targets = [output_dir / "videos" / stream / f"episode_{episode_index:06d}.mp4" for stream in stream_names]
        if not args.overwrite and all(path.exists() for path in video_targets):
            print(f"Skip episode_{episode_index:06d}: videos already exist.")
            continue
        total_frames += render_episode(
            dataset=dataset,
            predictor=predictor,
            output_dir=output_dir,
            image_keys=image_keys,
            stream_names=stream_names,
            episode_index=episode_index,
            start=start,
            end=end,
            fps=fps,
            colormap=args.colormap,
            invert=args.invert,
            overwrite=args.overwrite,
            save_npy=args.save_npy,
            limit_frames=args.limit_frames,
        )

    metadata_path = output_dir / "metadata.json"
    metadata_payload = {
        "repo_id": args.repo_id,
        "image_keys": list(image_keys),
        "stream_names": list(stream_names),
        "fps": fps,
        "model": args.model,
        "device": predictor.device,
        "dtype": "default" if predictor.dtype is None else str(predictor.dtype),
        "colormap": args.colormap,
        "invert": args.invert,
        "save_npy": args.save_npy,
        "episodes": [episode_index for episode_index, _, _ in indexed_ranges],
        "processed_frames_this_run": total_frames,
        "complete": True,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
