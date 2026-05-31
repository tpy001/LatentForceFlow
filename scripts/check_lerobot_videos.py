"""Check LeRobot video files against episode metadata.

This catches the common failure mode where LeRobot metadata asks for frame N
but the corresponding mp4 only contains fewer frames, e.g. torchcodec raising:
    Invalid frame index=145 for streamIndex=0 numFrames=70
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tqdm
except ImportError:
    tqdm = None


@dataclass(frozen=True)
class Episode:
    index: int
    length: int


@dataclass(frozen=True)
class Issue:
    status: str
    episode_index: int
    video_key: str
    expected_frames: int
    actual_frames: int | None
    path: pathlib.Path
    detail: str = ""


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_lerobot_home() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("HF_LEROBOT_HOME")
        or os.environ.get("LEROBOT_HOME")
        or pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"
    )


def resolve_dataset_root(repo_id: str | None, dataset_root: pathlib.Path | None) -> pathlib.Path:
    if dataset_root is not None:
        return dataset_root.expanduser().resolve()
    if repo_id is None:
        raise ValueError("Pass either --repo-id or --dataset-root.")
    return (default_lerobot_home() / repo_id).expanduser().resolve()


def iter_progress(items, **kwargs):
    if tqdm is None:
        return items
    return tqdm.tqdm(items, **kwargs)


def load_episodes(meta_dir: pathlib.Path) -> list[Episode]:
    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}")

    episodes = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            episodes.append(Episode(index=int(row["episode_index"]), length=int(row["length"])))
    return episodes


def parse_episode_filter(items: list[str] | None) -> set[int] | None:
    if not items:
        return None
    selected = set()
    for item in items:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if part.startswith("episode_"):
                selected.add(int(part.removeprefix("episode_")))
            elif "-" in part:
                start, end = part.split("-", 1)
                selected.update(range(int(start), int(end) + 1))
            else:
                selected.add(int(part))
    return selected


def infer_video_keys(info: dict[str, Any], requested: list[str] | None, only_generated: bool) -> list[str]:
    features = info.get("features", {})
    if requested:
        missing = [key for key in requested if key not in features]
        if missing:
            raise KeyError(f"Requested video keys not found in info.json features: {missing}")
        return requested

    keys = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if only_generated:
        keys = [key for key in keys if "flow" in key.lower() or "depth" in key.lower()]
    if not keys:
        raise ValueError("No video keys found in info.json.")
    return sorted(keys)


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def video_path_for(dataset_root: pathlib.Path, info: dict[str, Any], episode_index: int, video_key: str) -> pathlib.Path:
    template = info.get("video_path")
    if not template:
        raise ValueError("info.json does not define video_path.")
    rel_path = template.format(
        episode_chunk=episode_chunk(episode_index, int(info["chunks_size"])),
        video_key=video_key,
        episode_index=episode_index,
    )
    return dataset_root / rel_path


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_value = float(den)
        return None if den_value == 0 else float(num) / den_value
    return float(rate)


def probe_with_ffprobe(path: pathlib.Path, accurate: bool) -> tuple[int | None, str]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None, "ffprobe not found"

    entries = "stream=nb_frames,nb_read_frames,duration,r_frame_rate,avg_frame_rate"
    command = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if accurate:
        command.append("-count_frames")
    command += ["-show_entries", entries, "-of", "json", str(path)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr.strip() or "ffprobe failed"

    streams = json.loads(result.stdout or "{}").get("streams", [])
    if not streams:
        return None, "ffprobe found no video stream"
    stream = streams[0]

    for field in ("nb_read_frames", "nb_frames"):
        value = stream.get(field)
        if value not in (None, "N/A"):
            return int(value), f"ffprobe:{field}"

    duration = stream.get("duration")
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if duration not in (None, "N/A") and fps:
        return int(round(float(duration) * fps)), "ffprobe:duration*fps"

    return None, "ffprobe could not determine frame count"


def probe_with_cv2(path: pathlib.Path) -> tuple[int | None, str]:
    if cv2 is None:
        return None, "cv2 not installed"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, "cv2 failed to open video"
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        return None, "cv2 returned non-positive frame count"
    return count, "cv2:CAP_PROP_FRAME_COUNT"


def count_frames(path: pathlib.Path, accurate: bool) -> tuple[int | None, str]:
    count, detail = probe_with_ffprobe(path, accurate=accurate)
    if count is not None:
        return count, detail
    cv_count, cv_detail = probe_with_cv2(path)
    if cv_count is not None:
        return cv_count, cv_detail
    return None, f"{detail}; {cv_detail}"


def can_decode_frame(path: pathlib.Path, frame_index: int) -> tuple[bool, str]:
    if cv2 is None:
        return False, "cv2 not installed"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "cv2 failed to open video"
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False, f"cv2 failed to decode frame {frame_index}"
    return True, "cv2 decoded requested frame"


def write_csv(path: pathlib.Path, issues: list[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "episode_index",
                "video_key",
                "expected_frames",
                "actual_frames",
                "path",
                "detail",
            ],
        )
        writer.writeheader()
        for issue in issues:
            writer.writerow(
                {
                    "status": issue.status,
                    "episode_index": issue.episode_index,
                    "video_key": issue.video_key,
                    "expected_frames": issue.expected_frames,
                    "actual_frames": issue.actual_frames,
                    "path": str(issue.path),
                    "detail": issue.detail,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id", default=None, help="LeRobot dataset repo id.")
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=None,
        help="Explicit dataset root. Defaults to $HF_LEROBOT_HOME/REPO_ID or ~/.cache/huggingface/lerobot/REPO_ID.",
    )
    parser.add_argument(
        "--video-keys",
        nargs="*",
        default=None,
        help="Video keys to check. Defaults to all video keys, or only depth/flow keys with --only-generated.",
    )
    parser.add_argument(
        "--only-generated",
        action="store_true",
        help="Only check video keys whose name contains 'flow' or 'depth'.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="Episode ids/ranges to check, e.g. 0 5 10-20 episode_000145.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Check only the first N selected episodes. Useful for smoke tests.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use container metadata when possible instead of ffprobe -count_frames.",
    )
    parser.add_argument(
        "--decode-last",
        action="store_true",
        help="Also try to decode the last expected frame with OpenCV.",
    )
    parser.add_argument(
        "--report-csv",
        type=pathlib.Path,
        default=None,
        help="Optional CSV path for bad videos.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=50,
        help="Maximum number of issues to print.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit with code 0 even when issues are found.",
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.repo_id, args.dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    info = read_json(info_path)
    episodes = load_episodes(dataset_root / "meta")
    selected = parse_episode_filter(args.episodes)
    if selected is not None:
        episodes = [episode for episode in episodes if episode.index in selected]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    video_keys = infer_video_keys(info, args.video_keys, only_generated=args.only_generated)
    total_checks = len(episodes) * len(video_keys)
    issues: list[Issue] = []

    print(f"Dataset: {dataset_root}")
    print(f"Episodes: {len(episodes)}")
    print(f"Video keys: {', '.join(video_keys)}")
    print(f"Checks: {total_checks}")

    for episode in iter_progress(episodes, desc="checking episodes"):
        for video_key in video_keys:
            path = video_path_for(dataset_root, info, episode.index, video_key)
            if not path.exists():
                issues.append(
                    Issue("missing", episode.index, video_key, episode.length, None, path, "file does not exist")
                )
                continue

            actual_frames, detail = count_frames(path, accurate=not args.fast)
            if actual_frames is None:
                issues.append(Issue("unreadable", episode.index, video_key, episode.length, None, path, detail))
                continue

            if actual_frames < episode.length:
                issues.append(Issue("too_short", episode.index, video_key, episode.length, actual_frames, path, detail))
                continue

            if args.decode_last and episode.length > 0:
                ok, decode_detail = can_decode_frame(path, episode.length - 1)
                if not ok:
                    issues.append(
                        Issue(
                            "decode_failed",
                            episode.index,
                            video_key,
                            episode.length,
                            actual_frames,
                            path,
                            f"{detail}; {decode_detail}",
                        )
                    )

    if issues:
        print(f"\nFound {len(issues)} bad video checks:")
        for issue in issues[: args.max_print]:
            actual = "unknown" if issue.actual_frames is None else str(issue.actual_frames)
            print(
                f"{issue.status}: episode_{issue.episode_index:06d} {issue.video_key} "
                f"expected>={issue.expected_frames} actual={actual} "
                f"path={issue.path} detail={issue.detail}"
            )
        if len(issues) > args.max_print:
            print(f"... {len(issues) - args.max_print} more issues not printed")
    else:
        print("\nOK: all checked videos have at least the episode length frames.")

    if args.report_csv is not None:
        write_csv(args.report_csv.expanduser().resolve(), issues)
        print(f"CSV report: {args.report_csv.expanduser().resolve()}")

    if issues and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
