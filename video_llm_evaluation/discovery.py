"""Discovery and metadata helpers for squat video batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


@dataclass(frozen=True, slots=True)
class DiscoveredVideo:
    video_path: Path
    relative_path: Path
    output_dir: Path
    annotation_path: Path | None
    video_id: str
    duration_s: float
    fps: float
    num_frames: int


def discover_video_paths(videos_root: Path, extensions: Iterable[str] = (".mp4", ".mov", ".mkv", ".avi")) -> list[Path]:
    allowed = {ext.lower() for ext in extensions}
    return sorted(
        path
        for path in videos_root.rglob("*")
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in allowed
    )


def read_video_metadata(video_path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if fps <= 0 or num_frames <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}: fps={fps}, num_frames={num_frames}")

    duration_s = num_frames / fps
    return duration_s, fps, num_frames


def mirrored_output_dir(results_root: Path, videos_root: Path, video_path: Path) -> Path:
    relative_path = video_path.relative_to(videos_root)
    return results_root / relative_path.with_suffix("")


def discover_videos(videos_root: Path, results_root: Path) -> list[DiscoveredVideo]:
    items: list[DiscoveredVideo] = []
    for video_path in discover_video_paths(videos_root):
        duration_s, fps, num_frames = read_video_metadata(video_path)
        relative_path = video_path.relative_to(videos_root)
        annotation_path = video_path.with_name(f"{video_path.name}.txt")
        if not annotation_path.exists():
            annotation_path = None
        items.append(
            DiscoveredVideo(
                video_path=video_path,
                relative_path=relative_path,
                output_dir=mirrored_output_dir(results_root, videos_root, video_path),
                annotation_path=annotation_path,
                video_id=relative_path.stem,
                duration_s=duration_s,
                fps=fps,
                num_frames=num_frames,
            )
        )
    return items
