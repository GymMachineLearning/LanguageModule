"""Convert temporal error segments into per-frame binary labels."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from video_llm_evaluation.constants import ERROR_CLASSES, ERROR_TO_INDEX
from video_llm_evaluation.schemas import ErrorPrediction, ErrorSegment, VideoPrediction


def _frame_interval(frame_index: int, frame_rate: float) -> tuple[float, float]:
    start_s = frame_index / frame_rate
    end_s = (frame_index + 1) / frame_rate
    return start_s, end_s


def _segment_overlaps_frame(segment: ErrorSegment, frame_index: int, frame_rate: float) -> bool:
    frame_start_s, frame_end_s = _frame_interval(frame_index, frame_rate)
    return max(segment.start_s, frame_start_s) < min(segment.end_s, frame_end_s)


def segments_to_frame_labels(
    predictions: Sequence[ErrorPrediction],
    duration_s: float,
    fps: float,
    num_frames: int,
    *,
    use_effective_fps: bool = False,
) -> np.ndarray:
    """Convert predictions into a stable 6 x T binary matrix."""

    frame_rate = num_frames / duration_s if use_effective_fps else fps
    labels = np.zeros((len(ERROR_CLASSES), num_frames), dtype=np.uint8)

    for prediction in predictions:
        class_index = ERROR_TO_INDEX.get(prediction.error_type)
        if class_index is None or not prediction.present:
            continue

        for segment in prediction.segments:
            for frame_index in range(num_frames):
                if _segment_overlaps_frame(segment, frame_index, frame_rate):
                    labels[class_index, frame_index] = 1

    return labels


def video_prediction_to_frame_labels(
    prediction: VideoPrediction,
    fps: float,
    num_frames: int,
    *,
    use_effective_fps: bool = False,
) -> np.ndarray:
    """Convenience wrapper for VideoPrediction objects."""

    return segments_to_frame_labels(
        prediction.predictions,
        duration_s=prediction.duration_s,
        fps=fps,
        num_frames=num_frames,
        use_effective_fps=use_effective_fps,
    )
