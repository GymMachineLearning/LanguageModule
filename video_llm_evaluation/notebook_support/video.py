"""Video loading and annotation helpers for the squat results inspector notebook."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video_llm_evaluation.constants import ERROR_CLASSES


def load_video_frames(video_path: Path, max_frames: int | None = None, frame_step: int = 1) -> dict[str, object]:
    if not video_path.exists():
        raise FileNotFoundError(f'Cannot find video: {video_path}')

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    timestamps_s: list[float] = []
    frame_indices: list[int] = []

    frame_index = 0
    kept = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps_s.append(frame_index / fps if fps else 0.0)
            frame_indices.append(frame_index)
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        frame_index += 1

    capture.release()
    return {
        'fps': fps,
        'num_frames': num_frames,
        'frames': frames,
        'timestamps_s': timestamps_s,
        'frame_indices': frame_indices,
    }


def prediction_segments_by_error(payload: dict) -> dict[str, list[dict]]:
    segments_by_error: dict[str, list[dict]] = {error: [] for error in ERROR_CLASSES}
    for item in payload.get('predictions', []):
        error_type = str(item.get('error_type', ''))
        segments_by_error.setdefault(error_type, [])
        for segment in item.get('segments', []) or []:
            segments_by_error[error_type].append(segment)
    return segments_by_error


def active_errors_at_time(segments_by_error: dict[str, list[dict]], time_s: float) -> list[str]:
    active: list[str] = []
    for error_type, segments in segments_by_error.items():
        for segment in segments:
            if float(segment.get('start_s', 0.0)) <= time_s <= float(segment.get('end_s', 0.0)):
                active.append(error_type)
                break
    return active


def draw_error_overlay(frame_rgb: np.ndarray, *, time_s: float, llm_errors: list[str], gt_errors: list[str] | None = None) -> np.ndarray:
    from video_llm_evaluation.constants import ERROR_CLASSES as _ERROR_CLASSES

    error_colors = {
        'Squat-depth': '#e76f51',
        'Back-round': '#f4a261',
        'Taking-off-foot': '#2a9d8f',
        'Knee-collapse': '#e63946',
        'Dominant-hip': '#457b9d',
        'No-knee-outlet': '#6a4c93',
    }
    neutral_color = '#2b2d42'
    gt_color = '#264653'
    llm_color = '#9d4edd'

    frame = frame_rgb.copy()
    y = 30
    header = f't={time_s:0.2f}s'
    cv2.putText(frame, header, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, header, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y += 35

    def draw_line(prefix: str, labels: list[str], base_color: str, y_pos: int) -> int:
        line = prefix + (', '.join(labels) if labels else 'brak')
        rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr, 2, cv2.LINE_AA)
        return y_pos + 28

    y = draw_line('LLM: ', llm_errors, llm_color, y)
    if gt_errors is not None:
        y = draw_line('GT: ', gt_errors, gt_color, y)

    for idx, error_type in enumerate(llm_errors[:6]):
        color = error_colors.get(error_type, neutral_color)
        rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        text_y = frame.shape[0] - 20 - idx * 24
        cv2.putText(frame, error_type, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, error_type, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

    return frame


def build_annotated_frames(
    video_path: Path,
    prediction_payload: dict,
    *,
    max_frames: int | None = None,
    frame_step: int = 1,
    gt_labels_matrix: np.ndarray | None = None,
    gt_errors_for_video: list[str] | None = None,
) -> dict[str, object]:
    bundle = load_video_frames(video_path, max_frames=max_frames, frame_step=frame_step)
    fps = float(bundle['fps'])
    frames = bundle['frames']
    timestamps_s = bundle['timestamps_s']
    frame_indices = bundle.get('frame_indices', [])
    if not frames:
        raise RuntimeError(f'No frames loaded for {video_path}')

    segments_by_error = prediction_segments_by_error(prediction_payload)
    annotated_frames: list[np.ndarray] = []
    for frame_rgb, time_s, frame_index in zip(frames, timestamps_s, frame_indices):
        llm_errors = active_errors_at_time(segments_by_error, time_s)
        gt_errors: list[str] | None = None
        if gt_labels_matrix is not None:
            gt_errors = []
            max_rows = min(len(ERROR_CLASSES), gt_labels_matrix.shape[0])
            if frame_index >= gt_labels_matrix.shape[1]:
                frame_index = gt_labels_matrix.shape[1] - 1
            for row_idx in range(max_rows):
                if float(gt_labels_matrix[row_idx, frame_index]) == 1.0:
                    gt_errors.append(ERROR_CLASSES[row_idx])
            if not gt_errors:
                gt_errors = []
        elif gt_errors_for_video is not None:
            gt_errors = gt_errors_for_video
        annotated_frames.append(draw_error_overlay(frame_rgb, time_s=time_s, llm_errors=llm_errors, gt_errors=gt_errors))

    return {'fps': fps, 'frames': annotated_frames, 'timestamps_s': timestamps_s}
