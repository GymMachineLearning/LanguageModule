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


def _frame_content_bounds(frame_rgb: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > 20
    coords = np.argwhere(mask)
    if coords.size == 0:
        height, width = frame_rgb.shape[:2]
        return 0, 0, width, height
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return int(x0), int(y0), int(x1) + 1, int(y1) + 1


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return ['']
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current += ' ' + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_wrapped_lines(
    frame: np.ndarray,
    lines: list[str],
    *,
    x: int,
    y: int,
    max_width: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
    line_gap: int = 4,
) -> int:
    if max_width <= 0:
        return y

    estimated_chars = max(6, int(max_width / max(font_scale * 11, 1)))
    for line in lines:
        wrapped_lines = _wrap_text(line, estimated_chars)
        for wrapped_line in wrapped_lines:
            cv2.putText(frame, wrapped_line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, wrapped_line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            y += int(round(font_scale * 26)) + line_gap
    return y


def draw_error_overlay(frame_rgb: np.ndarray, *, time_s: float, llm_errors: list[str], gt_errors: list[str] | None = None) -> np.ndarray:
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
    x0, y0, x1, y1 = _frame_content_bounds(frame)
    height, width = frame.shape[:2]
    left_bar_width = max(0, x0)
    right_bar_width = max(0, width - x1)

    use_left_bar = left_bar_width >= right_bar_width
    if use_left_bar and left_bar_width > 30:
        text_x = 12
        text_width = left_bar_width - 24
    elif right_bar_width > 30:
        text_x = x1 + 12
        text_width = right_bar_width - 24
    else:
        text_x = 12
        text_width = width - 24

    header_color = (255, 255, 255)
    title_color = (255, 105, 180)
    gt_title_color = tuple(int(gt_color.lstrip('#')[i:i+2], 16)[::-1] if False else 0 for i in range(1))
    gt_rgb = tuple(int(gt_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    llm_rgb = tuple(int(llm_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    gt_bgr = (gt_rgb[2], gt_rgb[1], gt_rgb[0])
    llm_bgr = (llm_rgb[2], llm_rgb[1], llm_rgb[0])

    y = max(24, y0 + 24 if x0 <= 0 else 30)
    cv2.putText(frame, f't={time_s:0.2f}s', (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f't={time_s:0.2f}s', (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 1, cv2.LINE_AA)
    y += 26

    llm_line = 'LLM: ' + (', '.join(llm_errors) if llm_errors else 'brak')
    cv2.putText(frame, 'LLM:', (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, 'LLM:', (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, llm_bgr, 1, cv2.LINE_AA)
    y = _draw_wrapped_lines(frame, [llm_line[len('LLM: '):]], x=text_x + 44, y=y, max_width=max(0, text_width - 44), font_scale=0.43, color=llm_bgr, thickness=1)
    y += 6

    if gt_errors is not None:
        gt_line = 'GT: ' + (', '.join(gt_errors) if gt_errors else 'brak')
        cv2.putText(frame, 'GT:', (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'GT:', (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, gt_bgr, 1, cv2.LINE_AA)
        y = _draw_wrapped_lines(frame, [gt_line[len('GT: '):]], x=text_x + 38, y=y, max_width=max(0, text_width - 38), font_scale=0.43, color=gt_bgr, thickness=1)

    if llm_errors:
        y += 8
        _draw_wrapped_lines(
            frame,
            [f'- {error_type}' for error_type in llm_errors[:8]],
            x=text_x,
            y=y,
            max_width=text_width,
            font_scale=0.38,
            color=tuple(int(neutral_color.lstrip('#')[i:i+2], 16) for i in (2, 1, 0)),
            thickness=1,
            line_gap=2,
        )

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
