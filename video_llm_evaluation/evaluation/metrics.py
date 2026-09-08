"""Metrics for frame-level, video-level, and segment-level evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from video_llm_evaluation.constants import ERROR_CLASSES, ERROR_TO_INDEX
from video_llm_evaluation.schemas import ErrorPrediction, ErrorSegment


@dataclass(frozen=True)
class SegmentMatch:
    error_type: str
    tolerance_s: float
    tp: int
    fp: int
    fn: int
    mean_temporal_iou: float | None
    start_mae_s: float | None
    end_mae_s: float | None


def _as_binary_matrix(labels: np.ndarray) -> np.ndarray:
    matrix = np.asarray(labels)
    if matrix.ndim != 2:
        raise ValueError("labels must be a 2D array")
    if matrix.shape[0] != len(ERROR_CLASSES):
        raise ValueError(f"labels must have {len(ERROR_CLASSES)} rows")
    return matrix.astype(np.uint8, copy=False)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def frame_metrics(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, object]:
    """Compute frame-level metrics for the stable 6 x T matrix."""

    gt = _as_binary_matrix(gt_labels)
    pred = _as_binary_matrix(pred_labels)
    if gt.shape != pred.shape:
        raise ValueError(f"gt_labels and pred_labels must have the same shape, got {gt.shape} and {pred.shape}")

    per_class: list[dict[str, object]] = []
    total_tp = total_fp = total_fn = total_tn = 0
    weighted_support = 0
    weighted_f1_sum = 0.0

    for class_index, class_name in enumerate(ERROR_CLASSES):
        gt_row = gt[class_index]
        pred_row = pred[class_index]
        tp = int(np.sum((gt_row == 1) & (pred_row == 1)))
        fp = int(np.sum((gt_row == 0) & (pred_row == 1)))
        fn = int(np.sum((gt_row == 1) & (pred_row == 0)))
        tn = int(np.sum((gt_row == 0) & (pred_row == 0)))
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        support = int(np.sum(gt_row == 1))
        iou = _safe_divide(tp, tp + fp + fn)

        per_class.append(
            {
                "error_type": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        weighted_support += support
        weighted_f1_sum += f1 * support

    precision_micro, recall_micro, f1_micro = _precision_recall_f1(total_tp, total_fp, total_fn)
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0
    weighted_f1 = _safe_divide(weighted_f1_sum, weighted_support)
    mean_iou = float(np.mean([row["iou"] for row in per_class])) if per_class else 0.0

    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "micro_f1": f1_micro,
        "weighted_f1": weighted_f1,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "mean_iou": mean_iou,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


def video_level_metrics(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, object]:
    """Compute per-class video-level presence metrics."""

    gt = _as_binary_matrix(gt_labels)
    pred = _as_binary_matrix(pred_labels)
    if gt.shape != pred.shape:
        raise ValueError(f"gt_labels and pred_labels must have the same shape, got {gt.shape} and {pred.shape}")

    per_class: list[dict[str, object]] = []
    total_tp = total_fp = total_fn = total_tn = 0

    for class_index, class_name in enumerate(ERROR_CLASSES):
        gt_present = bool(gt[class_index].any())
        pred_present = bool(pred[class_index].any())
        tp = int(gt_present and pred_present)
        fp = int((not gt_present) and pred_present)
        fn = int(gt_present and (not pred_present))
        tn = int((not gt_present) and (not pred_present))
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

        per_class.append(
            {
                "error_type": class_name,
                "gt_present": gt_present,
                "pred_present": pred_present,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    precision_micro, recall_micro, f1_micro = _precision_recall_f1(total_tp, total_fp, total_fn)
    accuracy = _safe_divide(total_tp + total_tn, total_tp + total_tn + total_fp + total_fn)

    return {
        "per_class": per_class,
        "accuracy": accuracy,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "micro_f1": f1_micro,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


def labels_to_segments(labels: Sequence[int] | np.ndarray, fps: float) -> list[ErrorSegment]:
    """Convert a single class row of frame labels to contiguous temporal segments."""

    row = np.asarray(labels, dtype=np.uint8).ravel()
    segments: list[ErrorSegment] = []
    start_index: int | None = None

    for frame_index, value in enumerate(row):
        if value and start_index is None:
            start_index = frame_index
        elif not value and start_index is not None:
            segments.append(ErrorSegment(start_s=start_index / fps, end_s=frame_index / fps))
            start_index = None

    if start_index is not None:
        segments.append(ErrorSegment(start_s=start_index / fps, end_s=len(row) / fps))

    return segments


def _segment_iou(left: ErrorSegment, right: ErrorSegment) -> float:
    intersection = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
    union = max(left.end_s, right.end_s) - min(left.start_s, right.start_s)
    return _safe_divide(intersection, union)


def _match_segments_for_class(
    gt_segments: Sequence[ErrorSegment],
    pred_segments: Sequence[ErrorSegment],
    tolerance_s: float,
) -> tuple[int, int, int, list[tuple[ErrorSegment, ErrorSegment, float]]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_segment in enumerate(gt_segments):
        for pred_index, pred_segment in enumerate(pred_segments):
            if abs(gt_segment.start_s - pred_segment.start_s) <= tolerance_s and abs(gt_segment.end_s - pred_segment.end_s) <= tolerance_s:
                candidates.append((_segment_iou(gt_segment, pred_segment), gt_index, pred_index))

    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[ErrorSegment, ErrorSegment, float]] = []

    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_segments[gt_index], pred_segments[pred_index], iou))

    tp = len(matches)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp
    return tp, fp, fn, matches


def segment_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    fps: float,
    tolerances_s: Sequence[float] = (0.1, 0.5, 1.0),
) -> list[SegmentMatch]:
    """Compute segment-level metrics for each class and tolerance."""

    gt = _as_binary_matrix(gt_labels)
    pred = _as_binary_matrix(pred_labels)
    if gt.shape != pred.shape:
        raise ValueError(f"gt_labels and pred_labels must have the same shape, got {gt.shape} and {pred.shape}")

    rows: list[SegmentMatch] = []
    for tolerance_s in tolerances_s:
        for class_index, class_name in enumerate(ERROR_CLASSES):
            gt_segments = labels_to_segments(gt[class_index], fps)
            pred_segments = labels_to_segments(pred[class_index], fps)
            tp, fp, fn, matches = _match_segments_for_class(gt_segments, pred_segments, tolerance_s)

            precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
            if matches:
                mean_iou = float(np.mean([match[2] for match in matches]))
                start_mae = float(np.mean([abs(gt_segment.start_s - pred_segment.start_s) for gt_segment, pred_segment, _ in matches]))
                end_mae = float(np.mean([abs(gt_segment.end_s - pred_segment.end_s) for gt_segment, pred_segment, _ in matches]))
            else:
                mean_iou = None
                start_mae = None
                end_mae = None

            rows.append(
                SegmentMatch(
                    error_type=class_name,
                    tolerance_s=tolerance_s,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    mean_temporal_iou=mean_iou,
                    start_mae_s=start_mae,
                    end_mae_s=end_mae,
                )
            )

    return rows
