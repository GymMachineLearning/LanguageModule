"""High-level evaluation entrypoints for a single video or a batch."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from video_llm_evaluation.schemas import ManifestRow, VideoPrediction
from llm_api.gemini.response_parser import parse_video_prediction

from .metrics import frame_metrics, segment_metrics, video_level_metrics
from .persistence import (
    append_failure,
    save_labels_csv,
    save_labels_npy,
    save_metrics_rows,
    save_prediction_json,
    save_raw_response,
    save_segments_csv,
    save_summary_json,
)
from .segments_to_frames import video_prediction_to_frame_labels


def _mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def _load_ground_truth_labels(ground_truth_path: str) -> np.ndarray:
    path = Path(ground_truth_path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".csv":
        import csv

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            return np.zeros((0, 0), dtype=np.uint8)
        class_columns = [column for column in rows[0].keys() if column not in {"frame_idx", "time_s"}]
        matrix = np.array([[int(row[column]) for row in rows] for column in class_columns], dtype=np.uint8)
        return matrix
    raise ValueError(f"Unsupported ground truth format: {path.suffix}")


def evaluate_single_video(
    raw_response: str | Mapping[str, object],
    manifest_row: ManifestRow,
    *,
    run_dir: Path | str | None = None,
    model_name: str = "gemini-3.1-pro-preview",
    prompt_version: str = "v1",
    min_segment_duration_s: float = 0.2,
    merge_gap_s: float = 0.2,
    tolerances_s: Sequence[float] = (0.1, 0.5, 1.0),
) -> dict[str, object]:
    """Evaluate a single video response against its ground truth."""

    prediction = parse_video_prediction(
        raw_response,
        video_id=manifest_row.video_id,
        duration_s=manifest_row.duration_s,
        min_segment_duration_s=min_segment_duration_s,
        merge_gap_s=merge_gap_s,
    )

    pred_labels = video_prediction_to_frame_labels(prediction, fps=manifest_row.fps, num_frames=manifest_row.num_frames)

    if not manifest_row.ground_truth_path:
        raise ValueError("ground_truth_path is required for evaluation")

    gt_labels = _load_ground_truth_labels(manifest_row.ground_truth_path)
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(f"Ground truth shape {gt_labels.shape} does not match prediction shape {pred_labels.shape}")

    frame_level = frame_metrics(gt_labels, pred_labels)
    video_level = video_level_metrics(gt_labels, pred_labels)
    segment_level = segment_metrics(gt_labels, pred_labels, fps=manifest_row.fps, tolerances_s=tolerances_s)

    result = {
        "video_id": manifest_row.video_id,
        "prediction": prediction,
        "pred_labels": pred_labels,
        "gt_labels": gt_labels,
        "frame_metrics": frame_level,
        "video_metrics": video_level,
        "segment_metrics": segment_level,
    }

    if run_dir is not None:
        save_prediction_json(run_dir, prediction, model_name=model_name, prompt_version=prompt_version)
        save_raw_response(run_dir, manifest_row.video_id, raw_response)
        save_labels_npy(run_dir, manifest_row.video_id, pred_labels)
        save_labels_csv(run_dir, manifest_row.video_id, pred_labels, fps=manifest_row.fps)
        save_segments_csv(
            run_dir,
            [
                {
                    "video_id": manifest_row.video_id,
                    "error_type": item.error_type,
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                    "confidence": segment.confidence,
                    "model_name": model_name,
                    "prompt_version": prompt_version,
                }
                for item in prediction.predictions
                for segment in item.segments
            ],
        )
        save_metrics_rows(
            run_dir,
            "frame_metrics.csv",
            [
                {"video_id": manifest_row.video_id, **row}
                for row in frame_level["per_class"]
            ],
        )
        save_metrics_rows(
            run_dir,
            "video_level_metrics.csv",
            [
                {"video_id": manifest_row.video_id, **row}
                for row in video_level["per_class"]
            ],
        )
        save_metrics_rows(
            run_dir,
            "segment_metrics.csv",
            [
                {
                    "video_id": manifest_row.video_id,
                    "error_type": row.error_type,
                    "tolerance_s": row.tolerance_s,
                    "tp": row.tp,
                    "fp": row.fp,
                    "fn": row.fn,
                    "mean_temporal_iou": row.mean_temporal_iou,
                    "start_mae_s": row.start_mae_s,
                    "end_mae_s": row.end_mae_s,
                }
                for row in segment_level
            ],
        )
        save_summary_json(
            run_dir,
            {
                "video_id": manifest_row.video_id,
                "frame_metrics": {key: value for key, value in frame_level.items() if key != "per_class"},
                "video_metrics": {key: value for key, value in video_level.items() if key != "per_class"},
                "segment_metrics": [row.__dict__ for row in segment_level],
            },
        )

    return result


def evaluate_batch(
    items: Sequence[tuple[str | Mapping[str, object], ManifestRow]],
    *,
    run_dir: Path | str | None = None,
    model_name: str = "gemini-3.1-pro-preview",
    prompt_version: str = "v1",
    min_segment_duration_s: float = 0.2,
    merge_gap_s: float = 0.2,
    tolerances_s: Sequence[float] = (0.1, 0.5, 1.0),
) -> list[dict[str, object]]:
    """Evaluate many videos and continue after individual failures."""

    results: list[dict[str, object]] = []
    for raw_response, manifest_row in items:
        try:
            results.append(
                evaluate_single_video(
                    raw_response,
                    manifest_row,
                    run_dir=run_dir,
                    model_name=model_name,
                    prompt_version=prompt_version,
                    min_segment_duration_s=min_segment_duration_s,
                    merge_gap_s=merge_gap_s,
                    tolerances_s=tolerances_s,
                )
            )
        except Exception as exc:
            if run_dir is not None:
                append_failure(
                    run_dir,
                    video_id=manifest_row.video_id,
                    video_path=manifest_row.video_path,
                    status="VALIDATION_ERROR",
                    error_message=str(exc),
                )
            results.append({"video_id": manifest_row.video_id, "error": str(exc)})
    return results


def summarize_batch_results(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate a batch of per-video results into a compact run summary."""

    successful_results = [result for result in results if "frame_metrics" in result and "video_metrics" in result]
    if not successful_results:
        return {"num_videos": len(results), "num_successes": 0, "num_failures": len(results)}

    frame_macro_f1 = [_result["frame_metrics"]["macro_f1"] for _result in successful_results]
    frame_micro_f1 = [_result["frame_metrics"]["micro_f1"] for _result in successful_results]
    video_micro_f1 = [_result["video_metrics"]["micro_f1"] for _result in successful_results]
    video_accuracy = [_result["video_metrics"]["accuracy"] for _result in successful_results]

    return {
        "num_videos": len(results),
        "num_successes": len(successful_results),
        "num_failures": len(results) - len(successful_results),
        "mean_frame_macro_f1": _mean(frame_macro_f1),
        "mean_frame_micro_f1": _mean(frame_micro_f1),
        "mean_video_micro_f1": _mean(video_micro_f1),
        "mean_video_accuracy": _mean(video_accuracy),
    }
