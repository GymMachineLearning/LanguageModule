"""Persistence helpers for run artifacts and evaluation outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from video_llm_evaluation.constants import DEFAULT_MODEL_NAME, DEFAULT_PROMPT_VERSION, ERROR_CLASSES
from video_llm_evaluation.schemas import ErrorPrediction, ErrorSegment, ManifestRow, VideoPrediction


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_path: Path
    manifest_path: Path
    predictions_dir: Path
    raw_responses_dir: Path
    labels_npy_dir: Path
    labels_csv_dir: Path
    segments_dir: Path
    metrics_dir: Path
    logs_dir: Path


def make_run_paths(run_dir: Path | str) -> RunPaths:
    run_path = Path(run_dir)
    return RunPaths(
        run_dir=run_path,
        config_path=run_path / "config.yaml",
        manifest_path=run_path / "manifest_resolved.csv",
        predictions_dir=run_path / "predictions_json",
        raw_responses_dir=run_path / "raw_responses",
        labels_npy_dir=run_path / "labels_npy",
        labels_csv_dir=run_path / "labels_csv",
        segments_dir=run_path / "segments",
        metrics_dir=run_path / "metrics",
        logs_dir=run_path / "logs",
    )


def ensure_run_structure(run_dir: Path | str) -> RunPaths:
    paths = make_run_paths(run_dir)
    for directory in [
        paths.run_dir,
        paths.predictions_dir,
        paths.raw_responses_dir,
        paths.labels_npy_dir,
        paths.labels_csv_dir,
        paths.segments_dir,
        paths.metrics_dir,
        paths.logs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_run_metadata_structure(run_dir: Path | str) -> RunPaths:
    paths = make_run_paths(run_dir)
    for directory in [
        paths.run_dir,
        paths.metrics_dir,
        paths.logs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_config(run_dir: Path | str, payload: Mapping[str, object]) -> Path:
    paths = ensure_run_metadata_structure(run_dir)
    try:
        import yaml

        paths.config_path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        save_json(paths.config_path, payload)
    return paths.config_path


def save_manifest_resolved(run_dir: Path | str, rows: Sequence[ManifestRow]) -> Path:
    paths = ensure_run_structure(run_dir)
    with paths.manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "video_path", "duration_s", "fps", "num_frames", "ground_truth_path"])
        for row in rows:
            writer.writerow([row.video_id, row.video_path, row.duration_s, row.fps, row.num_frames, row.ground_truth_path or ""])
    return paths.manifest_path


def _prediction_to_json_payload(
    prediction: VideoPrediction,
    *,
    model_name: str,
    prompt_version: str,
    video_fps: float | None = None,
    media_processing: str | None = None,
) -> dict[str, object]:
    return {
        "video_id": prediction.video_id,
        "duration_s": prediction.duration_s,
        "model_name": model_name,
        "prompt_version": prompt_version,
        # Sampling conditions travel with the prediction: two runs of the same
        # model at different frame rates are not comparable, and a results tree
        # that does not record this cannot be audited later.
        "video_fps": video_fps,
        "media_processing": media_processing,
        "predictions": [
            {
                "error_type": item.error_type,
                "present": item.present,
                "segments": [segment.model_dump() if hasattr(segment, "model_dump") else segment.dict() for segment in item.segments],
            }
            for item in prediction.predictions
        ],
        "global_confidence": prediction.global_confidence,
        "notes": prediction.notes,
    }


def save_prediction_json(
    run_dir: Path | str,
    prediction: VideoPrediction,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    video_fps: float | None = None,
    media_processing: str | None = None,
) -> Path:
    paths = ensure_run_structure(run_dir)
    out_path = paths.predictions_dir / f"{prediction.video_id}.json"
    save_json(
        out_path,
        _prediction_to_json_payload(
            prediction,
            model_name=model_name,
            prompt_version=prompt_version,
            video_fps=video_fps,
            media_processing=media_processing,
        ),
    )
    return out_path


def save_raw_response(run_dir: Path | str, video_id: str, raw_response: object) -> Path:
    paths = ensure_run_structure(run_dir)
    out_path = paths.raw_responses_dir / f"{video_id}.raw.json"
    if isinstance(raw_response, str):
        out_path.write_text(raw_response, encoding="utf-8")
    else:
        save_json(out_path, {"raw_response": raw_response})
    return out_path


def save_labels_npy(run_dir: Path | str, video_id: str, labels: np.ndarray) -> Path:
    paths = ensure_run_structure(run_dir)
    out_path = paths.labels_npy_dir / f"{video_id}.npy"
    np.save(out_path, labels)
    return out_path


def save_labels_csv(run_dir: Path | str, video_id: str, labels: np.ndarray, fps: float) -> Path:
    paths = ensure_run_structure(run_dir)
    out_path = paths.labels_csv_dir / f"{video_id}.csv"
    labels = np.asarray(labels)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "time_s", *ERROR_CLASSES])
        for frame_index in range(labels.shape[1]):
            row = [frame_index, round(frame_index / fps, 3), *labels[:, frame_index].tolist()]
            writer.writerow(row)
    return out_path


def save_segments_csv(run_dir: Path | str, rows: Sequence[Mapping[str, object]]) -> Path:
    paths = ensure_run_structure(run_dir)
    out_path = paths.segments_dir / "all_segments.csv"
    fieldnames = ["video_id", "error_type", "start_s", "end_s", "confidence", "model_name", "prompt_version"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return out_path


def save_metrics_rows(run_dir: Path | str, filename: str, rows: Sequence[Mapping[str, object]]) -> Path:
    paths = ensure_run_structure(run_dir)
    out_path = paths.metrics_dir / filename
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return out_path
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def save_summary_json(run_dir: Path | str, payload: Mapping[str, object]) -> Path:
    paths = ensure_run_metadata_structure(run_dir)
    out_path = paths.metrics_dir / "summary.json"
    save_json(out_path, payload)
    return out_path


def append_case_result(run_dir: Path | str, row: Mapping[str, object]) -> Path:
    paths = ensure_run_metadata_structure(run_dir)
    out_path = paths.logs_dir / "cases.csv"
    fieldnames = [
        "video_id",
        "video_path",
        "duration_s",
        "fps",
        "num_frames",
        "status",
        "started_at",
        "finished_at",
        "elapsed_s",
        "model_name",
        "prompt_version",
        "prediction_path",
        "raw_response_path",
        "labels_npy_path",
        "labels_csv_path",
        "message",
        "traceback",
    ]
    row_data = {field: row.get(field, "") for field in fieldnames}
    file_exists = out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    return out_path


def append_failure(run_dir: Path | str, *, video_id: str, video_path: str, status: str, error_message: str) -> Path:
    paths = ensure_run_metadata_structure(run_dir)
    out_path = paths.logs_dir / "failures.csv"
    file_exists = out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["video_id", "video_path", "status", "error_message", "timestamp"])
        writer.writerow([video_id, video_path, status, error_message, datetime.utcnow().isoformat(timespec="seconds")])
    return out_path
