"""CLI entrypoints for Gemini-based squat video evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
import traceback
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from llm_api.gemini import GeminiClient, GeminiPromptBuilder, GeminiVideoConfig
from video_llm_evaluation.constants import (
    AGENTIC_CAPABLE_MODEL_PREFIXES,
    DEFAULT_MEDIA_PROCESSING,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_VIDEO_FPS,
    ERROR_CLASSES,
    MLPSD_ERROR_ROW_INDICES,
)
from video_llm_evaluation.discovery import discover_videos
from video_llm_evaluation.evaluation.persistence import (
    append_case_result,
    append_failure,
    save_config,
    save_json,
    save_labels_csv,
    save_labels_npy,
    save_metrics_rows,
    save_prediction_json,
    save_raw_response,
    save_summary_json,
)
from video_llm_evaluation.evaluation.segments_to_frames import video_prediction_to_frame_labels
from video_llm_evaluation.evaluation.metrics import frame_metrics, segment_metrics, video_level_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEOS_ROOT = (REPO_ROOT / "../../videos/Nagrania/Squat/preprocessed").resolve()
DEFAULT_RESULTS_ROOT = (REPO_ROOT / "results/llm_evaluation/squat").resolve()
DEFAULT_MLPSD_DATASET_PATH = (
    REPO_ROOT.parents[1] / "dataset/datasets/MLPSD/final_dataset/latest/MLPSD_v2.0_feature_extracted_v3.0.0_all_with_holistic_phases.pkl"
).resolve()

RESULTS_TO_MLPSD_FOLDER = {"formtext": "formcheck_text"}


LOGGER_NAME = "video_llm_evaluation"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _setup_logging(results_root: Path) -> tuple[logging.Logger, Path]:
    log_dir = results_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(file_handler)
    warnings_logger.addHandler(stream_handler)
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False

    return logger, log_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video_llm_evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Gemini requests over squat videos")
    run_parser.add_argument("--videos-root", type=Path, default=DEFAULT_VIDEOS_ROOT)
    run_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    run_parser.add_argument("--prompt-yaml", type=Path, default=None)
    run_parser.add_argument("--max-request", "--max_request", dest="max_request", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--model-name", type=str, default="gemini-3.1-pro-preview")
    run_parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY")
    run_parser.add_argument(
        "--preferred-folder",
        type=Path,
        default=None,
        help="Optional subfolder of videos-root to prioritize first when selecting videos.",
    )
    run_parser.add_argument("--skip-existing", action="store_true", default=False)
    run_parser.add_argument(
        "--video-fps",
        "--video_fps",
        dest="video_fps",
        type=float,
        default=DEFAULT_VIDEO_FPS,
        help=f"Frames per second Gemini samples from each video (default {DEFAULT_VIDEO_FPS}; API default is 1.0).",
    )
    run_parser.add_argument(
        "--media-processing",
        dest="media_processing",
        type=str,
        default=DEFAULT_MEDIA_PROCESSING,
        choices=["STATIC", "AGENTIC"],
        help="STATIC pins fixed-rate sampling so --video-fps applies; AGENTIC lets the model navigate and ignores it.",
    )
    run_parser.add_argument(
        "--media-resolution",
        dest="media_resolution",
        type=str,
        default=None,
        choices=["MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH", "MEDIA_RESOLUTION_ULTRA_HIGH"],
        help="Token resolution per frame. Unset leaves the API default.",
    )
    run_parser.add_argument(
        "--thinking-level",
        dest="thinking_level",
        type=str,
        default=DEFAULT_THINKING_LEVEL,
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH", "NONE"],
        help=f"Reasoning effort (default {DEFAULT_THINKING_LEVEL}); NONE leaves the API default.",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate saved LLM labels against MLPSD ground truth")
    evaluate_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    evaluate_parser.add_argument("--dataset-path", type=Path, default=DEFAULT_MLPSD_DATASET_PATH)
    evaluate_parser.add_argument(
        "--max-frame-difference",
        type=int,
        default=1,
        help="Trim only frame-count mismatches up to this size; skip larger mismatches.",
    )
    return parser


def _is_in_preferred_folder(video_path: Path, videos_root: Path, preferred_folder: Path) -> bool:
    preferred_root = preferred_folder if preferred_folder.is_absolute() else (videos_root / preferred_folder)
    try:
        return video_path.resolve().is_relative_to(preferred_root.resolve())
    except AttributeError:
        return str(video_path.resolve()).startswith(str(preferred_root.resolve()))


def _select_videos(items, max_request: int | None, seed: int, *, videos_root: Path, preferred_folder: Path | None = None):
    items = list(items)
    if preferred_folder is not None:
        preferred_items = [item for item in items if _is_in_preferred_folder(item.video_path, videos_root, preferred_folder)]
        remaining_items = [item for item in items if item not in preferred_items]
    else:
        preferred_items = []
        remaining_items = items

    if max_request is None or max_request >= len(items):
        return preferred_items + remaining_items

    rng = random.Random(seed)
    if preferred_folder is None:
        return rng.sample(items, k=max_request)

    if len(preferred_items) >= max_request:
        return rng.sample(preferred_items, k=max_request)

    selected = list(preferred_items)
    remaining_needed = max_request - len(selected)
    if remaining_needed > 0:
        selected.extend(rng.sample(remaining_items, k=remaining_needed))
    return selected


def _normalize_raw_response(response: object) -> str | Mapping[str, object]:
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return getattr(response, "text")
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return cast(str | Mapping[str, object], model_dump())
        except Exception:
            pass
    return {"repr": repr(response)}


def _run(args: argparse.Namespace) -> None:
    results_root = args.results_root
    videos_root = args.videos_root
    results_root.mkdir(parents=True, exist_ok=True)
    logger, log_path = _setup_logging(results_root)
    run_started_at = _utc_now()
    run_started_perf = time.perf_counter()
    logger.info(
        "Starting run: videos_root=%s results_root=%s max_request=%s seed=%s preferred_folder=%s skip_existing=%s model_name=%s",
        videos_root,
        results_root,
        args.max_request,
        args.seed,
        args.preferred_folder,
        args.skip_existing,
        args.model_name,
    )

    prompt_builder = GeminiPromptBuilder(template_path=args.prompt_yaml) if args.prompt_yaml else GeminiPromptBuilder()
    thinking_level = None if args.thinking_level == "NONE" else args.thinking_level
    client = GeminiClient(
        config=GeminiVideoConfig(
            model_name=args.model_name,
            api_key_env=args.api_key_env,
            video_fps=args.video_fps,
            media_processing=args.media_processing,
            media_resolution=args.media_resolution,
            thinking_level=thinking_level,
        ),
        prompt_builder=prompt_builder,
    )
    if client.client is None:
        raise RuntimeError(
            f"Gemini API key not found in environment variable {args.api_key_env}. "
            f"Set it before running, for example: export {args.api_key_env}=<your_key>"
        )

    client.verify_model_available()
    logger.info(
        "Model %s verified. media_processing=%s video_fps=%s media_resolution=%s thinking_level=%s",
        args.model_name,
        args.media_processing,
        args.video_fps,
        args.media_resolution or "api-default",
        thinking_level or "api-default",
    )
    if args.media_processing == "STATIC" and args.model_name.startswith(AGENTIC_CAPABLE_MODEL_PREFIXES):
        logger.info(
            "Model %s supports AGENTIC processing; pinning STATIC so --video-fps=%s is honoured.",
            args.model_name,
            args.video_fps,
        )

    discovered = discover_videos(videos_root, results_root)
    selected = _select_videos(discovered, args.max_request, args.seed, videos_root=videos_root, preferred_folder=args.preferred_folder)
    logger.info("Discovered %s videos, selected %s", len(discovered), len(selected))

    save_config(
        results_root,
        {
            "videos_root": str(videos_root),
            "results_root": str(results_root),
            "max_request": args.max_request,
            "seed": args.seed,
            "model_name": args.model_name,
            "video_fps": args.video_fps,
            "media_processing": args.media_processing,
            "media_resolution": args.media_resolution,
            "thinking_level": thinking_level,
            "prompt_version": client.prompt_builder.prompt_version,
            "prompt_yaml": str(args.prompt_yaml) if args.prompt_yaml else None,
            "preferred_folder": str(args.preferred_folder) if args.preferred_folder else None,
            "num_discovered": len(discovered),
            "num_selected": len(selected),
            "log_path": str(log_path),
        },
    )

    totals = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}
    for item in selected:
        run_paths = item.output_dir
        started_at = _utc_now()
        started_perf = time.perf_counter()
        if args.skip_existing and (run_paths / "predictions_json" / f"{item.video_id}.json").exists():
            elapsed_s = time.perf_counter() - started_perf
            totals["skipped"] += 1
            totals["processed"] += 1
            logger.info("SKIP video_id=%s elapsed=%.3fs reason=existing prediction", item.video_id, elapsed_s)
            append_case_result(
                results_root,
                {
                    "video_id": item.video_id,
                    "video_path": str(item.video_path),
                    "duration_s": item.duration_s,
                    "fps": item.fps,
                    "num_frames": item.num_frames,
                    "status": "skipped",
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "elapsed_s": round(elapsed_s, 3),
                    "model_name": args.model_name,
                    "prompt_version": client.prompt_builder.prompt_version,
                    "prediction_path": str(run_paths / "predictions_json" / f"{item.video_id}.json"),
                    "raw_response_path": str(run_paths / "raw_responses" / f"{item.video_id}.raw.json"),
                    "labels_npy_path": str(run_paths / "labels_npy" / f"{item.video_id}.npy"),
                    "labels_csv_path": str(run_paths / "labels_csv" / f"{item.video_id}.csv"),
                    "message": "existing prediction",
                },
            )
            continue

        logger.info("START video_id=%s video_path=%s", item.video_id, item.video_path)
        try:
            response = client.generate(video_path=str(item.video_path), video_id=item.video_id, duration_s=item.duration_s)
            raw_response = _normalize_raw_response(response)
            # Persist the response before parsing it: a malformed response is
            # exactly the case worth inspecting, and parsing it away first meant
            # the evidence was discarded with the exception.
            raw_response_path = save_raw_response(run_paths, item.video_id, raw_response)

            prediction = client.parse_response(raw_response, video_id=item.video_id, duration_s=item.duration_s)
            labels = video_prediction_to_frame_labels(prediction, fps=item.fps, num_frames=item.num_frames)

            prediction_path = save_prediction_json(
                run_paths,
                prediction,
                model_name=args.model_name,
                prompt_version=client.prompt_builder.prompt_version,
                video_fps=args.video_fps,
                media_processing=args.media_processing,
            )
            labels_npy_path = save_labels_npy(run_paths, item.video_id, labels)
            labels_csv_path = save_labels_csv(run_paths, item.video_id, labels, fps=item.fps)

            elapsed_s = time.perf_counter() - started_perf
            totals["processed"] += 1
            totals["succeeded"] += 1
            logger.info("OK video_id=%s elapsed=%.3fs prediction_path=%s", item.video_id, elapsed_s, prediction_path)
            append_case_result(
                results_root,
                {
                    "video_id": item.video_id,
                    "video_path": str(item.video_path),
                    "duration_s": item.duration_s,
                    "fps": item.fps,
                    "num_frames": item.num_frames,
                    "status": "success",
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "elapsed_s": round(elapsed_s, 3),
                    "model_name": args.model_name,
                    "prompt_version": client.prompt_builder.prompt_version,
                    "prediction_path": str(prediction_path),
                    "raw_response_path": str(raw_response_path),
                    "labels_npy_path": str(labels_npy_path),
                    "labels_csv_path": str(labels_csv_path),
                    "message": "",
                },
            )
        except Exception as exc:
            elapsed_s = time.perf_counter() - started_perf
            totals["processed"] += 1
            totals["failed"] += 1
            error_message = f"{type(exc).__name__}: {exc}"
            logger.exception("ERROR video_id=%s elapsed=%.3fs %s", item.video_id, elapsed_s, error_message)
            append_failure(results_root, video_id=item.video_id, video_path=str(item.video_path), status="error", error_message=error_message)
            append_case_result(
                results_root,
                {
                    "video_id": item.video_id,
                    "video_path": str(item.video_path),
                    "duration_s": item.duration_s,
                    "fps": item.fps,
                    "num_frames": item.num_frames,
                    "status": "error",
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "elapsed_s": round(elapsed_s, 3),
                    "model_name": args.model_name,
                    "prompt_version": client.prompt_builder.prompt_version,
                    "prediction_path": "",
                    "raw_response_path": "",
                    "labels_npy_path": "",
                    "labels_csv_path": "",
                    "message": error_message,
                    "traceback": traceback.format_exc(),
                },
            )

    manifest_path = results_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "video_path", "duration_s", "fps", "num_frames", "ground_truth_path"])
        for item in selected:
            writer.writerow([item.video_id, str(item.video_path), item.duration_s, item.fps, item.num_frames, ""])

    total_elapsed_s = round(time.perf_counter() - run_started_perf, 3)
    summary = {
        "run_started_at": run_started_at,
        "run_finished_at": _utc_now(),
        "total_elapsed_s": total_elapsed_s,
        "videos_root": str(videos_root),
        "results_root": str(results_root),
        "selected": len(selected),
        "processed": totals["processed"],
        "succeeded": totals["succeeded"],
        "failed": totals["failed"],
        "skipped": totals["skipped"],
        "log_path": str(log_path),
    }
    save_json(results_root / "metrics" / "run_summary.json", summary)
    logger.info("SUMMARY processed=%s succeeded=%s failed=%s skipped=%s total_elapsed=%.3fs log_path=%s", totals["processed"], totals["succeeded"], totals["failed"], totals["skipped"], total_elapsed_s, log_path)


def _normalise_video_key(value: str | Path) -> str:
    path = Path(str(value))
    return str(path.with_suffix("")).replace("\\", "/").casefold()


def _mlpsd_key_for_prediction(results_root: Path, labels_path: Path) -> str:
    relative_dir = labels_path.parent.parent.relative_to(results_root)
    relative_parts = list(relative_dir.parts)
    if len(relative_parts) < 2:
        raise ValueError(f"Unexpected prediction layout: {labels_path}")
    relative_parts[0] = RESULTS_TO_MLPSD_FOLDER.get(relative_parts[0], relative_parts[0])
    return "/".join(("squats", *relative_parts)).casefold()


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return precision, recall, _safe_divide(2 * precision * recall, precision + recall)


def _aggregate_video_metrics(per_video_rows: list[dict[str, object]]) -> dict[str, object]:
    per_class: list[dict[str, object]] = []
    total_tp = total_fp = total_fn = total_tn = 0

    for error_type in ERROR_CLASSES:
        rows = [row for row in per_video_rows if row["error_type"] == error_type]
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        tn = sum(int(row["tn"]) for row in rows)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        per_class.append(
            {
                "error_type": error_type,
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

    precision, recall, f1 = _precision_recall_f1(total_tp, total_fp, total_fn)
    accuracy = _safe_divide(total_tp + total_tn, total_tp + total_tn + total_fp + total_fn)
    return {
        "per_class": per_class,
        "accuracy": accuracy,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


def _aggregate_segment_metrics(per_video_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in per_video_rows:
        grouped.setdefault((str(row["error_type"]), float(row["tolerance_s"])), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for (error_type, tolerance_s), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        ious = [float(row["mean_temporal_iou"]) for row in rows if row["mean_temporal_iou"] is not None]
        start_maes = [float(row["start_mae_s"]) for row in rows if row["start_mae_s"] is not None]
        end_maes = [float(row["end_mae_s"]) for row in rows if row["end_mae_s"] is not None]
        aggregate_rows.append(
            {
                "error_type": error_type,
                "tolerance_s": tolerance_s,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_temporal_iou": float(np.mean(ious)) if ious else None,
                "start_mae_s": float(np.mean(start_maes)) if start_maes else None,
                "end_mae_s": float(np.mean(end_maes)) if end_maes else None,
            }
        )
    return aggregate_rows


def _evaluate(args: argparse.Namespace) -> None:
    if args.max_frame_difference < 0:
        raise ValueError("--max-frame-difference must be non-negative")
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Cannot find MLPSD dataset: {args.dataset_path}")

    import pandas as pd

    dataset_df = pd.read_pickle(args.dataset_path)
    required_columns = {"video_path", "labels"}
    missing_columns = required_columns - set(dataset_df.columns)
    if missing_columns:
        raise ValueError(f"MLPSD dataset is missing columns: {', '.join(sorted(missing_columns))}")

    dataset_by_key: dict[str, list[Any]] = {}
    for _, row in dataset_df.iterrows():
        dataset_by_key.setdefault(_normalise_video_key(row["video_path"]), []).append(row)

    labels_paths = sorted(args.results_root.rglob("labels_npy/*.npy"))
    if not labels_paths:
        raise FileNotFoundError(f"No saved LLM labels found below {args.results_root}")

    case_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    all_gt_labels: list[np.ndarray] = []
    all_pred_labels: list[np.ndarray] = []

    for labels_path in labels_paths:
        video_id = labels_path.stem
        try:
            dataset_key = _mlpsd_key_for_prediction(args.results_root, labels_path)
            matches = dataset_by_key.get(dataset_key, [])
            if not matches:
                raise ValueError(f"no MLPSD recording matches {dataset_key}")
            if len(matches) > 1:
                raise ValueError(f"ambiguous MLPSD match for {dataset_key}: {len(matches)} rows")

            pred_labels = np.asarray(np.load(labels_path), dtype=np.uint8)
            if pred_labels.ndim != 2 or pred_labels.shape[0] != len(MLPSD_ERROR_ROW_INDICES):
                raise ValueError(f"prediction must have shape (6, T), got {pred_labels.shape}")

            source_labels = np.asarray(matches[0]["labels"], dtype=np.uint8)
            if source_labels.ndim != 2 or source_labels.shape[0] <= max(MLPSD_ERROR_ROW_INDICES):
                raise ValueError(f"MLPSD labels must contain at least 8 rows, got {source_labels.shape}")
            gt_labels = source_labels[list(MLPSD_ERROR_ROW_INDICES)]

            frame_difference = abs(gt_labels.shape[1] - pred_labels.shape[1])
            alignment = "exact"
            if frame_difference:
                if frame_difference > args.max_frame_difference:
                    raise ValueError(
                        f"frame-count mismatch: prediction={pred_labels.shape[1]}, ground_truth={gt_labels.shape[1]}"
                    )
                shared_frames = min(gt_labels.shape[1], pred_labels.shape[1])
                gt_labels = gt_labels[:, :shared_frames]
                pred_labels = pred_labels[:, :shared_frames]
                alignment = f"trimmed_to_{shared_frames}"

            fps = float(matches[0].get("fps", 0.0))
            if fps <= 0:
                raise ValueError(f"invalid MLPSD fps: {fps}")

            frame_level = frame_metrics(gt_labels, pred_labels)
            video_level = video_level_metrics(gt_labels, pred_labels)
            segment_level = segment_metrics(gt_labels, pred_labels, fps=fps)
            all_gt_labels.append(gt_labels)
            all_pred_labels.append(pred_labels)

            frame_rows.extend({"video_id": video_id, **row} for row in frame_level["per_class"])
            video_rows.extend({"video_id": video_id, **row} for row in video_level["per_class"])
            segment_rows.extend(
                {
                    "video_id": video_id,
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
            )
            case_rows.append(
                {
                    "video_id": video_id,
                    "status": "evaluated",
                    "alignment": alignment,
                    "prediction_frames": pred_labels.shape[1],
                    "ground_truth_frames": gt_labels.shape[1],
                    "mlpsd_video_path": str(matches[0]["video_path"]),
                    "message": "",
                }
            )
        except Exception as exc:
            case_rows.append(
                {
                    "video_id": video_id,
                    "status": "skipped",
                    "alignment": "",
                    "prediction_frames": "",
                    "ground_truth_frames": "",
                    "mlpsd_video_path": "",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    if not all_gt_labels:
        raise RuntimeError("No recordings could be evaluated; inspect metrics/evaluation_cases.csv")

    aggregate_frame = frame_metrics(np.concatenate(all_gt_labels, axis=1), np.concatenate(all_pred_labels, axis=1))
    aggregate_video = _aggregate_video_metrics(video_rows)
    aggregate_segment = _aggregate_segment_metrics(segment_rows)

    save_metrics_rows(args.results_root, "evaluation_cases.csv", case_rows)
    save_metrics_rows(args.results_root, "frame_metrics_by_video.csv", frame_rows)
    save_metrics_rows(args.results_root, "video_level_metrics_by_video.csv", video_rows)
    save_metrics_rows(args.results_root, "segment_metrics_by_video.csv", segment_rows)
    save_metrics_rows(args.results_root, "frame_metrics.csv", aggregate_frame["per_class"])
    save_metrics_rows(args.results_root, "video_level_metrics.csv", aggregate_video["per_class"])
    save_metrics_rows(args.results_root, "segment_metrics.csv", aggregate_segment)

    evaluated = sum(row["status"] == "evaluated" for row in case_rows)
    summary = {
        "dataset_path": str(args.dataset_path),
        "results_root": str(args.results_root),
        "found_predictions": len(labels_paths),
        "evaluated": evaluated,
        "skipped": len(case_rows) - evaluated,
        "mlpsd_error_row_indices": list(MLPSD_ERROR_ROW_INDICES),
        "frame_metrics": {key: value for key, value in aggregate_frame.items() if key != "per_class"},
        "video_metrics": {key: value for key, value in aggregate_video.items() if key != "per_class"},
        "segment_metrics": aggregate_segment,
    }
    save_summary_json(args.results_root, summary)
    print(
        json.dumps(
            {
                "summary_path": str(args.results_root / "metrics" / "summary.json"),
                "found_predictions": len(labels_paths),
                "evaluated": evaluated,
                "skipped": len(case_rows) - evaluated,
                "frame_micro_f1": summary["frame_metrics"]["micro_f1"],
                "video_micro_f1": summary["video_metrics"]["micro_f1"],
            },
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args)
    elif args.command == "evaluate":
        _evaluate(args)


if __name__ == "__main__":
    main()
