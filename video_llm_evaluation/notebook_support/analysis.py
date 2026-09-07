"""Analysis helpers for the squat results inspector notebook."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from video_llm_evaluation.constants import ERROR_CLASSES
from video_llm_evaluation.notebook_support.data import ground_truth_errors_for_video, ground_truth_row_for_video


def predicted_errors_from_payload(payload: dict) -> list[str]:
    errors: list[str] = []
    for item in payload.get('predictions', []):
        if item.get('present'):
            errors.append(str(item.get('error_type', '')))
    return errors


def intervals_from_payload(payload: dict) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {error: [] for error in ERROR_CLASSES}
    for item in payload.get('predictions', []):
        error_type = str(item.get('error_type', ''))
        for segment in item.get('segments', []) or []:
            intervals.setdefault(error_type, []).append((float(segment.get('start_s', 0.0)), float(segment.get('end_s', 0.0))))
    return intervals


def gt_summary_for_video(dataset_df: pd.DataFrame, video_path: str | Path) -> dict[str, object]:
    subset = ground_truth_row_for_video(dataset_df, video_path)
    if subset.empty:
        return {'gt_errors': [], 'gt_labels_matrix': None, 'gt_active_rows': [], 'gt_row': None}
    row = subset.iloc[0]
    return {
        'gt_errors': list(row.get('gt_errors', [])),
        'gt_labels_matrix': row.get('gt_labels_matrix'),
        'gt_active_rows': list(row.get('gt_active_rows', [])),
        'gt_row': row,
    }


def build_summary_dataframe(results_df: pd.DataFrame, dataset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, result_row in results_df.iterrows():
        prediction_path = Path(result_row['prediction_path'])
        prediction_payload = json.loads(prediction_path.read_text(encoding='utf-8'))
        video_path_value = result_row.get('video_path', result_row.get('video_path_from_results', ''))
        gt_subset = ground_truth_row_for_video(dataset_df, video_path_value)
        gt_errors = ground_truth_errors_for_video(dataset_df, video_path_value)
        pred_errors = predicted_errors_from_payload(prediction_payload)
        matched_video_path = None if gt_subset.empty else str(gt_subset.iloc[0].get('video_path', ''))
        rows.append(
            {
                'video_id': result_row.get('video_id'),
                'video_path': video_path_value,
                'gt_match_found': not gt_subset.empty,
                'gt_matched_video_path': matched_video_path,
                'gt_errors': gt_errors,
                'pred_errors': pred_errors,
                'gt_present': bool(gt_errors),
                'pred_present': bool(pred_errors),
                'match_status': set(gt_errors) == set(pred_errors),
                'prediction_path': str(prediction_path),
            }
        )
    return pd.DataFrame(rows)
