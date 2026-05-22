"""Data-loading helpers for the squat results inspector notebook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEOS_ROOT = (REPO_ROOT / "../../videos/Nagrania/Squat/preprocessed").resolve()


def normalize_video_key(value: str | Path) -> str:
    path = Path(str(value))
    return str(path).replace('\\', '/').replace('.mp4', '').replace('.mov', '').replace('.mkv', '').replace('.avi', '')


def load_prediction_json(prediction_path: Path) -> dict:
    payload = json.loads(prediction_path.read_text(encoding='utf-8'))
    payload['_prediction_path'] = str(prediction_path)
    return payload


def load_manifest(results_root: Path) -> pd.DataFrame:
    manifest_path = results_root / 'manifest.csv'
    if not manifest_path.exists():
        return pd.DataFrame(columns=['video_id', 'video_path', 'duration_s', 'fps', 'num_frames', 'ground_truth_path'])
    return pd.read_csv(manifest_path)


def load_results_dataframe(results_root: Path, *, videos_root: Path) -> pd.DataFrame:
    manifest_df = load_manifest(results_root)
    rows: list[dict] = []
    for prediction_path in sorted(results_root.rglob('predictions_json/*.json')):
        payload = load_prediction_json(prediction_path)
        video_id = str(payload.get('video_id') or prediction_path.stem)
        relative_dir = prediction_path.parent.parent.relative_to(results_root)
        rows.append(
            {
                'video_id': video_id,
                'prediction_path': str(prediction_path),
                'relative_dir': str(relative_dir),
                'video_path_from_results': str(videos_root / Path(relative_dir)),
                'duration_s': payload.get('duration_s'),
                'model_name': payload.get('model_name'),
                'prompt_version': payload.get('prompt_version'),
                'predictions': payload.get('predictions', []),
                'global_confidence': payload.get('global_confidence'),
                'notes': payload.get('notes'),
            }
        )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    if not manifest_df.empty:
        manifest_df = manifest_df.copy()
        manifest_df['video_id'] = manifest_df['video_id'].astype(str)
        results_df['video_id'] = results_df['video_id'].astype(str)
        results_df = results_df.merge(manifest_df, on='video_id', how='left', suffixes=('', '_manifest'))
        if 'video_path_manifest' in results_df.columns:
            results_df['video_path'] = results_df['video_path_manifest'].fillna(results_df['video_path'])
            results_df = results_df.drop(columns=['video_path_manifest'])

    return results_df


def load_mlpsd_dataframe(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f'Cannot find MLPSD dataset at {dataset_path}')
    df = pd.read_pickle(dataset_path)
    df = df.copy()
    df['video_path'] = df['video_path'].astype(str)
    df['relative_video_key'] = df['video_path'].map(normalize_video_key)
    df['gt_labels_matrix'] = df['labels']
    df['gt_active_rows'] = df['errors_list'].apply(lambda value: [index for index, item in enumerate(value) if bool(item)] if isinstance(value, (list, tuple, np.ndarray)) else [])
    df['gt_errors'] = df['errors'].apply(lambda value: list(value) if isinstance(value, (list, tuple, set)) else ([] if pd.isna(value) else [value]))
    return df


def ground_truth_row_for_video(dataset_df: pd.DataFrame, video_path: str | Path) -> pd.DataFrame:
    key = normalize_video_key(video_path)
    subset = dataset_df[dataset_df['relative_video_key'].str.endswith(key) | dataset_df['relative_video_key'].eq(key)]
    return subset


def ground_truth_errors_for_video(dataset_df: pd.DataFrame, video_path: str | Path) -> list[str]:
    subset = ground_truth_row_for_video(dataset_df, video_path)
    if subset.empty:
        return []
    first = subset.iloc[0]
    return list(first.get('gt_errors', []))


def select_video(results_df: pd.DataFrame, dataset_df: pd.DataFrame, *, selected_video_id: str | None = None, selected_index: int = 0) -> tuple[pd.Series | None, pd.Series | None]:
    chosen_result = None
    if not results_df.empty:
        if selected_video_id is not None and 'video_id' in results_df.columns:
            matches = results_df[results_df['video_id'].astype(str) == str(selected_video_id)]
            if not matches.empty:
                chosen_result = matches.iloc[0]
        if chosen_result is None:
            chosen_result = results_df.iloc[min(selected_index, len(results_df) - 1)]

    chosen_dataset = None
    if chosen_result is not None and 'video_path' in chosen_result:
        matches = ground_truth_row_for_video(dataset_df, chosen_result.get('video_path', ''))
        if not matches.empty:
            chosen_dataset = matches.iloc[0]
    if chosen_dataset is None and not dataset_df.empty:
        chosen_dataset = dataset_df.iloc[min(selected_index, len(dataset_df) - 1)]

    return chosen_result, chosen_dataset


def resolve_video_path(video_path_value: str | Path, *, videos_root: Path) -> Path:
    path = Path(str(video_path_value))
    if path.is_absolute():
        return path
    return videos_root / path
