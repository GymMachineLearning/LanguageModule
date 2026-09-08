"""Data-loading helpers for the squat results inspector notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from video_llm_evaluation.constants import ERROR_CLASSES, MLPSD_ERROR_ROW_INDICES
from video_llm_evaluation.dataset_split import mlpsd_key_for_video, normalise_video_key

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEOS_ROOT = (REPO_ROOT / "../../videos/Nagrania/Squat/preprocessed").resolve()


def normalize_video_key(value: str | Path) -> str:
    path = Path(str(value))
    return str(path).replace('\\', '/').replace('.mp4', '').replace('.mov', '').replace('.mkv', '').replace('.avi', '')



def _select_pipeline_error_rows(labels: object) -> object:
    """Reduce an MLPSD label matrix to the six pipeline classes, in ERROR_CLASSES order.

    MLPSD stores ten label rows; only six of them correspond to the classes this
    pipeline predicts, and they are not the first six. Every consumer of
    ``gt_labels_matrix`` indexes it with ``ERROR_CLASSES`` positions, so the row
    selection has to happen here rather than at each call site.
    """
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        return labels
    if labels.shape[0] <= max(MLPSD_ERROR_ROW_INDICES):
        return labels
    return labels[list(MLPSD_ERROR_ROW_INDICES)]


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
        if 'video_path' in results_df.columns:
            results_df['video_path'] = results_df['video_path'].fillna(results_df['video_path_from_results'])

    return results_df


def load_mlpsd_dataframe(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f'Cannot find MLPSD dataset at {dataset_path}')
    df = pd.read_pickle(dataset_path)
    df = df.copy()
    df['video_path'] = df['video_path'].astype(str)
    df['relative_video_key'] = df['video_path'].map(normalize_video_key)
    #: The key the whole pipeline joins on. ``relative_video_key`` above is kept
    #: only because it is displayed; it must not be used for matching.
    df['mlpsd_key'] = df['video_path'].map(normalise_video_key)
    df['gt_labels_matrix_raw'] = df['labels']
    df['gt_labels_matrix'] = df['labels'].apply(_select_pipeline_error_rows)

    def _active_rows_from_labels(value: object) -> list[int]:
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            return []
        active_rows: list[int] = []
        for row_index in range(value.shape[0]):
            if np.any(np.asarray(value[row_index]) == 1):
                active_rows.append(row_index)
        return active_rows

    def _error_names_from_rows(active_rows: list[int]) -> list[str]:
        names: list[str] = []
        for row_index in active_rows:
            if 0 <= row_index < len(ERROR_CLASSES):
                names.append(ERROR_CLASSES[row_index])
            else:
                names.append(f'row_{row_index}')
        return names

    df['gt_active_rows'] = df['gt_labels_matrix'].apply(_active_rows_from_labels)
    df['gt_errors'] = df['gt_active_rows'].apply(_error_names_from_rows)
    df['gt_errors_legacy'] = df['errors'].apply(lambda value: list(value) if isinstance(value, (list, tuple, set)) else ([] if pd.isna(value) else [value]))
    df['gt_active_rows_legacy'] = df['errors_list'].apply(lambda value: [index for index, item in enumerate(value) if bool(item)] if isinstance(value, (list, tuple, np.ndarray)) else [])
    return df


def mlpsd_key_for_display_path(video_path: str | Path, *, videos_root: Path | None = None) -> str | None:
    """MLPSD key for a recording the notebook is about to show, or None.

    Deliberately the same derivation ``cli.py evaluate`` uses, from the same
    module: the notebook and the metrics must agree about which ground truth
    belongs to which recording, and a second, looser derivation here is exactly
    how they stopped agreeing before.

    Returns None when the path does not sit under the video library, because a
    key cannot be derived from it — better no ground truth than someone else's.
    """
    root = Path(videos_root) if videos_root is not None else DEFAULT_VIDEOS_ROOT
    path = Path(str(video_path))
    try:
        return mlpsd_key_for_video(root, path)
    except ValueError:
        return None


def ground_truth_row_for_video(
    dataset_df: pd.DataFrame,
    video_path: str | Path,
    *,
    videos_root: Path | None = None,
) -> pd.DataFrame:
    """The MLPSD row for one recording: exactly one, or none at all.

    Matching is on the canonical key, not on the file name. Names collide across
    folders -- ``formcheck/1-0063-...`` and ``stronglifts5x5/0-0063-...`` are
    different recordings by different people -- so a name-based match hands back
    another recording's annotation with nothing to signal it.
    """
    if dataset_df.empty:
        return dataset_df
    key = mlpsd_key_for_display_path(video_path, videos_root=videos_root)
    if key is None:
        return dataset_df.iloc[0:0]
    if 'mlpsd_key' not in dataset_df.columns:
        keys = dataset_df['video_path'].astype(str).map(normalise_video_key)
    else:
        keys = dataset_df['mlpsd_key']
    return dataset_df[keys == key]


def ground_truth_errors_for_video(
    dataset_df: pd.DataFrame,
    video_path: str | Path,
    *,
    videos_root: Path | None = None,
) -> list[str]:
    subset = ground_truth_row_for_video(dataset_df, video_path, videos_root=videos_root)
    if subset.empty:
        return []
    first = subset.iloc[0]
    return list(first.get('gt_errors', []))


def select_video(
    results_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    *,
    selected_video_id: str | None = None,
    selected_index: int = 0,
    videos_root: Path | None = None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Pick one prediction to inspect and the MLPSD row that belongs to it.

    The MLPSD row is None when the recording has none. There is deliberately no
    fallback row: standing in an unrelated recording's annotation renders a
    ground-truth overlay that looks authoritative and is not.
    """
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
        matches = ground_truth_row_for_video(
            dataset_df,
            chosen_result.get('video_path', ''),
            videos_root=videos_root,
        )
        if not matches.empty:
            chosen_dataset = matches.iloc[0]

    return chosen_result, chosen_dataset


def resolve_video_path(video_path_value: str | Path, *, videos_root: Path) -> Path:
    if video_path_value is None:
        raise ValueError('Missing video path value')

    if isinstance(video_path_value, str) and video_path_value.strip().lower() in {'', 'nan', 'none'}:
        raise ValueError('Missing video path value')

    path = Path(str(video_path_value))
    if str(path).strip().lower() in {'', 'nan', 'none'}:
        raise ValueError('Missing video path value')
    if path.is_absolute():
        if path.exists():
            return path
    else:
        candidate = videos_root / path
        if candidate.exists():
            return candidate

    normalized_key = normalize_video_key(path)
    search_roots = [videos_root]
    if path.is_absolute():
        search_roots.append(path.parent)

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in search_root.rglob('*'):
            if not candidate.is_file():
                continue
            candidate_key = normalize_video_key(candidate)
            candidate_stem = candidate.stem
            if (
                candidate_key == normalized_key
                or candidate_key.endswith(normalized_key)
                or candidate_stem == path.stem
                or candidate_stem.endswith(path.stem)
            ):
                return candidate

    if path.is_absolute():
        return path
    return videos_root / path
