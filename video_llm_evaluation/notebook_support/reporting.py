"""Per-error-class reporting over the artifacts written by ``cli.py evaluate``.

The evaluation CLI already computes every metric and persists it under
``<results_root>/metrics``. This module only reads those files and arranges them
for reading, so the notebook and the CLI can never disagree about a number.

The organising axis is the error class, not the run average: the question these
tables answer is "how does the model do on *this* error", and run-wide
aggregates are shown only as closing context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from video_llm_evaluation.constants import DEFAULT_SPLIT, ERROR_CLASSES
from video_llm_evaluation.dataset_split import (
    format_composition,
    load_split_index,
    mlpsd_key_for_prediction,
)
from video_llm_evaluation.evaluation.persistence import metrics_dir_for_split

#: Files written by ``cli.py evaluate`` that this report reads.
AGGREGATE_FILES = {
    'frame_by_class': 'frame_metrics.csv',
    'video_by_class': 'video_level_metrics.csv',
    'segment_by_class': 'segment_metrics.csv',
}
PER_VIDEO_FILES = {
    'frame_by_video': 'frame_metrics_by_video.csv',
    'video_by_video': 'video_level_metrics_by_video.csv',
    'segment_by_video': 'segment_metrics_by_video.csv',
}
CASES_FILE = 'evaluation_cases.csv'
SUMMARY_FILE = 'summary.json'

#: Segment tolerances the CLI evaluates, in seconds.
DEFAULT_TOLERANCES = (0.1, 0.5, 1.0)

_NA = '—'

#: Shown above the segment tables, because "tolerance" is not self-explanatory.
TOLERANCE_EXPLAINER = (
    'Tolerancja to luz czasowy na **obu** końcach segmentu: predykcja liczy się jako trafienie '
    'tylko wtedy, gdy jej początek **i** koniec mieszczą się w ±tolerancji od granic segmentu '
    'z ground truth. Dopasowanie jest 1:1, zachłannie po malejącym IoU; niedopasowane predykcje '
    'to FP, niedopasowane segmenty GT to FN.\n\n'
    'Przy 60 fps `0.1s` to około 6 klatek (trafienie co do klatki), `1.0s` to 60 klatek '
    '(„mniej więcej w tym miejscu").'
)


class MetricsNotAvailable(RuntimeError):
    """Raised when a results root has no evaluation artifacts to report on."""


@dataclass(frozen=True)
class MetricsReport:
    """Everything ``cli.py evaluate`` wrote for one run, loaded and nothing more."""

    results_root: Path
    summary: dict
    config: dict
    frame_by_class: pd.DataFrame
    video_by_class: pd.DataFrame
    segment_by_class: pd.DataFrame
    frame_by_video: pd.DataFrame
    video_by_video: pd.DataFrame
    segment_by_video: pd.DataFrame
    cases: pd.DataFrame
    evaluated_at: datetime | None
    newest_labels_at: datetime | None
    split: str | None = None

    @property
    def is_stale(self) -> bool:
        """True when predictions were written after the metrics were computed."""
        if self.evaluated_at is None or self.newest_labels_at is None:
            return False
        return self.newest_labels_at > self.evaluated_at

    @property
    def evaluated_video_ids(self) -> list[str]:
        if self.cases.empty:
            return []
        return sorted(self.cases.loc[self.cases['status'] == 'evaluated', 'video_id'].astype(str))

    @property
    def tolerances(self) -> list[float]:
        if self.segment_by_class.empty:
            return list(DEFAULT_TOLERANCES)
        return sorted(float(value) for value in self.segment_by_class['tolerance_s'].unique())


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _evaluate_command(results_root: Path, split: str | None = None) -> str:
    command = f'python -m video_llm_evaluation.cli evaluate --results-root {results_root}'
    return f'{command} --split {split}' if split else command


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_config(results_root: Path) -> dict:
    config_path = results_root / 'config.yaml'
    if not config_path.exists():
        return {}
    try:
        import yaml

        return dict(yaml.safe_load(config_path.read_text(encoding='utf-8')) or {})
    except Exception:
        try:
            return dict(json.loads(config_path.read_text(encoding='utf-8')))
        except Exception:
            return {}


def _newest_labels_mtime(results_root: Path) -> datetime | None:
    mtimes = [path.stat().st_mtime for path in results_root.rglob('labels_npy/*.npy')]
    return datetime.fromtimestamp(max(mtimes)) if mtimes else None


def load_metrics_report(results_root: Path | str, *, split: str | None = DEFAULT_SPLIT) -> MetricsReport:
    """Load the evaluation artifacts for one run and one split.

    ``split`` selects which metrics directory to read, never which rows to keep:
    filtering happens in ``cli.py evaluate``, so the notebook cannot report
    numbers the CLI never computed.

    Raises:
        MetricsNotAvailable: when the metrics are missing or incomplete, with the
            exact command that would produce them.
    """
    root = Path(results_root)
    metrics_dir = metrics_dir_for_split(root, split)

    if not metrics_dir.exists():
        raise MetricsNotAvailable(
            f'No metrics directory under {root}.\n'
            f'Run the evaluation first:\n    {_evaluate_command(root, split)}'
        )

    summary_path = metrics_dir / SUMMARY_FILE
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for attribute, filename in {**AGGREGATE_FILES, **PER_VIDEO_FILES}.items():
        path = metrics_dir / filename
        frame = _read_csv(path)
        if frame.empty:
            missing.append(filename)
        frames[attribute] = frame

    if not summary_path.exists():
        missing.append(SUMMARY_FILE)

    if len(missing) == len(AGGREGATE_FILES) + len(PER_VIDEO_FILES) + 1:
        raise MetricsNotAvailable(
            f'{metrics_dir} holds no evaluation output.\n'
            f'Run the evaluation first:\n    {_evaluate_command(root, split)}'
        )
    if missing:
        raise MetricsNotAvailable(
            f'{metrics_dir} is incomplete, missing: {", ".join(sorted(missing))}.\n'
            f'Re-run the evaluation:\n    {_evaluate_command(root, split)}'
        )

    summary = json.loads(summary_path.read_text(encoding='utf-8'))

    return MetricsReport(
        results_root=root,
        summary=summary,
        config=_read_config(root),
        cases=_read_csv(metrics_dir / CASES_FILE),
        split=summary.get('split', split),
        evaluated_at=datetime.fromtimestamp(summary_path.stat().st_mtime),
        newest_labels_at=_newest_labels_mtime(root),
        **frames,
    )



# --------------------------------------------------------------------------- #
# Run discovery and selection
# --------------------------------------------------------------------------- #

#: Runs written before --video-fps existed always sampled at the API default.
PRE_FLAG_VIDEO_FPS = 1.0


def _run_predictions_count(run_dir: Path) -> int:
    return sum(1 for _ in run_dir.rglob("predictions_json/*.json"))


def _available_metric_splits(run_dir: Path) -> list[str]:
    """Which splits already have metrics computed for this run."""
    metrics_dir = run_dir / 'metrics'
    if not metrics_dir.exists():
        return []
    available = ['all'] if (metrics_dir / SUMMARY_FILE).exists() else []
    available += sorted(
        path.name.removeprefix('split_')
        for path in metrics_dir.iterdir()
        if path.is_dir() and path.name.startswith('split_') and (path / SUMMARY_FILE).exists()
    )
    return available


def discover_runs(runs_root: Path | str, *, dataset_path: Path | str | None = None) -> pd.DataFrame:
    """List the evaluation runs under a directory, one row per run.

    A directory counts as a run when it holds at least one prediction, which
    keeps scratch directories and the review app out of the table.

    ``video_fps`` is read from ``config.yaml`` where the run recorded it. Runs
    predating the flag report the API default with ``video_fps_source`` set to
    ``inferred`` — the value was never written down, so it must not be presented
    as though it had been.

    Passing ``dataset_path`` adds ``content`` — what the directory actually holds,
    counted per MLPSD split. A run's recorded ``split`` says what was requested;
    ``content`` says what is there, and the two differ for every directory that
    was filled before the flag existed.
    """
    root = Path(runs_root)
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame(rows)

    split_index = load_split_index(dataset_path) if dataset_path is not None else None

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        prediction_count = _run_predictions_count(run_dir)
        if not prediction_count:
            continue

        config = _read_config(run_dir)
        recorded_fps = config.get("video_fps")
        content = _NA
        if split_index is not None:
            keys = []
            for labels_path in run_dir.rglob("labels_npy/*.npy"):
                try:
                    keys.append(mlpsd_key_for_prediction(run_dir, labels_path))
                except ValueError:
                    continue
            content = format_composition(split_index.compose(keys))
        rows.append(
            {
                "run": run_dir.name,
                "model": config.get("model_name", _NA),
                "video_fps": float(recorded_fps) if recorded_fps is not None else PRE_FLAG_VIDEO_FPS,
                "video_fps_source": "recorded" if recorded_fps is not None else "inferred",
                "media_processing": config.get("media_processing", _NA),
                "thinking_level": config.get("thinking_level", _NA),
                "prompt_version": config.get("prompt_version", _NA),
                "predictions": prediction_count,
                "split_requested": config.get("split", _NA),
                "content": content,
                "metrics_for": ", ".join(_available_metric_splits(run_dir)) or _NA,
                "results_root": run_dir,
            }
        )
    return pd.DataFrame(rows)


def _describe_runs(runs: pd.DataFrame) -> str:
    if runs.empty:
        return "  (none)"
    return "\n".join(
        f"  model={row['model']} video_fps={row['video_fps']:g} "
        f"metrics_for=[{row['metrics_for']}]  -> {row['run']}"
        for _, row in runs.iterrows()
    )


def select_run(
    runs: pd.DataFrame,
    *,
    model: str | None = None,
    video_fps: float | None = None,
) -> Path:
    """Resolve one run from the discovery table by model and frame rate.

    Raises:
        LookupError: when nothing matches, or when the filters leave more than
            one run — both cases list what is actually available, because a
            report silently bound to the wrong run is worse than no report.
    """
    matches = runs
    if model is not None:
        matches = matches[matches["model"] == model]
    if video_fps is not None:
        matches = matches[np.isclose(matches["video_fps"].astype(float), float(video_fps))]

    criteria = ", ".join(
        part for part in (f"model={model!r}" if model else "", f"video_fps={video_fps}" if video_fps else "") if part
    ) or "no filter"

    if matches.empty:
        raise LookupError(f"No run matches {criteria}. Available runs:\n{_describe_runs(runs)}")
    if len(matches) > 1:
        raise LookupError(f"{len(matches)} runs match {criteria}; narrow it down:\n{_describe_runs(matches)}")
    return Path(matches.iloc[0]["results_root"])


def load_run_report(
    runs_root: Path | str,
    *,
    model: str | None = None,
    video_fps: float | None = None,
    split: str | None = DEFAULT_SPLIT,
) -> MetricsReport:
    """Discover runs under ``runs_root`` and load the one matching model and fps."""
    chosen = select_run(discover_runs(runs_root), model=model, video_fps=video_fps)
    return load_metrics_report(chosen, split=split)



def annotate_results_with_split(
    results_df: pd.DataFrame,
    results_root: Path | str,
    *,
    dataset_path: Path | str,
) -> pd.DataFrame:
    """Add a ``dataset_split`` column to the prediction table used for browsing."""
    if results_df.empty:
        return results_df
    root = Path(results_root)
    split_index = load_split_index(dataset_path)

    def _split_for(prediction_path: object) -> str:
        try:
            return split_index.split_for(mlpsd_key_for_prediction(root, Path(str(prediction_path))))
        except ValueError:
            return 'unmatched'

    annotated = results_df.copy()
    annotated['dataset_split'] = annotated['prediction_path'].map(_split_for)
    return annotated


def filter_results_by_split(
    results_df: pd.DataFrame,
    results_root: Path | str,
    *,
    split: str | None,
    dataset_path: Path | str,
) -> pd.DataFrame:
    """Keep only the predictions belonging to ``split``.

    This filters the *browsing* table only. Metrics are never filtered here —
    they come pre-computed from ``cli.py evaluate``, so the notebook cannot show
    a number the CLI did not produce.
    """
    annotated = annotate_results_with_split(results_df, results_root, dataset_path=dataset_path)
    if annotated.empty or split in (None, 'all'):
        return annotated
    return annotated[annotated['dataset_split'] == split].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

_LOW = (250, 207, 207)
_MID = (253, 244, 200)
_HIGH = (201, 237, 208)


def _lerp(left: tuple[int, int, int], right: tuple[int, int, int], t: float) -> str:
    channels = tuple(int(round(a + (b - a) * t)) for a, b in zip(left, right))
    return 'background-color: #{:02x}{:02x}{:02x}'.format(*channels)


def score_background(value: object) -> str:
    """Pale red-to-green background for a 0..1 score; empty string for non-scores."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ''
    if not np.isfinite(score):
        return ''
    score = min(max(score, 0.0), 1.0)
    if score < 0.5:
        return _lerp(_LOW, _MID, score / 0.5)
    return _lerp(_MID, _HIGH, (score - 0.5) / 0.5)


def _style_scores(frame: pd.DataFrame, score_columns: list[str], *, precision: int = 3):
    present = [column for column in score_columns if column in frame.columns]
    styler = frame.style.format(precision=precision, na_rep=_NA)
    if present:
        styler = styler.map(score_background, subset=present)
    return styler.set_properties(**{'text-align': 'right'})


# --------------------------------------------------------------------------- #
# Per-class views
# --------------------------------------------------------------------------- #


def class_overview(report: MetricsReport, *, tolerance_s: float = 1.0) -> pd.DataFrame:
    """One row per error class: frame, video and segment performance side by side.

    This is the table that answers "which errors does the model actually detect".
    """
    frame = report.frame_by_class.set_index('error_type')
    video = report.video_by_class.set_index('error_type')

    segment = report.segment_by_class
    segment_at_tolerance = (
        segment[np.isclose(segment['tolerance_s'], tolerance_s)].set_index('error_type')
        if not segment.empty
        else pd.DataFrame()
    )

    gt_videos, pred_videos = _class_video_counts(report)

    rows = []
    for error_type in ERROR_CLASSES:
        frame_row = frame.loc[error_type] if error_type in frame.index else None
        video_row = video.loc[error_type] if error_type in video.index else None
        segment_row = (
            segment_at_tolerance.loc[error_type]
            if error_type in getattr(segment_at_tolerance, 'index', [])
            else None
        )
        rows.append(
            {
                'error_type': error_type,
                'gt_videos': gt_videos.get(error_type, 0),
                'pred_videos': pred_videos.get(error_type, 0),
                'frame_precision': None if frame_row is None else float(frame_row['precision']),
                'frame_recall': None if frame_row is None else float(frame_row['recall']),
                'frame_f1': None if frame_row is None else float(frame_row['f1']),
                'frame_iou': None if frame_row is None else float(frame_row['iou']),
                'frame_support': None if frame_row is None else int(frame_row['support']),
                'video_precision': None if video_row is None else float(video_row['precision']),
                'video_recall': None if video_row is None else float(video_row['recall']),
                'video_f1': None if video_row is None else float(video_row['f1']),
                f'segment_f1@{tolerance_s:g}s': None if segment_row is None else float(segment_row['f1']),
            }
        )
    return pd.DataFrame(rows).set_index('error_type')


def _class_video_counts(report: MetricsReport) -> tuple[dict[str, int], dict[str, int]]:
    """How many evaluated videos contain each class in GT, and in predictions."""
    per_video = report.video_by_video
    if per_video.empty:
        return {}, {}
    gt_counts = (
        per_video[per_video['gt_present'].astype(str).str.lower() == 'true']
        .groupby('error_type')
        .size()
        .to_dict()
    )
    pred_counts = (
        per_video[per_video['pred_present'].astype(str).str.lower() == 'true']
        .groupby('error_type')
        .size()
        .to_dict()
    )
    return gt_counts, pred_counts


def style_class_overview(report: MetricsReport, *, tolerance_s: float = 1.0):
    frame = class_overview(report, tolerance_s=tolerance_s)
    score_columns = [column for column in frame.columns if 'f1' in column or 'precision' in column or 'recall' in column or 'iou' in column]
    return _style_scores(frame, score_columns).set_caption(
        f'Per-class performance (segment F1 at {tolerance_s:g}s tolerance)'
    )


def segment_pivot(report: MetricsReport, value: str = 'f1') -> pd.DataFrame:
    """Error classes as rows, segment tolerances as columns."""
    segment = report.segment_by_class
    if segment.empty or value not in segment.columns:
        return pd.DataFrame()
    pivot = segment.pivot_table(index='error_type', columns='tolerance_s', values=value, dropna=False)
    pivot = pivot.reindex(ERROR_CLASSES)
    pivot.columns = [f'{float(column):g}s' for column in pivot.columns]
    pivot.columns.name = 'tolerance'
    return pivot


def style_segment_pivot(report: MetricsReport, value: str = 'f1'):
    pivot = segment_pivot(report, value=value)
    if pivot.empty:
        return pivot
    if value in {'f1', 'precision', 'recall', 'mean_temporal_iou'}:
        return _style_scores(pivot, list(pivot.columns)).set_caption(f'Segment-level {value} by tolerance')
    return pivot.style.format(precision=3, na_rep=_NA).set_caption(f'Segment-level {value} by tolerance')


def class_detail(report: MetricsReport, error_type: str) -> dict[str, pd.DataFrame]:
    """Every table the run holds for a single error class."""
    if error_type not in ERROR_CLASSES:
        raise ValueError(f'Unknown error type: {error_type}')

    def _for_class(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or 'error_type' not in frame.columns:
            return pd.DataFrame()
        return frame[frame['error_type'] == error_type].drop(columns=['error_type'])

    segment = _for_class(report.segment_by_class)
    if not segment.empty:
        segment = segment.set_index('tolerance_s')
        segment.index = [f'{float(index):g}s' for index in segment.index]
        segment.index.name = 'tolerance'

    per_video = _for_class(report.frame_by_video)
    if not per_video.empty:
        video_presence = _for_class(report.video_by_video)[['video_id', 'gt_present', 'pred_present']]
        per_video = per_video.merge(video_presence, on='video_id', how='left').set_index('video_id')
        per_video = per_video.sort_values(['support', 'f1'], ascending=[False, True])

    return {
        'frame': _for_class(report.frame_by_class),
        'video': _for_class(report.video_by_class),
        'segment': segment,
        'per_video': per_video,
    }


def per_video_class_matrix(report: MetricsReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Videos as rows, error classes as columns.

    Returns a display frame of formatted cells and a parallel frame of CSS
    backgrounds. A cell is the frame-level F1 where the class is present in
    ground truth, ``FP`` where the model predicted a class that is absent, and
    ``·`` where the model correctly stayed silent.
    """
    frame = report.frame_by_video
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    display = pd.DataFrame(index=sorted(frame['video_id'].astype(str).unique()), columns=ERROR_CLASSES, dtype=object)
    colors = pd.DataFrame('', index=display.index, columns=display.columns)

    for _, row in frame.iterrows():
        video_id = str(row['video_id'])
        error_type = str(row['error_type'])
        if error_type not in display.columns:
            continue
        support = int(row['support'])
        if support > 0:
            display.loc[video_id, error_type] = f'{float(row["f1"]):.2f}'
            colors.loc[video_id, error_type] = score_background(float(row['f1']))
        elif int(row['fp']) > 0:
            display.loc[video_id, error_type] = 'FP'
            colors.loc[video_id, error_type] = 'background-color: #ffe0c2'
        else:
            display.loc[video_id, error_type] = '·'
            colors.loc[video_id, error_type] = 'color: #b0b0b0'

    display = display.fillna(_NA)
    display.index.name = 'video_id'
    return display, colors


def style_per_video_class_matrix(report: MetricsReport):
    display, colors = per_video_class_matrix(report)
    if display.empty:
        return display
    return (
        display.style.apply(lambda _: colors, axis=None)
        .set_properties(**{'text-align': 'center'})
        .set_caption('Frame-level F1 per video and class — FP = class absent in GT but predicted, · = correctly silent')
    )


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #


def run_header(report: MetricsReport) -> pd.DataFrame:
    """Scope of the run: what was evaluated, by which model, against which dataset."""
    summary = report.summary
    config = report.config
    recorded_fps = config.get('video_fps')
    rows = [
        ('results_root', str(report.results_root)),
        ('model_name', config.get('model_name', _NA)),
        (
            'video_fps',
            f"{float(recorded_fps):g}" if recorded_fps is not None else f'{PRE_FLAG_VIDEO_FPS:g} (inferred, pre-flag run)',
        ),
        ('split', report.split or _NA),
        ('evaluated_in_split', summary.get('evaluated', _NA)),
        ('skipped_by_split', summary.get('skipped_by_split', _NA)),
        ('media_processing', config.get('media_processing', _NA)),
        ('thinking_level', config.get('thinking_level', _NA)),
        ('dataset_path', Path(str(summary.get('dataset_path', ''))).name or _NA),
        ('found_predictions', summary.get('found_predictions', _NA)),
        ('evaluated', summary.get('evaluated', _NA)),
        ('skipped', summary.get('skipped', _NA)),
        ('mlpsd_error_rows', str(summary.get('mlpsd_error_row_indices', _NA))),
        ('metrics_computed_at', report.evaluated_at.strftime('%Y-%m-%d %H:%M') if report.evaluated_at else _NA),
    ]
    if config.get('num_selected') is not None:
        rows.insert(4, ('config_num_selected', config.get('num_selected')))
    return pd.DataFrame(rows, columns=['field', 'value']).set_index('field')


def staleness_warning(report: MetricsReport) -> str | None:
    """Message to show when predictions are newer than the metrics, else None."""
    if not report.is_stale:
        return None
    return (
        f'Predictions changed at {report.newest_labels_at:%Y-%m-%d %H:%M}, after the metrics '
        f'were computed at {report.evaluated_at:%Y-%m-%d %H:%M}. The tables below are stale.\n'
        f'Re-run:\n    {_evaluate_command(report.results_root, report.split)}'
    )


def _classify_skip_reason(message: str) -> str:
    text = str(message).lower()
    if 'no mlpsd recording matches' in text:
        return 'no MLPSD match'
    if 'frame-count mismatch' in text:
        return 'frame-count mismatch'
    if 'ambiguous mlpsd match' in text:
        return 'ambiguous MLPSD match'
    if 'must have shape' in text or 'must contain at least' in text:
        return 'unexpected label shape'
    if 'invalid mlpsd fps' in text:
        return 'invalid fps'
    return 'other'


def skipped_overview(report: MetricsReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Videos excluded from the metrics, and the distribution of reasons."""
    cases = report.cases
    if cases.empty:
        return pd.DataFrame(), pd.DataFrame()

    skipped = cases[cases['status'] != 'evaluated'].copy()
    if skipped.empty:
        return skipped, pd.DataFrame()

    skipped['reason'] = skipped['message'].map(_classify_skip_reason)
    reasons = (
        skipped.groupby('reason').size().rename('videos').sort_values(ascending=False).to_frame()
    )
    detail = skipped[['video_id', 'reason', 'message']].set_index('video_id')
    return detail, reasons


def run_totals(report: MetricsReport) -> pd.DataFrame:
    """Run-wide aggregates, as closing context for the per-class tables."""
    frame = report.summary.get('frame_metrics', {})
    video = report.summary.get('video_metrics', {})
    rows = [
        ('frame', 'macro_f1', frame.get('macro_f1')),
        ('frame', 'micro_f1', frame.get('micro_f1')),
        ('frame', 'weighted_f1', frame.get('weighted_f1')),
        ('frame', 'micro_precision', frame.get('micro_precision')),
        ('frame', 'micro_recall', frame.get('micro_recall')),
        ('frame', 'mean_iou', frame.get('mean_iou')),
        ('video', 'accuracy', video.get('accuracy')),
        ('video', 'micro_f1', video.get('micro_f1')),
        ('video', 'micro_precision', video.get('micro_precision')),
        ('video', 'micro_recall', video.get('micro_recall')),
    ]
    return pd.DataFrame(rows, columns=['level', 'metric', 'value']).set_index(['level', 'metric'])


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def _save_figure(figure, save_dir: Path | None, filename: str) -> Path | None:
    if save_dir is None:
        return None
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    figure.savefig(path, dpi=150, bbox_inches='tight')
    return path


def plot_class_f1(report: MetricsReport, *, save_dir: Path | None = None):
    """Frame-level vs video-level F1, one pair of bars per error class."""
    import matplotlib.pyplot as plt

    overview = class_overview(report)
    positions = np.arange(len(overview.index))
    width = 0.38

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(positions - width / 2, overview['frame_f1'].fillna(0), width, label='frame-level F1', color='#4c78a8')
    axis.bar(positions + width / 2, overview['video_f1'].fillna(0), width, label='video-level F1', color='#f58518')
    axis.set_xticks(positions)
    axis.set_xticklabels(overview.index, rotation=20, ha='right')
    axis.set_ylabel('F1')
    axis.set_ylim(0, 1)
    axis.set_title('Detection quality per error class')
    axis.legend(frameon=False)
    axis.grid(axis='y', alpha=0.25)
    axis.set_axisbelow(True)
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()

    _save_figure(figure, save_dir, 'class_f1.png')
    return figure


def plot_f1_vs_tolerance(report: MetricsReport, *, save_dir: Path | None = None):
    """Segment-level F1 against matching tolerance, one line per error class."""
    import matplotlib.pyplot as plt

    pivot = segment_pivot(report, value='f1')
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if pivot.empty:
        axis.text(0.5, 0.5, 'No segment metrics available', ha='center', va='center')
        axis.set_axis_off()
        return figure

    tolerances = report.tolerances
    for error_type in pivot.index:
        axis.plot(tolerances, pivot.loc[error_type].to_numpy(dtype=float), marker='o', label=error_type)
    axis.set_xlabel('matching tolerance (s)')
    axis.set_ylabel('segment F1')
    axis.set_ylim(bottom=0)
    axis.set_xticks(tolerances)
    axis.set_title('Segment F1 vs temporal tolerance')
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()

    _save_figure(figure, save_dir, 'segment_f1_vs_tolerance.png')
    return figure


# --------------------------------------------------------------------------- #
# Notebook entry point
# --------------------------------------------------------------------------- #


def _markdown(text: str) -> None:
    from IPython.display import Markdown, display

    display(Markdown(text))


def show_report(
    results_root: Path | str | MetricsReport,
    *,
    tolerance_s: float = 1.0,
    save_figures: bool = True,
    show_class_cards: bool = True,
    per_video: bool = True,
) -> MetricsReport:
    """Render the whole per-class report in a notebook. Returns the loaded report.

    Args:
        tolerance_s: which segment tolerance the per-class overview column reports.
        save_figures: also write the charts as PNG under ``metrics/figures``.
        show_class_cards: render the detailed card for each of the six classes.
        per_video: when False, every table is aggregated over the whole run and the
            per-recording breakdowns are dropped — no ``video_id`` column, no
            recording x class matrix. Use it to read pure per-error statistics.
    """
    from IPython.display import display

    report = results_root if isinstance(results_root, MetricsReport) else load_metrics_report(results_root)
    figures_dir = report.results_root / 'metrics' / 'figures' if save_figures else None

    warning = staleness_warning(report)
    if warning:
        _markdown('> ⚠️ **Metryki są nieaktualne**\n>\n> ' + warning.replace('\n', '\n> '))

    _markdown('## Zakres runu')
    display(run_header(report))

    _markdown(
        '## Wyniki per klasa błędu\n\n'
        '`gt_videos` / `pred_videos` — w ilu nagraniach klasa występuje w ground truth '
        'i w ilu model ją zgłosił.'
    )
    display(style_class_overview(report, tolerance_s=tolerance_s))

    _markdown(
        '## Segment-level: F1 wobec tolerancji dopasowania\n\n'
        + TOLERANCE_EXPLAINER
    )
    display(style_segment_pivot(report, value='f1'))
    _markdown('Jakość dopasowania tam, gdzie w ogóle doszło do trafienia (`TP > 0`):')
    display(style_segment_pivot(report, value='mean_temporal_iou'))

    _markdown('## Wykresy')
    plot_class_f1(report, save_dir=figures_dir)
    plot_f1_vs_tolerance(report, save_dir=figures_dir)
    if figures_dir is not None:
        _markdown(f'Wykresy zapisane w `{figures_dir}`.')

    if show_class_cards:
        _markdown('## Karta każdego błędu')
        for error_type in ERROR_CLASSES:
            show_class_card(report, error_type, per_video=per_video)

    if per_video:
        _markdown('## Macierz nagranie × klasa błędu')
        display(style_per_video_class_matrix(report))

    _markdown('## Nagrania odrzucone z ewaluacji')
    detail, reasons = skipped_overview(report)
    if detail.empty:
        _markdown('Wszystkie nagrania weszły do ewaluacji.')
    else:
        display(reasons)
        if per_video:
            display(detail)

    _markdown('## Agregaty całego runu\n\nKontekst zamykający — właściwa analiza jest w tabelach per klasa.')
    display(run_totals(report).style.format(precision=3, na_rep=_NA))

    return report


def show_class_card(report: MetricsReport, error_type: str, *, per_video: bool = True) -> None:
    """Render the tables this run holds for one error class.

    Args:
        per_video: when False, only run-wide statistics for the class are shown;
            the per-recording table is omitted.
    """
    from IPython.display import display

    detail = class_detail(report, error_type)
    frame_row = detail['frame']
    headline = _NA
    if not frame_row.empty:
        row = frame_row.iloc[0]
        headline = (
            f'frame F1 **{float(row["f1"]):.3f}** · precision {float(row["precision"]):.3f} · '
            f'recall {float(row["recall"]):.3f} · support {int(row["support"])} klatek'
        )
    _markdown(f'### {error_type}\n\n{headline}')

    if not detail['video'].empty:
        display(_style_scores(detail['video'], ['precision', 'recall', 'f1']).hide(axis='index'))
    if not detail['segment'].empty:
        display(_style_scores(detail['segment'], ['precision', 'recall', 'f1', 'mean_temporal_iou']))
    if per_video and not detail['per_video'].empty:
        display(_style_scores(detail['per_video'], ['precision', 'recall', 'f1', 'iou']))
