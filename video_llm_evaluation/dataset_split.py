"""Mapping recordings on disk to their MLPSD train/val/test split.

The split lives in the MLPSD ``.pkl``, keyed by the dataset's own ``video_path``.
Both ends of the pipeline need it: ``run`` to avoid paying for requests on
recordings outside the split under evaluation, and ``evaluate`` to keep those
recordings out of the metrics. They reach a recording from different roots — the
video library and the results tree — so the key derivation lives here once
rather than being reimplemented on each side and drifting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

#: Splits MLPSD assigns. ``all`` is the pipeline's own value for "do not filter".
DATASET_SPLITS = ("train", "val", "test")
SPLIT_CHOICES = (*DATASET_SPLITS, "all")

#: Recordings whose folder is named differently on disk than in MLPSD.
RESULTS_TO_MLPSD_FOLDER = {"formtext": "formcheck_text"}

#: Label used when a recording has no matching MLPSD row at all.
UNMATCHED = "unmatched"


def normalise_video_key(value: str | Path) -> str:
    """Canonical key for a recording: path without suffix, lowercased."""
    path = Path(str(value))
    return str(path.with_suffix("")).replace("\\", "/").casefold()


def _mlpsd_key_from_relative_parts(parts: list[str]) -> str:
    parts = list(parts)
    parts[0] = RESULTS_TO_MLPSD_FOLDER.get(parts[0], parts[0])
    return normalise_video_key(Path("squats", *parts))


def mlpsd_key_for_video(videos_root: Path, video_path: Path) -> str:
    """Key for a recording sitting in the video library."""
    relative = video_path.relative_to(videos_root)
    return _mlpsd_key_from_relative_parts(list(relative.parts))


def mlpsd_key_for_prediction(results_root: Path, labels_path: Path) -> str:
    """Key for a recording reached through its saved labels in a results tree.

    Layout is ``<results_root>/<category>/<video_id>/labels_npy/<video_id>.npy``,
    so the recording's own directory is two levels above the labels file.
    """
    relative_dir = labels_path.parent.parent.relative_to(results_root)
    parts = list(relative_dir.parts)
    if len(parts) < 2:
        raise ValueError(f"Unexpected prediction layout: {labels_path}")
    return _mlpsd_key_from_relative_parts(parts)


@dataclass(frozen=True)
class SplitIndex:
    """Which MLPSD split each recording belongs to, keyed by canonical path."""

    by_key: Mapping[str, str]

    def split_for(self, key: str) -> str:
        """Split for a key, or ``unmatched`` when MLPSD does not know it."""
        return self.by_key.get(key, UNMATCHED)

    def matches(self, key: str, split: str) -> bool:
        """True when a recording belongs in a run filtered to ``split``.

        ``all`` admits everything, including recordings MLPSD has never heard of,
        because that is the pre-filter behaviour of the pipeline.
        """
        if split == "all":
            return True
        return self.split_for(key) == split

    def compose(self, keys: Iterable[str]) -> dict[str, int]:
        """Count recordings per split, for logging what a selection contains."""
        counts = Counter(self.split_for(key) for key in keys)
        ordered = {name: counts[name] for name in DATASET_SPLITS if counts[name]}
        if counts[UNMATCHED]:
            ordered[UNMATCHED] = counts[UNMATCHED]
        return ordered


def load_split_index(dataset_path: Path | str) -> SplitIndex:
    """Read the MLPSD pickle and index ``dataset_split`` by recording path."""
    import pandas as pd

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find MLPSD dataset: {path}")

    frame = pd.read_pickle(path)
    for column in ("video_path", "dataset_split"):
        if column not in frame.columns:
            raise ValueError(f"MLPSD dataset is missing column: {column}")

    return SplitIndex(
        by_key={
            normalise_video_key(row["video_path"]): str(row["dataset_split"])
            for _, row in frame.iterrows()
        }
    )


def format_composition(composition: Mapping[str, int]) -> str:
    """Render a split composition for logs and tables, e.g. ``test:42 train:8``."""
    if not composition:
        return "empty"
    return " ".join(f"{name}:{count}" for name, count in composition.items())
