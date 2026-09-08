import tempfile
import unittest
from pathlib import Path

import pandas as pd

from video_llm_evaluation.dataset_split import (
    SPLIT_CHOICES,
    SplitIndex,
    UNMATCHED,
    format_composition,
    load_split_index,
    mlpsd_key_for_prediction,
    mlpsd_key_for_video,
    normalise_video_key,
)
from video_llm_evaluation.evaluation.persistence import metrics_dir_for_split


class KeyDerivationTests(unittest.TestCase):
    """Both ends of the pipeline must land on the same key for one recording."""

    def test_video_library_and_results_tree_agree(self) -> None:
        videos_root = Path("/videos/preprocessed")
        results_root = Path("/results/run_a")
        from_library = mlpsd_key_for_video(videos_root, videos_root / "formcheck" / "clip_a.mp4")
        from_results = mlpsd_key_for_prediction(
            results_root, results_root / "formcheck" / "clip_a" / "labels_npy" / "clip_a.npy"
        )
        self.assertEqual(from_library, from_results)
        self.assertEqual(from_library, "squats/formcheck/clip_a")

    def test_the_renamed_folder_is_translated_on_both_paths(self) -> None:
        """On disk it is `formtext`; MLPSD calls it `formcheck_text`."""
        videos_root = Path("/videos")
        results_root = Path("/results")
        self.assertEqual(
            mlpsd_key_for_video(videos_root, videos_root / "formtext" / "c.mp4"),
            "squats/formcheck_text/c",
        )
        self.assertEqual(
            mlpsd_key_for_prediction(results_root, results_root / "formtext" / "c" / "labels_npy" / "c.npy"),
            "squats/formcheck_text/c",
        )

    def test_keys_are_case_insensitive_and_suffix_free(self) -> None:
        self.assertEqual(normalise_video_key("Squats/FormCheck/Clip.MP4"), "squats/formcheck/clip")

    def test_a_flat_results_layout_is_rejected(self) -> None:
        results_root = Path("/results")
        with self.assertRaises(ValueError):
            mlpsd_key_for_prediction(results_root, results_root / "labels_npy" / "c.npy")


class SplitIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = SplitIndex(by_key={"squats/a/one": "test", "squats/a/two": "train", "squats/a/three": "val"})

    def test_an_unknown_recording_reads_as_unmatched(self) -> None:
        self.assertEqual(self.index.split_for("squats/a/nope"), UNMATCHED)

    def test_matches_selects_only_the_requested_split(self) -> None:
        self.assertTrue(self.index.matches("squats/a/one", "test"))
        self.assertFalse(self.index.matches("squats/a/two", "test"))

    def test_all_admits_everything_including_unmatched(self) -> None:
        """`all` is the pre-filter behaviour, which never knew about splits."""
        self.assertTrue(self.index.matches("squats/a/two", "all"))
        self.assertTrue(self.index.matches("squats/a/nope", "all"))

    def test_unmatched_never_belongs_to_a_named_split(self) -> None:
        self.assertFalse(self.index.matches("squats/a/nope", "test"))

    def test_composition_counts_every_category(self) -> None:
        composition = self.index.compose(["squats/a/one", "squats/a/two", "squats/a/nope", "squats/a/nope"])
        self.assertEqual(composition, {"train": 1, "test": 1, UNMATCHED: 2})
        self.assertEqual(format_composition(composition), "train:1 test:1 unmatched:2")

    def test_an_empty_composition_is_readable(self) -> None:
        self.assertEqual(format_composition({}), "empty")


class LoadSplitIndexTests(unittest.TestCase):
    def test_the_index_is_keyed_by_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mlpsd.pkl"
            pd.DataFrame(
                [
                    {"video_path": "squats/formcheck/A.mp4", "dataset_split": "test"},
                    {"video_path": "squats/lifting/b.mp4", "dataset_split": "train"},
                ]
            ).to_pickle(path)
            index = load_split_index(path)
        self.assertEqual(index.split_for("squats/formcheck/a"), "test")
        self.assertEqual(index.split_for("squats/lifting/b"), "train")

    def test_a_dataset_without_the_split_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mlpsd.pkl"
            pd.DataFrame([{"video_path": "squats/a/b.mp4"}]).to_pickle(path)
            with self.assertRaises(ValueError) as context:
                load_split_index(path)
        self.assertIn("dataset_split", str(context.exception))

    def test_a_missing_dataset_is_reported_by_path(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_split_index("/definitely/not/here.pkl")


class MetricsDirectoryTests(unittest.TestCase):
    """Filtered metrics must not overwrite the full-set metrics."""

    def test_unfiltered_runs_keep_the_original_location(self) -> None:
        self.assertEqual(metrics_dir_for_split(Path("/r"), None), Path("/r/metrics"))
        self.assertEqual(metrics_dir_for_split(Path("/r"), "all"), Path("/r/metrics"))

    def test_each_split_gets_its_own_directory(self) -> None:
        self.assertEqual(metrics_dir_for_split(Path("/r"), "test"), Path("/r/metrics/split_test"))
        self.assertNotEqual(
            metrics_dir_for_split(Path("/r"), "test"), metrics_dir_for_split(Path("/r"), "val")
        )

    def test_split_choices_cover_the_dataset_plus_all(self) -> None:
        self.assertEqual(set(SPLIT_CHOICES), {"train", "val", "test", "all"})


if __name__ == "__main__":
    unittest.main()
