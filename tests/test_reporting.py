import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from video_llm_evaluation.constants import ERROR_CLASSES, MLPSD_ERROR_ROW_INDICES
from video_llm_evaluation.notebook_support import reporting
from video_llm_evaluation.notebook_support.data import load_mlpsd_dataframe


def _write_metrics(results_root: Path) -> None:
    """Write the artifact set that ``cli.py evaluate`` produces, in miniature."""
    metrics_dir = results_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    frame_by_class = pd.DataFrame(
        {
            "error_type": ERROR_CLASSES,
            "precision": [0.5, 0.0, 0.25, 0.0, 0.0, 0.0],
            "recall": [1.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            "f1": [0.666667, 0.0, 0.333333, 0.0, 0.0, 0.0],
            "iou": [0.5, 0.0, 0.2, 0.0, 0.0, 0.0],
            "support": [4, 0, 2, 0, 0, 0],
            "tp": [4, 0, 1, 0, 0, 0],
            "fp": [4, 0, 3, 0, 0, 0],
            "fn": [0, 0, 1, 0, 0, 0],
            "tn": [2, 10, 5, 10, 10, 10],
        }
    )
    frame_by_class.to_csv(metrics_dir / "frame_metrics.csv", index=False)

    video_by_class = frame_by_class[["error_type", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]]
    video_by_class.to_csv(metrics_dir / "video_level_metrics.csv", index=False)

    segment_rows = []
    for tolerance in (0.1, 0.5, 1.0):
        for index, error_type in enumerate(ERROR_CLASSES):
            matched = tolerance == 1.0 and index == 0
            segment_rows.append(
                {
                    "error_type": error_type,
                    "tolerance_s": tolerance,
                    "precision": 0.5 if matched else 0.0,
                    "recall": 0.5 if matched else 0.0,
                    "f1": 0.5 if matched else 0.0,
                    "tp": 1 if matched else 0,
                    "fp": 1,
                    "fn": 1,
                    "mean_temporal_iou": 0.8 if matched else None,
                    "start_mae_s": 0.1 if matched else None,
                    "end_mae_s": 0.2 if matched else None,
                }
            )
    pd.DataFrame(segment_rows).to_csv(metrics_dir / "segment_metrics.csv", index=False)

    per_video_frame = []
    per_video_video = []
    for video_id in ("clip_a", "clip_b"):
        for index, error_type in enumerate(ERROR_CLASSES):
            support = 4 if (video_id == "clip_a" and index == 0) else 0
            false_positive = 3 if (video_id == "clip_b" and index == 1) else 0
            per_video_frame.append(
                {
                    "video_id": video_id,
                    "error_type": error_type,
                    "precision": 0.5 if support else 0.0,
                    "recall": 1.0 if support else 0.0,
                    "f1": 0.666667 if support else 0.0,
                    "iou": 0.5 if support else 0.0,
                    "support": support,
                    "tp": 4 if support else 0,
                    "fp": false_positive,
                    "fn": 0,
                    "tn": 6,
                }
            )
            per_video_video.append(
                {
                    "video_id": video_id,
                    "error_type": error_type,
                    "gt_present": bool(support),
                    "pred_present": bool(support) or bool(false_positive),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 1,
                }
            )
    pd.DataFrame(per_video_frame).to_csv(metrics_dir / "frame_metrics_by_video.csv", index=False)
    pd.DataFrame(per_video_video).to_csv(metrics_dir / "video_level_metrics_by_video.csv", index=False)

    segment_by_video = [
        {
            "video_id": "clip_a",
            "error_type": ERROR_CLASSES[0],
            "tolerance_s": 1.0,
            "tp": 1,
            "fp": 1,
            "fn": 1,
            "mean_temporal_iou": 0.8,
            "start_mae_s": 0.1,
            "end_mae_s": 0.2,
        }
    ]
    pd.DataFrame(segment_by_video).to_csv(metrics_dir / "segment_metrics_by_video.csv", index=False)

    pd.DataFrame(
        [
            {
                "video_id": "clip_a",
                "status": "evaluated",
                "alignment": "exact",
                "prediction_frames": 10,
                "ground_truth_frames": 10,
                "mlpsd_video_path": "squats/formcheck/clip_a",
                "message": "",
            },
            {
                "video_id": "clip_b",
                "status": "evaluated",
                "alignment": "exact",
                "prediction_frames": 10,
                "ground_truth_frames": 10,
                "mlpsd_video_path": "squats/formcheck/clip_b",
                "message": "",
            },
            {
                "video_id": "clip_c",
                "status": "skipped",
                "alignment": "",
                "prediction_frames": "",
                "ground_truth_frames": "",
                "mlpsd_video_path": "",
                "message": "ValueError: no MLPSD recording matches squats/formcheck/clip_c",
            },
            {
                "video_id": "clip_d",
                "status": "skipped",
                "alignment": "",
                "prediction_frames": "",
                "ground_truth_frames": "",
                "mlpsd_video_path": "",
                "message": "ValueError: frame-count mismatch: prediction=10, ground_truth=40",
            },
        ]
    ).to_csv(metrics_dir / "evaluation_cases.csv", index=False)

    (metrics_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_path": "/tmp/MLPSD.pkl",
                "results_root": str(results_root),
                "found_predictions": 4,
                "evaluated": 2,
                "skipped": 2,
                "mlpsd_error_row_indices": list(MLPSD_ERROR_ROW_INDICES),
                "frame_metrics": {"macro_f1": 0.17, "micro_f1": 0.5, "weighted_f1": 0.2},
                "video_metrics": {"accuracy": 0.5, "micro_f1": 0.4},
                "segment_metrics": [],
            }
        ),
        encoding="utf-8",
    )



def _capture(call) -> list[str]:
    """Run a notebook-rendering call and return the text of everything displayed."""
    import IPython.display as ipython_display

    captured: list[str] = []

    def _record(*args, **kwargs):
        for item in args:
            if hasattr(item, "to_html"):
                captured.append(item.to_html())
            elif hasattr(item, "data"):
                captured.append(str(item.data))
            else:
                captured.append(str(item))

    original = ipython_display.display
    ipython_display.display = _record
    try:
        call()
    finally:
        ipython_display.display = original
    return captured


def _render(report, *, per_video: bool) -> list[str]:
    return _capture(lambda: reporting.show_report(report, save_figures=False, per_video=per_video))


def _render_card(report, error_type: str, *, per_video: bool) -> list[str]:
    return _capture(lambda: reporting.show_class_card(report, error_type, per_video=per_video))


class GroundTruthRowSelectionTests(unittest.TestCase):
    """The MLPSD label matrix has ten rows; only six belong to this pipeline."""

    def test_load_mlpsd_dataframe_selects_pipeline_rows_in_error_class_order(self) -> None:
        labels = np.zeros((10, 7), dtype=np.uint8)
        # Mark one distinct frame per pipeline row so a mis-slice cannot pass.
        for position, mlpsd_row in enumerate(MLPSD_ERROR_ROW_INDICES):
            labels[mlpsd_row, position] = 1
        # Rows that are NOT part of the pipeline must never leak into the matrix.
        labels[4, 6] = 1
        labels[6, 6] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "mlpsd.pkl"
            pd.DataFrame(
                [
                    {
                        "video_path": "squats/formcheck/clip_a.mp4",
                        "labels": labels,
                        "errors": [],
                        "errors_list": [],
                    }
                ]
            ).to_pickle(dataset_path)

            dataset_df = load_mlpsd_dataframe(dataset_path)

        matrix = dataset_df.iloc[0]["gt_labels_matrix"]
        self.assertEqual(matrix.shape, (len(ERROR_CLASSES), 7))
        np.testing.assert_array_equal(matrix, labels[list(MLPSD_ERROR_ROW_INDICES)])
        self.assertEqual(dataset_df.iloc[0]["gt_labels_matrix_raw"].shape, (10, 7))
        # The excluded rows were active only at frame 6, which must read as all-zero.
        self.assertEqual(int(matrix[:, 6].sum()), 0)

    def test_gt_errors_name_every_pipeline_class_when_all_rows_are_active(self) -> None:
        labels = np.zeros((10, 6), dtype=np.uint8)
        for position, mlpsd_row in enumerate(MLPSD_ERROR_ROW_INDICES):
            labels[mlpsd_row, position] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "mlpsd.pkl"
            pd.DataFrame(
                [{"video_path": "a.mp4", "labels": labels, "errors": [], "errors_list": []}]
            ).to_pickle(dataset_path)
            dataset_df = load_mlpsd_dataframe(dataset_path)

        self.assertEqual(list(dataset_df.iloc[0]["gt_active_rows"]), list(range(len(ERROR_CLASSES))))
        self.assertEqual(list(dataset_df.iloc[0]["gt_errors"]), ERROR_CLASSES)

    def test_gt_errors_uses_mlpsd_row_seven_for_the_last_class(self) -> None:
        """Row 7 -> No-knee-outlet; the naive row-index mapping would say row 5."""
        labels = np.zeros((10, 3), dtype=np.uint8)
        labels[7, 1] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "mlpsd.pkl"
            pd.DataFrame(
                [{"video_path": "a.mp4", "labels": labels, "errors": [], "errors_list": []}]
            ).to_pickle(dataset_path)
            dataset_df = load_mlpsd_dataframe(dataset_path)

        self.assertEqual(list(dataset_df.iloc[0]["gt_errors"]), ["No-knee-outlet"])

    def test_already_reduced_matrices_are_left_alone(self) -> None:
        labels = np.zeros((len(ERROR_CLASSES), 3), dtype=np.uint8)
        labels[2, 0] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "mlpsd.pkl"
            pd.DataFrame(
                [{"video_path": "a.mp4", "labels": labels, "errors": [], "errors_list": []}]
            ).to_pickle(dataset_path)
            dataset_df = load_mlpsd_dataframe(dataset_path)

        np.testing.assert_array_equal(dataset_df.iloc[0]["gt_labels_matrix"], labels)


class LoadMetricsReportTests(unittest.TestCase):
    def test_missing_metrics_directory_names_the_command_that_creates_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            results_root.mkdir()
            with self.assertRaises(reporting.MetricsNotAvailable) as context:
                reporting.load_metrics_report(results_root)
        self.assertIn("video_llm_evaluation.cli evaluate", str(context.exception))

    def test_empty_metrics_directory_is_reported_as_not_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            (results_root / "metrics").mkdir(parents=True)
            with self.assertRaises(reporting.MetricsNotAvailable):
                reporting.load_metrics_report(results_root)

    def test_incomplete_metrics_directory_names_the_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            _write_metrics(results_root)
            (results_root / "metrics" / "segment_metrics.csv").unlink()
            with self.assertRaises(reporting.MetricsNotAvailable) as context:
                reporting.load_metrics_report(results_root)
        self.assertIn("segment_metrics.csv", str(context.exception))

    def test_report_loads_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            _write_metrics(results_root)
            report = reporting.load_metrics_report(results_root)

        self.assertEqual(report.summary["evaluated"], 2)
        self.assertEqual(report.evaluated_video_ids, ["clip_a", "clip_b"])
        self.assertEqual(report.tolerances, [0.1, 0.5, 1.0])
        self.assertFalse(report.is_stale)

    def test_predictions_written_after_the_metrics_mark_the_report_stale(self) -> None:
        import os
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            _write_metrics(results_root)
            labels_path = results_root / "formcheck" / "clip_a" / "labels_npy" / "clip_a.npy"
            labels_path.parent.mkdir(parents=True)
            np.save(labels_path, np.zeros((6, 4), dtype=np.uint8))
            future = time.time() + 120
            os.utime(labels_path, (future, future))

            report = reporting.load_metrics_report(results_root)

        self.assertTrue(report.is_stale)
        self.assertIn("stale", reporting.staleness_warning(report))


class PerClassViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.results_root = Path(self._temp.name) / "results"
        _write_metrics(self.results_root)
        self.report = reporting.load_metrics_report(self.results_root)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_class_overview_has_one_row_per_error_class(self) -> None:
        overview = reporting.class_overview(self.report)
        self.assertEqual(list(overview.index), ERROR_CLASSES)

    def test_class_overview_counts_videos_where_the_class_is_in_ground_truth(self) -> None:
        overview = reporting.class_overview(self.report)
        self.assertEqual(overview.loc[ERROR_CLASSES[0], "gt_videos"], 1)
        # clip_b predicts class 1 without it being in ground truth.
        self.assertEqual(overview.loc[ERROR_CLASSES[1], "gt_videos"], 0)
        self.assertEqual(overview.loc[ERROR_CLASSES[1], "pred_videos"], 1)

    def test_class_overview_reads_segment_f1_at_the_requested_tolerance(self) -> None:
        at_one = reporting.class_overview(self.report, tolerance_s=1.0)
        at_tenth = reporting.class_overview(self.report, tolerance_s=0.1)
        self.assertAlmostEqual(at_one.loc[ERROR_CLASSES[0], "segment_f1@1s"], 0.5)
        self.assertAlmostEqual(at_tenth.loc[ERROR_CLASSES[0], "segment_f1@0.1s"], 0.0)

    def test_segment_pivot_puts_tolerances_in_columns(self) -> None:
        pivot = reporting.segment_pivot(self.report)
        self.assertEqual(list(pivot.index), ERROR_CLASSES)
        self.assertEqual(list(pivot.columns), ["0.1s", "0.5s", "1s"])
        self.assertAlmostEqual(pivot.loc[ERROR_CLASSES[0], "1s"], 0.5)

    def test_segment_pivot_keeps_unmatched_tolerances_empty_rather_than_zero(self) -> None:
        pivot = reporting.segment_pivot(self.report, value="mean_temporal_iou")
        self.assertTrue(pd.isna(pivot.loc[ERROR_CLASSES[0], "0.1s"]))
        self.assertAlmostEqual(pivot.loc[ERROR_CLASSES[0], "1s"], 0.8)

    def test_class_detail_rejects_an_unknown_error_type(self) -> None:
        with self.assertRaises(ValueError):
            reporting.class_detail(self.report, "Not-an-error")

    def test_class_detail_sorts_videos_with_ground_truth_first(self) -> None:
        detail = reporting.class_detail(self.report, ERROR_CLASSES[0])
        self.assertEqual(list(detail["per_video"].index)[0], "clip_a")
        self.assertEqual(list(detail["segment"].index), ["0.1s", "0.5s", "1s"])

    def test_per_video_matrix_marks_false_positives_apart_from_correct_silence(self) -> None:
        display, colors = reporting.per_video_class_matrix(self.report)
        self.assertEqual(display.loc["clip_a", ERROR_CLASSES[0]], "0.67")
        self.assertEqual(display.loc["clip_b", ERROR_CLASSES[1]], "FP")
        self.assertEqual(display.loc["clip_b", ERROR_CLASSES[2]], "·")
        self.assertNotEqual(colors.loc["clip_b", ERROR_CLASSES[1]], "")

    def test_skipped_overview_groups_reasons(self) -> None:
        detail, reasons = reporting.skipped_overview(self.report)
        self.assertEqual(sorted(detail.index), ["clip_c", "clip_d"])
        self.assertEqual(reasons.loc["no MLPSD match", "videos"], 1)
        self.assertEqual(reasons.loc["frame-count mismatch", "videos"], 1)

    def test_run_header_shows_both_prediction_counts(self) -> None:
        header = reporting.run_header(self.report)
        self.assertEqual(header.loc["found_predictions", "value"], 4)
        self.assertEqual(header.loc["skipped", "value"], 2)

    def test_score_background_maps_low_and_high_scores_to_different_colours(self) -> None:
        self.assertNotEqual(reporting.score_background(0.0), reporting.score_background(1.0))
        self.assertEqual(reporting.score_background(None), "")
        self.assertEqual(reporting.score_background(float("nan")), "")

    def test_styled_tables_render_to_html(self) -> None:
        self.assertIn("<table", reporting.style_class_overview(self.report).to_html())
        self.assertIn("<table", reporting.style_segment_pivot(self.report).to_html())
        self.assertIn("<table", reporting.style_per_video_class_matrix(self.report).to_html())

    def test_per_video_false_drops_every_recording_level_table(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        with_videos = _render(self.report, per_video=True)
        without_videos = _render(self.report, per_video=False)

        self.assertTrue(any("video_id" in text for text in with_videos))
        self.assertFalse(any("video_id" in text for text in without_videos))
        self.assertTrue(any("Macierz nagranie" in text for text in with_videos))
        self.assertFalse(any("Macierz nagranie" in text for text in without_videos))

    def test_per_video_false_keeps_the_per_class_tables(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        rendered = _render(self.report, per_video=False)
        joined = "\n".join(rendered)
        for error_type in ERROR_CLASSES:
            self.assertIn(error_type, joined)
        # Coverage still has to be visible: the reason breakdown is aggregate.
        self.assertIn("no MLPSD match", joined)

    def test_class_card_without_per_video_omits_the_recording_table(self) -> None:
        with_videos = _render_card(self.report, ERROR_CLASSES[0], per_video=True)
        without_videos = _render_card(self.report, ERROR_CLASSES[0], per_video=False)
        self.assertTrue(any("clip_a" in text for text in with_videos))
        self.assertFalse(any("clip_a" in text for text in without_videos))

    def test_charts_are_written_when_a_save_directory_is_given(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        figures_dir = self.results_root / "metrics" / "figures"
        reporting.plot_class_f1(self.report, save_dir=figures_dir)
        reporting.plot_f1_vs_tolerance(self.report, save_dir=figures_dir)
        self.assertTrue((figures_dir / "class_f1.png").exists())
        self.assertTrue((figures_dir / "segment_f1_vs_tolerance.png").exists())


if __name__ == "__main__":
    unittest.main()
