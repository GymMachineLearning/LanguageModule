import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from video_llm_evaluation.cli import DEFAULT_MLPSD_DATASET_PATH, REPO_ROOT, main


class EvaluateCliTests(unittest.TestCase):
    def test_default_mlpsd_dataset_path_is_relative_to_master_degree_directory(self) -> None:
        expected_path = REPO_ROOT.parents[1] / (
            "dataset/datasets/MLPSD/final_dataset/latest/"
            "MLPSD_v2.0_feature_extracted_v3.0.0_all_with_holistic_phases.pkl"
        )
        self.assertEqual(DEFAULT_MLPSD_DATASET_PATH, expected_path)

    def test_evaluate_maps_mlpsd_labels_and_writes_batch_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_root = temp_path / "results"
            labels_path = results_root / "formcheck" / "clip_a" / "labels_npy" / "clip_a.npy"
            labels_path.parent.mkdir(parents=True)

            predicted = np.zeros((6, 4), dtype=np.uint8)
            predicted[0, 0] = 1
            predicted[4, 1] = 1
            predicted[5, 2] = 1
            np.save(labels_path, predicted)

            ground_truth = np.zeros((10, 4), dtype=np.uint8)
            ground_truth[0, 0] = 1
            ground_truth[5, 1] = 1
            ground_truth[7, 2] = 1
            dataset_path = temp_path / "mlpsd.pkl"
            pd.DataFrame(
                [
                    {
                        "video_path": "squats/formcheck/clip_a.mp4",
                        "dataset_split": "test",
                        "labels": ground_truth,
                        "fps": 30.0,
                        "frames": 4,
                    }
                ]
            ).to_pickle(dataset_path)

            main(["evaluate", "--results-root", str(results_root), "--dataset-path", str(dataset_path)])

            # The default split is `test`, so metrics land beside the full-set
            # ones rather than replacing them.
            self.assertFalse((results_root / "metrics" / "summary.json").exists())
            summary_path = results_root / "metrics" / "split_test" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["evaluated"], 1)
            self.assertEqual(summary["skipped"], 0)
            self.assertEqual(summary["frame_metrics"]["micro_f1"], 1.0)
            self.assertEqual(summary["split"], "test")

            frame_rows = pd.read_csv(results_root / "metrics" / "split_test" / "frame_metrics.csv")
            self.assertEqual(
                set(frame_rows["error_type"]),
                {
                    "Squat-depth",
                    "Back-round",
                    "Taking-off-foot",
                    "Knee-collapse",
                    "Dominant-hip",
                    "No-knee-outlet",
                },
            )


if __name__ == "__main__":
    unittest.main()
