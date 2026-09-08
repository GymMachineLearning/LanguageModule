"""Tests for the certainty threshold and the present/segments invariant.

The threshold is the only knob that turns the model's own hedging into a decision,
so its edges matter: what happens to a class whose segments are all discarded, and
what happens to prompt v1 predictions that declare no certainty at all.
"""

import unittest

from llm_api.gemini.response_parser import GeminiResponseParser
from video_llm_evaluation.constants import ERROR_CLASSES


def _payload(*predictions):
    return {"video_id": "vid", "duration_s": 20.0, "predictions": list(predictions)}


def _segment(start_s, end_s, **extra):
    return {"start_s": start_s, "end_s": end_s, "rationale": "r", **extra}


def _present(prediction):
    return {item.error_type: item for item in prediction.predictions if item.present}


class CertaintyThresholdTests(unittest.TestCase):
    def test_no_threshold_keeps_every_certainty_level(self):
        parser = GeminiResponseParser()
        prediction = parser.parse(
            _payload(
                {
                    "error_type": "Squat-depth",
                    "present": True,
                    "segments": [_segment(1.0, 2.0, certainty="low")],
                }
            )
        )
        self.assertIn("Squat-depth", _present(prediction))

    def test_threshold_discards_segments_below_it(self):
        parser = GeminiResponseParser(min_certainty="medium")
        prediction = parser.parse(
            _payload(
                {
                    "error_type": "Dominant-hip",
                    "present": True,
                    "segments": [
                        _segment(1.0, 2.0, certainty="low"),
                        _segment(5.0, 6.0, certainty="high"),
                    ],
                }
            )
        )
        segments = _present(prediction)["Dominant-hip"].segments
        self.assertEqual([(s.start_s, s.end_s) for s in segments], [(5.0, 6.0)])

    def test_class_becomes_absent_when_the_threshold_empties_it(self):
        """The model claiming present=true must not survive losing every segment.

        ErrorPrediction forbids present=true with no segments, so without the veto
        this payload would raise instead of reading as "not detected".
        """
        parser = GeminiResponseParser(min_certainty="high")
        prediction = parser.parse(
            _payload(
                {
                    "error_type": "No-knee-outlet",
                    "present": True,
                    "segments": [_segment(1.0, 2.0, certainty="medium")],
                }
            )
        )
        self.assertEqual(_present(prediction), {})

    def test_merged_segment_takes_the_higher_certainty(self):
        parser = GeminiResponseParser()
        prediction = parser.parse(
            _payload(
                {
                    "error_type": "Knee-collapse",
                    "present": True,
                    "segments": [
                        _segment(1.0, 2.0, certainty="low"),
                        _segment(2.1, 3.0, certainty="high"),
                    ],
                }
            )
        )
        segments = _present(prediction)["Knee-collapse"].segments
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].certainty, "high")

    def test_prompt_v1_predictions_parse_unchanged_without_a_threshold(self):
        parser = GeminiResponseParser()
        prediction = parser.parse(
            _payload(
                {
                    "error_type": "Squat-depth",
                    "present": True,
                    "segments": [_segment(1.0, 2.0, confidence=0.72)],
                }
            )
        )
        segment = _present(prediction)["Squat-depth"].segments[0]
        self.assertEqual(segment.confidence, 0.72)
        self.assertIsNone(segment.certainty)

    def test_a_threshold_discards_prompt_v1_predictions_entirely(self):
        """Float confidence is not a certainty level, and is not silently promoted."""
        parser = GeminiResponseParser(min_certainty="low")
        prediction = parser.parse(
            _payload(
                {
                    "error_type": "Squat-depth",
                    "present": True,
                    "segments": [_segment(1.0, 2.0, confidence=0.95)],
                }
            )
        )
        self.assertEqual(_present(prediction), {})

    def test_unknown_certainty_level_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            GeminiResponseParser(min_certainty="bardzo")

    def test_absent_classes_are_filled_in_in_canonical_order(self):
        parser = GeminiResponseParser(min_certainty="medium")
        prediction = parser.parse(
            _payload(
                {
                    "error_type": "Taking-off-foot",
                    "present": True,
                    "segments": [_segment(3.0, 4.0, certainty="high")],
                }
            )
        )
        self.assertEqual([item.error_type for item in prediction.predictions], ERROR_CLASSES)


if __name__ == "__main__":
    unittest.main()
