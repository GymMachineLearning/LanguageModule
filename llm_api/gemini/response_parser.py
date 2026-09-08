"""Parse and normalize Gemini responses for squat video classification."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from video_llm_evaluation.constants import ERROR_CLASSES
from video_llm_evaluation.schemas import (
    CERTAINTY_LEVELS,
    ErrorPrediction,
    ErrorSegment,
    VideoPrediction,
)


def _certainty_rank(level: str | None) -> int:
    """Position of a certainty level, or -1 for anything unlabelled.

    A segment with no certainty sits below every threshold on purpose: the whole
    point of a threshold is to keep only what the model vouched for, and an
    unlabelled segment vouches for nothing. That also means a threshold applied to
    prompt v1 predictions, which carry a float ``confidence`` instead, drops all of
    them rather than quietly letting them through.
    """
    if level is None:
        return -1
    return CERTAINTY_LEVELS.index(level)


class GeminiResponseParser:
    def __init__(
        self,
        *,
        error_classes: Sequence[str] = ERROR_CLASSES,
        min_segment_duration_s: float = 0.2,
        merge_gap_s: float = 0.2,
        min_certainty: str | None = None,
    ) -> None:
        self.error_classes = tuple(error_classes)
        self.min_segment_duration_s = min_segment_duration_s
        self.merge_gap_s = merge_gap_s
        if min_certainty is not None and min_certainty not in CERTAINTY_LEVELS:
            raise ValueError(f"min_certainty must be one of: {', '.join(CERTAINTY_LEVELS)}")
        self.min_certainty = min_certainty

    @staticmethod
    def strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def extract_json_blob(self, text: str) -> str:
        cleaned = self.strip_code_fences(text)
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            return cleaned[first_brace : last_brace + 1]
        return cleaned

    def load_payload(self, raw_response: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw_response, Mapping):
            return dict(raw_response)

        candidate = self.extract_json_blob(raw_response)
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("Gemini response must decode to a JSON object")
        return payload

    def clean_segments(self, segments: Sequence[ErrorSegment], duration_s: float) -> list[ErrorSegment]:
        clipped_segments: list[ErrorSegment] = []

        threshold_rank = _certainty_rank(self.min_certainty) if self.min_certainty else None

        for segment in segments:
            start_s = max(0.0, float(segment.start_s))
            end_s = min(duration_s, float(segment.end_s))
            if end_s - start_s < self.min_segment_duration_s:
                continue
            if threshold_rank is not None and _certainty_rank(segment.certainty) < threshold_rank:
                continue
            clipped_segments.append(
                ErrorSegment(
                    start_s=start_s,
                    end_s=end_s,
                    certainty=segment.certainty,
                    confidence=segment.confidence,
                    rationale=segment.rationale,
                )
            )

        if not clipped_segments:
            return []

        clipped_segments.sort(key=lambda segment: (segment.start_s, segment.end_s))
        merged_segments: list[ErrorSegment] = [clipped_segments[0]]

        for segment in clipped_segments[1:]:
            previous = merged_segments[-1]
            if segment.start_s <= previous.end_s + self.merge_gap_s:
                merged_segments[-1] = ErrorSegment(
                    start_s=previous.start_s,
                    end_s=max(previous.end_s, segment.end_s),
                    # The merged span is as certain as the more certain of its parts.
                    certainty=max(
                        (previous.certainty, segment.certainty),
                        key=_certainty_rank,
                    ),
                    confidence=previous.confidence if previous.confidence is not None else segment.confidence,
                    rationale=previous.rationale or segment.rationale,
                )
            else:
                merged_segments.append(segment)

        return merged_segments

    def parse(
        self,
        raw_response: str | Mapping[str, Any],
        *,
        video_id: str | None = None,
        duration_s: float | None = None,
    ) -> VideoPrediction:
        payload = self.load_payload(raw_response)
        payload_video_id = payload.get("video_id", video_id)
        payload_duration_s = float(payload.get("duration_s", duration_s if duration_s is not None else 0))

        if not payload_video_id:
            raise ValueError("video_id is required in the payload or as an argument")
        if payload_duration_s <= 0:
            raise ValueError("duration_s is required in the payload or as an argument")

        predictions_payload = payload.get("predictions", [])
        if not isinstance(predictions_payload, list):
            raise ValueError("predictions must be a list")

        predictions: list[ErrorPrediction] = []
        for item in predictions_payload:
            if not isinstance(item, Mapping):
                raise ValueError("each prediction must be a JSON object")

            error_type = item.get("error_type")
            if not isinstance(error_type, str) or not error_type:
                raise ValueError("each prediction must include a string error_type")

            segments_payload = item.get("segments", []) or []
            segments = [ErrorSegment(**segment) for segment in segments_payload]
            cleaned_segments = self.clean_segments(segments, duration_s=payload_duration_s)

            # The model's own flag can only veto a class, never create one: a class is
            # present iff a segment survived cleaning. Without this, a payload whose
            # segments are all dropped -- too short, or below the certainty threshold --
            # would keep present=true and trip the ErrorPrediction validator.
            claimed_present = bool(item.get("present", bool(cleaned_segments)))
            present = claimed_present and bool(cleaned_segments)
            predictions.append(
                ErrorPrediction(
                    error_type=error_type,
                    present=present,
                    segments=cleaned_segments if present else [],
                )
            )

        existing_error_types = {prediction.error_type for prediction in predictions}
        for error_type in self.error_classes:
            if error_type not in existing_error_types:
                predictions.append(ErrorPrediction(error_type=error_type, present=False, segments=[]))

        predictions.sort(key=lambda prediction: self.error_classes.index(prediction.error_type))

        return VideoPrediction(
            video_id=payload_video_id,
            duration_s=payload_duration_s,
            predictions=predictions,
            global_confidence=payload.get("global_confidence"),
            notes=payload.get("notes"),
        )


_DEFAULT_PARSER = GeminiResponseParser()


def load_gemini_payload(raw_response: str | Mapping[str, Any]) -> dict[str, Any]:
    return _DEFAULT_PARSER.load_payload(raw_response)


def parse_video_prediction(
    raw_response: str | Mapping[str, Any],
    *,
    video_id: str | None = None,
    duration_s: float | None = None,
    min_segment_duration_s: float = 0.2,
    merge_gap_s: float = 0.2,
    min_certainty: str | None = None,
) -> VideoPrediction:
    parser = GeminiResponseParser(
        min_segment_duration_s=min_segment_duration_s,
        merge_gap_s=merge_gap_s,
        min_certainty=min_certainty,
    )
    return parser.parse(raw_response, video_id=video_id, duration_s=duration_s)
