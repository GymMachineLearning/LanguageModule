"""Parse and normalize Gemini responses for squat video classification."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from video_llm_evaluation.constants import ERROR_CLASSES
from video_llm_evaluation.schemas import ErrorPrediction, ErrorSegment, VideoPrediction


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1]
    return cleaned


def load_gemini_payload(raw_response: str | Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-like mapping from a Gemini raw response."""

    if isinstance(raw_response, Mapping):
        return dict(raw_response)

    candidate = _extract_json_blob(raw_response)
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("Gemini response must decode to a JSON object")
    return payload


def _clean_segments(
    segments: Sequence[ErrorSegment],
    duration_s: float,
    min_segment_duration_s: float,
    merge_gap_s: float,
) -> list[ErrorSegment]:
    clipped_segments: list[ErrorSegment] = []

    for segment in segments:
        start_s = max(0.0, float(segment.start_s))
        end_s = min(duration_s, float(segment.end_s))
        if end_s - start_s < min_segment_duration_s:
            continue
        clipped_segments.append(
            ErrorSegment(
                start_s=start_s,
                end_s=end_s,
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
        if segment.start_s <= previous.end_s + merge_gap_s:
            merged_segments[-1] = ErrorSegment(
                start_s=previous.start_s,
                end_s=max(previous.end_s, segment.end_s),
                confidence=previous.confidence if previous.confidence is not None else segment.confidence,
                rationale=previous.rationale or segment.rationale,
            )
        else:
            merged_segments.append(segment)

    return merged_segments


def parse_video_prediction(
    raw_response: str | Mapping[str, Any],
    *,
    video_id: str | None = None,
    duration_s: float | None = None,
    min_segment_duration_s: float = 0.2,
    merge_gap_s: float = 0.2,
) -> VideoPrediction:
    """Parse and normalize a Gemini video prediction payload."""

    payload = load_gemini_payload(raw_response)
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

        segments_payload = item.get("segments", []) or []
        segments = [ErrorSegment(**segment) for segment in segments_payload]
        cleaned_segments = _clean_segments(
            segments,
            duration_s=payload_duration_s,
            min_segment_duration_s=min_segment_duration_s,
            merge_gap_s=merge_gap_s,
        )

        present = bool(item.get("present", bool(cleaned_segments)))
        predictions.append(
            ErrorPrediction(
                error_type=item.get("error_type"),
                present=present,
                segments=cleaned_segments,
            )
        )

    missing_error_types = [error_type for error_type in ERROR_CLASSES if error_type not in {prediction.error_type for prediction in predictions}]
    for error_type in missing_error_types:
        predictions.append(ErrorPrediction(error_type=error_type, present=False, segments=[]))

    predictions.sort(key=lambda prediction: ERROR_CLASSES.index(prediction.error_type))

    return VideoPrediction(
        video_id=payload_video_id,
        duration_s=payload_duration_s,
        predictions=predictions,
        global_confidence=payload.get("global_confidence"),
        notes=payload.get("notes"),
    )
