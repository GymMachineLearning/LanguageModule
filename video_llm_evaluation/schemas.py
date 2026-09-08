"""Pydantic schemas used by the video LLM evaluation pipeline."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .constants import ERROR_CLASSES


class ManifestRow(BaseModel):
    video_id: str
    video_path: str
    duration_s: float = Field(gt=0)
    fps: float = Field(gt=0)
    num_frames: int = Field(gt=0)
    ground_truth_path: Optional[str] = None


#: Ordered from least to most certain, so a threshold is a slice of this tuple.
CERTAINTY_LEVELS = ("low", "medium", "high")

CertaintyLevel = Literal["low", "medium", "high"]


class ErrorSegment(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    #: Prompt v2 onwards. The float ``confidence`` it replaces turned out degenerate:
    #: every value a run produced sat in 0.65-0.74, with matched and unmatched
    #: predictions indistinguishable, so no threshold on it could separate them.
    certainty: Optional[CertaintyLevel] = None
    #: Prompt v1. Kept so v1 predictions stay readable and the baseline survives.
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    rationale: Optional[str] = None

    @model_validator(mode="after")
    def validate_time_order(self):
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        return self


class ErrorPrediction(BaseModel):
    error_type: str
    present: bool
    segments: List[ErrorSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_error_type_and_segments(self):
        error_type = self.error_type
        present = self.present
        segments = self.segments

        if error_type not in ERROR_CLASSES:
            raise ValueError(f"error_type must be one of: {', '.join(ERROR_CLASSES)}")
        if present and not segments:
            raise ValueError("segments must not be empty when present is true")
        if not present and segments:
            raise ValueError("segments must be empty when present is false")
        return self


class VideoPrediction(BaseModel):
    video_id: str
    duration_s: float = Field(gt=0)
    predictions: List[ErrorPrediction]
    global_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    notes: Optional[str] = None


class GroundTruthRecord(BaseModel):
    video_id: str
    duration_s: float = Field(gt=0)
    fps: float = Field(gt=0)
    num_frames: int = Field(gt=0)
    labels_path: str
