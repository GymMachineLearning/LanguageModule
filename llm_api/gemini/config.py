"""Configuration objects for Gemini-based workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from video_llm_evaluation.constants import (
    DEFAULT_MEDIA_PROCESSING,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_VIDEO_FPS,
    ERROR_CLASSES,
)


@dataclass(slots=True)
class GeminiVideoConfig:
    model_name: str = DEFAULT_MODEL_NAME
    prompt_version: str = DEFAULT_PROMPT_VERSION
    error_classes: tuple[str, ...] = field(default_factory=lambda: tuple(ERROR_CLASSES))
    temperature: float = 0.0
    max_retries: int = 3
    request_timeout_s: int = 180
    api_key_env: str = "GEMINI_API_KEY"

    #: Frames per second Gemini samples from the video. The API default is 1.0,
    #: which gives roughly 2-3 frames per squat repetition — too coarse to place
    #: error boundaries. Only honoured under STATIC media processing.
    video_fps: float = DEFAULT_VIDEO_FPS

    #: How the model consumes the video. The API default is "model-specific
    #: processing", which on Gemini 3.x Flash models may mean AGENTIC — the model
    #: navigates the timeline itself and ignores video_fps. STATIC is pinned so a
    #: run's frame sampling is a property of the config, not of the model.
    media_processing: str = DEFAULT_MEDIA_PROCESSING

    #: Token resolution per frame. None leaves the API default (low, ~65 tokens
    #: per frame as measured). Set explicitly to trade cost against detail.
    media_resolution: str | None = None

    #: One of MINIMAL / LOW / MEDIUM / HIGH, or None to leave the API default.
    thinking_level: str | None = DEFAULT_THINKING_LEVEL
