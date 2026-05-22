"""Configuration objects for Gemini-based workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from video_llm_evaluation.constants import DEFAULT_MODEL_NAME, DEFAULT_PROMPT_VERSION, ERROR_CLASSES


@dataclass(slots=True)
class GeminiVideoConfig:
    model_name: str = DEFAULT_MODEL_NAME
    prompt_version: str = DEFAULT_PROMPT_VERSION
    error_classes: tuple[str, ...] = field(default_factory=lambda: tuple(ERROR_CLASSES))
    temperature: float = 0.0
    max_retries: int = 3
    request_timeout_s: int = 180
    thinking_level: str = "low"
    api_key_env: str = "GEMINI_API_KEY"
