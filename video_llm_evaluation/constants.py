"""Shared constants for the squat video evaluation pipeline."""

ERROR_CLASSES = [
    "Squat-depth",
    "Back-round",
    "Taking-off-foot",
    "Knee-collapse",
    "Dominant-hip",
    "No-knee-outlet",
]

ERROR_TO_INDEX = {name: idx for idx, name in enumerate(ERROR_CLASSES)}

DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"
DEFAULT_PROMPT_VERSION = "v1"
