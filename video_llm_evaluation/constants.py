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

# MLPSD label matrices carry ten rows. These are the row indices corresponding to
# the six classes used by this pipeline, in ERROR_CLASSES order.
MLPSD_ERROR_ROW_INDICES = (0, 1, 2, 3, 5, 7)
