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

# Frames per second Gemini samples from a video. The API default of 1.0 yields
# only 2-3 frames per squat repetition, and 87% of the boundaries the model
# returned under it landed exactly on whole seconds.
DEFAULT_VIDEO_FPS = 2.0

# Pinned so frame sampling is a property of the run, not of the model: the API
# default is "model-specific processing", and Gemini 3.x Flash models may resolve
# it to AGENTIC, under which video_fps is ignored entirely.
DEFAULT_MEDIA_PROCESSING = "STATIC"

# Which MLPSD split the pipeline works on by default. Only recordings in this
# split are sent to the model, evaluated and reported; "all" disables filtering.
DEFAULT_SPLIT = "test"

# One of MINIMAL / LOW / MEDIUM / HIGH, or None for the API default.
DEFAULT_THINKING_LEVEL = "MEDIUM"

# Models that support AGENTIC media processing. Used to log, at run time, that
# STATIC is being pinned on a model that would otherwise be free to choose.
AGENTIC_CAPABLE_MODEL_PREFIXES = (
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
)

# MLPSD label matrices carry ten rows. These are the row indices corresponding to
# the six classes used by this pipeline, in ERROR_CLASSES order.
MLPSD_ERROR_ROW_INDICES = (0, 1, 2, 3, 5, 7)
