"""Gemini API client wrapper for reusable video-analysis workflows."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import GeminiVideoConfig
from .prompt_builder import GeminiPromptBuilder
from .response_parser import GeminiResponseParser

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def _load_dotenv(env_path: Path = DEFAULT_ENV_PATH) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        os.environ[key] = value


def _load_google_genai() -> Any:
    try:
        from google import genai as google_genai  # type: ignore[import-not-found]

        return google_genai
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("google-genai is required to use GeminiClient.generate") from exc


def _wait_for_active_file(client: Any, uploaded_file: Any, *, poll_interval_s: float = 1.0, max_attempts: int = 60) -> Any:
    state = getattr(uploaded_file, "state", None)
    if state == getattr(state, "ACTIVE", None):
        return uploaded_file

    file_name = getattr(uploaded_file, "name", None)
    if not file_name:
        return uploaded_file

    for _ in range(max_attempts):
        current_file = client.files.get(name=file_name)
        current_state = getattr(current_file, "state", None)
        if current_state == getattr(current_state, "ACTIVE", None):
            return current_file
        if current_state == getattr(current_state, "FAILED", None):
            raise RuntimeError(f"Uploaded Gemini file {file_name} failed to process")
        time.sleep(poll_interval_s)

    raise RuntimeError(f"Uploaded Gemini file {file_name} did not become ACTIVE within the expected time")


@dataclass(slots=True)
class GeminiClient:
    config: GeminiVideoConfig = field(default_factory=GeminiVideoConfig)
    prompt_builder: GeminiPromptBuilder = field(default_factory=GeminiPromptBuilder)
    response_parser: GeminiResponseParser = field(default_factory=GeminiResponseParser)
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            _load_dotenv()
            api_key = os.getenv(self.config.api_key_env)
            if api_key:
                google_genai = _load_google_genai()
                self.client = google_genai.Client(
                    api_key=api_key,
                    http_options={"timeout": self.config.request_timeout_s * 1000},
                )

    def build_prompts(self, *, video_id: str, duration_s: float) -> dict[str, str]:
        return self.prompt_builder.build_prompt_bundle(video_id=video_id, duration_s=duration_s)

    def parse_response(
        self,
        raw_response: str | Mapping[str, Any],
        *,
        video_id: str | None = None,
        duration_s: float | None = None,
    ):
        return self.response_parser.parse(raw_response, video_id=video_id, duration_s=duration_s)

    def available_model_names(self) -> list[str]:
        """Model ids the API currently exposes, without the ``models/`` prefix."""
        if self.client is None:
            raise RuntimeError("GeminiClient is not initialized with a live SDK client.")
        return [str(model.name).removeprefix("models/") for model in self.client.models.list()]

    def verify_model_available(self) -> None:
        """Fail before any upload if the configured model name is not served.

        Without this a typo surfaces only after the first video has been uploaded
        and processed, so the cost of the mistake scales with the batch.
        """
        available = self.available_model_names()
        if self.config.model_name in available:
            return
        near = [name for name in available if name.startswith(self.config.model_name.split("-preview")[0][:14])]
        hint = f" Did you mean one of: {', '.join(sorted(near)[:5])}?" if near else ""
        raise ValueError(f"Model {self.config.model_name!r} is not available on this API key.{hint}")

    def build_video_part(self, uploaded_video: Any) -> Any:
        """Wrap an uploaded file so frame sampling is explicit rather than inherited.

        ``video_fps`` only takes effect under STATIC media processing, so the two
        always travel together.
        """
        types = _load_google_genai().types
        part_kwargs: dict[str, Any] = {}
        if self.config.video_fps:
            part_kwargs["video_metadata"] = types.VideoMetadata(fps=float(self.config.video_fps))
        if self.config.media_processing:
            part_kwargs["media_processing"] = self.config.media_processing
        if self.config.media_resolution:
            # Part takes a PartMediaResolution object, unlike GenerateContentConfig
            # which takes the bare enum. Setting it per-part is the finer control
            # and is the only place that accepts ULTRA_HIGH.
            part_kwargs["media_resolution"] = types.PartMediaResolution(level=self.config.media_resolution)
        return types.Part(
            file_data=types.FileData(
                file_uri=uploaded_video.uri,
                mime_type=getattr(uploaded_video, "mime_type", "video/mp4"),
            ),
            **part_kwargs,
        )

    def build_request_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"temperature": self.config.temperature}
        if self.config.thinking_level:
            types = _load_google_genai().types
            config["thinking_config"] = types.ThinkingConfig(thinking_level=self.config.thinking_level)
        return config

    def generate(self, *, video_path: str, video_id: str, duration_s: float) -> Any:
        if self.client is None:
            raise RuntimeError(
                f"GeminiClient is not initialized with a live SDK client. "
                f"Set {self.config.api_key_env} or pass a configured client instance."
            )

        prompts = self.build_prompts(video_id=video_id, duration_s=duration_s)
        uploaded_video = self.client.files.upload(file=video_path, config={"mime_type": "video/mp4"})
        uploaded_video = _wait_for_active_file(self.client, uploaded_video)
        return self.client.models.generate_content(
            model=self.config.model_name,
            contents=[
                prompts["system_prompt"],
                prompts["user_prompt"],
                self.build_video_part(uploaded_video),
            ],
            config=self.build_request_config(),
        )
