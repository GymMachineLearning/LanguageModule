"""Prompt construction helpers for Gemini squat analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from video_llm_evaluation.constants import DEFAULT_PROMPT_VERSION, ERROR_CLASSES


DEFAULT_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().with_name("prompts") / "squat_v1.yaml"


@dataclass(slots=True)
class GeminiPromptTemplate:
    prompt_version: str
    system_prompt: str
    user_prompt: str


@dataclass(slots=True)
class GeminiPromptBuilder:
    template_path: Path = DEFAULT_PROMPT_TEMPLATE_PATH
    error_classes: tuple[str, ...] = tuple(ERROR_CLASSES)
    _template: GeminiPromptTemplate | None = field(default=None, init=False, repr=False)

    def load_template(self) -> GeminiPromptTemplate:
        if self._template is not None:
            return self._template

        payload = yaml.safe_load(self.template_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Prompt template at {self.template_path} must be a YAML mapping")

        prompt_version = str(payload.get("prompt_version", DEFAULT_PROMPT_VERSION))
        system_prompt = str(payload.get("system_prompt", "")).strip()
        user_prompt = str(payload.get("user_prompt", "")).strip()
        if not system_prompt or not user_prompt:
            raise ValueError(f"Prompt template at {self.template_path} must define system_prompt and user_prompt")

        self._template = GeminiPromptTemplate(
            prompt_version=prompt_version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return self._template

    def build_system_prompt(self) -> str:
        return self.load_template().system_prompt.replace("{error_classes}", ", ".join(self.error_classes))

    def build_user_prompt(self, *, video_id: str, duration_s: float) -> str:
        template = self.load_template()
        return (
            template.user_prompt.replace("{video_id}", video_id)
            .replace("{duration_s}", str(duration_s))
            .replace("{error_classes}", ", ".join(self.error_classes))
        )

    @property
    def prompt_version(self) -> str:
        return self.load_template().prompt_version

    def build_prompt_bundle(self, *, video_id: str, duration_s: float) -> dict[str, str]:
        return {
            "prompt_version": self.prompt_version,
            "system_prompt": self.build_system_prompt(),
            "user_prompt": self.build_user_prompt(video_id=video_id, duration_s=duration_s),
        }
