"""Tests that the video-sampling parameters actually reach the request.

A parameter that never leaves the process looks exactly like a parameter the API
ignores, so these assert on the shape of what the SDK is handed. Everything runs
against an injected stub — no API calls, no key required.
"""

import unittest
from types import SimpleNamespace

from google.genai import types

from llm_api.gemini import GeminiClient, GeminiPromptBuilder, GeminiVideoConfig
from video_llm_evaluation.constants import (
    DEFAULT_MEDIA_PROCESSING,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_VIDEO_FPS,
)


class _StubFiles:
    def __init__(self) -> None:
        self.uploaded: list[str] = []

    def upload(self, *, file, config):
        self.uploaded.append(str(file))
        return SimpleNamespace(
            name="files/stub",
            uri="https://generativelanguage.googleapis.com/v1beta/files/stub",
            mime_type="video/mp4",
            # The real SDK hands back a FileState enum member, and
            # _wait_for_active_file compares it against its own .ACTIVE attribute.
            state=types.FileState.ACTIVE,
        )

    def get(self, *, name):
        return SimpleNamespace(
            name=name,
            uri="https://x/files/stub",
            mime_type="video/mp4",
            state=types.FileState.ACTIVE,
        )


class _StubModels:
    def __init__(self, names=("gemini-3.1-pro-preview", "gemini-3.8-flash")) -> None:
        self.names = names
        self.calls: list[dict] = []

    def list(self):
        return [SimpleNamespace(name=f"models/{name}") for name in self.names]

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text="{}")


class _StubClient:
    def __init__(self, model_names=("gemini-3.1-pro-preview", "gemini-3.8-flash")) -> None:
        self.files = _StubFiles()
        self.models = _StubModels(model_names)


def _client(**config_kwargs) -> GeminiClient:
    return GeminiClient(config=GeminiVideoConfig(**config_kwargs), client=_StubClient())


class VideoPartTests(unittest.TestCase):
    def test_video_fps_defaults_to_two_frames_per_second(self) -> None:
        self.assertEqual(DEFAULT_VIDEO_FPS, 2.0)
        self.assertEqual(GeminiVideoConfig().video_fps, 2.0)

    def test_the_part_carries_the_configured_frame_rate(self) -> None:
        client = _client(video_fps=2.0)
        part = client.build_video_part(SimpleNamespace(uri="https://x/f", mime_type="video/mp4"))
        self.assertIsNotNone(part.video_metadata)
        self.assertEqual(part.video_metadata.fps, 2.0)

    def test_static_processing_is_pinned_by_default(self) -> None:
        """video_fps is only honoured under STATIC, so it must not be left to the model."""
        self.assertEqual(DEFAULT_MEDIA_PROCESSING, "STATIC")
        client = _client()
        part = client.build_video_part(SimpleNamespace(uri="https://x/f", mime_type="video/mp4"))
        self.assertEqual(part.media_processing, types.MediaProcessing.STATIC)

    def test_agentic_processing_can_be_selected(self) -> None:
        client = _client(media_processing="AGENTIC")
        part = client.build_video_part(SimpleNamespace(uri="https://x/f", mime_type="video/mp4"))
        self.assertEqual(part.media_processing, types.MediaProcessing.AGENTIC)

    def test_media_resolution_is_omitted_unless_configured(self) -> None:
        default_part = _client().build_video_part(SimpleNamespace(uri="https://x/f", mime_type="video/mp4"))
        self.assertIsNone(default_part.media_resolution)

        explicit = _client(media_resolution="MEDIA_RESOLUTION_LOW")
        part = explicit.build_video_part(SimpleNamespace(uri="https://x/f", mime_type="video/mp4"))
        self.assertEqual(part.media_resolution.level, types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW)

    def test_media_resolution_is_not_also_sent_in_the_request_config(self) -> None:
        """Two places accept it with different shapes; sending both invites a conflict."""
        config = _client(media_resolution="MEDIA_RESOLUTION_LOW").build_request_config()
        self.assertNotIn("media_resolution", config)

    def test_the_part_points_at_the_uploaded_file(self) -> None:
        part = _client().build_video_part(SimpleNamespace(uri="https://x/files/abc", mime_type="video/mp4"))
        self.assertEqual(part.file_data.file_uri, "https://x/files/abc")
        self.assertEqual(part.file_data.mime_type, "video/mp4")


class RequestConfigTests(unittest.TestCase):
    def test_thinking_level_defaults_to_medium_and_is_sent(self) -> None:
        self.assertEqual(DEFAULT_THINKING_LEVEL, "MEDIUM")
        config = _client().build_request_config()
        self.assertIn("thinking_config", config)
        self.assertEqual(config["thinking_config"].thinking_level, types.ThinkingLevel.MEDIUM)

    def test_thinking_config_is_omitted_when_level_is_none(self) -> None:
        config = _client(thinking_level=None).build_request_config()
        self.assertNotIn("thinking_config", config)

    def test_temperature_is_always_sent(self) -> None:
        self.assertEqual(_client(temperature=0.0).build_request_config()["temperature"], 0.0)


class GenerateTests(unittest.TestCase):
    def test_generate_sends_prompts_then_the_configured_video_part(self) -> None:
        client = _client(video_fps=2.0)
        client.generate(video_path="/tmp/clip.mp4", video_id="clip", duration_s=12.0)

        call = client.client.models.calls[0]
        self.assertEqual(call["model"], "gemini-3.1-pro-preview")
        system_prompt, user_prompt, video_part = call["contents"]
        self.assertIn("Squat-depth", system_prompt)
        self.assertIn("clip", user_prompt)
        self.assertEqual(video_part.video_metadata.fps, 2.0)
        self.assertEqual(video_part.media_processing, types.MediaProcessing.STATIC)
        self.assertEqual(call["config"]["thinking_config"].thinking_level, types.ThinkingLevel.MEDIUM)

    def test_generate_uploads_the_file_it_was_given(self) -> None:
        client = _client()
        client.generate(video_path="/tmp/clip.mp4", video_id="clip", duration_s=12.0)
        self.assertEqual(client.client.files.uploaded, ["/tmp/clip.mp4"])

    def test_generate_without_a_live_client_fails_before_doing_anything(self) -> None:
        client = GeminiClient(config=GeminiVideoConfig(api_key_env="DEFINITELY_NOT_SET_12345"), client=None)
        with self.assertRaises(RuntimeError):
            client.generate(video_path="/tmp/clip.mp4", video_id="clip", duration_s=12.0)


class PromptTemplateTests(unittest.TestCase):
    """The template is substituted with str.replace(), not str.format()."""

    def test_the_prompt_contains_no_doubled_braces(self) -> None:
        """Doubled braces were copied verbatim by gemini-3.8-flash, breaking JSON."""
        builder = GeminiPromptBuilder()
        system_prompt = builder.build_system_prompt()
        user_prompt = builder.build_user_prompt(video_id="clip", duration_s=12.0)
        self.assertNotIn("{{", system_prompt)
        self.assertNotIn("}}", system_prompt)
        self.assertNotIn("{{", user_prompt)
        self.assertNotIn("}}", user_prompt)

    def test_the_example_response_in_the_prompt_is_valid_json(self) -> None:
        import json
        import re

        system_prompt = GeminiPromptBuilder().build_system_prompt()
        blob = re.search(r"\{.*\}", system_prompt, re.S)
        self.assertIsNotNone(blob)
        payload = json.loads(blob.group(0))
        self.assertEqual(len(payload["predictions"]), 6)

    def test_placeholders_are_substituted(self) -> None:
        builder = GeminiPromptBuilder()
        user_prompt = builder.build_user_prompt(video_id="clip_xyz", duration_s=12.0)
        self.assertIn("clip_xyz", user_prompt)
        self.assertNotIn("{video_id}", user_prompt)
        self.assertNotIn("{error_classes}", builder.build_system_prompt())


class ModelPreflightTests(unittest.TestCase):
    def test_a_served_model_passes(self) -> None:
        _client(model_name="gemini-3.8-flash").verify_model_available()

    def test_an_unknown_model_is_rejected_before_any_upload(self) -> None:
        client = _client(model_name="gemini-9.9-flash")
        with self.assertRaises(ValueError) as context:
            client.verify_model_available()
        self.assertIn("gemini-9.9-flash", str(context.exception))
        self.assertEqual(client.client.files.uploaded, [])

    def test_a_near_miss_suggests_the_real_model_names(self) -> None:
        client = _client(model_name="gemini-3.8-flashh")
        with self.assertRaises(ValueError) as context:
            client.verify_model_available()
        self.assertIn("gemini-3.8-flash", str(context.exception))

    def test_available_model_names_drop_the_models_prefix(self) -> None:
        self.assertIn("gemini-3.8-flash", _client().available_model_names())


if __name__ == "__main__":
    unittest.main()
