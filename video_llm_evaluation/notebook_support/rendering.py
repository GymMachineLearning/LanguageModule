"""Rendering helpers for the squat results inspector notebook."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


def frame_to_png_bytes(frame_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray(frame_rgb).save(buffer, format='PNG')
    return buffer.getvalue()


def load_raw_response_text(prediction_path: Path) -> str | None:
    raw_path = prediction_path.parent.parent / 'raw_responses' / f'{prediction_path.stem}.raw.json'
    if not raw_path.exists():
        return None
    return raw_path.read_text(encoding='utf-8')


def pretty_json(payload: object) -> str:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return payload
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
