"""Standalone browser review page generator for squat evaluation results."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

from video_llm_evaluation.notebook_support.analysis import gt_summary_for_video
from video_llm_evaluation.notebook_support.data import resolve_video_path
from video_llm_evaluation.notebook_support.rendering import load_raw_response_text, pretty_json
from video_llm_evaluation.notebook_support.video import build_annotated_frames
from video_llm_evaluation.constants import ERROR_CLASSES


def build_review_items(
    results_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    *,
    videos_root: Path,
    max_frames: int,
    frame_step: int,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for item_index, (_, result_row) in enumerate(results_df.iterrows()):
        prediction_path = Path(result_row['prediction_path'])
        video_path_value = result_row.get('video_path', result_row.get('video_path_from_results', ''))
        video_path = resolve_video_path(video_path_value, videos_root=videos_root)
        prediction_payload = json.loads(prediction_path.read_text(encoding='utf-8'))
        gt_info = gt_summary_for_video(dataset_df, video_path)
        if gt_info.get('gt_labels_matrix') is None and not dataset_df.empty:
            fallback_row = dataset_df.iloc[min(item_index, len(dataset_df) - 1)]
            gt_info = {
                'gt_errors': list(fallback_row.get('gt_errors', [])),
                'gt_labels_matrix': fallback_row.get('gt_labels_matrix'),
                'gt_active_rows': list(fallback_row.get('gt_active_rows', [])),
                'gt_row': fallback_row,
            }
        gt_labels_matrix = gt_info.get('gt_labels_matrix')
        gt_active_rows = gt_info.get('gt_active_rows', [])
        gt_error_names = [ERROR_CLASSES[index] for index in gt_active_rows if 0 <= index < len(ERROR_CLASSES)]
        gt_shape = list(getattr(gt_labels_matrix, 'shape', ())) if gt_labels_matrix is not None else []
        raw_response_text = load_raw_response_text(prediction_path)
        if raw_response_text is None:
            raw_response_text = pretty_json(json.loads(prediction_path.read_text(encoding='utf-8')))

        annotated_bundle = build_annotated_frames(
            video_path,
            prediction_payload,
            max_frames=max_frames,
            frame_step=frame_step,
            gt_labels_matrix=gt_labels_matrix,
            gt_errors_for_video=[f'row {index}' for index in gt_active_rows] if gt_labels_matrix is None else None,
        )

        items.append(
            {
                'video_id': str(result_row.get('video_id', '')),
                'video_path': str(video_path),
                'annotated_video_path': None,
                'annotated_bundle': annotated_bundle,
                'raw_response_text': raw_response_text,
                'gt_shape': gt_shape,
                'gt_active_rows': gt_active_rows,
                'gt_error_names': gt_error_names,
                'prediction_path': str(prediction_path),
            }
        )
    return items


def write_review_page(
    results_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    *,
    videos_root: Path,
    output_dir: Path,
    max_frames: int,
    frame_step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / 'media'
    media_dir.mkdir(parents=True, exist_ok=True)

    items = build_review_items(
        results_df,
        dataset_df,
        videos_root=videos_root,
        max_frames=max_frames,
        frame_step=frame_step,
    )

    # Materialize mp4 files after the frames are prepared.
    for item in items:
        bundle = item.pop('annotated_bundle')
        fps = float(bundle['fps'])
        frames = bundle['frames']
        if not frames:
            continue
        video_id = item['video_id'] or 'video'
        safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(video_id))
        out_path = media_dir / f'{safe_name}.mp4'
        from cv2 import VideoWriter, VideoWriter_fourcc, cvtColor, COLOR_RGB2BGR, resize, INTER_AREA

        height, width = frames[0].shape[:2]
        max_dimension = 600
        scale = min(1.0, max_dimension / width, max_dimension / height)
        if scale < 1.0:
            target_width = int(round(width * scale))
            target_height = int(round(height * scale))
        else:
            target_width = width
            target_height = height

        codec_candidates = ['avc1', 'H264', 'mp4v']
        writer = None
        for codec in codec_candidates:
            candidate = VideoWriter(str(out_path), VideoWriter_fourcc(*codec), fps / max(frame_step, 1), (target_width, target_height))
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is None:
            raise RuntimeError('Could not open any MP4 video writer for annotated video')

        for frame_rgb in frames:
            if scale < 1.0:
                frame_rgb = resize(frame_rgb, (target_width, target_height), interpolation=INTER_AREA)
            writer.write(cvtColor(frame_rgb, COLOR_RGB2BGR))
        writer.release()
        item['annotated_video_path'] = str(out_path.relative_to(output_dir))
        item['annotated_video_data_uri'] = 'data:video/mp4;base64,' + base64.b64encode(out_path.read_bytes()).decode('ascii')
        item['gt_error_names'] = item.get('gt_error_names', [])

    payload = json.dumps(items, ensure_ascii=False)
    html_path = output_dir / 'index.html'
    html_path.write_text(
        f"""<!doctype html>
<html lang=\"pl\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Squat review app</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f4f7fb; color: #172033; }}
    .shell {{ display: grid; grid-template-columns: 1.6fr 1fr; gap: 16px; padding: 16px; min-height: 100vh; box-sizing: border-box; }}
    .panel {{ background: white; border-radius: 14px; padding: 14px; box-shadow: 0 8px 30px rgba(0,0,0,.08); overflow: auto; }}
    .video-wrap {{ display: flex; justify-content: center; }}
    .video-wrap video {{ width: 100%; max-width: 900px; max-height: 900px; height: auto; border-radius: 12px; background: #111; }}
    .controls {{ display:flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 12px 0; }}
    button, input[type=range] {{ accent-color: #1f6feb; }}
    button {{ border: 0; border-radius: 999px; padding: 8px 12px; background: #1f6feb; color: white; cursor: pointer; }}
    button.secondary {{ background: #e6ebf5; color: #172033; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 12px; max-height: 320px; overflow: auto; }}
    .meta {{ display:grid; gap: 8px; font-size: 14px; }}
    .label {{ color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-weight: 600; }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"panel\">
      <div class=\"video-wrap\">
        <video id=\"video\" controls playsinline></video>
      </div>
      <div class=\"controls\">
        <button id=\"prev\" class=\"secondary\">Poprzednie nagranie</button>
        <button id=\"next\" class=\"secondary\">Następne nagranie</button>
        <label>Prędkość <input id=\"speed\" type=\"range\" min=\"0.5\" max=\"3\" step=\"0.1\" value=\"1\" /></label>
        <span id=\"speedValue\">1.0x</span>
      </div>
      <div class=\"meta\">
        <div><div class=\"label\">Video</div><div class=\"value\" id=\"videoId\"></div></div>
        <div><div class=\"label\">GT matrix shape</div><div class=\"value\" id=\"gtShape\"></div></div>
        <div><div class=\"label\">GT errors present</div><div class=\"value\" id=\"gtErrorNames\"></div></div>
        <div><div class=\"label\">Path</div><div class=\"value\" id=\"videoPath\"></div></div>
      </div>
    </div>
    <div class=\"panel\">
      <h3 style=\"margin-top: 0;\">Raw LLM output</h3>
      <pre id=\"rawOutput\"></pre>
    </div>
  </div>
  <script>
    const items = {payload};
    const video = document.getElementById('video');
    const rawOutput = document.getElementById('rawOutput');
    const videoId = document.getElementById('videoId');
    const gtShape = document.getElementById('gtShape');
    const gtErrorNames = document.getElementById('gtErrorNames');
    const videoPath = document.getElementById('videoPath');
    const speed = document.getElementById('speed');
    const speedValue = document.getElementById('speedValue');
    let index = 0;

    function render() {{
      const item = items[index];
      video.src = item.annotated_video_path || item.annotated_video_data_uri;
      video.load();
      video.playbackRate = parseFloat(speed.value);
      rawOutput.textContent = item.raw_response_text;
      videoId.textContent = item.video_id;
      gtShape.textContent = item.gt_shape && item.gt_shape.length ? item.gt_shape.join(' x ') : 'n/a';
      gtErrorNames.textContent = item.gt_error_names && item.gt_error_names.length ? item.gt_error_names.join(', ') : 'none';
      videoPath.textContent = item.video_path;
      speedValue.textContent = video.playbackRate.toFixed(1) + 'x';
      document.title = 'Squat review app - ' + item.video_id;
    }}

    document.getElementById('prev').onclick = () => {{ index = (index - 1 + items.length) % items.length; render(); }};
    document.getElementById('next').onclick = () => {{ index = (index + 1) % items.length; render(); }};
    speed.oninput = () => {{
      video.playbackRate = parseFloat(speed.value);
      speedValue.textContent = video.playbackRate.toFixed(1) + 'x';
    }};

    render();
  </script>
</body>
</html>
""",
        encoding='utf-8',
    )
    return html_path
