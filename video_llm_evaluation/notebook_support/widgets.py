"""Interactive notebook widgets for reviewing squat evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

import ipywidgets as widgets
from IPython.display import clear_output

from video_llm_evaluation.notebook_support.data import ground_truth_row_for_video, resolve_video_path
from video_llm_evaluation.notebook_support.rendering import frame_to_png_bytes, load_raw_response_text, pretty_json
from video_llm_evaluation.notebook_support.video import build_annotated_frames


def build_video_review_widget(
    results_df,
    dataset_df,
    *,
    videos_root,
    max_frames: int,
    frame_step: int,
    start_index: int = 0,
):
    if results_df.empty:
        return widgets.VBox([widgets.HTML('<b>Brak wyników do wyświetlenia.</b>')])

    state = {'index': max(0, min(start_index, len(results_df) - 1)), 'frames': [], 'timestamps_s': [], 'fps': 0.0}
    title = widgets.HTML()
    meta = widgets.HTML()
    image = widgets.Image(format='png')
    raw_output = widgets.Output(layout=widgets.Layout(max_height='320px', overflow='auto', border='1px solid #ddd', padding='8px'))
    frame_slider = widgets.IntSlider(value=0, min=0, max=0, step=1, description='Klatka', continuous_update=False)
    play = widgets.Play(value=0, min=0, max=0, step=1, interval=250, description='Odtwarzanie')
    speed_slider = widgets.FloatSlider(value=1.0, min=0.5, max=3.0, step=0.1, description='Prędkość', readout_format='.1f', continuous_update=False)
    prev_button = widgets.Button(description='Poprzednie nagranie', icon='arrow-left')
    next_button = widgets.Button(description='Następne nagranie', icon='arrow-right')

    widgets.jslink((play, 'value'), (frame_slider, 'value'))

    def show_frame(frame_index: int) -> None:
        if not state['frames']:
            image.value = b''
            return
        frame_index = max(0, min(frame_index, len(state['frames']) - 1))
        image.value = frame_to_png_bytes(state['frames'][frame_index])
        frame_slider.value = frame_index

    def refresh_raw_output(prediction_path) -> None:
        with raw_output:
            clear_output(wait=True)
            raw_response_text = load_raw_response_text(prediction_path)
            if raw_response_text is not None:
                print(raw_response_text)
            else:
                print(pretty_json(json.loads(prediction_path.read_text(encoding='utf-8'))))

    def refresh_video(index: int) -> None:
        index = max(0, min(index, len(results_df) - 1))
        state['index'] = index
        result_row = results_df.iloc[index]
        video_path_value = result_row.get('video_path', result_row.get('video_path_from_results', ''))
        video_path = resolve_video_path(video_path_value, videos_root=videos_root)
        prediction_path = Path(result_row['prediction_path'])
        prediction_payload = json.loads(prediction_path.read_text(encoding='utf-8'))
        dataset_row = ground_truth_row_for_video(dataset_df, video_path)
        gt_errors_for_video = [] if dataset_row.empty else list(dataset_row.iloc[0].get('gt_errors', []))
        annotated_bundle = build_annotated_frames(
            video_path,
            prediction_payload,
            max_frames=max_frames,
            frame_step=frame_step,
            gt_errors_for_video=gt_errors_for_video,
        )
        state['frames'] = annotated_bundle['frames']
        state['timestamps_s'] = annotated_bundle['timestamps_s']
        state['fps'] = float(annotated_bundle['fps'])
        frame_slider.max = max(0, len(state['frames']) - 1)
        play.max = frame_slider.max
        frame_slider.value = 0
        play.value = 0
        play.interval = max(40, int(1000 / max(speed_slider.value * max(state['fps'], 1e-6), 1e-6)))
        title.value = f'<b>{result_row.get("video_id", "")}</b> &nbsp; | &nbsp; index: {index + 1}/{len(results_df)}'
        meta.value = (
            f'<div><b>video_path:</b> {video_path}</div>'
            f'<div><b>GT labels:</b> {", ".join(gt_errors_for_video) if gt_errors_for_video else "brak"}</div>'
            f'<div><b>fps:</b> {state["fps"]:.2f} &nbsp; | &nbsp; <b>frames:</b> {len(state["frames"])}</div>'
        )
        show_frame(0)
        refresh_raw_output(prediction_path)

    def on_frame_change(change: dict) -> None:
        if change.get('name') == 'value' and state['frames']:
            show_frame(int(change['new']))

    def on_speed_change(change: dict) -> None:
        if change.get('name') == 'value':
            play.interval = max(40, int(1000 / max(float(change['new']) * max(state['fps'], 1e-6), 1e-6)))

    def on_prev(_button: widgets.Button) -> None:
        refresh_video(state['index'] - 1)

    def on_next(_button: widgets.Button) -> None:
        refresh_video(state['index'] + 1)

    frame_slider.observe(on_frame_change, names='value')
    speed_slider.observe(on_speed_change, names='value')
    prev_button.on_click(on_prev)
    next_button.on_click(on_next)

    controls = widgets.HBox([prev_button, next_button, speed_slider, play])
    container = widgets.VBox([title, meta, controls, frame_slider, image, raw_output])
    refresh_video(state['index'])
    return container
