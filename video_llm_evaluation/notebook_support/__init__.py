"""Notebook helper utilities for squat result inspection."""

from .analysis import build_summary_dataframe, gt_summary_for_video, intervals_from_payload, predicted_errors_from_payload
from .data import (
    ground_truth_errors_for_video,
    ground_truth_row_for_video,
    load_mlpsd_dataframe,
    load_manifest,
    load_prediction_json,
    load_results_dataframe,
    normalize_video_key,
    resolve_video_path,
    select_video,
)
from .reporting import (
    MetricsNotAvailable,
    MetricsReport,
    class_detail,
    annotate_results_with_split,
    class_overview,
    discover_runs,
    filter_results_by_split,
    load_metrics_report,
    load_run_report,
    per_video_class_matrix,
    plot_class_f1,
    plot_f1_vs_tolerance,
    run_header,
    run_totals,
    segment_pivot,
    select_run,
    show_class_card,
    show_report,
    skipped_overview,
    staleness_warning,
    style_class_overview,
    style_per_video_class_matrix,
    style_segment_pivot,
)
from .review_app import write_review_page
from .rendering import frame_to_png_bytes, load_raw_response_text, pretty_json
from .video import active_errors_at_time, build_annotated_frames, draw_error_overlay, load_video_frames, prediction_segments_by_error
from .widgets import build_video_review_widget
