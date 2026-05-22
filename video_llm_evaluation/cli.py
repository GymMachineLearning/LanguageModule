"""CLI entrypoints for Gemini-based squat video evaluation runs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from llm_api.gemini import GeminiClient, GeminiPromptBuilder, GeminiVideoConfig
from video_llm_evaluation.discovery import discover_videos
from video_llm_evaluation.evaluation.persistence import (
    save_config,
    save_labels_csv,
    save_labels_npy,
    save_prediction_json,
    save_raw_response,
)
from video_llm_evaluation.evaluation.segments_to_frames import video_prediction_to_frame_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEOS_ROOT = (REPO_ROOT / "../../videos/Nagrania/Squat/preprocessed").resolve()
DEFAULT_RESULTS_ROOT = (REPO_ROOT / "results/llm_evaluation/squat").resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video_llm_evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Gemini requests over squat videos")
    run_parser.add_argument("--videos-root", type=Path, default=DEFAULT_VIDEOS_ROOT)
    run_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    run_parser.add_argument("--prompt-yaml", type=Path, default=None)
    run_parser.add_argument("--max_request", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--model-name", type=str, default="gemini-3.1-pro-preview")
    run_parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY")
    run_parser.add_argument("--skip-existing", action="store_true", default=False)
    return parser


def _select_videos(items, max_request: int | None, seed: int):
    if max_request is None or max_request >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(list(items), k=max_request)


def _normalize_raw_response(response: object) -> object:
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return getattr(response, "text")
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    return {"repr": repr(response)}


def _run(args: argparse.Namespace) -> None:
    results_root = args.results_root
    videos_root = args.videos_root
    results_root.mkdir(parents=True, exist_ok=True)

    prompt_builder = GeminiPromptBuilder(template_path=args.prompt_yaml) if args.prompt_yaml else GeminiPromptBuilder()
    client = GeminiClient(
        config=GeminiVideoConfig(model_name=args.model_name, api_key_env=args.api_key_env),
        prompt_builder=prompt_builder,
    )
    if client.client is None:
        raise RuntimeError(
            f"Gemini API key not found in environment variable {args.api_key_env}. "
            f"Set it before running, for example: export {args.api_key_env}=<your_key>"
        )

    discovered = discover_videos(videos_root, results_root)
    selected = _select_videos(discovered, args.max_request, args.seed)

    save_config(
        results_root,
        {
            "videos_root": str(videos_root),
            "results_root": str(results_root),
            "max_request": args.max_request,
            "seed": args.seed,
            "model_name": args.model_name,
            "prompt_yaml": str(args.prompt_yaml) if args.prompt_yaml else None,
            "num_discovered": len(discovered),
            "num_selected": len(selected),
        },
    )

    for item in selected:
        run_paths = item.output_dir
        if args.skip_existing and (run_paths / "predictions_json" / f"{item.video_id}.json").exists():
            continue

        response = client.generate(video_path=str(item.video_path), video_id=item.video_id, duration_s=item.duration_s)
        raw_response = _normalize_raw_response(response)
        prediction = client.parse_response(raw_response, video_id=item.video_id, duration_s=item.duration_s)
        labels = video_prediction_to_frame_labels(prediction, fps=item.fps, num_frames=item.num_frames)

        save_prediction_json(run_paths, prediction, model_name=args.model_name, prompt_version=client.prompt_builder.prompt_version)
        save_raw_response(run_paths, item.video_id, raw_response)
        save_labels_npy(run_paths, item.video_id, labels)
        save_labels_csv(run_paths, item.video_id, labels, fps=item.fps)

    manifest_path = results_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "video_path", "duration_s", "fps", "num_frames", "ground_truth_path"])
        for item in selected:
            writer.writerow([item.video_id, str(item.video_path), item.duration_s, item.fps, item.num_frames, ""])


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args)


if __name__ == "__main__":
    main()
