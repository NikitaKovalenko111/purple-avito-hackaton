from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, List


def _optional_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


torch = _optional_import("torch")
AdamW = getattr(_optional_import("torch.optim"), "AdamW", None)

from model import (
    DraftSplitPipeline,
    PipelineSettings,
    TrainingSettings,
    evaluate_retrieval_recall,
    evaluate_split_quality,
    evaluate_split_probability_threshold,
    load_labeled_items_csv,
    load_labeled_items_jsonl,
    load_microcategories_from_csv,
    search_best_class_probability_thresholds,
    search_best_probability_threshold,
    search_best_reranking_controls,
    split_dataset,
    to_response_json,
)


def _require_torch() -> None:
    if torch is None or AdamW is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")


DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (0.08, 0.12, 0.16, 0.20, 0.24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Avito split-draft pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--transformer-name", type=str, default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--tfidf-threshold", type=float, default=0.03)
    parser.add_argument("--tfidf-top-k", type=int, default=11)
    parser.add_argument("--prob-threshold", type=float, default=0.08)
    parser.add_argument("--split-threshold", type=float, default=0.5)
    parser.add_argument("--max-drafts", type=int, default=0, help="Max number of generated drafts (0 = no cap).")
    parser.add_argument("--score-margin", type=float, default=1.0, help="Keep classes within top_score - margin.")
    parser.add_argument(
        "--relative-ratio",
        type=float,
        default=0.0,
        help="Keep classes with score >= top_score * ratio (0 = disabled).",
    )
    parser.add_argument(
        "--score-blend-alpha",
        type=float,
        default=1.0,
        help="Blend weight for model score vs TF-IDF score (1.0 = model-only).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--split-threshold-grid",
        type=float,
        nargs="+",
        default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        help="Candidate shouldSplit thresholds to search on validation.",
    )
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLD_GRID),
        help="Candidate probability thresholds to search on validation.",
    )
    parser.add_argument(
        "--class-threshold-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLD_GRID),
        help="Candidate per-class probability thresholds to search on validation.",
    )
    parser.add_argument(
        "--max-drafts-grid",
        type=int,
        nargs="+",
        default=[0, 2, 3],
        help="Grid for max generated drafts (0 = no cap).",
    )
    parser.add_argument(
        "--score-margin-grid",
        type=float,
        nargs="+",
        default=[1.0, 0.20, 0.12, 0.08],
        help="Grid for absolute margin filtering from the top score.",
    )
    parser.add_argument(
        "--relative-ratio-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.7],
        help="Grid for relative top-score ratio filtering.",
    )
    parser.add_argument(
        "--score-blend-alpha-grid",
        type=float,
        nargs="+",
        default=[1.0, 0.9, 0.8],
        help="Grid for blending transformer and TF-IDF scores.",
    )
    parser.add_argument(
        "--optimize-for",
        choices=["f1", "precision", "recall"],
        default="f1",
        help="Metric used for threshold/control search on validation.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="Minimum recall constraint during validation searches.",
    )
    parser.add_argument("--use-jsonl", action="store_true", help="Load dataset from JSONL instead of CSV.")
    return parser.parse_args()


def load_items(data_dir: Path, use_jsonl: bool) -> List[object]:
    dataset_path = data_dir / ("rnc_dataset.jsonl" if use_jsonl else "rnc_dataset.csv")
    if use_jsonl:
        return load_labeled_items_jsonl(str(dataset_path))
    return load_labeled_items_csv(str(dataset_path))


def resolve_data_dir(data_dir: Path) -> Path:
    if data_dir.exists():
        return data_dir

    fallback = Path(__file__).resolve().parent.parent / "data"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Data directory not found: {data_dir}")


def main() -> None:
    _require_torch()
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    micro_path = data_dir / "rnc_mic_key_phrases.csv"
    dataset_label = "JSONL" if args.use_jsonl else "CSV"
    print(f"Loading microcategories from {micro_path}...")
    microcategories = load_microcategories_from_csv(str(micro_path))
    print(f"Loading dataset ({dataset_label}) from {data_dir}...")
    items = load_items(data_dir, args.use_jsonl)
    buckets = split_dataset(items)

    train_items = buckets.get("train", [])
    val_items = buckets.get("val", [])
    test_items = buckets.get("test", [])

    print(f"Loaded: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    pipeline_settings = PipelineSettings(
        transformer_name=args.transformer_name,
        tfidf_threshold=args.tfidf_threshold,
        tfidf_top_k=args.tfidf_top_k,
        prob_threshold=args.prob_threshold,
        split_threshold=args.split_threshold,
        max_drafts=args.max_drafts,
        score_margin=args.score_margin,
        relative_ratio=args.relative_ratio,
        score_blend_alpha=args.score_blend_alpha,
        device=args.device,
    )
    training_settings = TrainingSettings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        threshold_grid=tuple(args.threshold_grid),
    )

    pipeline = DraftSplitPipeline(
        microcategories=microcategories,
        settings=pipeline_settings,
    )

    optimizer = AdamW(pipeline.model.parameters(), lr=training_settings.lr, weight_decay=training_settings.weight_decay)

    if train_items and val_items:
        print("Training with early stopping...")
        fit_report = pipeline.fit_with_early_stopping(
            train_items=train_items,
            val_items=val_items,
            optimizer=optimizer,
            training_settings=training_settings,
        )
        print("Training report:")
        print(json.dumps(fit_report, ensure_ascii=False, indent=2))
    elif train_items:
        print("Training without validation set...")
        losses = pipeline.train_on_labeled_items(
            items=train_items,
            optimizer=optimizer,
            batch_size=training_settings.batch_size,
            epochs=training_settings.epochs,
        )
        print(f"Loss history: {losses}")
    else:
        print("No training data found.")

    if val_items:
        print("Validation metrics with tuned threshold:")
        threshold_report = search_best_probability_threshold(
            pipeline,
            val_items,
            training_settings.threshold_grid,
            optimize_for=args.optimize_for,
            min_recall=args.min_recall,
        )
        print(json.dumps(threshold_report, ensure_ascii=False, indent=2))
        pipeline.prob_threshold = threshold_report["threshold"]

        split_threshold_report = evaluate_split_probability_threshold(pipeline, val_items, args.split_threshold_grid)
        print("Validation split-threshold tuning:")
        print(json.dumps(split_threshold_report, ensure_ascii=False, indent=2))
        pipeline.split_threshold = split_threshold_report["threshold"]

        print("Validation per-class threshold tuning:")
        class_threshold_report = search_best_class_probability_thresholds(
            pipeline,
            val_items,
            args.class_threshold_grid,
            optimize_for=args.optimize_for,
            min_recall=args.min_recall,
        )
        pipeline.set_class_prob_thresholds(class_threshold_report["class_thresholds"])
        print(json.dumps(class_threshold_report, ensure_ascii=False, indent=2))

        print("Validation reranking-controls tuning:")
        reranking_report = search_best_reranking_controls(
            pipeline,
            val_items,
            args.max_drafts_grid,
            args.score_margin_grid,
            args.relative_ratio_grid,
            args.score_blend_alpha_grid,
            optimize_for=args.optimize_for,
            min_recall=args.min_recall,
        )
        best_controls = reranking_report["best_controls"]
        pipeline.set_reranking_controls(
            max_drafts=best_controls["max_drafts"],
            score_margin=best_controls["score_margin"],
            relative_ratio=best_controls["relative_ratio"],
            score_blend_alpha=best_controls["score_blend_alpha"],
        )
        print(json.dumps(reranking_report, ensure_ascii=False, indent=2))

        print(f"Retrieval recall on val: {evaluate_retrieval_recall(pipeline, val_items):.4f}")
        print("Final val metrics:")
        print(json.dumps(evaluate_split_quality(pipeline, val_items), ensure_ascii=False, indent=2))

    if test_items:
        print("Test sample prediction:")
        sample = test_items[0]
        result = pipeline.predict(sample)
        print(json.dumps(to_response_json(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
