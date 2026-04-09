from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _optional_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


torch = _optional_import("torch")
sklearn_model_selection = _optional_import("sklearn.model_selection")

from model import (
    DraftSplitPipeline,
    PipelineSettings,
    evaluate_detect_quality,
    evaluate_retrieval_recall,
    evaluate_split_quality,
    load_labeled_items_csv,
    load_labeled_items_jsonl,
    load_microcategories_from_csv,
    split_dataset,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")


def _require_stratified_kfold() -> None:
    if sklearn_model_selection is None or not hasattr(sklearn_model_selection, "StratifiedKFold"):
        raise ImportError("scikit-learn is required for StratifiedKFold. Install with: pip install scikit-learn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on dataset split and print metrics.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "model_checkpoint.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory with dataset and microcategories files",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Specific dataset file (CSV or JSONL). If omitted, defaults to rnc_dataset.csv/jsonl in data-dir",
    )
    parser.add_argument("--use-jsonl", action="store_true", help="Load dataset from JSONL instead of CSV")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--stratified-kfold",
        type=int,
        default=0,
        help="If >1, run StratifiedKFold by shouldSplit over selected split and report mean/std metrics",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for StratifiedKFold shuffle",
    )
    parser.add_argument(
        "--microcategories-csv",
        type=Path,
        default=None,
        help="Path to microcategories CSV (defaults to data-dir/rnc_mic_key_phrases.csv)",
    )
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save full evaluation report as JSON",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(raw_path: Path) -> Path:
    checkpoint_path = Path(raw_path)
    if checkpoint_path.exists() and checkpoint_path.suffix.lower() == ".json":
        candidates = [
            checkpoint_path.parent / "cv_best_fold_checkpoint.pt",
            checkpoint_path.parent / "model_checkpoint.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                print(f"Resolved JSON report to checkpoint: {candidate}")
                return candidate
        raise ValueError(
            f"Checkpoint path points to JSON report: {checkpoint_path}. "
            "Could not find cv_best_fold_checkpoint.pt or model_checkpoint.pt next to it."
        )
    return checkpoint_path


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        if "invalid load key, '{'" in str(exc):
            raise ValueError(
                f"File is not a PyTorch checkpoint: {checkpoint_path}. "
                "Looks like a JSON file; pass a .pt checkpoint path instead."
            ) from exc
        if "Weights only load failed" not in str(exc):
            raise
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    return checkpoint


def _load_items(data_dir: Path, use_jsonl: bool, dataset_file: Path | None = None) -> List[Any]:
    if dataset_file is not None:
        dataset_file = Path(dataset_file)
        if dataset_file.is_absolute():
            dataset_path = dataset_file
        elif dataset_file.exists():
            dataset_path = dataset_file
        else:
            dataset_path = data_dir / dataset_file
        suffix = dataset_path.suffix.lower()
        if suffix == ".jsonl":
            return load_labeled_items_jsonl(str(dataset_path))
        if suffix == ".csv":
            return load_labeled_items_csv(str(dataset_path))
        if use_jsonl:
            return load_labeled_items_jsonl(str(dataset_path))
        return load_labeled_items_csv(str(dataset_path))

    default_name = "rnc_dataset.jsonl" if use_jsonl else "rnc_dataset.csv"
    dataset_path = data_dir / default_name
    if use_jsonl:
        return load_labeled_items_jsonl(str(dataset_path))
    return load_labeled_items_csv(str(dataset_path))


def _infer_split_target_mode(items: List[Any]) -> str:
    has_positive_split = False
    for item in items:
        target_split = set(getattr(item, "target_split_mc_ids", []))
        if not target_split:
            continue
        has_positive_split = True
        target_detected = set(getattr(item, "target_detected_mc_ids", []))
        if target_split != target_detected:
            return "split"
    return "detected" if has_positive_split else "split"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _compute_should_split_confusion(pipeline: DraftSplitPipeline, items: List[Any]) -> Dict[str, float]:
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for item in items:
        pred = pipeline.predict(item)
        pred_label = bool(pred.should_split)
        true_label = bool(item.should_split)
        if pred_label and true_label:
            tp += 1
        elif pred_label and (not true_label):
            fp += 1
        elif (not pred_label) and true_label:
            fn += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = _safe_div(tp + tn, total)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "pred_positive_rate": _safe_div(tp + fp, total),
        "true_positive_rate": _safe_div(tp + fn, total),
        "count": total,
    }


def _print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n{title}")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def _build_pipeline(
    checkpoint: Dict[str, Any],
    checkpoint_config: Dict[str, Any],
    microcategories: List[Any],
    items: List[Any],
    device: str | None,
) -> tuple[DraftSplitPipeline, PipelineSettings]:
    split_target_mode = str(checkpoint_config.get("split_target_mode", _infer_split_target_mode(items)))

    settings = PipelineSettings(
        transformer_name=str(checkpoint.get("transformer_name", "DeepPavlov/rubert-base-cased")),
        prob_threshold=float(checkpoint_config.get("prob_threshold", 0.08)),
        split_threshold=float(checkpoint_config.get("split_threshold", 0.5)),
        tfidf_threshold=float(checkpoint_config.get("tfidf_threshold", 0.03)),
        tfidf_top_k=int(checkpoint_config.get("tfidf_top_k", 11)),
        max_length=int(checkpoint_config.get("max_length", 512)),
        long_text_mode=str(checkpoint_config.get("long_text_mode", "chunks")),
        long_text_window_tokens=int(checkpoint_config.get("long_text_window_tokens", 256)),
        long_text_stride_tokens=int(checkpoint_config.get("long_text_stride_tokens", 192)),
        long_text_max_windows=int(checkpoint_config.get("long_text_max_windows", 4)),
        top_k_drafts=int(checkpoint_config.get("top_k_drafts", 5)),
        use_cross_encoder=bool(checkpoint_config.get("use_cross_encoder", False)),
        cross_encoder_alpha=float(checkpoint_config.get("cross_encoder_alpha", 0.5)),
        score_margin=float(checkpoint_config.get("score_margin", 1.0)),
        relative_ratio=float(checkpoint_config.get("relative_ratio", 0.0)),
        score_blend_alpha=float(checkpoint_config.get("score_blend_alpha", 1.0)),
        max_drafts=int(checkpoint_config.get("max_drafts", 0)),
        split_target_mode=split_target_mode,
        split_equals_detected_when_should_split=bool(
            checkpoint_config.get("split_equals_detected_when_should_split", False)
        ),
        device=device,
        use_llm_drafts=False,
    )

    pipeline = DraftSplitPipeline(microcategories=microcategories, settings=settings)
    pipeline.model.load_state_dict(checkpoint["model_state"], strict=False)
    class_prob_thresholds = checkpoint_config.get("class_prob_thresholds", {})
    if isinstance(class_prob_thresholds, dict) and class_prob_thresholds:
        pipeline.set_class_prob_thresholds({int(k): float(v) for k, v in class_prob_thresholds.items()})
    pipeline.model.eval()
    return pipeline, settings


def _evaluate_metrics(pipeline: DraftSplitPipeline, eval_items: List[Any]) -> Dict[str, Any]:
    split_metrics = evaluate_split_quality(pipeline, eval_items)
    detect_metrics = evaluate_detect_quality(pipeline, eval_items)
    retrieval_recall = evaluate_retrieval_recall(pipeline, eval_items)
    should_split_confusion = _compute_should_split_confusion(pipeline, eval_items)
    return {
        "split": split_metrics,
        "detect": detect_metrics,
        "retrieval_recall": retrieval_recall,
        "should_split": should_split_confusion,
    }


def _collect_numeric_metrics(prefix: str, value: Any, out: Dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_numeric_metrics(next_prefix, nested, out)
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _aggregate_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    series: Dict[str, List[float]] = {}
    for metric in fold_metrics:
        flat: Dict[str, float] = {}
        _collect_numeric_metrics("", metric, flat)
        for key, value in flat.items():
            series.setdefault(key, []).append(value)

    mean_metrics = {key: _mean(values) for key, values in series.items()}
    std_metrics = {key: _std(values) for key, values in series.items()}
    return {"mean": mean_metrics, "std": std_metrics}


def main() -> None:
    _require_torch()
    args = parse_args()

    started_at = time.time()
    resolved_checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    checkpoint = _load_checkpoint(resolved_checkpoint_path)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    microcategories_csv = args.microcategories_csv or (data_dir / "rnc_mic_key_phrases.csv")
    if not microcategories_csv.exists():
        raise FileNotFoundError(f"Microcategories CSV not found: {microcategories_csv}")

    microcategories = load_microcategories_from_csv(str(microcategories_csv))
    items = _load_items(data_dir, args.use_jsonl, args.dataset_file)
    buckets = split_dataset(items)

    eval_items = list(buckets.get(args.split, []))
    if not eval_items:
        raise ValueError(f"No items in split '{args.split}'. Available: train={len(buckets.get('train', []))}, val={len(buckets.get('val', []))}, test={len(buckets.get('test', []))}")

    eval_items = list(items if args.split == "all" else buckets.get(args.split, []))
    if not eval_items:
        raise ValueError(
            f"No items in split '{args.split}'. Available: "
            f"train={len(buckets.get('train', []))}, val={len(buckets.get('val', []))}, test={len(buckets.get('test', []))}"
        )

    if args.stratified_kfold > 1:
        _require_stratified_kfold()
        labels = [1 if bool(item.should_split) else 0 for item in eval_items]
        positives = sum(labels)
        negatives = len(labels) - positives
        if positives < args.stratified_kfold or negatives < args.stratified_kfold:
            raise ValueError(
                "Not enough samples per class for StratifiedKFold: "
                f"positives={positives}, negatives={negatives}, n_splits={args.stratified_kfold}"
            )

        skf = sklearn_model_selection.StratifiedKFold(
            n_splits=args.stratified_kfold,
            shuffle=True,
            random_state=args.random_state,
        )

        fold_reports: List[Dict[str, Any]] = []
        for fold_idx, (_, val_idx) in enumerate(skf.split(eval_items, labels), start=1):
            fold_eval_items = [eval_items[i] for i in val_idx]
            fold_pipeline, fold_settings = _build_pipeline(
                checkpoint=checkpoint,
                checkpoint_config=checkpoint_config,
                microcategories=microcategories,
                items=items,
                device=args.device,
            )
            fold_metric = _evaluate_metrics(fold_pipeline, fold_eval_items)
            fold_reports.append(
                {
                    "fold": fold_idx,
                    "size": len(fold_eval_items),
                    "settings": {
                        "split_target_mode": fold_settings.split_target_mode,
                        "split_threshold": fold_settings.split_threshold,
                        "prob_threshold": fold_settings.prob_threshold,
                    },
                    "metrics": fold_metric,
                }
            )

        aggregate = _aggregate_fold_metrics([f["metrics"] for f in fold_reports])
        elapsed_sec = time.time() - started_at
        settings = fold_settings
        report = {
            "checkpoint_path": str(resolved_checkpoint_path),
            "data_dir": str(data_dir),
            "dataset_file": str(args.dataset_file) if args.dataset_file is not None else None,
            "split": args.split,
            "stratified_kfold": args.stratified_kfold,
            "random_state": args.random_state,
            "counts": {
                "all": len(items),
                "train": len(buckets.get("train", [])),
                "val": len(buckets.get("val", [])),
                "test": len(buckets.get("test", [])),
                "evaluated": len(eval_items),
            },
            "settings": {
                "transformer_name": settings.transformer_name,
                "prob_threshold": settings.prob_threshold,
                "split_threshold": settings.split_threshold,
                "tfidf_threshold": settings.tfidf_threshold,
                "tfidf_top_k": settings.tfidf_top_k,
                "split_target_mode": settings.split_target_mode,
                "split_equals_detected_when_should_split": settings.split_equals_detected_when_should_split,
                "top_k_drafts": settings.top_k_drafts,
                "max_drafts": settings.max_drafts,
                "score_margin": settings.score_margin,
                "relative_ratio": settings.relative_ratio,
                "score_blend_alpha": settings.score_blend_alpha,
                "device": settings.device,
            },
            "folds": fold_reports,
            "aggregate": aggregate,
            "elapsed_sec": elapsed_sec,
        }

        print("=== StratifiedKFold Evaluation Report ===")
        print(f"Checkpoint: {resolved_checkpoint_path}")
        print(f"Dataset split: {args.split} ({len(eval_items)} items)")
        print(f"n_splits: {args.stratified_kfold}")
        print(f"Split target mode: {settings.split_target_mode}")
        print(f"Device: {settings.device}")
        print("\nAggregate mean metrics:")
        for key, value in sorted(aggregate["mean"].items()):
            std = aggregate["std"].get(key, 0.0)
            print(f"  {key}: {value:.4f} ± {std:.4f}")
        print(f"\nElapsed: {elapsed_sec:.2f} sec")
    else:
        pipeline, settings = _build_pipeline(
            checkpoint=checkpoint,
            checkpoint_config=checkpoint_config,
            microcategories=microcategories,
            items=items,
            device=args.device,
        )
        metrics = _evaluate_metrics(pipeline, eval_items)
        elapsed_sec = time.time() - started_at

        report = {
            "checkpoint_path": str(resolved_checkpoint_path),
            "data_dir": str(data_dir),
            "dataset_file": str(args.dataset_file) if args.dataset_file is not None else None,
            "split": args.split,
            "counts": {
                "all": len(items),
                "train": len(buckets.get("train", [])),
                "val": len(buckets.get("val", [])),
                "test": len(buckets.get("test", [])),
                "evaluated": len(eval_items),
            },
            "settings": {
                "transformer_name": settings.transformer_name,
                "prob_threshold": settings.prob_threshold,
                "split_threshold": settings.split_threshold,
                "tfidf_threshold": settings.tfidf_threshold,
                "tfidf_top_k": settings.tfidf_top_k,
                "split_target_mode": settings.split_target_mode,
                "split_equals_detected_when_should_split": settings.split_equals_detected_when_should_split,
                "top_k_drafts": settings.top_k_drafts,
                "max_drafts": settings.max_drafts,
                "score_margin": settings.score_margin,
                "relative_ratio": settings.relative_ratio,
                "score_blend_alpha": settings.score_blend_alpha,
                "device": settings.device,
            },
            "metrics": metrics,
            "elapsed_sec": elapsed_sec,
        }

        print("=== Evaluation Report ===")
        print(f"Checkpoint: {resolved_checkpoint_path}")
        print(f"Dataset split: {args.split} ({len(eval_items)} items)")
        print(f"Split target mode: {settings.split_target_mode}")
        print(f"Device: {settings.device}")

        _print_metrics("Split metrics", metrics["split"])
        _print_metrics("Detect metrics", metrics["detect"])
        print(f"\nRetrieval recall: {float(metrics['retrieval_recall']):.4f}")
        _print_metrics("ShouldSplit confusion", metrics["should_split"])
        print(f"\nElapsed: {elapsed_sec:.2f} sec")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved report to {args.output_json}")


if __name__ == "__main__":
    main()
