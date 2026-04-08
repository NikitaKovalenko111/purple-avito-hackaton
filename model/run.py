from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List


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
    evaluate_detect_quality,
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
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Optional dataset file path (csv/jsonl). Relative paths are resolved against --data-dir.",
    )
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length for transformer encoder (recommended 384 for 8GB GPU).",
    )
    parser.add_argument(
        "--long-text-mode",
        type=str,
        choices=["head", "head_tail", "chunks"],
        default="chunks",
        help="Strategy for long ads: head truncation, head+tail, or chunk windows.",
    )
    parser.add_argument(
        "--long-text-window-tokens",
        type=int,
        default=256,
        help="Chunk window size in tokens for --long-text-mode chunks.",
    )
    parser.add_argument(
        "--long-text-stride-tokens",
        type=int,
        default=192,
        help="Chunk stride in tokens for --long-text-mode chunks.",
    )
    parser.add_argument(
        "--long-text-max-windows",
        type=int,
        default=4,
        help="Max windows per text for --long-text-mode chunks.",
    )
    parser.add_argument(
        "--use-cross-encoder",
        action="store_true",
        help="Enable cross-encoder reranking on shortlisted candidates.",
    )
    parser.add_argument(
        "--cross-encoder-alpha",
        type=float,
        default=0.5,
        help="Blend weight for cross-encoder reranking scores.",
    )
    parser.add_argument(
        "--cross-encoder-loss-weight",
        type=float,
        default=0.5,
        help="Loss weight for cross-encoder pairwise training.",
    )
    parser.add_argument(
        "--disable-text-normalization",
        action="store_true",
        help="Disable text normalization (emoji/symbol cleanup and whitespace normalization).",
    )
    parser.add_argument(
        "--noise-min-words",
        type=int,
        default=3,
        help="Drop short noisy sentences with fewer than N words.",
    )
    parser.add_argument(
        "--noise-min-unique-ratio",
        type=float,
        default=0.35,
        help="Drop noisy sentences with low unique-word ratio.",
    )
    parser.add_argument(
        "--disable-collapse-repeated-words",
        action="store_true",
        help="Disable collapsing repeated neighboring words like 'доставка доставка доставка'.",
    )
    parser.add_argument(
        "--sentence-head-count",
        type=int,
        default=1,
        help="Always keep first N informative sentences.",
    )
    parser.add_argument(
        "--sentence-tail-count",
        type=int,
        default=1,
        help="Always keep last N informative sentences.",
    )
    parser.add_argument(
        "--sentence-top-k",
        type=int,
        default=7,
        help="Keep top-K additional informative sentences by heuristic score.",
    )
    parser.add_argument(
        "--split-threshold-grid",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50],
        help="Candidate shouldSplit thresholds to search on validation.",
    )
    parser.add_argument(
        "--split-target-mode",
        type=str,
        choices=["auto", "split", "detected"],
        default="auto",
        help="Which labels to use as the draft target: raw split labels, detected labels, or auto-detect from data.",
    )
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=[0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
        help="Candidate probability thresholds to search on validation.",
    )
    parser.add_argument(
        "--class-threshold-grid",
        type=float,
        nargs="+",
        default=[0.03, 0.04, 0.05, 0.06, 0.08],
        help="Candidate per-class probability thresholds to search on validation.",
    )
    parser.add_argument(
        "--max-drafts-grid",
        type=int,
        nargs="+",
        default=[0, 3, 5, 7],
        help="Grid for max generated drafts (0 = no cap).",
    )
    parser.add_argument(
        "--top-k-drafts-grid",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="Grid for top-k candidate selection before reranking filters.",
    )
    parser.add_argument(
        "--score-margin-grid",
        type=float,
        nargs="+",
        default=[0.30, 0.20, 0.12],
        help="Grid for absolute margin filtering from the top score.",
    )
    parser.add_argument(
        "--relative-ratio-grid",
        type=float,
        nargs="+",
        default=[0.4, 0.6],
        help="Grid for relative top-score ratio filtering.",
    )
    parser.add_argument(
        "--score-blend-alpha-grid",
        type=float,
        nargs="+",
        default=[1.0, 0.9],
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
        default=0.45,
        help="Minimum recall constraint during validation searches.",
    )
    parser.add_argument("--use-jsonl", action="store_true", help="Load dataset from JSONL instead of CSV.")
    parser.add_argument(
        "--use-llm-drafts",
        action="store_true",
        help="Generate draft texts via OpenRouter LLM (fallback to template on errors).",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="qwen/qwen3.6-plus:free",
        help="OpenRouter model name for draft generation.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="OpenRouter API key. If omitted, OPENROUTER_API_KEY env var is used.",
    )
    parser.add_argument(
        "--openrouter-base-url",
        type=str,
        default="https://openrouter.ai/api/v1/chat/completions",
        help="OpenRouter Chat Completions endpoint.",
    )
    parser.add_argument(
        "--openrouter-timeout-sec",
        type=float,
        default=30.0,
        help="Timeout for OpenRouter requests in seconds.",
    )
    parser.add_argument(
        "--openrouter-site-url",
        type=str,
        default=None,
        help="Optional site URL sent as HTTP-Referer to OpenRouter.",
    )
    parser.add_argument(
        "--openrouter-app-name",
        type=str,
        default="purple-avito-hackaton",
        help="Optional app title sent as X-Title header to OpenRouter.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to saved model checkpoint (.pt). If provided, training is skipped.",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint (.pt) to initialize model weights before training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Directory to save model checkpoints and results.",
    )
    parser.add_argument(
        "--split-pos-weight",
        type=float,
        default=1.8,
        help="Positive class weight for shouldSplit BCE loss (>1 increases split recall).",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Train for all epochs without validation-based early stopping.",
    )
    return parser.parse_args()


def load_items(data_dir: Path, use_jsonl: bool, dataset_file: Path | None = None) -> List[object]:
    if dataset_file is not None:
        dataset_path = dataset_file if dataset_file.is_absolute() else data_dir / dataset_file
        suffix = dataset_path.suffix.lower()
        if suffix == ".jsonl":
            return load_labeled_items_jsonl(str(dataset_path))
        if suffix == ".csv":
            return load_labeled_items_csv(str(dataset_path))
        # Fallback for files without extension: use --use-jsonl toggle.
        if use_jsonl:
            return load_labeled_items_jsonl(str(dataset_path))
        return load_labeled_items_csv(str(dataset_path))

    dataset_path = data_dir / ("rnc_dataset.jsonl" if use_jsonl else "rnc_dataset.csv")
    if use_jsonl:
        return load_labeled_items_jsonl(str(dataset_path))
    return load_labeled_items_csv(str(dataset_path))


def infer_split_target_mode(items: List[object]) -> str:
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


def resolve_data_dir(data_dir: Path) -> Path:
    if data_dir.exists():
        return data_dir

    fallback = Path(__file__).resolve().parent / "data"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Data directory not found: {data_dir}")


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # PyTorch 2.6 changed default torch.load(..., weights_only=True).
    # Our checkpoints include metadata (e.g. custom classes), so we fall back
    # to weights_only=False for trusted local files.
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    return checkpoint


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue

            # Do not override already exported environment variables.
            os.environ.setdefault(key, value)


def main() -> None:
    _require_torch()

    # Support local secrets/config via .env without external dependencies.
    project_root = Path(__file__).resolve().parent.parent
    _load_env_file(project_root / ".env")

    args = parse_args()
    if args.checkpoint_path is not None and args.init_from_checkpoint is not None:
        raise ValueError("Use either --checkpoint-path or --init-from-checkpoint, not both.")

    checkpoint_mode = args.checkpoint_path is not None
    checkpoint: Dict[str, Any] = {}
    checkpoint_config: Dict[str, Any] = {}
    init_checkpoint: Dict[str, Any] = {}

    if checkpoint_mode:
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        checkpoint = _load_checkpoint(args.checkpoint_path)
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}

    if args.init_from_checkpoint is not None:
        print(f"Loading initialization checkpoint from {args.init_from_checkpoint}...")
        init_checkpoint = _load_checkpoint(args.init_from_checkpoint)

    data_dir = resolve_data_dir(args.data_dir)

    micro_path = data_dir / "rnc_mic_key_phrases.csv"
    dataset_label = "JSONL" if args.use_jsonl else "CSV"
    print(f"Loading microcategories from {micro_path}...")
    microcategories = load_microcategories_from_csv(str(micro_path))
    print(f"Loading dataset ({dataset_label}) from {data_dir}...")
    items = load_items(data_dir, args.use_jsonl, args.dataset_file)
    buckets = split_dataset(items)
    inferred_split_target_mode = infer_split_target_mode(items)
    split_target_mode = (
        inferred_split_target_mode if args.split_target_mode == "auto" else args.split_target_mode
    )
    print(f"Split target mode: {split_target_mode} (inferred={inferred_split_target_mode})")

    train_items = buckets.get("train", [])
    val_items = buckets.get("val", [])
    test_items = buckets.get("test", [])

    print(f"Loaded: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    transformer_name = str(checkpoint.get("transformer_name", args.transformer_name))
    use_llm_drafts = bool(args.use_llm_drafts or checkpoint_config.get("use_llm_drafts", False))
    openrouter_api_key = args.openrouter_api_key or checkpoint_config.get("openrouter_api_key")
    pipeline_settings = PipelineSettings(
        transformer_name=transformer_name,
        tfidf_threshold=float(checkpoint_config.get("tfidf_threshold", args.tfidf_threshold)),
        tfidf_top_k=int(checkpoint_config.get("tfidf_top_k", args.tfidf_top_k)),
        prob_threshold=float(checkpoint_config.get("prob_threshold", args.prob_threshold)),
        split_threshold=float(checkpoint_config.get("split_threshold", args.split_threshold)),
        max_drafts=int(checkpoint_config.get("max_drafts", args.max_drafts)),
        score_margin=float(checkpoint_config.get("score_margin", args.score_margin)),
        relative_ratio=float(checkpoint_config.get("relative_ratio", args.relative_ratio)),
        score_blend_alpha=float(checkpoint_config.get("score_blend_alpha", args.score_blend_alpha)),
        max_length=int(checkpoint_config.get("max_length", args.max_length)),
        long_text_mode=str(checkpoint_config.get("long_text_mode", args.long_text_mode)),
        long_text_window_tokens=int(
            checkpoint_config.get("long_text_window_tokens", args.long_text_window_tokens)
        ),
        long_text_stride_tokens=int(
            checkpoint_config.get("long_text_stride_tokens", args.long_text_stride_tokens)
        ),
        long_text_max_windows=int(
            checkpoint_config.get("long_text_max_windows", args.long_text_max_windows)
        ),
        top_k_drafts=int(checkpoint_config.get("top_k_drafts", 5)),
        use_cross_encoder=bool(checkpoint_config.get("use_cross_encoder", args.use_cross_encoder)),
        cross_encoder_alpha=float(checkpoint_config.get("cross_encoder_alpha", args.cross_encoder_alpha)),
        cross_encoder_loss_weight=float(
            checkpoint_config.get("cross_encoder_loss_weight", args.cross_encoder_loss_weight)
        ),
        device=args.device,
        use_llm_drafts=use_llm_drafts,
        openrouter_model=str(args.openrouter_model),
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=str(args.openrouter_base_url),
        openrouter_timeout_sec=float(args.openrouter_timeout_sec),
        openrouter_site_url=args.openrouter_site_url,
        openrouter_app_name=str(args.openrouter_app_name),
        normalize_text=bool(checkpoint_config.get("normalize_text", not args.disable_text_normalization)),
        noise_min_words=int(checkpoint_config.get("noise_min_words", args.noise_min_words)),
        noise_min_unique_ratio=float(
            checkpoint_config.get("noise_min_unique_ratio", args.noise_min_unique_ratio)
        ),
        collapse_repeated_words=bool(
            checkpoint_config.get("collapse_repeated_words", not args.disable_collapse_repeated_words)
        ),
        sentence_head_count=int(checkpoint_config.get("sentence_head_count", args.sentence_head_count)),
        sentence_tail_count=int(checkpoint_config.get("sentence_tail_count", args.sentence_tail_count)),
        sentence_top_k=int(checkpoint_config.get("sentence_top_k", args.sentence_top_k)),
        split_target_mode=str(checkpoint_config.get("split_target_mode", split_target_mode)),
    )
    training_settings = TrainingSettings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        split_pos_weight=float(args.split_pos_weight),
        threshold_grid=tuple(args.threshold_grid),
        split_threshold_grid=tuple(args.split_threshold_grid),
        optimize_for=args.optimize_for,
        min_recall=float(args.min_recall),
        cross_encoder_loss_weight=float(args.cross_encoder_loss_weight),
    )

    pipeline = DraftSplitPipeline(
        microcategories=microcategories,
        settings=pipeline_settings,
    )

    if args.init_from_checkpoint is not None:
        pipeline.model.load_state_dict(init_checkpoint["model_state"], strict=False)
        print("Model weights initialized from checkpoint. Training will continue from this state.")

    llm_requested = bool(pipeline.use_llm_drafts)
    # Metrics should be deterministic and fast: avoid LLM calls on full val/test sets.
    pipeline.use_llm_drafts = False
    print(f"LLM drafts enabled: {llm_requested}")
    if llm_requested:
        print(f"OpenRouter model: {pipeline.openrouter_model}")
        print(f"OpenRouter API key present: {bool(pipeline.openrouter_api_key)}")

    if not checkpoint_mode and llm_requested:
        # Keep training/tuning deterministic and fast: use template drafts in this phase.
        print("LLM drafts are disabled for training/validation/test metrics and enabled only for sample output.")

    if checkpoint_mode:
        pipeline.model.load_state_dict(checkpoint["model_state"], strict=False)
        class_prob_thresholds = checkpoint_config.get("class_prob_thresholds", {})
        if isinstance(class_prob_thresholds, dict) and class_prob_thresholds:
            pipeline.set_class_prob_thresholds({int(k): float(v) for k, v in class_prob_thresholds.items()})
        pipeline.model.eval()
        print("Checkpoint loaded. Running in inference/evaluation mode (training skipped).")

    optimizer = AdamW(pipeline.model.parameters(), lr=training_settings.lr, weight_decay=training_settings.weight_decay)

    if checkpoint_mode:
        print("Skipping training because --checkpoint-path is provided.")
    elif train_items and val_items:
        if args.no_early_stopping:
            print("Training without early stopping...")
            losses: List[float] = []
            for epoch in range(1, training_settings.epochs + 1):
                epoch_loss = pipeline.train_on_labeled_items(
                    items=train_items,
                    optimizer=optimizer,
                    batch_size=training_settings.batch_size,
                    epochs=1,
                    split_loss_weight=training_settings.split_loss_weight,
                    split_pos_weight=training_settings.split_pos_weight,
                    cross_encoder_loss_weight=training_settings.cross_encoder_loss_weight,
                    verbose=False,
                )[-1]
                losses.append(epoch_loss)
                val_metrics = evaluate_split_quality(pipeline, val_items)
                print(
                    f"Epoch {epoch}/{training_settings.epochs} | "
                    f"train_loss={epoch_loss:.4f} | "
                    f"val_f1={val_metrics['f1_micro']:.4f} | "
                    f"val_precision={val_metrics['precision_micro']:.4f} | "
                    f"val_recall={val_metrics['recall_micro']:.4f} | "
                    f"shouldSplit_acc={val_metrics['should_split_accuracy']:.4f} | "
                    f"threshold={pipeline.prob_threshold:.2f} | "
                    f"split_threshold={pipeline.split_threshold:.2f}"
                )
            print(f"Loss history: {losses}")
        else:
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
            split_loss_weight=training_settings.split_loss_weight,
            split_pos_weight=training_settings.split_pos_weight,
            # cross-encoder loss is handled inside train_step through pipeline settings
            # cross-encoder loss is handled inside train_step through pipeline settings
            verbose=True,
        )
        print(f"Loss history: {losses}")
    else:
        print("No training data found.")

    if val_items and checkpoint_mode:
        print("Validation metrics (loaded checkpoint):")
        print(json.dumps(evaluate_split_quality(pipeline, val_items), ensure_ascii=False, indent=2))
        print("Validation detect metrics:")
        print(json.dumps(evaluate_detect_quality(pipeline, val_items), ensure_ascii=False, indent=2))
        print(f"Retrieval recall on val: {evaluate_retrieval_recall(pipeline, val_items):.4f}")
    elif val_items:
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

        split_threshold_report = evaluate_split_probability_threshold(
            pipeline,
            val_items,
            args.split_threshold_grid,
            optimize_for=args.optimize_for,
            min_recall=args.min_recall,
        )
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
            args.top_k_drafts_grid,
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
        print("Final val detect metrics:")
        print(json.dumps(evaluate_detect_quality(pipeline, val_items), ensure_ascii=False, indent=2))

    if test_items:
        print(f"\nTest set evaluation ({len(test_items)} items):")
        test_metrics = evaluate_split_quality(pipeline, test_items)
        print("Test metrics:")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
        print("Test detect metrics:")
        print(json.dumps(evaluate_detect_quality(pipeline, test_items), ensure_ascii=False, indent=2))
        
        print(f"\nRetreval recall on test: {evaluate_retrieval_recall(pipeline, test_items):.4f}")
        
        print("\nTest sample prediction (first item):")
        sample = test_items[0]
        enable_llm_for_sample = llm_requested
        pipeline.llm_success_count = 0
        pipeline.llm_fallback_count = 0
        pipeline.last_llm_error = None
        if enable_llm_for_sample:
            pipeline.use_llm_drafts = True
        result = pipeline.predict(sample)
        if enable_llm_for_sample:
            pipeline.use_llm_drafts = False
        print(json.dumps(to_response_json(result), ensure_ascii=False, indent=2))
        print("LLM diagnostics:")
        print(json.dumps(pipeline.get_llm_diagnostics(), ensure_ascii=False, indent=2))

    if not checkpoint_mode:
        # Save model checkpoint
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_checkpoint_path = output_dir / "model_checkpoint.pt"

        checkpoint_config = {
            "transformer_name": transformer_name,
            "tfidf_threshold": args.tfidf_threshold,
            "tfidf_top_k": args.tfidf_top_k,
            "prob_threshold": pipeline.prob_threshold,
            "split_threshold": pipeline.split_threshold,
            "max_drafts": pipeline.max_drafts,
            "score_margin": pipeline.score_margin,
            "relative_ratio": pipeline.relative_ratio,
            "score_blend_alpha": pipeline.score_blend_alpha,
            "max_length": args.max_length,
            "long_text_mode": pipeline.model.long_text_mode,
            "long_text_window_tokens": pipeline.model.long_text_window_tokens,
            "long_text_stride_tokens": pipeline.model.long_text_stride_tokens,
            "long_text_max_windows": pipeline.model.long_text_max_windows,
            "top_k_drafts": pipeline.top_k_drafts,
            "use_cross_encoder": pipeline.use_cross_encoder,
            "cross_encoder_alpha": pipeline.cross_encoder_alpha,
            "cross_encoder_loss_weight": pipeline.cross_encoder_loss_weight,
            "class_prob_thresholds": pipeline.class_prob_thresholds,
            "use_llm_drafts": llm_requested,
            "openrouter_model": pipeline.openrouter_model,
            "openrouter_base_url": pipeline.openrouter_base_url,
            "openrouter_timeout_sec": pipeline.openrouter_timeout_sec,
            "openrouter_site_url": pipeline.openrouter_site_url,
            "openrouter_app_name": pipeline.openrouter_app_name,
            "normalize_text": pipeline.normalize_text,
            "noise_min_words": pipeline.noise_min_words,
            "noise_min_unique_ratio": pipeline.noise_min_unique_ratio,
            "collapse_repeated_words": pipeline.collapse_repeated_words,
            "sentence_head_count": pipeline.sentence_head_count,
            "sentence_tail_count": pipeline.sentence_tail_count,
            "sentence_top_k": pipeline.sentence_top_k,
            "split_target_mode": pipeline.split_target_mode,
        }

        print(f"\nSaving model checkpoint to {model_checkpoint_path}...")
        torch.save(
            {
                "model_state": pipeline.model.state_dict(),
                "config": checkpoint_config,
                "transformer_name": transformer_name,
                "microcategories": microcategories,
            },
            str(model_checkpoint_path),
        )
        print("Model saved")

        # Save pipeline settings as JSON for reference
        settings_path = output_dir / "pipeline_settings.json"
        settings_json = checkpoint_config
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings_json, f, ensure_ascii=False, indent=2)
        print(f"Settings saved to {settings_path}")


if __name__ == "__main__":
    main()
