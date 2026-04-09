from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _optional_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


torch = _optional_import("torch")

from model import (
    DraftSplitPipeline,
    Item,
    PipelineSettings,
    load_microcategories_from_csv,
    to_response_json,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model inference on a CSV file.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV. Supported formats: (1) request/response, (2) flat sourceMcId/sourceMcTitle/description",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "model_checkpoint.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--microcategories-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "rnc_mic_key_phrases.csv",
        help="Path to microcategories CSV",
    )
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda). If omitted, checkpoint/default is used")
    parser.add_argument(
        "--use-llm-drafts",
        action="store_true",
        help="Generate drafts via OpenRouter LLM (requires OPENROUTER_API_KEY and OPENROUTER_MODEL).",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default=None,
        help="Override OpenRouter model for LLM draft generation.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="Override OpenRouter API key for LLM draft generation.",
    )
    parser.add_argument(
        "--openrouter-base-url",
        type=str,
        default=None,
        help="Override OpenRouter base URL for LLM draft generation.",
    )
    parser.add_argument(
        "--openrouter-timeout-sec",
        type=float,
        default=None,
        help="Override OpenRouter request timeout in seconds.",
    )
    parser.add_argument("--encoding", type=str, default="utf-8-sig", help="Input/output CSV encoding")
    parser.add_argument("--print-every", type=int, default=100, help="Log progress every N rows")
    return parser.parse_args()


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    return checkpoint


def _pick(row: Dict[str, Any], names: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if name in row and row[name] is not None and str(row[name]).strip() != "":
            return str(row[name])
    return default


def _to_int(value: Optional[str], default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _parse_request_payload(raw_request: str) -> Dict[str, Any]:
    try:
        obj = json.loads(raw_request)
    except Exception as exc:
        raise ValueError(f"Invalid JSON in request column: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("Request JSON must be an object")
    return obj


def _compute_detected_mc_ids(pipeline: DraftSplitPipeline, item: Item, prediction_payload: Dict[str, Any]) -> List[int]:
    probabilities = prediction_payload.get("probabilities") or {}

    ranked_detected = sorted(
        (
            (int(mc_id), float(score))
            for mc_id, score in probabilities.items()
            if int(mc_id) != int(item.mc_id)
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    detected_mc_ids = [
        mc_id
        for mc_id, score in ranked_detected
        if score >= pipeline.get_class_prob_threshold(mc_id)
    ]

    if not detected_mc_ids and ranked_detected:
        detected_mc_ids = [ranked_detected[0][0]]

    if pipeline.top_k_drafts > 0:
        detected_mc_ids = detected_mc_ids[: pipeline.top_k_drafts]

    if pipeline.max_drafts > 0:
        detected_mc_ids = detected_mc_ids[: pipeline.max_drafts]

    drafts = prediction_payload.get("drafts")
    should_split = bool(prediction_payload.get("shouldSplit"))
    if should_split and isinstance(drafts, list) and drafts:
        detected_mc_ids = list(
            dict.fromkeys(int(d["mcId"]) for d in drafts if isinstance(d, dict) and "mcId" in d)
        )

    return detected_mc_ids


def main() -> None:
    _require_torch()
    args = parse_args()

    microcategories = load_microcategories_from_csv(str(args.microcategories_csv))
    title_by_id = {mc.mc_id: mc.mc_title for mc in microcategories}

    checkpoint = _load_checkpoint(args.checkpoint_path)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}

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
        split_target_mode=str(checkpoint_config.get("split_target_mode", "split")),
        split_equals_detected_when_should_split=bool(
            checkpoint_config.get("split_equals_detected_when_should_split", False)
        ),
        device=args.device,
        use_llm_drafts=bool(args.use_llm_drafts),
        openrouter_model=str(
            args.openrouter_model
            if args.openrouter_model is not None
            else checkpoint_config.get("openrouter_model")
            or os.getenv("OPENROUTER_MODEL", "qwen/qwen3.6-plus:free")
        ),
        openrouter_api_key=args.openrouter_api_key,
        openrouter_base_url=str(
            args.openrouter_base_url
            if args.openrouter_base_url is not None
            else checkpoint_config.get("openrouter_base_url")
            or os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
        ),
        openrouter_timeout_sec=float(
            args.openrouter_timeout_sec
            if args.openrouter_timeout_sec is not None
            else checkpoint_config.get("openrouter_timeout_sec", 30.0)
        ),
    )

    pipeline = DraftSplitPipeline(
        microcategories=microcategories,
        settings=settings,
    )
    pipeline.model.load_state_dict(checkpoint["model_state"], strict=False)
    class_prob_thresholds = checkpoint_config.get("class_prob_thresholds", {})
    if isinstance(class_prob_thresholds, dict) and class_prob_thresholds:
        pipeline.set_class_prob_thresholds({int(k): float(v) for k, v in class_prob_thresholds.items()})
    pipeline.model.eval()

    if args.use_llm_drafts:
        print(f"LLM drafts enabled: {pipeline.use_llm_drafts}")
        print(f"OpenRouter model: {pipeline.openrouter_model}")
        print(f"OpenRouter API key present: {bool(pipeline.openrouter_api_key)}")

    with args.input_csv.open("r", encoding=args.encoding, newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    request_response_mode = "request" in fieldnames and "response" in fieldnames

    output_rows: List[Dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        if request_response_mode:
            request_payload = _parse_request_payload(str(row.get("request", "")))
            source_mc_id = _to_int(str(request_payload.get("mcId", request_payload.get("sourceMcId", 0))))
            source_mc_title = str(
                request_payload.get(
                    "mcTitle",
                    request_payload.get("sourceMcTitle", title_by_id.get(source_mc_id, "")),
                )
            )
            description = str(request_payload.get("description", ""))
            item_id = _to_int(str(request_payload.get("itemId", index)), default=index)
        else:
            source_mc_id = _to_int(_pick(row, ["sourceMcId", "source_mc_id", "mcId", "mc_id"]))
            source_mc_title = _pick(
                row,
                ["sourceMcTitle", "source_mc_title", "mcTitle", "mc_title"],
                title_by_id.get(source_mc_id, ""),
            )
            description = _pick(row, ["description", "text", "ad_text"], "") or ""
            item_id = _to_int(_pick(row, ["itemId", "item_id"]), default=index)

        item = Item(
            item_id=item_id,
            mc_id=source_mc_id,
            mc_title=source_mc_title,
            description=description,
        )

        prediction = pipeline.predict(item)
        payload = to_response_json(prediction)
        payload["detectedMcIds"] = _compute_detected_mc_ids(
            pipeline=pipeline,
            item=item,
            prediction_payload={
                "probabilities": prediction.probabilities,
                "shouldSplit": payload.get("shouldSplit", False),
                "drafts": payload.get("drafts", []),
            },
        )

        if request_response_mode:
            output_rows.append(
                {
                    "request": row.get("request", ""),
                    "response": json.dumps(payload, ensure_ascii=False),
                }
            )
        else:
            output_rows.append(
                {
                    "itemId": item_id,
                    "sourceMcId": source_mc_id,
                    "sourceMcTitle": source_mc_title,
                    "description": description,
                    "detectedMcIds": json.dumps(payload["detectedMcIds"], ensure_ascii=False),
                    "shouldSplit": payload["shouldSplit"],
                    "splitCategories": json.dumps(
                        [{"mcId": d["mcId"], "mcTitle": d["mcTitle"]} for d in payload["drafts"]],
                        ensure_ascii=False,
                    ),
                    "drafts": json.dumps(payload["drafts"], ensure_ascii=False),
                }
            )

        if args.print_every > 0 and (index % args.print_every == 0 or index == total):
            print(f"Processed {index}/{total}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding=args.encoding, newline="") as f_out:
        if request_response_mode:
            out_fieldnames = ["request", "response"]
        else:
            out_fieldnames = [
                "itemId",
                "sourceMcId",
                "sourceMcTitle",
                "description",
                "detectedMcIds",
                "shouldSplit",
                "splitCategories",
                "drafts",
            ]
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved {len(output_rows)} rows to {args.output_csv}")
    if args.use_llm_drafts:
        llm_diag = pipeline.get_llm_diagnostics()
        print(
            "LLM diagnostics: "
            f"success={llm_diag.get('success_count', 0)} | "
            f"fallback={llm_diag.get('fallback_count', 0)} | "
            f"last_error={llm_diag.get('last_error')}"
        )


if __name__ == "__main__":
    main()
