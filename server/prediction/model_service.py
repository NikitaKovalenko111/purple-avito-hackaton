import sys
import os
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model.model import (
    Item,
    PipelineSettings,
    load_microcategories_from_csv,
    DraftSplitPipeline,
    MicroCategory,
    LabeledItem,
    PredictionResult,
    Draft,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

MICRO_CATEGORIES_CSV = ROOT_DIR / "model" / "data" / "rnc_mic_key_phrases.csv"
DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "model" / "checkpoints" / "model_checkpoint.pt"

_pipeline = None
_micro_by_id = {}
_pipeline_lock = threading.Lock()


def _load_checkpoint(checkpoint_path: Path) -> dict:
    if torch is None:
        raise RuntimeError("PyTorch is required to load model checkpoint")

    # Prefer safe tensor-only load to avoid legacy pickle symbol issues.
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions may not support weights_only argument.
        pass
    except Exception:
        pass

    # Backward-compatible fallback for legacy checkpoints that pickle dataclasses
    # under module name "model" instead of "model.model".
    legacy_model_module = sys.modules.get("model")
    if legacy_model_module is not None:
        setattr(legacy_model_module, "MicroCategory", MicroCategory)
        setattr(legacy_model_module, "LabeledItem", LabeledItem)
        setattr(legacy_model_module, "PredictionResult", PredictionResult)
        setattr(legacy_model_module, "Draft", Draft)
        setattr(legacy_model_module, "PipelineSettings", PipelineSettings)

    return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)


def get_pipeline():
    global _pipeline, _micro_by_id

    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        microcategories = load_microcategories_from_csv(str(MICRO_CATEGORIES_CSV))
        _micro_by_id = {mc.mc_id: mc for mc in microcategories}

        checkpoint_path = Path(os.getenv("MODEL_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
        checkpoint = {}
        checkpoint_config = {}
        if checkpoint_path.exists():
            checkpoint = _load_checkpoint(checkpoint_path)
            checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}
            print(f"[MODEL] Loaded checkpoint config from {checkpoint_path}")
        else:
            print(f"[MODEL] Checkpoint not found: {checkpoint_path}. Using randomly initialized weights.")

        settings = PipelineSettings(
            transformer_name=str(checkpoint.get("transformer_name", "DeepPavlov/rubert-base-cased")),
            use_llm_drafts=False,
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
        )

        _pipeline = DraftSplitPipeline(
            microcategories=microcategories,
            settings=settings,
        )

        if checkpoint and "model_state" in checkpoint:
            _pipeline.model.load_state_dict(checkpoint["model_state"], strict=False)
            class_prob_thresholds = checkpoint_config.get("class_prob_thresholds", {})
            if isinstance(class_prob_thresholds, dict) and class_prob_thresholds:
                _pipeline.set_class_prob_thresholds({int(k): float(v) for k, v in class_prob_thresholds.items()})
            _pipeline.model.eval()
            print("[MODEL] Model weights loaded for inference")

    return _pipeline


def warmup_pipeline() -> None:
    try:
        get_pipeline()
        print("[MODEL] Warm-up completed")
    except Exception as exc:
        print(f"[MODEL] Warm-up failed: {exc}")


def get_micro_title(mc_id: int) -> str:
    mc = _micro_by_id.get(mc_id)
    return mc.mc_title if mc else f"mc_{mc_id}"


def run_prediction(payload: dict) -> dict:
    pipeline = get_pipeline()

    item = Item(
        item_id=0,
        mc_id=payload["sourceMcId"],
        mc_title=payload["sourceMcTitle"],
        description=payload["description"],
    )

    result = pipeline.predict(item)

    # API detection output should not include the source category and should be confidence-filtered.
    ranked_detected = sorted(
        (
            (mc_id, float(score))
            for mc_id, score in result.probabilities.items()
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
        # Keep at least one best candidate for UX stability.
        detected_mc_ids = [ranked_detected[0][0]]

    if pipeline.top_k_drafts > 0:
        detected_mc_ids = detected_mc_ids[: pipeline.top_k_drafts]

    if pipeline.max_drafts > 0:
        detected_mc_ids = detected_mc_ids[: pipeline.max_drafts]

    should_split = result.should_split

    split_categories = [
        {
            "mcId": draft.mc_id,
            "mcTitle": draft.mc_title,
        }
        for draft in result.drafts
    ]

    return {
        "detectedMcIds": detected_mc_ids,
        "shouldSplit": should_split,
        "splitCategories": split_categories,
    }