import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model.model import (
    Item,
    PipelineSettings,
    load_microcategories_from_csv,
    DraftSplitPipeline,
)

MICRO_CATEGORIES_CSV = ROOT_DIR / "model" / "data" / "rnc_mic_key_phrases.csv"

_pipeline = None
_micro_by_id = {}


def get_pipeline():
    global _pipeline, _micro_by_id

    if _pipeline is None:
        microcategories = load_microcategories_from_csv(str(MICRO_CATEGORIES_CSV))
        _micro_by_id = {mc.mc_id: mc for mc in microcategories}

        settings = PipelineSettings(
            use_llm_drafts=False,   # важно: drafts не генерим внутри /predict
            prob_threshold=0.08,
            split_threshold=0.5,
        )

        _pipeline = DraftSplitPipeline(
            microcategories=microcategories,
            settings=settings,
        )

    return _pipeline


def get_micro_title(mc_id: int) -> str:
    mc = _micro_by_id.get(mc_id)
    return mc.mc_title if mc else f"mc_{mc_id}"


def run_prediction(payload: dict) -> dict:
    pipeline = get_pipeline()

    item = Item(
        item_id=payload["itemId"],
        mc_id=payload["sourceMcId"],
        mc_title=payload["sourceMcTitle"],
        description=payload["description"],
    )

    result = pipeline.predict(item)

    detected_mc_ids = result.detected_mc_ids
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