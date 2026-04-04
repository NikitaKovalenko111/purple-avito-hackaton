# Backend Guide: запуск модели на сервере и API

Этот документ для backend-разработчика: как поднять модель как HTTP-сервис, принимать запросы на предикт и отдавать ответы в формате хакатона.

## 1. Что уже есть в проекте

- Обучение и оффлайн-оценка: model/run.py
- Модель и пайплайн: model/model.py
- Формат ответа API: функция to_response_json(...) в model/model.py
- Чекпоинт после обучения: checkpoints/model_checkpoint.pt

## 2. Подготовка сервера

Пример для Linux (Python 3.10+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers scikit-learn fastapi uvicorn
```

Для Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch transformers scikit-learn fastapi uvicorn
```

## 3. Переменные окружения (.env)

В корне проекта создайте файл .env:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=qwen/qwen3-6-plus:free
```

Примечания:
- Если OPENROUTER_API_KEY не задан, генерация через LLM не упадет, пайплайн использует шаблонный fallback.
- Для прода рекомендуется хранить ключ в секретах окружения (Vault/K8s Secret), а не в файле.

## 4. Минимальный API-сервер (FastAPI)

Создайте файл model/api_server.py:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import DraftSplitPipeline, Item, PipelineSettings, load_microcategories_from_csv, to_response_json


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_PATH = ROOT / "checkpoints" / "model_checkpoint.pt"


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class PredictRequest(BaseModel):
    itemId: int = Field(default=0)
    sourceMcId: int
    sourceMcTitle: str = ""
    description: str


class HealthResponse(BaseModel):
    status: str


app = FastAPI(title="Avito Split Draft API", version="1.0.0")
pipeline: Optional[DraftSplitPipeline] = None


@app.on_event("startup")
def startup() -> None:
    global pipeline

    load_env(ROOT / ".env")

    microcategories = load_microcategories_from_csv(str(DATA_DIR / "rnc_mic_key_phrases.csv"))

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    config = checkpoint.get("config", {})

    settings = PipelineSettings(
        transformer_name=str(checkpoint.get("transformer_name", "DeepPavlov/rubert-base-cased")),
        tfidf_threshold=float(config.get("tfidf_threshold", 0.03)),
        tfidf_top_k=int(config.get("tfidf_top_k", 11)),
        prob_threshold=float(config.get("prob_threshold", 0.08)),
        split_threshold=float(config.get("split_threshold", 0.5)),
        max_drafts=int(config.get("max_drafts", 0)),
        score_margin=float(config.get("score_margin", 1.0)),
        relative_ratio=float(config.get("relative_ratio", 0.0)),
        score_blend_alpha=float(config.get("score_blend_alpha", 1.0)),
        use_llm_drafts=True,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3-6-plus:free"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    pipeline = DraftSplitPipeline(microcategories=microcategories, settings=settings)
    pipeline.model.load_state_dict(checkpoint["model_state"])
    class_thresholds = config.get("class_prob_thresholds", {})
    if isinstance(class_thresholds, dict) and class_thresholds:
        pipeline.set_class_prob_thresholds({int(k): float(v) for k, v in class_thresholds.items()})

    pipeline.model.eval()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict")
def predict(payload: PredictRequest):
    if pipeline is None:
        return {"error": "Pipeline is not initialized"}

    item = Item(
        item_id=payload.itemId,
        mc_id=payload.sourceMcId,
        mc_title=payload.sourceMcTitle,
        description=payload.description,
    )

    result = pipeline.predict(item)
    return to_response_json(result)
```

## 5. Запуск сервера

```bash
uvicorn model.api_server:app --host 0.0.0.0 --port 8000
```

Проверка health:

```bash
curl http://localhost:8000/health
```

## 6. Формат запроса/ответа

Запрос на предикт:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "itemId": 2,
    "sourceMcId": 101,
    "sourceMcTitle": "Ремонт квартир и домов под ключ",
    "description": "Косметический ремонт под ключ в доме..."
  }'
```

Ответ:

```json
{
  "detectedMcIds": [101, 110, 111, 108, 104, 102],
  "shouldSplit": true,
  "drafts": [
    {
      "mcId": 110,
      "mcTitle": "Гипсокартон",
      "text": "..."
    }
  ]
}
```

## 7. Что важно для production

- Загружайте чекпоинт один раз на старте приложения.
- Не выполняйте обучение в API-процессе.
- Держите OPENROUTER_API_KEY в секретах окружения.
- Если OpenRouter временно недоступен, пайплайн автоматически вернет шаблонный текст (fallback).
- Для масштабирования запускайте несколько worker-процессов и выносите логирование/метрики (Prometheus/Grafana).

## 8. Быстрый чек-лист перед релизом

- Есть файл checkpoints/model_checkpoint.pt
- Есть data/rnc_mic_key_phrases.csv
- OPENROUTER_API_KEY задан
- /health отвечает 200
- /predict возвращает JSON со схемой detectedMcIds/shouldSplit/drafts
