import os
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()


def _build_prompt(item_data: dict, category: dict) -> str:
    source_title = item_data.get("sourceMcTitle", "")
    description = item_data.get("description", "").strip()
    target_title = category["mcTitle"]

    return f"""
Ты помогаешь генерировать короткий черновик объявления для микрокатегории.

Исходная категория:
{source_title}

Целевая микрокатегория:
{target_title}

Описание исходного объявления:
{description}

Сгенерируй только текст черновика для микрокатегории "{target_title}".

Требования:
- пиши по-русски
- 2-4 предложения
- без списков
- без кавычек вокруг ответа
- без пояснений вроде "вот черновик"
- текст должен быть похож на черновик объявления услуги
- упоминай только то, что действительно есть в исходном описании
""".strip()


def generate_draft_with_llm(item_data: dict, category: dict) -> str:
    """
    Возвращает только текст черновика через OpenRouter.
    Если ключ или модель не заданы, возвращает fallback-текст.
    """

    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        return (
            f"Черновик для категории '{category['mcTitle']}'. "
            f"Описание: {item_data.get('description', '')[:160]}"
        )

    prompt = _build_prompt(item_data, category)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-OpenRouter-Title": "purple-avito-hackaton",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Ты пишешь краткие и аккуратные черновики объявлений на русском языке.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.4,
        "max_tokens": 180,
        "stream": False,
    }

    response = requests.post(
        LLM_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(f"OpenRouter {response.status_code}: {response.text}")

    data = response.json()

    text = data["choices"][0]["message"]["content"].strip()
    return text