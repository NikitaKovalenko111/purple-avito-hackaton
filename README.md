# Purple Avito Hackathon

Главный README проекта. Здесь собрана навигация по фронтенду, серверу и модели, а подробные технические заметки по каждому блоку лежат в отдельных README внутри папок.

# ВАЖНО

Для генерации черновиков используются бесплатные LLM с OpenRouter, в связи с этим могут быть проблемы с генерацией (превышение Rate Limit и т.д), в таком случае генерируется шаблонный вариант. Следует иметь это в виду. Токен OpenRouter и название модели можно менять в .env (в корне и в server).

## Навигация

- [Обзор проекта](#обзор-проекта)
- [Структура](#структура)
- [Быстрый старт](#быстрый-старт)
- [Фронтенд](#фронтенд)
- [Сервер](#сервер)
- [Модель](#модель)
- [Данные и артефакты](#данные-и-артефакты)
- [Полезные команды](#полезные-команды)
- [README блоков](#readme-блоков)
- [Куда смотреть дальше](#куда-смотреть-дальше)

## Обзор проекта

Проект решает задачу разделения объявления на микрокатегории и генерации черновиков для найденных категорий.

## Структура

| Блок     | Назначение                                                      | README                               |
| -------- | --------------------------------------------------------------- | ------------------------------------ |
| `client` | UI для ввода объявления, просмотра результатов и истории сессий | [client/README.md](client/README.md) |
| `server` | API и WebSocket-слой для предикта и генерации черновиков        | [server/README.md](server/README.md) |
| `model`  | Обучение, оценка, инференс и чекпоинты модели                   | [model/README.md](model/README.md)   |

## Архитектура

Поток данных выглядит так:

1. Пользователь вводит описание объявления во фронтенде.
2. Клиент отправляет запрос на сервер.
3. Сервер вызывает модель и получает:
    - `detectedMcIds`
    - `shouldSplit`
    - `splitCategories`
4. Если используется потоковый режим, черновики приходят по WebSocket.
5. Если используется синхронный режим, черновики возвращаются сразу в HTTP-ответе.

Модельный пайплайн:

1. Предобработка текста.
2. TF-IDF retriever для отбора кандидатов.
3. Transformer encoder + softmax scoring.
4. Отдельная split-head для `shouldSplit`.
5. Thresholding и reranking.
6. Формирование черновиков.

## Быстрый старт

### Сервер

```bash
cd server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Фронтенд

```bash
cd client
npm install
npm run dev
```

### Модель

```bash
cd model
python run.py --dataset-file rnc_dataset_markup_balanced.csv --split-pos-weight 3 --optimize-for f1
```

## Фронтенд

UI находится в `client`.

Что умеет:

- вводить описание объявления;
- выбирать исходную категорию;
- получать предсказание и черновики;
- сохранять сессии в `localStorage`;
- открывать недавние сессии из истории.

Подробности и команды запуска: [client/README.md](client/README.md)

## Сервер

Сервер находится в `server`.

Что умеет:

- `GET /health/` — healthcheck;
- `POST /predict/` — асинхронный прогноз с WebSocket-черновиками;
- `POST /predict/sync/` — синхронный прогноз сразу со всеми черновиками.

Основные компоненты:

- `server/prediction/model_service.py` — загрузка модели и вызов инференса;
- `server/prediction/views.py` — HTTP endpoints;
- `server/prediction/consumers.py` — WebSocket consumer;
- `server/prediction/tasks.py` — генерация черновиков;
- `server/config/settings.py` — настройки Django и CORS.

Подробный README сервера: [server/README.md](server/README.md)

## Модель

Папка `model` содержит:

- обучение;
- оценку на test;
- CSV-инференс;
- описание пайплайна и конфигурации;
- сохранение checkpoint'ов.

Ключевые скрипты:

- `model/run.py` — обучение, k-fold, tuning, сохранение checkpoint;
- `model/evaluate_test.py` — оценка checkpoint на split/test;
- `model/infer_csv.py` — batch-инференс по CSV.
- `run_rnc_test.py` — запуск инференса по `rnc_test.csv` с авто-подбором checkpoint и python-интерпретатора.

Подробный README модели: [model/README.md](model/README.md)

## Данные и артефакты

Основные датасеты лежат в `model/data`:

- `rnc_dataset.csv`
- `rnc_dataset.jsonl`
- `rnc_dataset_corrected.csv`
- `rnc_dataset_markup_balanced.csv`
- `rnc_mic_key_phrases.csv`

Основные артефакты обучения:

- `model/checkpoints/model_checkpoint.pt`
- `model/checkpoints/pipeline_settings.json`
- `model/checkpoints/cv_training_report.json`
- `model/checkpoints/cv_best_fold_checkpoint.pt`

## Полезные команды

### Тренировка с k-fold

```bash
cd model
python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --epochs 10 --batch-size 4 --lr 3e-5 --weight-decay 0.01 --max-length 512 --long-text-mode chunks --long-text-window-tokens 256 --long-text-stride-tokens 192 --long-text-max-windows 8 --split-pos-weight 1.5 --optimize-for composite --min-recall 0.0 --patience 3 --split-target-mode auto --device cuda --train-stratified-kfold 3 --cv-source-split train_val --cv-random-state 42 --balance-should-split-batches --batch-false-ratio 0.6 --batch-true-ratio 0.4 --output-dir .\checkpoints --split-equals-detected-when-should-split
```

````

### Оценка test

```bash
cd model
python evaluate_test.py --checkpoint-path ./checkpoints/model_checkpoint.pt --dataset-file data/rnc_dataset_markup_balanced.csv --split test --output-json checkpoints/test_metrics.json
````

### CSV-инференс

```bash
cd model
python infer_csv.py --input-csv ..\rnc_test.csv --output-csv ..\rnc_test_filled.csv --checkpoint-path .\checkpoints\model_checkpoint.pt
```

### Запуск прогона rnc_test.csv

```bash
python run_rnc_test.py --output-csv .\test_results.csv
```

### Синхронный серверный прогноз

```bash
POST /predict/sync/
```

## README блоков

Если нужен глубокий разбор конкретной части проекта, открывайте README соответствующего блока:

- [client/README.md](client/README.md) — краткий обзор фронтенда, структура и команды запуска.
- [server/README.md](server/README.md) — серверные endpoints, WebSocket-режим и стек.
- [model/README.md](model/README.md) — архитектура модели, параметры, датасет и команды обучения.
