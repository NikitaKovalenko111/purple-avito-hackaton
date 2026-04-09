# Purple Avito Hackathon

Главный README проекта. Здесь собрана навигация по фронтенду, серверу и модели, а подробные технические заметки по каждому блоку лежат в отдельных README внутри папок.

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
python run.py --dataset-file rnc_dataset_markup_balanced.csv --train-stratified-kfold 5 --cv-source-split train_val --optimize-for f1 --split-pos-weight 3 --balance-should-split-batches --batch-false-ratio 0.7 --batch-true-ratio 0.3 --output-dir checkpoints
```

### Быстрый k-fold режим

```bash
cd model
python run.py --dataset-file rnc_dataset_markup_balanced.csv --train-stratified-kfold 5 --cv-source-split train_val --optimize-for f1 --split-pos-weight 3 --balance-should-split-batches --batch-false-ratio 0.7 --batch-true-ratio 0.3 --output-dir checkpoints
```

### Оценка test

```bash
cd model
python evaluate_test.py --checkpoint-path ./checkpoints/model_checkpoint.pt --dataset-file data/rnc_dataset_markup_balanced.csv --split test --output-json checkpoints/test_metrics.json
```

### CSV-инференс

```bash
cd model
python infer_csv.py --input-csv ..\rnc_test.csv --output-csv ..\rnc_test_filled.csv --checkpoint-path .\checkpoints\model_checkpoint.pt
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
