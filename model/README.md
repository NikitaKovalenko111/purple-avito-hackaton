# Модель: детекция микрокатегорий и split-draft

Этот документ описывает текущую модельную часть проекта: архитектуру, пайплайн, параметры, датасет, инференс, генерацию черновиков, масштабируемость и команды запуска.

## 1. Как работает модель: архитектура и пайплайн

### 1.1 Общая схема

Модель решает 2 подзадачи:

1. Детекция релевантных микрокатегорий (`detectedMcIds`)
2. Решение, нужно ли делить объявление (`shouldSplit`)

Пайплайн состоит из этапов:

1. Предобработка текста объявления
2. TF-IDF retriever (кандидаты категорий)
3. Transformer-encoder + softmax scoring по кандидатам
4. Отдельная split-голова (`shouldSplit`) с учетом:
    - скрытого представления текста
    - keyword/surface признаков
5. Пороговая фильтрация и reranking
6. Формирование `drafts` (template или LLM)

### 1.2 Архитектурные компоненты

- `TfidfMicroCategoryRetriever`:
    - кандидаты по key phrases микрокатегорий
    - word n-grams + optional char n-grams
- `TransformerSoftmaxSplitModel`:
    - текстовый энкодер (`DeepPavlov/rubert-base-cased` по умолчанию)
    - softmax логиты по candidate категориям
    - split-голова для `shouldSplit`
- `DraftSplitPipeline`:
    - объединяет retriever, нейросеть, пороги, reranking и генерацию черновиков

### 1.3 Что оптимизируется в обучении

Итоговый loss:

1. KLDiv loss для распределения по target категориям
2. BCEWithLogits loss для `shouldSplit`

Сигнал `shouldSplit` усилен через `split_pos_weight` (положительный класс весится сильнее).

## 2. Настраиваемые параметры

Ниже ключевые параметры CLI (`run.py`) и что они делают.

### 2.1 Данные и модель

- `--data-dir`: путь к папке данных
- `--dataset-file`: конкретный CSV/JSONL датасет
- `--transformer-name`: базовый transformer
- `--max-length`: max token length
- `--long-text-mode`: `head` | `head_tail` | `chunks`
- `--long-text-window-tokens`, `--long-text-stride-tokens`, `--long-text-max-windows`: контроль chunking

### 2.2 Candidate retrieval и классификация

- `--tfidf-threshold`, `--tfidf-top-k`: чувствительность retriever
- `--prob-threshold`: порог по category probability
- `--class-threshold-grid`: поиск per-class порогов на val
- `--score-blend-alpha`: blend neural score и TF-IDF score

### 2.3 shouldSplit

- `--split-threshold`: порог решения `shouldSplit`
- `--split-threshold-grid`: тюнинг split порога на val
- `--split-pos-weight`: усиление positive класса в BCE (ключевой параметр для recall)

### 2.4 Reranking и число draft

- `--top-k-drafts-grid`: top-k до фильтров
- `--max-drafts`, `--max-drafts-grid`: максимум draft
- `--score-margin`, `--score-margin-grid`: абсолютный margin от top score
- `--relative-ratio`, `--relative-ratio-grid`: относительный фильтр

### 2.5 Обучение

- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`
- `--patience`: early stopping patience
- `--no-early-stopping`: отключить early stopping
- `--optimize-for`: `f1` | `precision` | `recall` | `composite`
- `--min-recall`: ограничение при поиске порогов

### 2.6 Чекпоинты

- `--output-dir`: куда сохранять чекпоинт
- `--checkpoint-path`: загрузить чекпоинт и пропустить training
- `--init-from-checkpoint`: старт обучения из весов

## 3. Кратко про датасет

Основной рабочий датасет в текущих экспериментах:

- `data/rnc_dataset_markup_balanced.csv`

Ключевые поля:

- `description`: текст объявления
- `targetDetectedMcIds`: целевые найденные категории
- `targetSplitMcIds`: целевые категории для split
- `shouldSplit`: бинарная цель деления
- `split`: train/val/test

Важно для `markup_balanced`:

- в этой ветке при non-empty `targetSplitMcIds` он обычно совпадает с `targetDetectedMcIds`
- поэтому задача практически сводится к:
    - детекция категорий
    - бинарное решение `shouldSplit`

## 4. Время инференса

Точное время зависит от:

- устройства (`cpu`/`cuda`)
- длины текста
- режима long text (`head` быстрее, `chunks` медленнее)
- количества кандидатов

Практический ориентир:

- CPU: обычно десятки-сотни миллисекунд на объявление после прогрева
- GPU: обычно быстрее, но сильно зависит от батчинга и PCIe overhead

Рекомендуется измерять локально тем же конфигом, что в проде.

Мини-бенчмарк можно сделать через 100-1000 одинаковых вызовов `pipeline.predict(...)` после warm-up и посчитать p50/p95.

## 5. Генерация черновиков

Поддерживается 2 режима:

1. Template generation (по умолчанию)
2. LLM generation через OpenRouter (`--use-llm-drafts`)

Логика:

- для train/val/test метрик LLM обычно отключается для детерминизма
- для sample/production можно включить LLM
- при ошибке LLM используется template fallback

Ключевые параметры:

- `--use-llm-drafts`
- `--openrouter-model`
- `--openrouter-api-key`
- `--openrouter-base-url`
- `--openrouter-timeout-sec`

## 6. Масштабируемость

### 6.1 Увеличение числа категорий

Это относительно просто:

1. Добавить категории и key phrases в `rnc_mic_key_phrases.csv`
2. Переобучить модель

Ограничения:

- с ростом числа категорий ухудшается разделимость и растет стоимость reranking
- может потребоваться:
    - более строгий retriever
    - более агрессивные thresholds
    - larger model / больше данных

### 6.2 Увеличение длины текстов

Поддерживается через `long_text_mode=chunks`, но это увеличивает latency и память.

### 6.3 Прод-сервер

Для API рекомендуется:

- загружать пайплайн один раз при старте процесса
- держать модель прогретой
- не делать повторную инициализацию на каждый запрос

## 7. Команды для работы с моделью

Все команды ниже запускать из папки `model`.

### 7.1 Обучение (markup_balanced, early stopping)

```bash
python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --epochs 10 --batch-size 4 --lr 3e-5 --weight-decay 0.01 --max-length 512 --long-text-mode chunks --long-text-window-tokens 256 --long-text-stride-tokens 192 --long-text-max-windows 8 --split-pos-weight 1.5 --optimize-for composite --min-recall 0.0 --patience 3 --split-target-mode auto --device cuda --train-stratified-kfold 3 --cv-source-split train_val --cv-random-state 42 --balance-should-split-batches --batch-false-ratio 0.6 --batch-true-ratio 0.4 --output-dir .\checkpoints --split-equals-detected-when-should-split
```

Эта команда является финальной командой обучения.

### 7.2 Обучение без early stopping

```bash
python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --epochs 10 --batch-size 4 --lr 3e-5 --weight-decay 0.01 --max-length 512 --long-text-mode chunks --long-text-window-tokens 256 --long-text-stride-tokens 192 --long-text-max-windows 4 --split-pos-weight 3.0 --optimize-for recall --min-recall 0.0 --no-early-stopping --split-target-mode auto --device cuda
```

### 7.3 Инференс из чекпоинта (без обучения)

```bash
python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --checkpoint-path .\checkpoints\model_checkpoint.pt --split-target-mode auto --device cuda
```

### 7.4 Дообучение из существующих весов

```bash
python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --init-from-checkpoint .\checkpoints\model_checkpoint.pt --epochs 5 --batch-size 4 --lr 1e-5 --patience 3 --device cuda
```

### 7.5 CPU вариант

Заменить в любой команде:

```bash
--device cuda
```

на

```bash
--device cpu
```

### 7.6 Прогон rnc_test.csv через root-скрипт

Запускать из корня репозитория:

```bash
python run_rnc_test.py --output-csv .\test_results.csv
```

Полезные варианты:

```bash
python run_rnc_test.py --no-llm-drafts --output-csv .\test_results.csv
python run_rnc_test.py --checkpoint-path .\model\checkpoints\cv_best_fold_checkpoint.pt --output-csv .\test_results.csv
```

---

Если в логах видно, что val recall падает после ранних эпох, обычно лучший checkpoint находится на 1-3 эпохе, и имеет смысл сохранять/использовать именно его.
