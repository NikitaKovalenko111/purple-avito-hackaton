# Model Training Presets For Organizers

Этот файл содержит готовые команды запуска обучения под разные цели метрик, чтобы можно было быстро выбрать режим без повторной разработки конфигурации.

Все команды ниже запускаются из папки model:

cd D:\Github\purple-avito-hackaton\model

Если на машине нет CUDA, замените в командах --device cuda на --device cpu.

## 1) Баланс по F1 (рекомендуемый базовый режим)

python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --epochs 10 --no-early-stopping --batch-size 2 --lr 3e-5 --weight-decay 0.01 --max-length 384 --long-text-mode chunks --long-text-window-tokens 256 --long-text-stride-tokens 192 --long-text-max-windows 4 --use-cross-encoder --cross-encoder-alpha 0.5 --cross-encoder-loss-weight 0.5 --noise-min-words 3 --noise-min-unique-ratio 0.4 --sentence-head-count 1 --sentence-tail-count 1 --sentence-top-k 5 --top-k-drafts-grid 11 --max-drafts-grid 0 --score-margin-grid 1.0 --relative-ratio-grid 0.0 --score-blend-alpha-grid 1.0 --threshold-grid 0.03 0.04 0.05 0.06 0.08 0.10 --split-threshold-grid 0.30 0.35 0.40 0.45 0.50 --class-threshold-grid 0.03 0.04 0.05 0.06 0.08 --optimize-for f1 --min-recall 0.0 --device cuda

Когда использовать:

- Нужен компромисс между precision и recall
- Нужен отчетный baseline для сравнения

## 2) Recall-first (ловим больше релевантных микрокатегорий)

python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --epochs 10 --no-early-stopping --batch-size 2 --lr 3e-5 --weight-decay 0.01 --max-length 384 --long-text-mode chunks --long-text-window-tokens 256 --long-text-stride-tokens 192 --long-text-max-windows 4 --use-cross-encoder --cross-encoder-alpha 0.4 --cross-encoder-loss-weight 0.6 --noise-min-words 2 --noise-min-unique-ratio 0.35 --sentence-head-count 1 --sentence-tail-count 1 --sentence-top-k 7 --top-k-drafts-grid 11 --max-drafts-grid 0 --score-margin-grid 1.0 --relative-ratio-grid 0.0 --score-blend-alpha-grid 0.9 1.0 --threshold-grid 0.02 0.03 0.04 0.05 0.06 --split-threshold-grid 0.25 0.30 0.35 0.40 0.45 --class-threshold-grid 0.02 0.03 0.04 0.05 0.06 --optimize-for recall --min-recall 0.0 --device cuda

Когда использовать:

- Важнее не пропустить класс, чем уменьшить ложные срабатывания
- Приоритет для этапа candidate recall

## 3) Precision-first (минимум лишних черновиков)

python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --epochs 10 --no-early-stopping --batch-size 2 --lr 3e-5 --weight-decay 0.01 --max-length 384 --long-text-mode chunks --long-text-window-tokens 256 --long-text-stride-tokens 192 --long-text-max-windows 4 --use-cross-encoder --cross-encoder-alpha 0.6 --cross-encoder-loss-weight 0.4 --noise-min-words 3 --noise-min-unique-ratio 0.5 --sentence-head-count 1 --sentence-tail-count 1 --sentence-top-k 3 --top-k-drafts-grid 5 7 --max-drafts-grid 3 5 --score-margin-grid 0.2 0.12 --relative-ratio-grid 0.4 0.6 --score-blend-alpha-grid 1.0 --threshold-grid 0.06 0.08 0.10 0.12 --split-threshold-grid 0.40 0.45 0.50 0.55 --class-threshold-grid 0.06 0.08 0.10 --optimize-for precision --min-recall 0.0 --device cuda

Когда использовать:

- Важнее качество каждого предложенного драфта
- Нужно меньше ложных классов в выдаче

## 4) Быстрый sanity-run (проверка пайплайна)

python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --epochs 3 --no-early-stopping --batch-size 2 --max-length 256 --long-text-mode head_tail --use-cross-encoder --top-k-drafts-grid 7 --max-drafts-grid 0 --score-margin-grid 1.0 --relative-ratio-grid 0.0 --score-blend-alpha-grid 1.0 --optimize-for f1 --min-recall 0.0 --device cuda

Когда использовать:

- Быстро проверить, что код, данные и сохранение чекпоинта работают
- Не использовать как финальный отчетный прогон

## 5) Curriculum learning (сначала простой датасет, потом тяжелый)

Этот режим полезен, если на тяжелом датасете recall сильно проседает.

Шаг 1 (предобучение на более простом датасете):

python run.py --data-dir .\data --dataset-file rnc_dataset.csv --epochs 5 --batch-size 4 --lr 3e-5 --weight-decay 0.01 --max-length 384 --long-text-mode chunks --long-text-window-tokens 256 --long-text-stride-tokens 192 --long-text-max-windows 64 --top-k-drafts-grid 11 --max-drafts-grid 0 --score-margin-grid 1.0 --relative-ratio-grid 0.0 --score-blend-alpha-grid 1.0 --threshold-grid 0.03 0.04 0.05 0.06 0.08 --split-threshold-grid 0.30 0.35 0.40 0.45 0.50 --class-threshold-grid 0.03 0.04 0.05 0.06 0.08 --optimize-for f1 --min-recall 0.0 --output-dir .\checkpoints\stage1_simple --device cuda

Шаг 2 (дообучение на тяжелом датасете из весов шага 1):

python run.py --data-dir .\data --dataset-file rnc_dataset_markup_balanced.csv --init-from-checkpoint .\checkpoints\stage1_simple\model_checkpoint.pt --epochs 7 --batch-size 4 --lr 1e-5 --weight-decay 0.01 --max-length 384 --long-text-mode chunks --long-text-window-tokens 256 --long-text-stride-tokens 192 --long-text-max-windows 64 --noise-min-words 2 --noise-min-unique-ratio 0.35 --sentence-head-count 1 --sentence-tail-count 1 --sentence-top-k 7 --top-k-drafts-grid 11 --max-drafts-grid 0 --score-margin-grid 1.0 --relative-ratio-grid 0.0 --score-blend-alpha-grid 1.0 --threshold-grid 0.02 0.03 0.04 0.05 0.06 --split-threshold-grid 0.25 0.30 0.35 0.40 0.45 --class-threshold-grid 0.02 0.03 0.04 0.05 0.06 --optimize-for recall --min-recall 0.0 --output-dir .\checkpoints\stage2_heavy --device cuda

Примечания:

- Используйте --checkpoint-path только для инференса без обучения.
- Для обучения из готовых весов используйте --init-from-checkpoint.
- Если простой датасет называется иначе, замените rnc_dataset.csv на нужный файл.

## Интерпретация результатов

- Если retrieval recall высокий, а micro recall низкий: проблема в порогах и reranking, а не в retriever.
- Если precision высокий, recall низкий: ослабить thresholds, увеличить sentence-top-k, не ограничивать max-drafts.
- Если recall высокий, precision низкий: ужесточить thresholds и включить более жесткий reranking.

## Что сдавать организаторам

- Основной режим: раздел 1 (баланс по F1)
- Дополнительно: один из режимов раздела 2 или 3, если нужна аргументация trade-off
- В отчете указывать не только F1, но и отдельно precision, recall, should_split_accuracy
