#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_PATH="/app/model/checkpoints/model_checkpoint.pt"

echo "Миграция базы данных..."

python manage.py migrate --noinput
python manage.py collectstatic --noinput

echo "Проверка наличия модели..."

if [ -f "$MODEL_PATH" ]; then
  echo "Модель найдена на хосте — пропускаем обучение"
else
  echo "Модели нет — запускаем обучение..."
  mkdir -p /app/model/checkpoints
  python /app/model/run.py       # или python manage.py train_model
  echo "Обучение завершено! Веса сохранены в ${WEIGHTS_PATH}"
fi

echo "Запускаем Daphne + Django Channels..."
exec daphne -b 0.0.0.0 -p 8000 myproject.asgi:application