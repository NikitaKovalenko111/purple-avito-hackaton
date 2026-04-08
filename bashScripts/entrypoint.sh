#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="//app/models/checkpoints/model_checkpoint.pt"

echo "Проверка наличия модели..."

if [ -f "$MODEL_PATH" ]; then
  echo "Модель найдена на хосте — пропускаем обучение"
else
  echo "Модели нет — запускаем обучение..."
  python ../model.py          # или python manage.py train_model
fi