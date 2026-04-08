!#!/usr/bin/env bash
set -euo pipefail

CHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$CHECK_DIR/../model"

if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "Python не найден. Установите python3." >&2
    exit 1
fi

echo "Проверка модели..."



if [ ! -f "$MODEL_DIR\checkpoints\model_checkpoint.pt" ]; then
    echo "Создание модели..."
    $PY "$MODEL_DIR/run.py"
    echo "Модель обучена"
else
    echo "Модель уже существует"
fi
