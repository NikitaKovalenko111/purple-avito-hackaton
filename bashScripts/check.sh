#!/usr/bin/env bash
set -euo pipefail

CHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$CHECK_DIR/../model"

if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    log_error "Python не найден. Установите python3." >&2
    exit 1
fi

log_info "Проверка модели..."



if [ ! -f "$MODEL_DIR/checkpoints/model_checkpoint.pt" ]; then
    log_info "Создание модели..."
    $PY "$MODEL_DIR/run.py"
    log_info "Модель обучена"
else
    log_info "Модель уже существует"
fi
