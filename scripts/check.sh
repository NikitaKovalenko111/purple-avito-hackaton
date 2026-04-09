#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

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
