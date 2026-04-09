#!/usr/bin/env bash
set -euo pipefail

CHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$CHECK_DIR/../server"

if command -v python3 &>/dev/null; then
    PY=python3
    echo "Python установлен(python3)." >&2
elif command -v python &>/dev/null; then
    PY=python
    echo "Python установлен(python)." >&2
else
    echo "Python не найден. Установите python3." >&2
    exit 1
fi

echo "Создание окружения..."

cd "$BACKEND_DIR"

$PY -m venv .venv
source .venv/bin/activate



log_info "Установка библиотек..."

pip install -r requirements.txt

log_info "Миграция базы данных..."

$PY manage.py migrate

log_info "Запуск сервера..."

$PY manage.py runserver

cd $OLDPWD

log_info "Сервер запущен"