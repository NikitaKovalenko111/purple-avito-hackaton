!#!/usr/bin/env bash
set -euo pipefail

if command -v npm &>/dev/null; then
    echo "npm установлен (версия: $(npm -v))"
else
    echo "npm не найден. Установите Node.js или npm." >&2
    exit 1
fi

echo "Запуск клиента..."

npm run dev

echo "Клиент запущен"
