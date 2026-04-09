#!/usr/bin/env bash
set -euo pipefail

if command -v npm &>/dev/null; then
    log_info "npm установлен (версия: $(npm -v))"
else
    log_error "npm не найден. Установите Node.js или npm." >&2
    exit 1
fi

log_info "Запуск клиента..."

cd ./client
npm install 
npm run dev
cd ..

log_info "Клиент запущен"
