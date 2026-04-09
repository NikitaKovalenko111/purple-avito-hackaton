#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

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