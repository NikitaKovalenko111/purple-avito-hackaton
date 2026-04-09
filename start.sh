#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/bashScripts"

cleanup() {
    log_warn "Получен сигнал завершения. Останавливаем сервисы..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    log_info "Все процессы остановлены."
    exit 0
}
trap cleanup SIGINT SIGTERM

log_info "Запуск проекта..."

# chmod +x "$SCRIPTS_DIR/check.sh"
# bash "$SCRIPTS_DIR/check.sh" &

chmod +x "$SCRIPTS_DIR/backend.sh"
bash "$SCRIPTS_DIR/backend.sh" &

chmod +x "$SCRIPTS_DIR/frontend.sh"
bash "$SCRIPTS_DIR/frontend.sh" &

wait