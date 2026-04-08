#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/bashScripts"

echo "Запуск проекта..."

if [ ! -f "nginx.conf" ]; then
    echo "Создаём nginx.conf..."
  
    echo "nginx.conf создан"
else
    echo "nginx.conf уже существует"
fi

# chmod +x "$SCRIPTS_DIR/check.sh"
# bash "$SCRIPTS_DIR/check.sh"

# chmod +x "$SCRIPTS_DIR/backend.sh"
# bash "$SCRIPTS_DIR/backend.sh"

chmod +x "$SCRIPTS_DIR/frontend.sh"
bash "$SCRIPTS_DIR/frontend.sh"