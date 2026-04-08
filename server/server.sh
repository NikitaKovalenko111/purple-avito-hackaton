#!/usr/bin/env bash
set -euo pipefail

echo "Запуск сервера..."

python manage.py migrate

python manage.py collectstatic --noinput
