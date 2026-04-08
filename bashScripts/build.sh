#!/bin/bash
# client/build.sh

set -e
echo "🔨 Сборка фронтенда..."

cd "$(dirname "$0")"

# Очистка старой сборки
if [ -d "dist" ]; then
  echo "🗑️  Очистка старой сборки..."
  rm -rf dist
fi

# Установка зависимостей (если нужно)
if [ ! -d "node_modules" ]; then
  echo "📦 Установка зависимостей..."
  npm ci --silent
fi

# Сборка
echo "⚡ Запуск vite build..."
npm run build

# Проверка результата
if [ -f "dist/index.html" ]; then
  echo "✅ Сборка успешна!"
  echo "📁 Размер dist/:"
  du -sh dist/
else
  echo "❌ Ошибка: dist/index.html не создан!"
  exit 1
fi