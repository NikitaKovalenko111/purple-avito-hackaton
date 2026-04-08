#!/bin/bash
# start.sh в корне проекта

set -e
echo "=== 🚀 Purple Avito Hackathon — запуск ==="

# Создаём необходимые директории
mkdir -p db model/checkpoints bashScripts

# Собираем фронтенд, если есть package.json
if [ -f "client/package.json" ]; then
  echo "🔨 Сборка фронтенда..."
  if command -v npm &> /dev/null; then
    (cd client && npm ci --silent && npm run build)
  else
    echo "⚠️  npm не найден — пропускаем сборку фронтенда"
    echo "   Убедись, что client/dist/ уже содержит собранные файлы"
  fi
fi

# Запускаем Docker
echo "🐳 Запуск Docker Compose..."
docker compose up --build -d

echo ""
echo "✅ Сервер запущен!"
echo "🌐 HTTP → http://localhost"
echo "🔌 WebSocket → ws://localhost/ws/predict/<request_id>/"
echo "📁 Веса модели: ./model/checkpoints/model_checkpoints.pt"
echo ""
echo "📊 Логи:"
echo "   backend:  docker compose logs -f backend"
echo "   nginx:    docker compose logs -f nginx"
echo "   redis:    docker compose logs -f redis"
echo ""
echo "🔧 Остановить: docker compose down"