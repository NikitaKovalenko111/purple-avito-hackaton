# Frontend (React + Vite)

Краткое описание фронтенда проекта `purple-avito-hackaton`.

## Что здесь

- React + TypeScript + Vite
- Redux Toolkit для состояния
- React Router для страниц
- Axios для запросов к backend

## Быстрый старт

Из папки `client`:

```bash
npm install
npm run dev
```

Локальный адрес по умолчанию: `http://localhost:5173`.

## Полезные команды

```bash
npm run dev      # запуск в режиме разработки
npm run build    # production-сборка
npm run preview  # локальный просмотр production-сборки
npm run lint     # проверка линтером
```

## Структура

- `src/pages/main` — главная страница
- `src/pages/project` — описание проекта
- `src/pages/dataset` — данные/датасет
- `src/pages/metrics` — метрики
- `src/components/header` — шапка
- `src/components/footer` — подвал
- `src/redux` — store и slices
- `src/api` — клиентские API-вызовы

## Интеграция с backend

Фронтенд отправляет запросы на backend endpoint `/predict` и показывает результат предсказаний. Если меняется адрес backend, обнови базовый URL в API-слое (`src/api`).
