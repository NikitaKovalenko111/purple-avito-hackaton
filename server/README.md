# Purple Avito Hackathon — Server

Серверная часть для предсказания микрокатегорий и асинхронной генерации черновиков через WebSocket.

---

## 🚀 Что умеет сервер

Сервер выполняет следующий pipeline:

1. Принимает запрос на `/predict`
2. Прогоняет модель и определяет:
   - найденные микрокатегории
   - нужно ли делить объявление (`shouldSplit`)
3. Сразу возвращает HTTP-ответ с:
   - `request_id`
   - `detectedMcIds`
   - `shouldSplit`
4. Если `shouldSplit = true`:
   - запускает фоновую генерацию черновиков
   - отправляет их по WebSocket по мере готовности
5. В конце отправляет событие `predict_done`

---

## 🧱 Стек

- Python 3.11
- Django
- Django REST Framework
- Django Channels
- Daphne (ASGI сервер)
- PyTorch
- Transformers
- scikit-learn
- OpenRouter API (LLM для генерации текста)

---

## ⚙️ Запуск сервера

### Windows (cmd / PowerShell)

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python manage.py migrate
python manage.py runserver

После запуска сервер доступен по адресу:

http://127.0.0.1:8000/
🔌 Endpoints
1. Healthcheck

GET /health/

Ответ:
{
  "status": "ok"
}
2. Predict

POST /predict/
Content-Type: application/json

Тело запроса:
{
  "itemId": 2,
  "sourceMcId": 101,
  "sourceMcTitle": "Ремонт квартир и домов под ключ",
  "description": "Выполняем комплексный ремонт квартир и домов под ключ..."
}
Ответ:
{
  "request_id": "8d0357fd-f3d3-4cfa-9dd5-7315994b8a05",
  "detectedMcIds": [105, 102, 101, 107, 106, 110, 103, 108, 109, 104],
  "shouldSplit": true,
  "drafts": []
}

📌 Важно:

drafts всегда пустой в HTTP-ответе
сами черновики приходят через WebSocket
🔄 WebSocket
Подключение
ws://127.0.0.1:8000/ws/predict/<request_id>/

где <request_id> — из ответа /predict

📡 События WebSocket
1. draft_ready

Приходит, когда готов один черновик:

{
  "event": "draft_ready",
  "request_id": "8d0357fd-f3d3-4cfa-9dd5-7315994b8a05",
  "draft": {
    "mcId": 107,
    "mcTitle": "Малярные работы",
    "text": "Выполняем малярные работы, включая ..."
  }
}
2. draft_error

Если произошла ошибка при генерации:

{
  "event": "draft_error",
  "request_id": "8d0357fd-f3d3-4cfa-9dd5-7315994b8a05",
  "mcId": 107,
  "error": "OpenRouter 400: ..."
}
3. predict_done

Когда всё завершено:

{
  "event": "predict_done",
  "request_id": "8d0357fd-f3d3-4cfa-9dd5-7315994b8a05"
}