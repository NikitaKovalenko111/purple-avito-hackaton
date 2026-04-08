from django.apps import AppConfig
import os
import sys
import threading


class PredictionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prediction'

    def ready(self):
        warmup_enabled = os.getenv("MODEL_WARMUP_ON_START", "1").strip().lower() not in {"0", "false", "no"}
        if not warmup_enabled:
            return

        # Avoid duplicate warmups with Django autoreload parent process.
        if "runserver" in sys.argv and os.getenv("RUN_MAIN") != "true":
            return

        # Run warm-up only for server-like commands.
        if not any(cmd in sys.argv for cmd in ("runserver", "daphne", "gunicorn", "uvicorn")):
            return

        from .model_service import warmup_pipeline

        threading.Thread(target=warmup_pipeline, daemon=True, name="model-warmup").start()
