from django.urls import path
from .views import PredictSyncView, PredictView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('predict/sync/', PredictSyncView.as_view(), name='predict_sync'),
]