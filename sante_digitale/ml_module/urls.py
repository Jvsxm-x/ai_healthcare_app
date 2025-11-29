from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import AnomalyDetectionViewSet, HealthPredictionViewSet

router = DefaultRouter()
router.register(r'anomaly-detection', AnomalyDetectionViewSet)
router.register(r'health-prediction', HealthPredictionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
