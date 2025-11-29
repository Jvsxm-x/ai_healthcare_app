from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    MedicalMeasurementViewSet,
    MedicalDocumentViewSet,
    MedicationViewSet,
    MedicalAppointmentViewSet,
    AlertViewSet,
    MedicalHistoryViewSet
)

router = DefaultRouter()
router.register(r'measurements', MedicalMeasurementViewSet, basename='medical-measurement')
router.register(r'documents', MedicalDocumentViewSet, basename='medical-document')
router.register(r'medications', MedicationViewSet, basename='medication')
router.register(r'appointments', MedicalAppointmentViewSet, basename='medical-appointment')
router.register(r'alerts', AlertViewSet, basename='alert')
router.register(r'history', MedicalHistoryViewSet, basename='medical-history')

urlpatterns = [
    path('', include(router.urls)),
]
