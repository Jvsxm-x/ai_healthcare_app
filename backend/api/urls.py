from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    # === AUTH VIEWS ===
    register_view, login_view, profile_view, my_profile_view,

    # === RECORDS VIEWS ===
    RecordsView, record_detail, latest_record, RecordsStatsView,

    # === ALERTS VIEWS ===
    alerts_view, acknowledge_alert,

    # === ML VIEWS ===
    predict_view, retrain_view,

    # === DOCTOR VIEWS ===
    patients_list, patient_profile_view,

    # === UPGRADE VIEWS ===
    upgrade_to_pro,

    # === VIEWSETS ===
    PatientProfileViewSet, AlertViewSet,
)

# Router for ViewSets
router = DefaultRouter()
router.register(r'patients', PatientProfileViewSet, basename='patient')
router.register(r'alerts', AlertViewSet, basename='alert')

urlpatterns = [
    # ==================== AUTH ENDPOINTS ====================
    path('auth/register/', register_view, name='register'),
    path('auth/login/', login_view, name='login'),
    path('auth/profile/', profile_view, name='profile'),
    path('auth/my-profile/', my_profile_view, name='my-profile'),

    # ==================== RECORDS ENDPOINTS ====================
    path('records/', RecordsView.as_view(), name='records'),
    path('records/<str:record_id>/', record_detail, name='record-detail'),
    path('records/latest/', latest_record, name='latest-record'),
    path('records/stats/', RecordsStatsView.as_view(), name='records-stats'),
    path('records/stats/<int:days>/', RecordsStatsView.as_view(), name='records-stats-days'),

    # ==================== ALERTS ENDPOINTS ====================
    path('alerts/', alerts_view, name='alerts-list'),
    path('alerts/<str:alert_id>/acknowledge/', acknowledge_alert, name='acknowledge-alert'),

    # ==================== ML ENDPOINTS ====================
    path('predict/', predict_view, name='predict'),
    path('retrain/', retrain_view, name='retrain'),

    # ==================== DOCTOR ENDPOINTS ====================
    path('doctor/patients/', patients_list, name='patients-list'),
    path('patients/profile/', patient_profile_view, name='patient-profile'),

    # ==================== UPGRADE ENDPOINTS ====================
    path('upgrade-to-pro/', upgrade_to_pro, name='upgrade-to-pro'),
    path('upgrade/pro/', upgrade_to_pro, name='upgrade-pro'),

    # ==================== VIEWSET ROUTERS ====================
    path('', include(router.urls)),
]
