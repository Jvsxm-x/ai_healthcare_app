from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DoctorDashboardViewSet

router = DefaultRouter()
router.register(r'dashboard', DoctorDashboardViewSet, basename='doctor-dashboard')

app_name = 'doctor_dashboard'

urlpatterns = [
    path('', include(router.urls)),
]
