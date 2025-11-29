from django.urls import path, include
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

app_name = 'authentication'

urlpatterns = [
    # Authentication endpoints
    path('auth/login/', views.CustomTokenObtainPairView.as_view(), name='login'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/register/', views.RegisterView.as_view(), name='register'),
    path('auth/password/change/', views.PasswordChangeView.as_view(), name='password_change'),
    path('auth/password/reset/', views.password_reset, name='password_reset'),
    
    # Profile endpoints
    path('profile/', views.ProfileView.as_view(), name='profile'),
    path('profile/patient/', views.PatientProfileView.as_view(), name='patient_profile'),
    path('profile/doctor/', views.DoctorProfileView.as_view(), name='doctor_profile'),
    
    # Doctor endpoints
    path('doctors/', views.doctors_list, name='doctors_list'),
    path('doctors/specialties/', views.doctor_specialties, name='doctor_specialties'),
]
