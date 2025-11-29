from rest_framework import status, generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from django.conf import settings
import uuid

from .serializers import (
    UserRegistrationSerializer, UserSerializer, LoginSerializer,
    PasswordResetSerializer, PasswordChangeSerializer,
    PatientProfileSerializer, DoctorProfileSerializer
)
from .models import PatientProfile, DoctorProfile

User = get_user_model()


class CustomTokenObtainPairView(TokenObtainPairView):
    """Custom JWT token view with additional user info"""
    serializer_class = LoginSerializer
    
    @extend_schema(
        summary="Obtain JWT token",
        description="Authenticate user and return JWT tokens",
        responses={200: UserSerializer}
    )
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': UserSerializer(user).data
        })


class RegisterView(generics.CreateAPIView):
    """User registration endpoint"""
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny]
    
    @extend_schema(
        summary="Register new user",
        description="Create a new user account with patient or doctor role",
        responses={201: UserSerializer}
    )
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': UserSerializer(user).data
        }, status=status.HTTP_201_CREATED)


class ProfileView(generics.RetrieveUpdateAPIView):
    """Get/update current user profile"""
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_object(self):
        return self.request.user
    
    @extend_schema(
        summary="Get user profile",
        description="Retrieve current user profile information"
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)
    
    @extend_schema(
        summary="Update user profile",
        description="Update current user profile information"
    )
    def patch(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)


class PatientProfileView(generics.RetrieveUpdateAPIView):
    """Get/update patient profile"""
    serializer_class = PatientProfileSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_object(self):
        try:
            return self.request.user.patient_profile
        except PatientProfile.DoesNotExist:
            from rest_framework.exceptions import NotFound
            raise NotFound("Patient profile not found")
    
    @extend_schema(
        summary="Get patient profile",
        description="Retrieve patient medical profile"
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)
    
    @extend_schema(
        summary="Update patient profile",
        description="Update patient medical profile information"
    )
    def patch(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)


class DoctorProfileView(generics.RetrieveUpdateAPIView):
    """Get/update doctor profile"""
    serializer_class = DoctorProfileSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_object(self):
        try:
            return self.request.user.doctor_profile
        except DoctorProfile.DoesNotExist:
            from rest_framework.exceptions import NotFound
            raise NotFound("Doctor profile not found")
    
    @extend_schema(
        summary="Get doctor profile",
        description="Retrieve doctor professional profile"
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)
    
    @extend_schema(
        summary="Update doctor profile",
        description="Update doctor professional profile information"
    )
    def patch(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)


class PasswordChangeView(generics.GenericAPIView):
    """Change user password"""
    serializer_class = PasswordChangeSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    @extend_schema(
        summary="Change password",
        description="Change current user password"
    )
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user = request.user
        user.set_password(serializer.validated_data['new_password'])
        user.save()
        
        return Response({'message': 'Password changed successfully'})


@api_view(['POST'])
@permission_classes([permissions.AllowAny])
@extend_schema(
    summary="Reset password",
    description="Send password reset email"
)
def password_reset(request):
    """Send password reset email"""
    serializer = PasswordResetSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    
    email = serializer.validated_data['email']
    user = User.objects.get(email=email)
    
    # Generate reset token
    token = str(uuid.uuid4())
    user.password_reset_token = token
    user.save()
    
    # Send email (in production, use proper email backend)
    reset_link = f"{settings.FRONTEND_URL}/reset-password/{token}"
    send_mail(
        'Password Reset',
        f'Click the link to reset your password: {reset_link}',
        settings.DEFAULT_FROM_EMAIL,
        [email],
        fail_silently=False,
    )
    
    return Response({'message': 'Password reset email sent'})


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
@extend_schema(
    summary="Get doctors list",
    description="Get list of verified doctors"
)
def doctors_list(request):
    """Get list of verified doctors"""
    doctors = User.objects.filter(
        role='doctor', 
        doctor_profile__is_verified=True
    ).select_related('doctor_profile')
    
    serializer = UserSerializer(doctors, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
@extend_schema(
    summary="Get doctor specialties",
    description="Get list of available doctor specialties"
)
def doctor_specialties(request):
    """Get list of doctor specialties"""
    specialties = DoctorProfile.SPECIALTY_CHOICES
    return Response([{'value': choice[0], 'label': choice[1]} for choice in specialties])
