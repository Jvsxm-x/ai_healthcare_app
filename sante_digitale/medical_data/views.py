from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from django.db.models import Q, Count, Avg
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter

from .models import (
    MedicalMeasurement, MedicalDocument, Medication, 
    MedicalAppointment, Alert, MedicalHistory
)
from .serializers import (
    MedicalMeasurementSerializer, MedicalMeasurementCreateSerializer,
    MedicalDocumentSerializer, MedicationSerializer,
    MedicalAppointmentSerializer, AlertSerializer, AlertCreateSerializer,
    MedicalHistorySerializer,
    MedicalMeasurementSummarySerializer, MedicalDocumentSummarySerializer,
    MedicationSummarySerializer, MedicalAppointmentSummarySerializer,
    AlertSummarySerializer, MedicalHistorySummarySerializer
)
from authentication.permissions import IsOwner, IsDoctorOrOwner, IsAdminOrDoctor
from authentication.models import PatientProfile, DoctorProfile


class IsPatientOrDoctor(permissions.BasePermission):
    """Custom permission to only allow patients or doctors to access medical data"""
    
    def has_permission(self, request, view):
        return request.user.is_authenticated and (
            hasattr(request.user, 'patientprofile') or 
            hasattr(request.user, 'doctorprofile') or 
            request.user.is_staff
        )


class IsPatientOwner(permissions.BasePermission):
    """Custom permission to only allow patients to access their own data"""
    
    def has_object_permission(self, request, view, obj):
        if request.user.is_staff:
            return True
        if hasattr(request.user, 'doctorprofile'):
            return True
        if hasattr(request.user, 'patientprofile'):
            return obj.patient.user == request.user
        return False


class MedicalMeasurementViewSet(viewsets.ModelViewSet):
    """ViewSet for medical measurements"""
    
    queryset = MedicalMeasurement.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['patient', 'measurement_type', 'anomaly_status']
    search_fields = ['notes', 'device_used', 'location']
    ordering_fields = ['measured_at', 'created_at', 'value']
    ordering = ['-measured_at']
    
    def get_serializer_class(self):
        if self.action == 'create':
            return MedicalMeasurementCreateSerializer
        elif self.action in ['list', 'retrieve']:
            return MedicalMeasurementSummarySerializer
        return MedicalMeasurementSerializer
    
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [IsPatientOrDoctor, IsPatientOwner]
        else:
            permission_classes = [IsPatientOrDoctor]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        """Filter measurements based on user role"""
        user = self.request.user
        if user.is_staff:
            return MedicalMeasurement.objects.all()
        elif hasattr(user, 'doctorprofile'):
            # Doctors can see measurements of their patients
            doctor_patients = user.doctorprofile.patients.all()
            return MedicalMeasurement.objects.filter(patient__in=doctor_patients)
        elif hasattr(user, 'patientprofile'):
            # Patients can only see their own measurements
            return MedicalMeasurement.objects.filter(patient=user.patientprofile)
        return MedicalMeasurement.objects.none()
    
    def perform_create(self, serializer):
        """Set patient automatically for patients creating their own measurements"""
        user = self.request.user
        if hasattr(user, 'patientprofile'):
            serializer.save(patient=user.patientprofile)
        else:
            serializer.save()
    
    @action(detail=False, methods=['get'])
    def latest(self, request):
        """Get latest measurements for each type"""
        queryset = self.get_queryset()
        measurements = {}
        
        for measurement_type, _ in MedicalMeasurement.MEASUREMENT_TYPES:
            latest = queryset.filter(measurement_type=measurement_type).first()
            if latest:
                measurements[measurement_type] = MedicalMeasurementSummarySerializer(latest).data
        
        return Response(measurements)
    
    @action(detail=False, methods=['get'])
    def trends(self, request):
        """Get measurement trends over time"""
        measurement_type = request.query_params.get('type')
        days = int(request.query_params.get('days', 30))
        
        if not measurement_type:
            return Response({'error': 'Measurement type is required'}, status=400)
        
        start_date = timezone.now() - timezone.timedelta(days=days)
        queryset = self.get_queryset().filter(
            measurement_type=measurement_type,
            measured_at__gte=start_date
        ).order_by('measured_at')
        
        serializer = MedicalMeasurementSerializer(queryset, many=True)
        return Response(serializer.data)


class MedicalDocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for medical documents"""
    
    queryset = MedicalDocument.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['patient', 'document_type', 'is_confidential']
    search_fields = ['title', 'description']
    ordering_fields = ['created_at', 'title']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        if self.action in ['list', 'retrieve']:
            return MedicalDocumentSummarySerializer
        return MedicalDocumentSerializer
    
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [IsPatientOrDoctor, IsPatientOwner]
        else:
            permission_classes = [IsPatientOrDoctor]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        """Filter documents based on user role"""
        user = self.request.user
        if user.is_staff:
            return MedicalDocument.objects.all()
        elif hasattr(user, 'doctorprofile'):
            # Doctors can see documents of their patients
            doctor_patients = user.doctorprofile.patients.all()
            return MedicalDocument.objects.filter(patient__in=doctor_patients)
        elif hasattr(user, 'patientprofile'):
            # Patients can only see their own documents
            return MedicalDocument.objects.filter(patient=user.patientprofile)
        return MedicalDocument.objects.none()
    
    def perform_create(self, serializer):
        """Set patient and uploaded_by automatically"""
        user = self.request.user
        if hasattr(user, 'patientprofile'):
            serializer.save(patient=user.patientprofile, uploaded_by=user)
        else:
            serializer.save(uploaded_by=user)


class MedicationViewSet(viewsets.ModelViewSet):
    """ViewSet for medications"""
    
    queryset = Medication.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['patient', 'is_active', 'prescribed_by']
    search_fields = ['name', 'generic_name', 'notes']
    ordering_fields = ['created_at', 'start_date', 'name']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        if self.action in ['list', 'retrieve']:
            return MedicationSummarySerializer
        return MedicationSerializer
    
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [IsPatientOrDoctor, IsPatientOwner]
        else:
            permission_classes = [IsPatientOrDoctor]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        """Filter medications based on user role"""
        user = self.request.user
        if user.is_staff:
            return Medication.objects.all()
        elif hasattr(user, 'doctorprofile'):
            # Doctors can see medications of their patients
            doctor_patients = user.doctorprofile.patients.all()
            return Medication.objects.filter(patient__in=doctor_patients)
        elif hasattr(user, 'patientprofile'):
            # Patients can only see their own medications
            return Medication.objects.filter(patient=user.patientprofile)
        return Medication.objects.none()
    
    def perform_create(self, serializer):
        """Set patient automatically for patients creating their own medications"""
        user = self.request.user
        if hasattr(user, 'patientprofile'):
            serializer.save(patient=user.patientprofile)
        else:
            serializer.save()
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get active medications"""
        queryset = self.get_queryset().filter(is_active=True)
        serializer = MedicationSummarySerializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def upcoming(self, request):
        """Get medications that need attention (expiring soon, etc.)"""
        queryset = self.get_queryset().filter(is_active=True)
        # Add logic for medications needing attention
        serializer = MedicationSummarySerializer(queryset, many=True)
        return Response(serializer.data)


class MedicalAppointmentViewSet(viewsets.ModelViewSet):
    """ViewSet for medical appointments"""
    
    queryset = MedicalAppointment.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['patient', 'doctor', 'appointment_type', 'status']
    search_fields = ['title', 'description', 'location', 'notes']
    ordering_fields = ['scheduled_date', 'created_at', 'title']
    ordering = ['scheduled_date']
    
    def get_serializer_class(self):
        if self.action in ['list', 'retrieve']:
            return MedicalAppointmentSummarySerializer
        return MedicalAppointmentSerializer
    
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [IsPatientOrDoctor, IsPatientOwner]
        else:
            permission_classes = [IsPatientOrDoctor]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        """Filter appointments based on user role"""
        user = self.request.user
        if user.is_staff:
            return MedicalAppointment.objects.all()
        elif hasattr(user, 'doctorprofile'):
            # Doctors can see their own appointments
            return MedicalAppointment.objects.filter(doctor=user.doctorprofile)
        elif hasattr(user, 'patientprofile'):
            # Patients can only see their own appointments
            return MedicalAppointment.objects.filter(patient=user.patientprofile)
        return MedicalAppointment.objects.none()
    
    def perform_create(self, serializer):
        """Set patient or doctor automatically"""
        user = self.request.user
        if hasattr(user, 'patientprofile'):
            serializer.save(patient=user.patientprofile)
        elif hasattr(user, 'doctorprofile'):
            serializer.save(doctor=user.doctorprofile)
        else:
            serializer.save()
    
    @action(detail=False, methods=['get'])
    def upcoming(self, request):
        """Get upcoming appointments"""
        queryset = self.get_queryset().filter(
            scheduled_date__gte=timezone.now(),
            status__in=['scheduled', 'confirmed']
        ).order_by('scheduled_date')
        serializer = MedicalAppointmentSummarySerializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def today(self, request):
        """Get today's appointments"""
        today = timezone.now().date()
        queryset = self.get_queryset().filter(
            scheduled_date__date=today
        ).order_by('scheduled_date')
        serializer = MedicalAppointmentSummarySerializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel an appointment"""
        appointment = self.get_object()
        cancellation_reason = request.data.get('cancellation_reason', '')
        
        appointment.status = 'cancelled'
        appointment.cancellation_reason = cancellation_reason
        appointment.save()
        
        return Response({'status': 'cancelled'})
    
    @action(detail=True, methods=['post'])
    def confirm(self, request, pk=None):
        """Confirm an appointment"""
        appointment = self.get_object()
        appointment.status = 'confirmed'
        appointment.save()
        
        return Response({'status': 'confirmed'})


class AlertViewSet(viewsets.ModelViewSet):
    """ViewSet for alerts"""
    
    queryset = Alert.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['patient', 'alert_type', 'severity', 'is_read']
    search_fields = ['title', 'message']
    ordering_fields = ['created_at', 'severity']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        if self.action == 'create':
            return AlertCreateSerializer
        elif self.action in ['list', 'retrieve']:
            return AlertSummarySerializer
        return AlertSerializer
    
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [IsPatientOrDoctor, IsPatientOwner]
        else:
            permission_classes = [IsPatientOrDoctor]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        """Filter alerts based on user role"""
        user = self.request.user
        if user.is_staff:
            return Alert.objects.all()
        elif hasattr(user, 'doctorprofile'):
            # Doctors can see alerts of their patients
            doctor_patients = user.doctorprofile.patients.all()
            return Alert.objects.filter(patient__in=doctor_patients)
        elif hasattr(user, 'patientprofile'):
            # Patients can only see their own alerts
            return Alert.objects.filter(patient=user.patientprofile)
        return Alert.objects.none()
    
    def perform_create(self, serializer):
        """Set patient automatically for patients creating their own alerts"""
        user = self.request.user
        if hasattr(user, 'patientprofile'):
            serializer.save(patient=user.patientprofile)
        else:
            serializer.save()
    
    @action(detail=False, methods=['get'])
    def unread(self, request):
        """Get unread alerts"""
        queryset = self.get_queryset().filter(is_read=False)
        serializer = AlertSummarySerializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def critical(self, request):
        """Get critical alerts"""
        queryset = self.get_queryset().filter(severity='critical')
        serializer = AlertSummarySerializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def mark_read(self, request, pk=None):
        """Mark alert as read"""
        alert = self.get_object()
        alert.is_read = True
        alert.read_at = timezone.now()
        alert.save()
        
        return Response({'status': 'marked as read'})
    
    @action(detail=True, methods=['post'])
    def acknowledge(self, request, pk=None):
        """Acknowledge alert"""
        alert = self.get_object()
        action_taken = request.data.get('action_taken', '')
        
        alert.acknowledged_by = request.user
        alert.acknowledged_at = timezone.now()
        alert.action_taken = action_taken
        
        if action_taken:
            alert.resolved_at = timezone.now()
        
        alert.save()
        
        return Response({'status': 'acknowledged'})


class MedicalHistoryViewSet(viewsets.ModelViewSet):
    """ViewSet for medical history"""
    
    queryset = MedicalHistory.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['patient', 'history_type', 'is_active']
    search_fields = ['condition', 'description']
    ordering_fields = ['diagnosis_date', 'created_at', 'condition']
    ordering = ['-diagnosis_date']
    
    def get_serializer_class(self):
        if self.action in ['list', 'retrieve']:
            return MedicalHistorySummarySerializer
        return MedicalHistorySerializer
    
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [IsPatientOrDoctor, IsPatientOwner]
        else:
            permission_classes = [IsPatientOrDoctor]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        """Filter medical history based on user role"""
        user = self.request.user
        if user.is_staff:
            return MedicalHistory.objects.all()
        elif hasattr(user, 'doctorprofile'):
            # Doctors can see medical history of their patients
            doctor_patients = user.doctorprofile.patients.all()
            return MedicalHistory.objects.filter(patient__in=doctor_patients)
        elif hasattr(user, 'patientprofile'):
            # Patients can only see their own medical history
            return MedicalHistory.objects.filter(patient=user.patientprofile)
        return MedicalHistory.objects.none()
    
    def perform_create(self, serializer):
        """Set patient automatically for patients creating their own history"""
        user = self.request.user
        if hasattr(user, 'patientprofile'):
            serializer.save(patient=user.patientprofile)
        else:
            serializer.save()
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get active medical conditions"""
        queryset = self.get_queryset().filter(is_active=True)
        serializer = MedicalHistorySummarySerializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_type(self, request):
        """Get medical history grouped by type"""
        history_type = request.query_params.get('type')
        if not history_type:
            return Response({'error': 'History type is required'}, status=400)
        
        queryset = self.get_queryset().filter(history_type=history_type)
        serializer = MedicalHistorySummarySerializer(queryset, many=True)
        return Response(serializer.data)
