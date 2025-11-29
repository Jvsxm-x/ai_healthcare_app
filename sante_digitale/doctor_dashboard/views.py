from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from django.db.models import Q, Count, Avg, F
from django.utils.translation import gettext_lazy as _
from authentication.models import PatientProfile, DoctorProfile
from medical_data.models import MedicalMeasurement, MedicalAppointment, Alert, Medication
from ml_module.anomaly_detector import AnomalyDetector, HealthRiskPredictor
from ml_module.serializers import (
    DoctorDashboardSerializer, PatientHealthSummarySerializer,
    MeasurementTrendSerializer, HealthAnalyticsSerializer
)
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DoctorDashboardViewSet(viewsets.ViewSet):
    """Doctor Dashboard API endpoints"""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def list(self, request):
        """Get doctor dashboard overview"""
        try:
            user = request.user
            if not hasattr(user, 'doctor_profile'):
                return Response(
                    {'error': 'User is not a doctor'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            doctor_profile = user.doctor_profile
            
            # Get dashboard data
            data = self._get_dashboard_data(doctor_profile)
            
            serializer = DoctorDashboardSerializer(data)
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"Error getting doctor dashboard: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'], url_path='(?P<doctor_id>[^/.]+)/analytics')
    def analytics(self, request, doctor_id=None):
        """Get doctor analytics data"""
        try:
            user = request.user
            
            # Check permissions
            if str(user.id) != doctor_id and not user.is_staff:
                if hasattr(user, 'doctorprofile'):
                    doctor_id = str(user.doctorprofile.id)
                else:
                    return Response(
                        {'error': 'Access denied'},
                        status=status.HTTP_403_FORBIDDEN
                    )
            
            doctor = get_object_or_404(DoctorProfile, id=doctor_id)
            
            # Get analytics data
            analytics = self._get_analytics_data(doctor)
            
            return Response(analytics)
            
        except Exception as e:
            logger.error(f"Error getting doctor analytics: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'], url_path='(?P<doctor_id>[^/.]+)/patients')
    def simplified_patients(self, request, doctor_id=None):
        """Get simplified patient list for doctor"""
        try:
            user = request.user
            
            # Check permissions
            if str(user.id) != doctor_id and not user.is_staff:
                if hasattr(user, 'doctorprofile'):
                    doctor_id = str(user.doctorprofile.id)
                else:
                    return Response(
                        {'error': 'Access denied'},
                        status=status.HTTP_403_FORBIDDEN
                    )
            
            doctor = get_object_or_404(DoctorProfile, id=doctor_id)
            
            # Get simplified patient data
            patients = self._get_simplified_patients(doctor)
            
            return Response(patients)
            
        except Exception as e:
            logger.error(f"Error getting simplified patients: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_analytics_data(self, doctor):
        """Get comprehensive analytics for doctor dashboard"""
        now = timezone.now()
        today = now.date()
        
        # Patient counts
        total_patients = doctor.patients.count()
        active_patients = doctor.patients.filter(
            user__is_active=True
        ).count()
        
        # Critical alerts
        patient_ids = doctor.patients.values_list('id', flat=True)
        critical_alerts = Alert.objects.filter(
            patient_id__in=patient_ids,
            severity='critical',
            is_resolved=False
        ).count()
        
        # Last 24 hours stats
        last_24h = now - timedelta(hours=24)
        recent_measurements_24h = MedicalMeasurement.objects.filter(
            patient_id__in=patient_ids,
            created_at__gte=last_24h
        ).count()
        
        recent_alerts_24h = Alert.objects.filter(
            patient_id__in=patient_ids,
            created_at__gte=last_24h
        ).count()
        
        # Last 7 days stats
        last_7d = now - timedelta(days=7)
        recent_measurements_7d = MedicalMeasurement.objects.filter(
            patient_id__in=patient_ids,
            created_at__gte=last_7d
        ).count()
        
        recent_alerts_7d = Alert.objects.filter(
            patient_id__in=patient_ids,
            created_at__gte=last_7d
        ).count()
        
        # Highest measurements (last 30 days)
        last_30d = now - timedelta(days=30)
        highest_measurements = self._get_highest_measurements(patient_ids, last_30d)
        
        return {
            'patient_count': {
                'total': total_patients,
                'active': active_patients,
                'new_this_month': doctor.patients.filter(
                    created_at__gte=today.replace(day=1)
                ).count()
            },
            'critical_alerts': {
                'total': critical_alerts,
                'new_24h': Alert.objects.filter(
                    patient_id__in=patient_ids,
                    severity='critical',
                    created_at__gte=last_24h
                ).count()
            },
            'stats_24h': {
                'measurements': recent_measurements_24h,
                'alerts': recent_alerts_24h,
                'appointments': MedicalAppointment.objects.filter(
                    doctor=doctor,
                    scheduled_date__gte=last_24h,
                    scheduled_date__lte=now
                ).count()
            },
            'stats_7d': {
                'measurements': recent_measurements_7d,
                'alerts': recent_alerts_7d,
                'appointments': MedicalAppointment.objects.filter(
                    doctor=doctor,
                    scheduled_date__gte=last_7d,
                    scheduled_date__lte=now
                ).count()
            },
            'highest_measurements': highest_measurements
        }
    
    def _get_simplified_patients(self, doctor):
        """Get simplified patient list"""
        patients = []
        
        for patient in doctor.patients.select_related('user').all():
            # Get last measurement
            last_measurement = MedicalMeasurement.objects.filter(
                patient=patient
            ).order_by('-created_at').first()
            
            # Calculate risk level based on recent alerts and measurements
            risk_level = self._calculate_patient_risk(patient)
            
            patient_data = {
                'id': str(patient.id),
                'name': patient.user.get_full_name(),
                'age': patient.age,
                'last_measure': {
                    'type': last_measurement.get_measurement_type_display() if last_measurement else None,
                    'value': last_measurement.value if last_measurement else None,
                    'date': last_measurement.created_at.isoformat() if last_measurement else None
                },
                'risk_level': risk_level
            }
            
            patients.append(patient_data)
        
        return patients
    
    def _get_highest_measurements(self, patient_ids, since_date):
        """Get highest measurements by type"""
        measurements = MedicalMeasurement.objects.filter(
            patient_id__in=patient_ids,
            created_at__gte=since_date
        )
        
        highest_by_type = {}
        
        for measurement in measurements:
            measurement_type = measurement.get_measurement_type_display()
            
            if measurement_type not in highest_by_type:
                highest_by_type[measurement_type] = measurement
            elif measurement.value > highest_by_type[measurement_type].value:
                highest_by_type[measurement_type] = measurement
        
        # Format for response
        result = []
        for measurement_type, measurement in highest_by_type.items():
            result.append({
                'type': measurement_type,
                'value': measurement.value,
                'patient': measurement.patient.user.get_full_name(),
                'date': measurement.created_at.isoformat(),
                'patient_id': str(measurement.patient.id)
            })
        
        return sorted(result, key=lambda x: x['value'], reverse=True)[:10]  # Top 10
    
    def _calculate_patient_risk(self, patient):
        """Calculate patient risk level based on recent data"""
        now = timezone.now()
        last_7d = now - timedelta(days=7)
        
        # Check for recent critical alerts
        critical_alerts = Alert.objects.filter(
            patient=patient,
            severity='critical',
            created_at__gte=last_7d
        ).count()
        
        if critical_alerts > 0:
            return 'high'
        
        # Check for recent anomalies
        anomaly_measurements = MedicalMeasurement.objects.filter(
            patient=patient,
            anomaly_status='anomaly',
            created_at__gte=last_7d
        ).count()
        
        if anomaly_measurements > 2:
            return 'medium'
        
        # Check for any recent alerts
        any_alerts = Alert.objects.filter(
            patient=patient,
            created_at__gte=last_7d
        ).count()
        
        if any_alerts > 0:
            return 'medium'
        
        return 'low'

    def _get_dashboard_data(self, doctor_profile):
        """Get comprehensive dashboard data for doctor"""
        today = timezone.now().date()
        
        # Total patients
        total_patients = doctor_profile.patients.count()
        
        # Upcoming appointments
        upcoming_appointments = MedicalAppointment.objects.filter(
            doctor=doctor_profile,
            scheduled_date__gte=timezone.now(),
            status__in=['scheduled', 'confirmed']
        ).order_by('scheduled_date')[:10]
        
        # Recent measurements from patients
        patient_ids = doctor_profile.patients.values_list('id', flat=True)
        recent_measurements = MedicalMeasurement.objects.filter(
            patient_id__in=patient_ids,
            created_at__gte=timezone.now() - timedelta(days=7)
        ).order_by('-created_at')[:20]
        
        # Critical alerts
        critical_alerts = Alert.objects.filter(
            patient_id__in=patient_ids,
            severity='critical',
            is_read=False
        ).order_by('-created_at')[:10]
        
        # Patient anomalies
        patient_anomalies = MedicalMeasurement.objects.filter(
            patient_id__in=patient_ids,
            anomaly_status='critical',
            created_at__gte=timezone.now() - timedelta(days=7)
        ).count()
        
        # Pending reviews (measurements needing doctor attention)
        pending_reviews = MedicalMeasurement.objects.filter(
            patient_id__in=patient_ids,
            anomaly_status__in=['warning', 'critical'],
            created_at__gte=timezone.now() - timedelta(days=1)
        ).count()
        
        return {
            'doctor_id': doctor_profile.id,
            'doctor_name': doctor_profile.user.get_full_name(),
            'total_patients': total_patients,
            'upcoming_appointments': upcoming_appointments,
            'recent_measurements': recent_measurements,
            'critical_alerts': critical_alerts,
            'patient_anomalies': patient_anomalies,
            'pending_reviews': pending_reviews
        }
    
    @action(detail=False, methods=['get'])
    def patient_list(self, request):
        """Get list of doctor's patients with summary data"""
        try:
            doctor_profile = request.user.doctor_profile
            
            patients = []
            for patient in doctor_profile.patients.all():
                # Get latest measurements
                latest_measurements = MedicalMeasurement.objects.filter(
                    patient=patient
                ).order_by('-measured_at')[:5]
                
                # Get active medications
                active_medications = Medication.objects.filter(
                    patient=patient,
                    is_active=True
                )
                
                # Get recent alerts
                recent_alerts = Alert.objects.filter(
                    patient=patient,
                    is_read=False
                ).order_by('-created_at')[:3]
                
                # Get upcoming appointments
                upcoming_appointments = MedicalAppointment.objects.filter(
                    patient=patient,
                    doctor=doctor_profile,
                    scheduled_date__gte=timezone.now()
                ).order_by('scheduled_date')[:3]
                
                patients.append({
                    'patient_id': patient.id,
                    'patient_name': patient.user.get_full_name(),
                    'age': patient.age,
                    'gender': patient.gender,
                    'latest_measurements': latest_measurements,
                    'active_medications': active_medications,
                    'recent_alerts': recent_alerts,
                    'upcoming_appointments': upcoming_appointments,
                    'last_visit': self._get_last_visit(patient, doctor_profile)
                })
            
            return Response({'patients': patients})
            
        except Exception as e:
            logger.error(f"Error getting patient list: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_last_visit(self, patient, doctor):
        """Get patient's last visit with this doctor"""
        last_appointment = MedicalAppointment.objects.filter(
            patient=patient,
            doctor=doctor,
            status='completed'
        ).order_by('-scheduled_date').first()
        
        return last_appointment.scheduled_date if last_appointment else None
