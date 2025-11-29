from rest_framework import serializers
from django.utils.translation import gettext_lazy as _
from authentication.serializers import UserSerializer
from medical_data.models import MedicalMeasurement, MedicalDocument, Medication, MedicalAppointment, Alert


class MedicalMeasurementSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    measurement_type_display = serializers.CharField(source='get_measurement_type_display', read_only=True)
    anomaly_status_display = serializers.CharField(source='get_anomaly_status_display', read_only=True)
    
    class Meta:
        model = MedicalMeasurement
        fields = [
            'id', 'patient', 'patient_name', 'measurement_type', 'measurement_type_display',
            'value', 'unit', 'measured_at', 'notes', 'anomaly_status', 'anomaly_status_display',
            'anomaly_score', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'anomaly_status', 'anomaly_score']


class MedicalMeasurementCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicalMeasurement
        fields = [
            'patient', 'measurement_type', 'value', 'unit', 'measured_at', 'notes'
        ]
    
    def validate_value(self, value):
        """Validate measurement value based on type"""
        measurement_type = self.initial_data.get('measurement_type')
        
        # Define normal ranges for different measurement types
        normal_ranges = {
            'blood_pressure_systolic': (70, 200),
            'blood_pressure_diastolic': (40, 130),
            'blood_glucose': (50, 400),
            'heart_rate': (30, 250),
            'weight': (20, 300),
            'temperature': (35, 42),
            'oxygen_saturation': (70, 100),
            'cholesterol': (100, 500),
            'triglycerides': (10, 1000),
            'hemoglobin_a1c': (3, 15),
        }
        
        if measurement_type in normal_ranges:
            min_val, max_val = normal_ranges[measurement_type]
            if not (min_val <= value <= max_val):
                raise serializers.ValidationError(
                    f"Value {value} is outside normal range ({min_val}-{max_val}) for {measurement_type}"
                )
        
        return value


class MedicalDocumentSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    uploaded_by_name = serializers.CharField(source='uploaded_by.get_full_name', read_only=True)
    document_type_display = serializers.CharField(source='get_document_type_display', read_only=True)
    
    class Meta:
        model = MedicalDocument
        fields = [
            'id', 'patient', 'patient_name', 'document_type', 'document_type_display',
            'title', 'description', 'file', 'uploaded_by', 'uploaded_by_name',
            'document_date', 'is_confidential', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'uploaded_by']


class MedicationSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    prescribed_by_name = serializers.CharField(source='prescribed_by.user.get_full_name', read_only=True)
    frequency_display = serializers.CharField(source='get_frequency_display', read_only=True)
    
    class Meta:
        model = Medication
        fields = [
            'id', 'patient', 'patient_name', 'name', 'dosage', 'frequency', 'frequency_display',
            'start_date', 'end_date', 'prescribed_by', 'prescribed_by_name', 'notes',
            'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class MedicalAppointmentSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    doctor_name = serializers.CharField(source='doctor.user.get_full_name', read_only=True)
    appointment_type_display = serializers.CharField(source='get_appointment_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = MedicalAppointment
        fields = [
            'id', 'patient', 'patient_name', 'doctor', 'doctor_name',
            'appointment_type', 'appointment_type_display', 'scheduled_date',
            'duration_minutes', 'status', 'status_display', 'reason', 'notes',
            'is_telemedicine', 'meeting_link', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class MedicalAppointmentCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicalAppointment
        fields = [
            'patient', 'doctor', 'appointment_type', 'scheduled_date',
            'duration_minutes', 'reason', 'notes', 'is_telemedicine'
        ]
    
    def validate_scheduled_date(self, value):
        """Validate appointment date is in the future"""
        from django.utils import timezone
        if value <= timezone.now():
            raise serializers.ValidationError("Appointment date must be in the future")
        return value


class AlertSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    alert_type_display = serializers.CharField(source='get_alert_type_display', read_only=True)
    severity_display = serializers.CharField(source='get_severity_display', read_only=True)
    
    class Meta:
        model = Alert
        fields = [
            'id', 'patient', 'patient_name', 'alert_type', 'alert_type_display',
            'severity', 'severity_display', 'title', 'message', 'is_read',
            'measurement', 'appointment', 'doctor_notified', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class AnomalyDetectionSerializer(serializers.Serializer):
    measurement_id = serializers.IntegerField()
    is_anomaly = serializers.BooleanField()
    anomaly_score = serializers.FloatField()
    anomaly_status = serializers.CharField()
    confidence = serializers.FloatField()


class HealthRiskPredictionSerializer(serializers.Serializer):
    risk_type = serializers.CharField()
    score = serializers.FloatField(min_value=0, max_value=1)
    level = serializers.CharField()
    recommendations = serializers.ListField(child=serializers.CharField())


class PatientHealthSummarySerializer(serializers.Serializer):
    patient_id = serializers.IntegerField()
    patient_name = serializers.CharField()
    latest_measurements = MedicalMeasurementSerializer(many=True)
    active_medications = MedicationSerializer(many=True)
    upcoming_appointments = MedicalAppointmentSerializer(many=True)
    recent_alerts = AlertSerializer(many=True)
    health_risks = HealthRiskPredictionSerializer(many=True)
    anomaly_results = AnomalyDetectionSerializer(many=True)


class DoctorDashboardSerializer(serializers.Serializer):
    doctor_id = serializers.IntegerField()
    doctor_name = serializers.CharField()
    total_patients = serializers.IntegerField()
    upcoming_appointments = MedicalAppointmentSerializer(many=True)
    recent_measurements = MedicalMeasurementSerializer(many=True)
    critical_alerts = AlertSerializer(many=True)
    patient_anomalies = serializers.IntegerField()
    pending_reviews = serializers.IntegerField()


class MeasurementTrendSerializer(serializers.Serializer):
    measurement_type = serializers.CharField()
    trend_data = serializers.ListField(
        child=serializers.DictField(
            child=serializers.FloatField()
        )
    )
    trend_direction = serializers.CharField()
    average_value = serializers.FloatField()
    anomaly_count = serializers.IntegerField()


class HealthAnalyticsSerializer(serializers.Serializer):
    period = serializers.CharField()
    total_measurements = serializers.IntegerField()
    anomaly_rate = serializers.FloatField()
    most_common_anomalies = serializers.ListField(
        child=serializers.CharField()
    )
    patient_health_scores = serializers.ListField(
        child=serializers.DictField()
    )
    medication_adherence = serializers.FloatField()
    appointment_attendance_rate = serializers.FloatField()
