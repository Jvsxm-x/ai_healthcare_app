from rest_framework import serializers
from django.utils.translation import gettext_lazy as _
from django.core.exceptions import ValidationError
from django.utils import timezone
from .models import (
    MedicalMeasurement, MedicalDocument, Medication, 
    MedicalAppointment, Alert, MedicalHistory
)


class MedicalMeasurementSerializer(serializers.ModelSerializer):
    """Serializer for medical measurements"""
    
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    measurement_type_display = serializers.CharField(source='get_measurement_type_display', read_only=True)
    anomaly_status_display = serializers.CharField(source='get_anomaly_status_display', read_only=True)
    
    class Meta:
        model = MedicalMeasurement
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at', 'anomaly_status', 'anomaly_score')
    
    def validate_value(self, value):
        """Validate measurement values based on type"""
        measurement_type = self.initial_data.get('measurement_type')
        
        if measurement_type == 'blood_pressure_systolic':
            if value < 60 or value > 250:
                raise ValidationError(_('Systolic blood pressure must be between 60 and 250 mmHg'))
        elif measurement_type == 'blood_pressure_diastolic':
            if value < 30 or value > 150:
                raise ValidationError(_('Diastolic blood pressure must be between 30 and 150 mmHg'))
        elif measurement_type == 'heart_rate':
            if value < 30 or value > 220:
                raise ValidationError(_('Heart rate must be between 30 and 220 bpm'))
        elif measurement_type == 'blood_sugar':
            if value < 20 or value > 600:
                raise ValidationError(_('Blood sugar must be between 20 and 600 mg/dL'))
        elif measurement_type == 'temperature':
            if value < 35 or value > 42:
                raise ValidationError(_('Temperature must be between 35°C and 42°C'))
        elif measurement_type == 'oxygen_saturation':
            if value < 70 or value > 100:
                raise ValidationError(_('Oxygen saturation must be between 70% and 100%'))
        elif measurement_type == 'weight':
            if value < 1 or value > 500:
                raise ValidationError(_('Weight must be between 1 and 500 kg'))
        elif measurement_type == 'height':
            if value < 30 or value > 250:
                raise ValidationError(_('Height must be between 30 and 250 cm'))
        elif measurement_type == 'cholesterol':
            if value < 50 or value > 500:
                raise ValidationError(_('Cholesterol must be between 50 and 500 mg/dL'))
        elif measurement_type == 'bmi':
            if value < 10 or value > 50:
                raise ValidationError(_('BMI must be between 10 and 50'))
        
        return value
    
    def validate_measured_at(self, value):
        """Validate measurement date is not in the future"""
        if value > timezone.now():
            raise ValidationError(_('Measurement date cannot be in the future'))
        return value


class MedicalMeasurementCreateSerializer(MedicalMeasurementSerializer):
    """Serializer for creating medical measurements with patient validation"""
    
    class Meta(MedicalMeasurementSerializer.Meta):
        read_only_fields = ('id', 'created_at', 'updated_at', 'anomaly_status', 'anomaly_score')


class MedicalDocumentSerializer(serializers.ModelSerializer):
    """Serializer for medical documents"""
    
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    document_type_display = serializers.CharField(source='get_document_type_display', read_only=True)
    uploaded_by_name = serializers.CharField(source='uploaded_by.get_full_name', read_only=True)
    file_size_display = serializers.SerializerMethodField()
    
    class Meta:
        model = MedicalDocument
        fields = '__all__'
        read_only_fields = ('id', 'file_size', 'mime_type', 'uploaded_by', 'created_at', 'updated_at')
    
    def get_file_size_display(self, obj):
        """Get human-readable file size"""
        if obj.file_size:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if obj.file_size < 1024.0:
                    return f"{obj.file_size:.1f} {unit}"
                obj.file_size /= 1024.0
            return f"{obj.file_size:.1f} TB"
        return "0 B"
    
    def validate_file(self, value):
        """Validate file size and type"""
        max_size = 10 * 1024 * 1024  # 10MB
        if value.size > max_size:
            raise ValidationError(_('File size cannot exceed 10MB'))
        
        allowed_types = [
            'application/pdf', 'image/jpeg', 'image/png', 
            'image/gif', 'text/plain', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]
        
        if value.content_type not in allowed_types:
            raise ValidationError(_('File type not allowed'))
        
        return value
    
    def validate_expires_at(self, value):
        """Validate expiration date is in the future"""
        if value and value <= timezone.now().date():
            raise ValidationError(_('Expiration date must be in the future'))
        return value


class MedicationSerializer(serializers.ModelSerializer):
    """Serializer for medications"""
    
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    prescribed_by_name = serializers.CharField(source='prescribed_by.user.get_full_name', read_only=True)
    frequency_display = serializers.CharField(source='get_frequency_display', read_only=True)
    
    class Meta:
        model = Medication
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')
    
    def validate_end_date(self, value):
        """Validate end date is after start date"""
        start_date = self.initial_data.get('start_date')
        if start_date and value and value < start_date:
            raise ValidationError(_('End date must be after start date'))
        return value
    
    def validate_start_date(self, value):
        """Validate start date is not too far in the past"""
        if value < timezone.now().date() - timezone.timedelta(days=365):
            raise ValidationError(_('Start date cannot be more than 1 year in the past'))
        return value


class MedicalAppointmentSerializer(serializers.ModelSerializer):
    """Serializer for medical appointments"""
    
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    doctor_name = serializers.CharField(source='doctor.user.get_full_name', read_only=True)
    appointment_type_display = serializers.CharField(source='get_appointment_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = MedicalAppointment
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')
    
    def validate_scheduled_date(self, value):
        """Validate appointment date is not too far in the past"""
        if value < timezone.now() - timezone.timedelta(hours=1):
            raise ValidationError(_('Appointment cannot be scheduled more than 1 hour in the past'))
        return value
    
    def validate_duration_minutes(self, value):
        """Validate appointment duration"""
        if value < 5 or value > 480:  # 5 min to 8 hours
            raise ValidationError(_('Duration must be between 5 and 480 minutes'))
        return value
    
    def validate(self, data):
        """Validate appointment conflicts"""
        patient = data.get('patient')
        doctor = data.get('doctor')
        scheduled_date = data.get('scheduled_date')
        duration = data.get('duration_minutes', 30)
        
        # Check for patient conflicts
        if patient and scheduled_date:
            end_time = scheduled_date + timezone.timedelta(minutes=duration)
            conflicts = MedicalAppointment.objects.filter(
                patient=patient,
                scheduled_date__lt=end_time,
                scheduled_date__gte=scheduled_date - timezone.timedelta(minutes=duration)
            ).exclude(status='cancelled')
            
            if self.instance:
                conflicts = conflicts.exclude(pk=self.instance.pk)
            
            if conflicts.exists():
                raise ValidationError(_('Patient has another appointment at this time'))
        
        # Check for doctor conflicts
        if doctor and scheduled_date:
            end_time = scheduled_date + timezone.timedelta(minutes=duration)
            conflicts = MedicalAppointment.objects.filter(
                doctor=doctor,
                scheduled_date__lt=end_time,
                scheduled_date__gte=scheduled_date - timezone.timedelta(minutes=duration)
            ).exclude(status='cancelled')
            
            if self.instance:
                conflicts = conflicts.exclude(pk=self.instance.pk)
            
            if conflicts.exists():
                raise ValidationError(_('Doctor has another appointment at this time'))
        
        return data


class AlertSerializer(serializers.ModelSerializer):
    """Serializer for alerts"""
    
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    alert_type_display = serializers.CharField(source='get_alert_type_display', read_only=True)
    severity_display = serializers.CharField(source='get_severity_display', read_only=True)
    acknowledged_by_name = serializers.CharField(source='acknowledged_by.get_full_name', read_only=True)
    
    class Meta:
        model = Alert
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at', 'read_at', 'acknowledged_at', 'resolved_at')
    
    def validate_expires_at(self, value):
        """Validate expiration date is in the future"""
        if value and value <= timezone.now():
            raise ValidationError(_('Expiration date must be in the future'))
        return value
    
    def validate(self, data):
        """Validate alert resolution logic"""
        is_read = data.get('is_read', False)
        acknowledged_by = data.get('acknowledged_by')
        action_taken = data.get('action_taken')
        resolved_at = data.get('resolved_at')
        
        if is_read and not data.get('read_at'):
            data['read_at'] = timezone.now()
        
        if acknowledged_by and not data.get('acknowledged_at'):
            data['acknowledged_at'] = timezone.now()
        
        if action_taken and not resolved_at:
            data['resolved_at'] = timezone.now()
        
        return data


class AlertCreateSerializer(AlertSerializer):
    """Serializer for creating alerts with automatic timestamps"""
    
    class Meta(AlertSerializer.Meta):
        read_only_fields = ('id', 'created_at', 'updated_at', 'read_at', 'acknowledged_at', 'resolved_at')


class MedicalHistorySerializer(serializers.ModelSerializer):
    """Serializer for medical history"""
    
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    history_type_display = serializers.CharField(source='get_history_type_display', read_only=True)
    treating_doctor_name = serializers.CharField(source='treating_doctor.user.get_full_name', read_only=True)
    
    class Meta:
        model = MedicalHistory
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')
    
    def validate_diagnosis_date(self, value):
        """Validate diagnosis date is not in the future"""
        if value > timezone.now().date():
            raise ValidationError(_('Diagnosis date cannot be in the future'))
        return value
    
    def validate_resolved_date(self, value):
        """Validate resolved date is after diagnosis date"""
        diagnosis_date = self.initial_data.get('diagnosis_date')
        if diagnosis_date and value and value < diagnosis_date:
            raise ValidationError(_('Resolved date must be after diagnosis date'))
        return value
    
    def validate(self, data):
        """Validate medical history logic"""
        resolved_date = data.get('resolved_date')
        is_active = data.get('is_active', True)
        
        if resolved_date and is_active:
            data['is_active'] = False
        
        return data


# Summary serializers for dashboard and list views
class MedicalMeasurementSummarySerializer(MedicalMeasurementSerializer):
    """Summary serializer for medical measurements in lists"""
    
    class Meta:
        model = MedicalMeasurement
        fields = ('id', 'measurement_type', 'value', 'unit', 'measured_at', 'anomaly_status', 'patient_name')


class MedicalDocumentSummarySerializer(MedicalDocumentSerializer):
    """Summary serializer for medical documents in lists"""
    
    class Meta:
        model = MedicalDocument
        fields = ('id', 'title', 'document_type', 'created_at', 'patient_name', 'file_size_display')


class MedicationSummarySerializer(MedicationSerializer):
    """Summary serializer for medications in lists"""
    
    class Meta:
        model = Medication
        fields = ('id', 'name', 'dosage', 'frequency', 'is_active', 'start_date', 'end_date', 'patient_name')


class MedicalAppointmentSummarySerializer(MedicalAppointmentSerializer):
    """Summary serializer for appointments in lists"""
    
    class Meta:
        model = MedicalAppointment
        fields = ('id', 'title', 'appointment_type', 'scheduled_date', 'status', 'patient_name', 'doctor_name')


class AlertSummarySerializer(AlertSerializer):
    """Summary serializer for alerts in lists"""
    
    class Meta:
        model = Alert
        fields = ('id', 'title', 'alert_type', 'severity', 'is_read', 'created_at', 'patient_name')


class MedicalHistorySummarySerializer(MedicalHistorySerializer):
    """Summary serializer for medical history in lists"""
    
    class Meta:
        model = MedicalHistory
        fields = ('id', 'condition', 'history_type', 'is_active', 'diagnosis_date', 'patient_name')
