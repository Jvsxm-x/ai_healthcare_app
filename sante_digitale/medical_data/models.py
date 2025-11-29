from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid

User = get_user_model()


class MedicalMeasurement(models.Model):
    """Medical measurements for patients"""
    
    MEASUREMENT_TYPES = [
        ('blood_pressure_systolic', _('Blood Pressure - Systolic')),
        ('blood_pressure_diastolic', _('Blood Pressure - Diastolic')),
        ('heart_rate', _('Heart Rate')),
        ('blood_sugar', _('Blood Sugar')),
        ('weight', _('Weight')),
        ('height', _('Height')),
        ('temperature', _('Temperature')),
        ('oxygen_saturation', _('Oxygen Saturation')),
        ('cholesterol', _('Cholesterol')),
        ('bmi', _('BMI')),
    ]
    
    ANOMALY_STATUS = [
        ('normal', _('Normal')),
        ('warning', _('Warning')),
        ('critical', _('Critical')),
        ('unknown', _('Unknown')),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        'authentication.PatientProfile',
        on_delete=models.CASCADE,
        related_name='measurements',
        verbose_name=_('Patient')
    )
    measurement_type = models.CharField(
        max_length=50,
        choices=MEASUREMENT_TYPES,
        verbose_name=_('Measurement Type')
    )
    value = models.FloatField(
        validators=[MinValueValidator(0)],
        verbose_name=_('Value')
    )
    unit = models.CharField(
        max_length=20,
        verbose_name=_('Unit')
    )
    measured_at = models.DateTimeField(
        verbose_name=_('Measured At')
    )
    anomaly_status = models.CharField(
        max_length=20,
        choices=ANOMALY_STATUS,
        default='unknown',
        verbose_name=_('Anomaly Status')
    )
    anomaly_score = models.FloatField(
        null=True,
        blank=True,
        verbose_name=_('Anomaly Score')
    )
    notes = models.TextField(
        blank=True,
        verbose_name=_('Notes')
    )
    device_used = models.CharField(
        max_length=100,
        blank=True,
        verbose_name=_('Device Used')
    )
    location = models.CharField(
        max_length=200,
        blank=True,
        verbose_name=_('Location')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created At')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated At')
    )

    class Meta:
        db_table = 'medical_measurements'
        verbose_name = _('Medical Measurement')
        verbose_name_plural = _('Medical Measurements')
        ordering = ['-measured_at']
        indexes = [
            models.Index(fields=['patient', 'measurement_type']),
            models.Index(fields=['patient', 'measured_at']),
            models.Index(fields=['anomaly_status']),
            models.Index(fields=['measured_at']),
        ]

    def __str__(self):
        return f"{self.patient.user.get_full_name()} - {self.get_measurement_type_display()}: {self.value} {self.unit}"


class MedicalDocument(models.Model):
    """Medical documents and records"""
    
    DOCUMENT_TYPES = [
        ('prescription', _('Prescription')),
        ('lab_result', _('Lab Result')),
        ('imaging', _('Imaging')),
        ('report', _('Medical Report')),
        ('consent', _('Consent Form')),
        ('insurance', _('Insurance Document')),
        ('other', _('Other')),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        'authentication.PatientProfile',
        on_delete=models.CASCADE,
        related_name='documents',
        verbose_name=_('Patient')
    )
    document_type = models.CharField(
        max_length=20,
        choices=DOCUMENT_TYPES,
        verbose_name=_('Document Type')
    )
    title = models.CharField(
        max_length=200,
        verbose_name=_('Title')
    )
    description = models.TextField(
        blank=True,
        verbose_name=_('Description')
    )
    file = models.FileField(
        upload_to='medical_documents/%Y/%m/',
        verbose_name=_('File')
    )
    file_size = models.PositiveIntegerField(
        verbose_name=_('File Size (bytes)')
    )
    mime_type = models.CharField(
        max_length=100,
        verbose_name=_('MIME Type')
    )
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='uploaded_documents',
        verbose_name=_('Uploaded By')
    )
    is_confidential = models.BooleanField(
        default=False,
        verbose_name=_('Is Confidential')
    )
    expires_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name=_('Expires At')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created At')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated At')
    )

    class Meta:
        db_table = 'medical_documents'
        verbose_name = _('Medical Document')
        verbose_name_plural = _('Medical Documents')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['patient', 'document_type']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.patient.user.get_full_name()} - {self.title}"


class Medication(models.Model):
    """Patient medications"""
    
    FREQUENCY_CHOICES = [
        ('once_daily', _('Once Daily')),
        ('twice_daily', _('Twice Daily')),
        ('three_times_daily', _('Three Times Daily')),
        ('four_times_daily', _('Four Times Daily')),
        ('weekly', _('Weekly')),
        ('monthly', _('Monthly')),
        ('as_needed', _('As Needed')),
        ('other', _('Other')),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        'authentication.PatientProfile',
        on_delete=models.CASCADE,
        related_name='medications',
        verbose_name=_('Patient')
    )
    name = models.CharField(
        max_length=200,
        verbose_name=_('Medication Name')
    )
    generic_name = models.CharField(
        max_length=200,
        blank=True,
        verbose_name=_('Generic Name')
    )
    dosage = models.CharField(
        max_length=100,
        verbose_name=_('Dosage')
    )
    frequency = models.CharField(
        max_length=20,
        choices=FREQUENCY_CHOICES,
        verbose_name=_('Frequency')
    )
    route = models.CharField(
        max_length=50,
        verbose_name=_('Route')
    )
    start_date = models.DateField(
        verbose_name=_('Start Date')
    )
    end_date = models.DateField(
        null=True,
        blank=True,
        verbose_name=_('End Date')
    )
    prescribed_by = models.ForeignKey(
        'authentication.DoctorProfile',
        on_delete=models.SET_NULL,
        null=True,
        related_name='prescribed_medications',
        verbose_name=_('Prescribed By')
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name=_('Is Active')
    )
    notes = models.TextField(
        blank=True,
        verbose_name=_('Notes')
    )
    side_effects = models.TextField(
        blank=True,
        verbose_name=_('Side Effects')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created At')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated At')
    )

    class Meta:
        db_table = 'medications'
        verbose_name = _('Medication')
        verbose_name_plural = _('Medications')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['patient', 'is_active']),
            models.Index(fields=['start_date']),
            models.Index(fields=['prescribed_by']),
        ]

    def __str__(self):
        return f"{self.patient.user.get_full_name()} - {self.name}"


class MedicalAppointment(models.Model):
    """Medical appointments"""
    
    APPOINTMENT_TYPES = [
        ('consultation', _('Consultation')),
        ('follow_up', _('Follow-up')),
        ('emergency', _('Emergency')),
        ('checkup', _('Check-up')),
        ('surgery', _('Surgery')),
        ('therapy', _('Therapy')),
        ('other', _('Other')),
    ]
    
    STATUS_CHOICES = [
        ('scheduled', _('Scheduled')),
        ('confirmed', _('Confirmed')),
        ('cancelled', _('Cancelled')),
        ('completed', _('Completed')),
        ('no_show', _('No Show')),
        ('rescheduled', _('Rescheduled')),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        'authentication.PatientProfile',
        on_delete=models.CASCADE,
        related_name='appointments',
        verbose_name=_('Patient')
    )
    doctor = models.ForeignKey(
        'authentication.DoctorProfile',
        on_delete=models.CASCADE,
        related_name='appointments',
        verbose_name=_('Doctor')
    )
    appointment_type = models.CharField(
        max_length=20,
        choices=APPOINTMENT_TYPES,
        verbose_name=_('Appointment Type')
    )
    title = models.CharField(
        max_length=200,
        verbose_name=_('Title')
    )
    description = models.TextField(
        blank=True,
        verbose_name=_('Description')
    )
    scheduled_date = models.DateTimeField(
        verbose_name=_('Scheduled Date')
    )
    duration_minutes = models.PositiveIntegerField(
        default=30,
        verbose_name=_('Duration (minutes)')
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='scheduled',
        verbose_name=_('Status')
    )
    location = models.CharField(
        max_length=200,
        blank=True,
        verbose_name=_('Location')
    )
    notes = models.TextField(
        blank=True,
        verbose_name=_('Notes')
    )
    cancellation_reason = models.TextField(
        blank=True,
        verbose_name=_('Cancellation Reason')
    )
    rescheduled_from = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='rescheduled_to',
        verbose_name=_('Rescheduled From')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created At')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated At')
    )

    class Meta:
        db_table = 'medical_appointments'
        verbose_name = _('Medical Appointment')
        verbose_name_plural = _('Medical Appointments')
        ordering = ['-scheduled_date']
        indexes = [
            models.Index(fields=['patient', 'status']),
            models.Index(fields=['doctor', 'status']),
            models.Index(fields=['scheduled_date']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"{self.patient.user.get_full_name()} - {self.doctor.user.get_full_name()} - {self.scheduled_date}"


class Alert(models.Model):
    """Medical alerts and notifications"""
    
    ALERT_TYPES = [
        ('medication', _('Medication')),
        ('appointment', _('Appointment')),
        ('measurement', _('Measurement')),
        ('lab_result', _('Lab Result')),
        ('emergency', _('Emergency')),
        ('system', _('System')),
        ('other', _('Other')),
    ]
    
    SEVERITY_CHOICES = [
        ('low', _('Low')),
        ('medium', _('Medium')),
        ('high', _('High')),
        ('critical', _('Critical')),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        'authentication.PatientProfile',
        on_delete=models.CASCADE,
        related_name='alerts',
        verbose_name=_('Patient')
    )
    alert_type = models.CharField(
        max_length=20,
        choices=ALERT_TYPES,
        verbose_name=_('Alert Type')
    )
    severity = models.CharField(
        max_length=20,
        choices=SEVERITY_CHOICES,
        verbose_name=_('Severity')
    )
    title = models.CharField(
        max_length=200,
        verbose_name=_('Title')
    )
    message = models.TextField(
        verbose_name=_('Message')
    )
    is_read = models.BooleanField(
        default=False,
        verbose_name=_('Is Read')
    )
    read_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name=_('Read At')
    )
    acknowledged_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='acknowledged_alerts',
        verbose_name=_('Acknowledged By')
    )
    acknowledged_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name=_('Acknowledged At')
    )
    requires_action = models.BooleanField(
        default=False,
        verbose_name=_('Requires Action')
    )
    action_taken = models.TextField(
        blank=True,
        verbose_name=_('Action Taken')
    )
    resolved_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name=_('Resolved At')
    )
    expires_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name=_('Expires At')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created At')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated At')
    )

    class Meta:
        db_table = 'alerts'
        verbose_name = _('Alert')
        verbose_name_plural = _('Alerts')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['patient', 'is_read']),
            models.Index(fields=['severity']),
            models.Index(fields=['alert_type']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.patient.user.get_full_name()} - {self.title}"


class MedicalHistory(models.Model):
    """Patient medical history"""
    
    HISTORY_TYPES = [
        ('diagnosis', _('Diagnosis')),
        ('surgery', _('Surgery')),
        ('hospitalization', _('Hospitalization')),
        ('allergy', _('Allergy')),
        ('condition', _('Condition')),
        ('vaccination', _('Vaccination')),
        ('family_history', _('Family History')),
        ('other', _('Other')),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        'authentication.PatientProfile',
        on_delete=models.CASCADE,
        related_name='medical_history',
        verbose_name=_('Patient')
    )
    history_type = models.CharField(
        max_length=20,
        choices=HISTORY_TYPES,
        verbose_name=_('History Type')
    )
    condition = models.CharField(
        max_length=200,
        verbose_name=_('Condition')
    )
    description = models.TextField(
        verbose_name=_('Description')
    )
    diagnosis_date = models.DateField(
        null=True,
        blank=True,
        verbose_name=_('Diagnosis Date')
    )
    resolved_date = models.DateField(
        null=True,
        blank=True,
        verbose_name=_('Resolved Date')
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name=_('Is Active')
    )
    treating_doctor = models.ForeignKey(
        'authentication.DoctorProfile',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='patient_conditions',
        verbose_name=_('Treating Doctor')
    )
    notes = models.TextField(
        blank=True,
        verbose_name=_('Notes')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created At')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated At')
    )

    class Meta:
        db_table = 'medical_history'
        verbose_name = _('Medical History')
        verbose_name_plural = _('Medical Histories')
        ordering = ['-diagnosis_date']
        indexes = [
            models.Index(fields=['patient', 'history_type']),
            models.Index(fields=['is_active']),
            models.Index(fields=['diagnosis_date']),
        ]

    def __str__(self):
        return f"{self.patient.user.get_full_name()} - {self.condition}"
