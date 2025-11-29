from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _


class User(AbstractUser):
    ROLE_CHOICES = [
        ('patient', 'Patient'),
        ('doctor', 'Doctor'),
        ('admin', 'Administrator'),
    ]
    
    role = models.CharField(
        max_length=20, 
        choices=ROLE_CHOICES, 
        default='patient',
        verbose_name=_('Role')
    )
    phone = models.CharField(
        max_length=20, 
        blank=True, 
        null=True,
        verbose_name=_('Phone number')
    )
    date_of_birth = models.DateField(
        blank=True, 
        null=True,
        verbose_name=_('Date of birth')
    )
    is_verified = models.BooleanField(
        default=False,
        verbose_name=_('Is verified')
    )
    firebase_token = models.TextField(
        blank=True, 
        null=True,
        verbose_name=_('Firebase token')
    )
    password_reset_token = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        verbose_name=_('Password reset token')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created at')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated at')
    )

    class Meta:
        db_table = 'users'
        verbose_name = _('User')
        verbose_name_plural = _('Users')

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"


class PatientProfile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    user = models.OneToOneField(
        User, 
        on_delete=models.CASCADE, 
        related_name='patient_profile',
        verbose_name=_('User')
    )
    medical_id = models.CharField(
        max_length=50, 
        unique=True,
        verbose_name=_('Medical ID')
    )
    gender = models.CharField(
        max_length=1, 
        choices=GENDER_CHOICES, 
        blank=True, 
        null=True,
        verbose_name=_('Gender')
    )
    height = models.FloatField(
        blank=True, 
        null=True,
        help_text=_('Height in cm'),
        verbose_name=_('Height')
    )
    weight = models.FloatField(
        blank=True, 
        null=True,
        help_text=_('Weight in kg'),
        verbose_name=_('Weight')
    )
    blood_type = models.CharField(
        max_length=5, 
        blank=True, 
        null=True,
        verbose_name=_('Blood type')
    )
    allergies = models.TextField(
        blank=True, 
        null=True,
        verbose_name=_('Allergies')
    )
    chronic_diseases = models.TextField(
        blank=True, 
        null=True,
        verbose_name=_('Chronic diseases')
    )
    emergency_contact_name = models.CharField(
        max_length=100, 
        blank=True, 
        null=True,
        verbose_name=_('Emergency contact name')
    )
    emergency_contact_phone = models.CharField(
        max_length=20, 
        blank=True, 
        null=True,
        verbose_name=_('Emergency contact phone')
    )
    assigned_doctor = models.ForeignKey(
        'DoctorProfile', 
        on_delete=models.SET_NULL, 
        blank=True, 
        null=True,
        related_name='patients',
        verbose_name=_('Assigned doctor')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created at')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated at')
    )

    class Meta:
        db_table = 'patient_profiles'
        verbose_name = _('Patient Profile')
        verbose_name_plural = _('Patient Profiles')

    def __str__(self):
        return f"Patient: {self.user.get_full_name()}"


class DoctorProfile(models.Model):
    SPECIALTY_CHOICES = [
        ('cardiology', 'Cardiology'),
        ('endocrinology', 'Endocrinology'),
        ('general', 'General Practice'),
        ('internal', 'Internal Medicine'),
        ('nephrology', 'Nephrology'),
        ('neurology', 'Neurology'),
        ('pediatrics', 'Pediatrics'),
        ('psychiatry', 'Psychiatry'),
    ]
    
    user = models.OneToOneField(
        User, 
        on_delete=models.CASCADE, 
        related_name='doctor_profile',
        verbose_name=_('User')
    )
    license_number = models.CharField(
        max_length=50, 
        unique=True,
        verbose_name=_('License number')
    )
    specialty = models.CharField(
        max_length=50, 
        choices=SPECIALTY_CHOICES,
        verbose_name=_('Specialty')
    )
    years_of_experience = models.PositiveIntegerField(
        blank=True, 
        null=True,
        verbose_name=_('Years of experience')
    )
    hospital = models.CharField(
        max_length=200, 
        blank=True, 
        null=True,
        verbose_name=_('Hospital')
    )
    consultation_fee = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        blank=True, 
        null=True,
        verbose_name=_('Consultation fee')
    )
    availability = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Availability schedule in JSON format'),
        verbose_name=_('Availability')
    )
    is_verified = models.BooleanField(
        default=False,
        verbose_name=_('Is verified')
    )
    verification_documents = models.TextField(
        blank=True, 
        null=True,
        verbose_name=_('Verification documents')
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_('Created at')
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Updated at')
    )

    class Meta:
        db_table = 'doctor_profiles'
        verbose_name = _('Doctor Profile')
        verbose_name_plural = _('Doctor Profiles')

    def __str__(self):
        return f"Dr. {self.user.get_full_name()} ({self.get_specialty_display()})"
