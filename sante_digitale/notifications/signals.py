from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
from .service import notification_service
from .models import Notification
from medical_data.models import Alert, MedicalMeasurement


@receiver(post_save, sender=Alert)
def handle_critical_alert(sender, instance, created, **kwargs):
    """Handle critical alert creation"""
    if created and instance.severity == 'critical':
        patient = instance.patient
        alert_data = {
            'message': instance.message,
            'type': instance.alert_type,
        }
        notification_service.send_critical_alert(patient, alert_data)


@receiver(post_save, sender=MedicalMeasurement)
def handle_medical_measurement(sender, instance, created, **kwargs):
    """Handle new medical measurement creation"""
    if created and instance.anomaly_status in ['anomaly', 'critical']:
        patient = instance.patient
        measurement_data = {
            'type': instance.get_measurement_type_display(),
            'value': instance.value,
            'anomaly_status': instance.anomaly_status,
        }
        notification_service.send_anomaly_notification(patient, measurement_data)
