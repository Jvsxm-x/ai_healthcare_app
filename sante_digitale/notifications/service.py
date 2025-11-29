import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from django.conf import settings
from django.utils import timezone
from django.core.mail import send_mail
from django.template.loader import render_to_string
from celery import shared_task

try:
    from firebase_admin import credentials, messaging, initialize_app
    FCM_AVAILABLE = True
except ImportError:
    FCM_AVAILABLE = False
    print("Firebase Admin SDK not available. FCM notifications will be disabled.")

from .models import Notification, NotificationQueue, NotificationLog, NotificationPreference
from authentication.models import User, PatientProfile, DoctorProfile

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications across multiple channels"""
    
    def __init__(self):
        self.fcm_app = None
        if FCM_AVAILABLE and hasattr(settings, 'FIREBASE_CREDENTIALS_PATH'):
            try:
                cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
                self.fcm_app = initialize_app(cred, name='medical_notifications')
            except Exception as e:
                logger.error(f"Failed to initialize Firebase: {e}")
    
    def send_notification(self, notification: Notification, channels: List[str] = None) -> bool:
        """Send notification through specified channels"""
        if not channels:
            channels = ['push', 'email']
        
        success = True
        
        for channel in channels:
            try:
                if channel == 'push' and self._should_send_push(notification):
                    success &= self._send_push_notification(notification)
                elif channel == 'email' and self._should_send_email(notification):
                    success &= self._send_email_notification(notification)
                elif channel == 'web_push' and self._should_send_web_push(notification):
                    success &= self._send_web_push_notification(notification)
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")
                success &= False
        
        # Log the notification attempt
        NotificationLog.objects.create(
            notification=notification,
            channel=', '.join(channels),
            status='sent' if success else 'failed',
            sent_at=timezone.now()
        )
        
        return success
    
    def _should_send_push(self, notification: Notification) -> bool:
        """Check if push notification should be sent"""
        preference = self._get_user_preference(notification.recipient, 'push')
        return preference and preference.is_enabled and notification.recipient.fcm_token
    
    def _should_send_email(self, notification: Notification) -> bool:
        """Check if email notification should be sent"""
        preference = self._get_user_preference(notification.recipient, 'email')
        return preference and preference.is_enabled and notification.recipient.email
    
    def _should_send_web_push(self, notification: Notification) -> bool:
        """Check if web push notification should be sent"""
        preference = self._get_user_preference(notification.recipient, 'web_push')
        return preference and preference.is_enabled
    
    def _get_user_preference(self, user: User, channel: str) -> Optional[NotificationPreference]:
        """Get user notification preference for channel"""
        try:
            return NotificationPreference.objects.get(
                user=user,
                notification_type=channel
            )
        except NotificationPreference.DoesNotExist:
            return None
    
    def _send_push_notification(self, notification: Notification) -> bool:
        """Send FCM push notification"""
        if not FCM_AVAILABLE or not self.fcm_app:
            logger.warning("FCM not available")
            return False
        
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=notification.title,
                    body=notification.message,
                ),
                data={
                    'notification_id': str(notification.id),
                    'type': notification.notification_type,
                    'priority': notification.priority,
                },
                token=notification.recipient.fcm_token,
                android=messaging.AndroidConfig(
                    priority='high' if notification.priority == 'high' else 'normal',
                    notification=messaging.AndroidNotification(
                        sound='default',
                        click_action='FLUTTER_NOTIFICATION_CLICK',
                    ),
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            sound='default',
                            badge=1,
                        ),
                    ),
                ),
            )
            
            response = messaging.send(message, app=self.fcm_app)
            logger.info(f"FCM notification sent: {response}")
            return True
            
        except Exception as e:
            logger.error(f"FCM send failed: {e}")
            return False
    
    def _send_email_notification(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            subject = notification.title
            
            # Render email template if available
            html_content = render_to_string('notifications/email_template.html', {
                'notification': notification,
                'user': notification.recipient,
            })
            
            # Fallback to plain text
            text_content = notification.message
            
            send_mail(
                subject=subject,
                message=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[notification.recipient.email],
                html_message=html_content,
                fail_silently=False,
            )
            
            logger.info(f"Email sent to {notification.recipient.email}")
            return True
            
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False
    
    def _send_web_push_notification(self, notification: Notification) -> bool:
        """Send web push notification (simplified implementation)"""
        # This would integrate with a web push service like VAPID
        # For now, we'll just log it
        logger.info(f"Web push notification for {notification.recipient.username}")
        return True
    
    def send_bulk_notification(self, recipients: List[User], title: str, message: str, 
                             notification_type: str = 'info', channels: List[str] = None) -> int:
        """Send bulk notification to multiple users"""
        success_count = 0
        
        for recipient in recipients:
            notification = Notification.objects.create(
                recipient=recipient,
                title=title,
                message=message,
                notification_type=notification_type,
            )
            
            if self.send_notification(notification, channels):
                success_count += 1
        
        return success_count
    
    def queue_notification(self, notification: Notification, scheduled_at: datetime = None) -> NotificationQueue:
        """Queue notification for later sending"""
        if not scheduled_at:
            scheduled_at = timezone.now() + timedelta(minutes=5)
        
        queue_item = NotificationQueue.objects.create(
            notification=notification,
            scheduled_at=scheduled_at,
        )
        
        return queue_item
    
    def send_critical_alert(self, patient: PatientProfile, alert_data: Dict[str, Any]) -> bool:
        """Send critical alert notification immediately"""
        # Send to patient
        patient_notification = Notification.objects.create(
            recipient=patient.user,
            title="🚨 Alert Critique",
            message=alert_data.get('message', 'Une alerte critique a été détectée'),
            notification_type='alert',
            priority='high',
        )
        
        # Send to assigned doctor
        doctor_notifications = []
        if patient.assigned_doctor:
            doctor_notification = Notification.objects.create(
                recipient=patient.assigned_doctor.user,
                title=f"🚨 Alert Critique - {patient.user.get_full_name()}",
                message=alert_data.get('message', f'Alerte critique pour {patient.user.get_full_name()}'),
                notification_type='alert',
                priority='high',
            )
            doctor_notifications.append(doctor_notification)
        
        # Send immediately with all channels
        success = self.send_notification(patient_notification, ['push', 'email', 'web_push'])
        
        for doc_notif in doctor_notifications:
            success &= self.send_notification(doc_notif, ['push', 'email', 'web_push'])
        
        return success
    
    def send_anomaly_notification(self, patient: PatientProfile, measurement_data: Dict[str, Any]) -> bool:
        """Send notification for detected anomaly"""
        notification = Notification.objects.create(
            recipient=patient.user,
            title="⚠️ Anomalie Détectée",
            message=f"Une anomalie a été détectée dans vos mesures de {measurement_data.get('type', 'santé')}",
            notification_type='anomaly',
            priority='medium',
        )
        
        # Also notify doctor if assigned
        if patient.assigned_doctor:
            doctor_notification = Notification.objects.create(
                recipient=patient.assigned_doctor.user,
                title=f"⚠️ Anomalie - {patient.user.get_full_name()}",
                message=f"Anomalie détectée dans les mesures de {measurement_data.get('type', 'santé')}",
                notification_type='anomaly',
                priority='medium',
            )
            return self.send_notification(notification, ['push']) and self.send_notification(doctor_notification, ['push'])
        
        return self.send_notification(notification, ['push'])


# Global notification service instance
notification_service = NotificationService()


@shared_task(bind=True, max_retries=3)
def send_notification_async(self, notification_id: str, channels: List[str] = None):
    """Async task to send notification"""
    try:
        notification = Notification.objects.get(id=notification_id)
        notification_service.send_notification(notification, channels)
    except Notification.DoesNotExist:
        logger.error(f"Notification {notification_id} not found")
    except Exception as exc:
        logger.error(f"Failed to send notification {notification_id}: {exc}")
        raise self.retry(exc=exc, countdown=60)


@shared_task
def process_notification_queue():
    """Process queued notifications"""
    now = timezone.now()
    queue_items = NotificationQueue.objects.filter(
        scheduled_at__lte=now,
        status='pending'
    )
    
    for queue_item in queue_items:
        try:
            success = notification_service.send_notification(
                queue_item.notification
            )
            
            queue_item.status = 'sent' if success else 'failed'
            queue_item.processed_at = now
            queue_item.save()
            
        except Exception as e:
            logger.error(f"Failed to process queued notification {queue_item.id}: {e}")
            queue_item.status = 'failed'
            queue_item.processed_at = now
            queue_item.save()


@shared_task
def send_daily_health_reminders():
    """Send daily health reminders to patients"""
    patients = PatientProfile.objects.filter(
        user__is_active=True
    )
    
    for patient in patients:
        notification = Notification.objects.create(
            recipient=patient.user,
            title="📊 Rappel Santé Quotidien",
            message="N'oubliez pas de prendre vos mesures aujourd'hui!",
            notification_type='reminder',
            priority='low',
        )
        
        # Queue for morning delivery
        notification_service.queue_notification(
            notification,
            timezone.now().replace(hour=9, minute=0, second=0)
        )


# Signal handlers for automatic notifications
def on_critical_alert_created(alert):
    """Handle critical alert creation"""
    from medical_data.models import Alert
    
    if isinstance(alert, Alert) and alert.severity == 'critical':
        patient = alert.patient
        alert_data = {
            'message': alert.message,
            'type': alert.alert_type,
        }
        notification_service.send_critical_alert(patient, alert_data)


def on_medical_measurement_created(measurement):
    """Handle new medical measurement creation"""
    from medical_data.models import MedicalMeasurement
    
    if isinstance(measurement, MedicalMeasurement):
        # Check for anomaly
        if measurement.anomaly_status in ['anomaly', 'critical']:
            patient = measurement.patient
            measurement_data = {
                'type': measurement.get_measurement_type_display(),
                'value': measurement.value,
                'anomaly_status': measurement.anomaly_status,
            }
            notification_service.send_anomaly_notification(patient, measurement_data)
