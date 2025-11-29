from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(max_length=50, choices=[
        ('appointment', 'Appointment'),
        ('medication', 'Medication'),
        ('measurement', 'Measurement'),
        ('alert', 'Alert'),
        ('system', 'System'),
    ])
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.title}"

class NotificationTemplate(models.Model):
    name = models.CharField(max_length=100, unique=True)
    title_template = models.CharField(max_length=200)
    message_template = models.TextField()
    notification_type = models.CharField(max_length=50, choices=[
        ('appointment', 'Appointment'),
        ('medication', 'Medication'),
        ('measurement', 'Measurement'),
        ('alert', 'Alert'),
        ('system', 'System'),
    ])
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class NotificationPreference(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='notification_preferences')
    email_notifications = models.BooleanField(default=True)
    push_notifications = models.BooleanField(default=True)
    sms_notifications = models.BooleanField(default=False)
    appointment_reminders = models.BooleanField(default=True)
    medication_reminders = models.BooleanField(default=True)
    measurement_alerts = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - Preferences"

class NotificationQueue(models.Model):
    notification = models.ForeignKey(Notification, on_delete=models.CASCADE, related_name='queue_items')
    scheduled_at = models.DateTimeField(null=True, blank=True)
    is_sent = models.BooleanField(default=False)
    attempts = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['scheduled_at', '-created_at']

    def __str__(self):
        return f"Queue: {self.notification.title} - {self.scheduled_at}"

class NotificationLog(models.Model):
    notification = models.ForeignKey(Notification, on_delete=models.CASCADE, related_name='logs')
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ])
    channel = models.CharField(max_length=20, choices=[
        ('email', 'Email'),
        ('push', 'Push'),
        ('sms', 'SMS'),
        ('in_app', 'In App'),
    ])
    error_message = models.TextField(blank=True, null=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Log: {self.notification.title} - {self.status} via {self.channel}"
