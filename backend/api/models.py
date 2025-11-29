from django.db import models
from django.contrib.auth.models import User

class PatientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    birth_date = models.DateField(null=True, blank=True)
    phone = models.CharField(max_length=30, blank=True)
    role = models.CharField(max_length=20, default='patient')  # patient / doctor / admin

    def __str__(self):
        return f"{self.user.username} ({self.role})"

class Alert(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    level = models.CharField(max_length=20)
    message = models.TextField()
    acknowledged = models.BooleanField(default=False)
    def __str__(self):
        return f"Alert({self.level} - {self.patient})"
