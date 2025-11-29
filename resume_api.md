# Résumé API - Santé Digitale

Ce document liste tous les endpoints de l'API Santé Digitale, avec des exemples de corps de requête et de réponse.

## Base URL
`http://localhost:8000/api/`

## Authentification (`/auth/`)

### POST /api/auth/register/
**Description:** Inscription d'un nouvel utilisateur.

**Corps de requête (exemple):**
```json
{
  "email": "user@example.com",
  "password": "password123",
  "password_confirm": "password123",
  "role": "patient",
  "first_name": "John",
  "last_name": "Doe"
}
```

**Corps de réponse (exemple):**
```json
{
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "role": "patient",
    "is_active": true,
    "date_joined": "2023-10-01T00:00:00Z"
  }
}
```

### POST /api/auth/login/
**Description:** Connexion utilisateur.

**Corps de requête (exemple):**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Corps de réponse (exemple):**
```json
{
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "role": "patient",
    "is_active": true,
    "date_joined": "2023-10-01T00:00:00Z"
  }
}
```

### POST /api/auth/logout/
**Description:** Déconnexion (nécessite token).

**Corps de requête:** Aucun (utilise le token dans les headers).

**Corps de réponse (exemple):**
```json
{
  "message": "Successfully logged out"
}
```

### GET /api/auth/profile/
**Description:** Récupérer le profil utilisateur.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "role": "patient",
  "is_active": true,
  "date_joined": "2023-10-01T00:00:00Z"
}
```

### PATCH /api/auth/profile/
**Description:** Mettre à jour le profil utilisateur.

**Corps de requête (exemple):**
```json
{
  "first_name": "Jane"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "first_name": "Jane",
  "last_name": "Doe",
  "role": "patient",
  "is_active": true,
  "date_joined": "2023-10-01T00:00:00Z"
}
```

### POST /api/auth/change-password/
**Description:** Changer le mot de passe.

**Corps de requête (exemple):**
```json
{
  "old_password": "oldpassword123",
  "new_password": "newpassword123",
  "new_password_confirm": "newpassword123"
}
```

**Corps de réponse (exemple):**
```json
{
  "message": "Password changed successfully"
}
```

### POST /api/auth/password-reset/
**Description:** Réinitialiser le mot de passe.

**Corps de requête (exemple):**
```json
{
  "email": "user@example.com"
}
```

**Corps de réponse (exemple):**
```json
{
  "message": "Password reset email sent"
}
```

### GET /api/auth/doctors/
**Description:** Liste des médecins vérifiés.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 2,
    "email": "doctor@example.com",
    "first_name": "Dr. Smith",
    "last_name": "John",
    "role": "doctor",
    "is_active": true,
    "date_joined": "2023-09-01T00:00:00Z"
  }
]
```

### GET /api/auth/specialties/
**Description:** Liste des spécialités médicales.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {"value": "cardiology", "label": "Cardiologie"},
  {"value": "dermatology", "label": "Dermatologie"}
]
```

## Notifications (`/notifications/`)

### GET /api/notifications/notifications/
**Description:** Liste des notifications.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "user": 1,
    "title": "Rappel de rendez-vous",
    "message": "Vous avez un rendez-vous demain.",
    "notification_type": "appointment",
    "notification_type_display": "Appointment",
    "is_read": false,
    "created_at": "2023-10-01T10:00:00Z",
    "updated_at": "2023-10-01T10:00:00Z"
  }
]
```

### POST /api/notifications/notifications/
**Description:** Créer une notification.

**Corps de requête (exemple):**
```json
{
  "user": 1,
  "title": "Nouvelle notification",
  "message": "Ceci est une notification.",
  "notification_type": "info"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "user": 1,
  "title": "Nouvelle notification",
  "message": "Ceci est une notification.",
  "notification_type": "info",
  "notification_type_display": "Info",
  "is_read": false,
  "created_at": "2023-10-01T11:00:00Z",
  "updated_at": "2023-10-01T11:00:00Z"
}
```

### GET /api/notifications/notifications/{id}/
**Description:** Détails d'une notification.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "user": 1,
  "title": "Rappel de rendez-vous",
  "message": "Vous avez un rendez-vous demain.",
  "notification_type": "appointment",
  "notification_type_display": "Appointment",
  "is_read": false,
  "created_at": "2023-10-01T10:00:00Z",
  "updated_at": "2023-10-01T10:00:00Z"
}
```

### PUT /api/notifications/notifications/{id}/
**Description:** Mettre à jour une notification.

**Corps de requête (exemple):**
```json
{
  "is_read": true
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "user": 1,
  "title": "Rappel de rendez-vous",
  "message": "Vous avez un rendez-vous demain.",
  "notification_type": "appointment",
  "notification_type_display": "Appointment",
  "is_read": true,
  "created_at": "2023-10-01T10:00:00Z",
  "updated_at": "2023-10-01T11:00:00Z"
}
```

### PATCH /api/notifications/notifications/{id}/
**Description:** Mise à jour partielle d'une notification.

**Corps de requête (exemple):**
```json
{
  "is_read": true
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "user": 1,
  "title": "Rappel de rendez-vous",
  "message": "Vous avez un rendez-vous demain.",
  "notification_type": "appointment",
  "notification_type_display": "Appointment",
  "is_read": true,
  "created_at": "2023-10-01T10:00:00Z",
  "updated_at": "2023-10-01T11:00:00Z"
}
```

### DELETE /api/notifications/notifications/{id}/
**Description:** Supprimer une notification.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

### GET /api/notifications/templates/
**Description:** Liste des templates de notifications.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "name": "Appointment Reminder",
    "title_template": "Rappel de rendez-vous",
    "message_template": "Vous avez un rendez-vous le {date}.",
    "notification_type": "appointment",
    "notification_type_display": "Appointment",
    "is_active": true,
    "created_at": "2023-10-01T09:00:00Z",
    "updated_at": "2023-10-01T09:00:00Z"
  }
]
```

### POST /api/notifications/templates/
**Description:** Créer un template de notification.

**Corps de requête (exemple):**
```json
{
  "name": "New Template",
  "title_template": "Titre {variable}",
  "message_template": "Message {variable}",
  "notification_type": "info"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "name": "New Template",
  "title_template": "Titre {variable}",
  "message_template": "Message {variable}",
  "notification_type": "info",
  "notification_type_display": "Info",
  "is_active": true,
  "created_at": "2023-10-01T12:00:00Z",
  "updated_at": "2023-10-01T12:00:00Z"
}
```

### GET /api/notifications/templates/{id}/
**Description:** Détails d'un template.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "name": "Appointment Reminder",
  "title_template": "Rappel de rendez-vous",
  "message_template": "Vous avez un rendez-vous le {date}.",
  "notification_type": "appointment",
  "notification_type_display": "Appointment",
  "is_active": true,
  "created_at": "2023-10-01T09:00:00Z",
  "updated_at": "2023-10-01T09:00:00Z"
}
```

### PUT /api/notifications/templates/{id}/
**Description:** Mettre à jour un template.

**Corps de requête (exemple):**
```json
{
  "is_active": false
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "name": "Appointment Reminder",
  "title_template": "Rappel de rendez-vous",
  "message_template": "Vous avez un rendez-vous le {date}.",
  "notification_type": "appointment",
  "notification_type_display": "Appointment",
  "is_active": false,
  "created_at": "2023-10-01T09:00:00Z",
  "updated_at": "2023-10-01T12:00:00Z"
}
```

### PATCH /api/notifications/templates/{id}/
**Description:** Mise à jour partielle d'un template.

**Corps de requête (exemple):**
```json
{
  "is_active": false
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "name": "Appointment Reminder",
  "title_template": "Rappel de rendez-vous",
  "message_template": "Vous avez un rendez-vous le {date}.",
  "notification_type": "appointment",
  "notification_type_display": "Appointment",
  "is_active": false,
  "created_at": "2023-10-01T09:00:00Z",
  "updated_at": "2023-10-01T12:00:00Z"
}
```

### DELETE /api/notifications/templates/{id}/
**Description:** Supprimer un template.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

### GET /api/notifications/preferences/
**Description:** Liste des préférences de notifications.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "user": 1,
    "email_notifications": true,
    "push_notifications": true,
    "sms_notifications": false,
    "appointment_reminders": true,
    "medication_reminders": true,
    "measurement_alerts": true,
    "created_at": "2023-10-01T08:00:00Z",
    "updated_at": "2023-10-01T08:00:00Z"
  }
]
```

### POST /api/notifications/preferences/
**Description:** Créer des préférences de notifications.

**Corps de requête (exemple):**
```json
{
  "user": 1,
  "email_notifications": true,
  "push_notifications": false,
  "sms_notifications": true,
  "appointment_reminders": true,
  "medication_reminders": false,
  "measurement_alerts": true
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "user": 1,
  "email_notifications": true,
  "push_notifications": false,
  "sms_notifications": true,
  "appointment_reminders": true,
  "medication_reminders": false,
  "measurement_alerts": true,
  "created_at": "2023-10-01T13:00:00Z",
  "updated_at": "2023-10-01T13:00:00Z"
}
```

### GET /api/notifications/preferences/{id}/
**Description:** Détails des préférences.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "user": 1,
  "email_notifications": true,
  "push_notifications": true,
  "sms_notifications": false,
  "appointment_reminders": true,
  "medication_reminders": true,
  "measurement_alerts": true,
  "created_at": "2023-10-01T08:00:00Z",
  "updated_at": "2023-10-01T08:00:00Z"
}
```

### PUT /api/notifications/preferences/{id}/
**Description:** Mettre à jour les préférences.

**Corps de requête (exemple):**
```json
{
  "push_notifications": false
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "user": 1,
  "email_notifications": true,
  "push_notifications": false,
  "sms_notifications": false,
  "appointment_reminders": true,
  "medication_reminders": true,
  "measurement_alerts": true,
  "created_at": "2023-10-01T08:00:00Z",
  "updated_at": "2023-10-01T13:00:00Z"
}
```

### PATCH /api/notifications/preferences/{id}/
**Description:** Mise à jour partielle des préférences.

**Corps de requête (exemple):**
```json
{
  "push_notifications": false
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "user": 1,
  "email_notifications": true,
  "push_notifications": false,
  "sms_notifications": false,
  "appointment_reminders": true,
  "medication_reminders": true,
  "measurement_alerts": true,
  "created_at": "2023-10-01T08:00:00Z",
  "updated_at": "2023-10-01T13:00:00Z"
}
```

### DELETE /api/notifications/preferences/{id}/
**Description:** Supprimer les préférences.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

## Doctor Dashboard (`/doctor/`)

### GET /api/doctor/dashboard/
**Description:** Liste du tableau de bord médecin.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "doctor": 2,
    "patient_count": 50,
    "appointment_count": 20,
    "created_at": "2023-10-01T07:00:00Z"
  }
]
```

### POST /api/doctor/dashboard/
**Description:** Créer une entrée tableau de bord.

**Corps de requête (exemple):**
```json
{
  "doctor": 2,
  "patient_count": 50,
  "appointment_count": 20
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "doctor": 2,
  "patient_count": 50,
  "appointment_count": 20,
  "created_at": "2023-10-01T14:00:00Z"
}
```

### GET /api/doctor/dashboard/{id}/
**Description:** Détails du tableau de bord.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "doctor": 2,
  "patient_count": 50,
  "appointment_count": 20,
  "created_at": "2023-10-01T07:00:00Z"
}
```

### PUT /api/doctor/dashboard/{id}/
**Description:** Mettre à jour le tableau de bord.

**Corps de requête (exemple):**
```json
{
  "appointment_count": 25
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "doctor": 2,
  "patient_count": 50,
  "appointment_count": 25,
  "created_at": "2023-10-01T07:00:00Z"
}
```

### PATCH /api/doctor/dashboard/{id}/
**Description:** Mise à jour partielle du tableau de bord.

**Corps de requête (exemple):**
```json
{
  "appointment_count": 25
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "doctor": 2,
  "patient_count": 50,
  "appointment_count": 25,
  "created_at": "2023-10-01T07:00:00Z"
}
```

### DELETE /api/doctor/dashboard/{id}/
**Description:** Supprimer une entrée tableau de bord.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

## Medical Data (`/medical/`)

### GET /api/medical/measurements/
**Description:** Liste des mesures médicales.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "patient": 1,
    "measurement_type": "blood_pressure",
    "value": "120/80",
    "unit": "mmHg",
    "measured_at": "2023-10-01T06:00:00Z",
    "created_at": "2023-10-01T06:00:00Z"
  }
]
```

### POST /api/medical/measurements/
**Description:** Créer une mesure médicale.

**Corps de requête (exemple):**
```json
{
  "patient": 1,
  "measurement_type": "blood_pressure",
  "value": "120/80",
  "unit": "mmHg",
  "measured_at": "2023-10-01T06:00:00Z"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "patient": 1,
  "measurement_type": "blood_pressure",
  "value": "120/80",
  "unit": "mmHg",
  "measured_at": "2023-10-01T06:00:00Z",
  "created_at": "2023-10-01T15:00:00Z"
}
```

### GET /api/medical/measurements/{id}/
**Description:** Détails d'une mesure.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "measurement_type": "blood_pressure",
  "value": "120/80",
  "unit": "mmHg",
  "measured_at": "2023-10-01T06:00:00Z",
  "created_at": "2023-10-01T06:00:00Z"
}
```

### PUT /api/medical/measurements/{id}/
**Description:** Mettre à jour une mesure.

**Corps de requête (exemple):**
```json
{
  "value": "130/85"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "measurement_type": "blood_pressure",
  "value": "130/85",
  "unit": "mmHg",
  "measured_at": "2023-10-01T06:00:00Z",
  "created_at": "2023-10-01T06:00:00Z"
}
```

### PATCH /api/medical/measurements/{id}/
**Description:** Mise à jour partielle d'une mesure.

**Corps de requête (exemple):**
```json
{
  "value": "130/85"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "measurement_type": "blood_pressure",
  "value": "130/85",
  "unit": "mmHg",
  "measured_at": "2023-10-01T06:00:00Z",
  "created_at": "2023-10-01T06:00:00Z"
}
```

### DELETE /api/medical/measurements/{id}/
**Description:** Supprimer une mesure.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

### GET /api/medical/documents/
**Description:** Liste des documents médicaux.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "patient": 1,
    "document_type": "prescription",
    "title": "Prescription Antibiotiques",
    "file": "path/to/file.pdf",
    "uploaded_at": "2023-10-01T05:00:00Z",
    "created_at": "2023-10-01T05:00:00Z"
  }
]
```

### POST /api/medical/documents/
**Description:** Créer un document médical.

**Corps de requête (exemple):**
```json
{
  "patient": 1,
  "document_type": "prescription",
  "title": "Prescription Antibiotiques",
  "file": "path/to/file.pdf"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "patient": 1,
  "document_type": "prescription",
  "title": "Prescription Antibiotiques",
  "file": "path/to/file.pdf",
  "uploaded_at": "2023-10-01T16:00:00Z",
  "created_at": "2023-10-01T16:00:00Z"
}
```

### GET /api/medical/documents/{id}/
**Description:** Détails d'un document.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "document_type": "prescription",
  "title": "Prescription Antibiotiques",
  "file": "path/to/file.pdf",
  "uploaded_at": "2023-10-01T05:00:00Z",
  "created_at": "2023-10-01T05:00:00Z"
}
```

### PUT /api/medical/documents/{id}/
**Description:** Mettre à jour un document.

**Corps de requête (exemple):**
```json
{
  "title": "Prescription Mise à Jour"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "document_type": "prescription",
  "title": "Prescription Mise à Jour",
  "file": "path/to/file.pdf",
  "uploaded_at": "2023-10-01T05:00:00Z",
  "created_at": "2023-10-01T05:00:00Z"
}
```

### PATCH /api/medical/documents/{id}/
**Description:** Mise à jour partielle d'un document.

**Corps de requête (exemple):**
```json
{
  "title": "Prescription Mise à Jour"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "document_type": "prescription",
  "title": "Prescription Mise à Jour",
  "file": "path/to/file.pdf",
  "uploaded_at": "2023-10-01T05:00:00Z",
  "created_at": "2023-10-01T05:00:00Z"
}
```

### DELETE /api/medical/documents/{id}/
**Description:** Supprimer un document.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

### GET /api/medical/medications/
**Description:** Liste des médicaments.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "patient": 1,
    "name": "Aspirine",
    "dosage": "100mg",
    "frequency": "once daily",
    "start_date": "2023-10-01",
    "end_date": "2023-10-10",
    "created_at": "2023-10-01T04:00:00Z"
  }
]
```

### POST /api/medical/medications/
**Description:** Créer un médicament.

**Corps de requête (exemple):**
```json
{
  "patient": 1,
  "name": "Aspirine",
  "dosage": "100mg",
  "frequency": "once daily",
  "start_date": "2023-10-01",
  "end_date": "2023-10-10"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "patient": 1,
  "name": "Aspirine",
  "dosage": "100mg",
  "frequency": "once daily",
  "start_date": "2023-10-01",
  "end_date": "2023-10-10",
  "created_at": "2023-10-01T17:00:00Z"
}
```

### GET /api/medical/medications/{id}/
**Description:** Détails d'un médicament.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "name": "Aspirine",
  "dosage": "100mg",
  "frequency": "once daily",
  "start_date": "2023-10-01",
  "end_date": "2023-10-10",
  "created_at": "2023-10-01T04:00:00Z"
}
```

### PUT /api/medical/medications/{id}/
**Description:** Mettre à jour un médicament.

**Corps de requête (exemple):**
```json
{
  "dosage": "200mg"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "name": "Aspirine",
  "dosage": "200mg",
  "frequency": "once daily",
  "start_date": "2023-10-01",
  "end_date": "2023-10-10",
  "created_at": "2023-10-01T04:00:00Z"
}
```

### PATCH /api/medical/medications/{id}/
**Description:** Mise à jour partielle d'un médicament.

**Corps de requête (exemple):**
```json
{
  "dosage": "200mg"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "name": "Aspirine",
  "dosage": "200mg",
  "frequency": "once daily",
  "start_date": "2023-10-01",
  "end_date": "2023-10-10",
  "created_at": "2023-10-01T04:00:00Z"
}
```

### DELETE /api/medical/medications/{id}/
**Description:** Supprimer un médicament.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

### GET /api/medical/appointments/
**Description:** Liste des rendez-vous médicaux.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "patient": 1,
    "doctor": 2,
    "appointment_date": "2023-10-05T10:00:00Z",
    "reason": "Consultation générale",
    "status": "scheduled",
    "created_at": "2023-10-01T03:00:00Z"
  }
]
```

### POST /api/medical/appointments/
**Description:** Créer un rendez-vous.

**Corps de requête (exemple):**
```json
{
  "patient": 1,
  "doctor": 2,
  "appointment_date": "2023-10-05T10:00:00Z",
  "reason": "Consultation générale",
  "status": "scheduled"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "patient": 1,
  "doctor": 2,
  "appointment_date": "2023-10-05T10:00:00Z",
  "reason": "Consultation générale",
  "status": "scheduled",
  "created_at": "2023-10-01T18:00:00Z"
}
```

### GET /api/medical/appointments/{id}/
**Description:** Détails d'un rendez-vous.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "doctor": 2,
  "appointment_date": "2023-10-05T10:00:00Z",
  "reason": "Consultation générale",
  "status": "scheduled",
  "created_at": "2023-10-01T03:00:00Z"
}
```

### PUT /api/medical/appointments/{id}/
**Description:** Mettre à jour un rendez-vous.

**Corps de requête (exemple):**
```json
{
  "status": "completed"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "doctor": 2,
  "appointment_date": "2023-10-05T10:00:00Z",
  "reason": "Consultation générale",
  "status": "completed",
  "created_at": "2023-10-01T03:00:00Z"
}
```

### PATCH /api/medical/appointments/{id}/
**Description:** Mise à jour partielle d'un rendez-vous.

**Corps de requête (exemple):**
```json
{
  "status": "completed"
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "doctor": 2,
  "appointment_date": "2023-10-05T10:00:00Z",
  "reason": "Consultation générale",
  "status": "completed",
  "created_at": "2023-10-01T03:00:00Z"
}
```

### DELETE /api/medical/appointments/{id}/
**Description:** Supprimer un rendez-vous.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

### GET /api/medical/alerts/
**Description:** Liste des alertes médicales.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "patient": 1,
    "alert_type": "high_blood_pressure",
    "message": "Tension artérielle élevée détectée",
    "severity": "high",
    "is_active": true,
    "created_at": "2023-10-01T02:00:00Z"
  }
]
```

### POST /api/medical/alerts/
**Description:** Créer une alerte.

**Corps de requête (exemple):**
```json
{
  "patient": 1,
  "alert_type": "high_blood_pressure",
  "message": "Tension artérielle élevée détectée",
  "severity": "high",
  "is_active": true
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 2,
  "patient": 1,
  "alert_type": "high_blood_pressure",
  "message": "Tension artérielle élevée détectée",
  "severity": "high",
  "is_active": true,
  "created_at": "2023-10-01T19:00:00Z"
}
```

### GET /api/medical/alerts/{id}/
**Description:** Détails d'une alerte.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "alert_type": "high_blood_pressure",
  "message": "Tension artérielle élevée détectée",
  "severity": "high",
  "is_active": true,
  "created_at": "2023-10-01T02:00:00Z"
}
```

### PUT /api/medical/alerts/{id}/
**Description:** Mettre à jour une alerte.

**Corps de requête (exemple):**
```json
{
  "is_active": false
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "alert_type": "high_blood_pressure",
  "message": "Tension artérielle élevée détectée",
  "severity": "high",
  "is_active": false,
  "created_at": "2023-10-01T02:00:00Z"
}
```

### PATCH /api/medical/alerts/{id}/
**Description:** Mise à jour partielle d'une alerte.

**Corps de requête (exemple):**
```json
{
  "is_active": false
}
```

**Corps de réponse (exemple):**
```json
{
  "id": 1,
  "patient": 1,
  "alert_type": "high_blood_pressure",
  "message": "Tension artérielle élevée détectée",
  "severity": "high",
  "is_active": false,
  "created_at": "2023-10-01T02:00:00Z"
}
```

### DELETE /api/medical/alerts/{id}/
**Description:** Supprimer une alerte.

**Corps de requête:** Aucun.

**Corps de réponse:** 204 No Content

### GET /api/medical/history/
**Description:** Liste de l'historique médical.

**Corps de requête:** Aucun.

**Corps de réponse (exemple):**
```json
[
  {
    "id": 1,
    "patient": 1,
    "condition": "Hypertension",
    "diagnosis_date": "2023-09-01",
    "notes": "Diagnostiqué lors de la visite annuelle",
    "
