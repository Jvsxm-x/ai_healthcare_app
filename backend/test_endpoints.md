# Commandes curl pour tester les endpoints API Dawini

## Authentification

### 1. Register
```bash
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123",
    "email": "test@example.com",
    "first_name": "Test",
    "last_name": "User"
  }'
```
**Attendu**: 201 Created avec données utilisateur

### 2. Login
```bash
curl -X POST http://localhost:8000/api/token/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'
```
**Attendu**: 200 OK avec access_token et refresh_token

### 3. Refresh Token
```bash
curl -X POST http://localhost:8000/api/token/refresh/ \
  -H "Content-Type: application/json" \
  -d '{
    "refresh": "votre_refresh_token_ici"
  }'
```
**Attendu**: 200 OK avec nouvel access_token

## Profils Patients

### 4. List Patients
```bash
curl -X GET http://localhost:8000/api/patients/ \
  -H "Authorization: Bearer votre_access_token_ici"
```
**Attendu**: 200 OK avec liste des patients

### 5. Create Patient
```bash
curl -X POST http://localhost:8000/api/patients/ \
  -H "Authorization: Bearer votre_access_token_ici" \
  -H "Content-Type: application/json" \
  -d '{
    "date_of_birth": "1990-01-01",
    "phone_number": "+33612345678",
    "address": "123 Rue Test, 75001 Paris",
    "emergency_contact": "+33687654321",
    "medical_history": "Aucun antécédent"
  }'
```
**Attendu**: 201 Created avec données patient

### 6. Get Patient
```bash
curl -X GET http://localhost:8000/api/patients/{patient_id}/ \
  -H "Authorization: Bearer votre_access_token_ici"
```
**Attendu**: 200 OK avec détails patient

### 7. Update Patient
```bash
curl -X PUT http://localhost:8000/api/patients/{patient_id}/ \
  -H "Authorization: Bearer votre_access_token_ici" \
  -H "Content-Type: application/json" \
  -d '{
    "date_of_birth": "1990-01-01",
    "phone_number": "+33699999999",
    "address": "456 Rue Modifiée, 75002 Paris",
    "emergency_contact": "+33687654321",
    "medical_history": "Antécédents mis à jour"
  }'
```
**Attendu**: 200 OK avec patient mis à jour

### 8. Delete Patient
```bash
curl -X DELETE http://localhost:8000/api/patients/{patient_id}/ \
  -H "Authorization: Bearer votre_access_token_ici"
```
**Attendu**: 204 No Content

## Données Médicales

### 9. Get Records
```bash
curl -X GET http://localhost:8000/api/records/ \
  -H "Authorization: Bearer votre_access_token_ici"
```
**Attendu**: 200 OK avec liste des enregistrements médicaux

### 10. Create Record
```bash
curl -X POST http://localhost:8000/api/records/ \
  -H "Authorization: Bearer votre_access_token_ici" \
  -H "Content-Type: application/json" \
  -d '{
    "systolic": 120,
    "diastolic": 80,
    "glucose": 95,
    "heart_rate": 72,
    "notes": "Mesure normale"
  }'
```
**Attendu**: 201 Created avec données enregistrement

### 11. Get Records Stats
```bash
curl -X GET http://localhost:8000/api/records/stats/ \
  -H "Authorization: Bearer votre_access_token_ici"
```
**Attendu**: 200 OK avec statistiques (avg_systolic, avg_diastolic, avg_glucose, avg_heart_rate)

## Alertes

### 12. List Alerts
```bash
curl -X GET http://localhost:8000/api/alerts/ \
  -H "Authorization: Bearer votre_access_token_ici"
```
**Attendu**: 200 OK avec liste des alertes

### 13. Create Alert
```bash
curl -X POST http://localhost:8000/api/alerts/ \
  -H "Authorization: Bearer votre_access_token_ici" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tension artérielle élevée",
    "severity": "high",
    "is_read": false
  }'
```
**Attendu**: 201 Created avec données alerte

### 14. Get Alert
```bash
curl -X GET http://localhost:8000/api/alerts/{alert_id}/ \
  -H "Authorization: Bearer votre_access_token_ici"
```
**Attendu**: 200 OK avec détails alerte

### 15. Update Alert
```bash
curl -X PUT http://localhost:8000/api/alerts/{alert_id}/ \
  -H "Authorization: Bearer votre_access_token_ici" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tension artérielle élevée - mise à jour",
    "severity": "medium",
    "is_read": true
  }'
```
**Attendu**: 200 OK avec alerte mise à jour

### 16. Delete Alert
```bash
curl -X DELETE http://localhost:8000/api/alerts/{alert_id}/ \
  -H "Authorization: Bearer votre_access_token_ici"
```
**Attendu**: 204 No Content

## Machine Learning

### 17. Predict
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Authorization: Bearer votre_access_token_ici" \
  -H "Content-Type: application/json" \
  -d '{
    "systolic": 140,
    "diastolic": 90,
    "glucose": 110,
    "heart_rate": 85
  }'
```
**Attendu**: 200 OK avec prediction et confidence

### 18. Retrain
```bash
curl -X POST http://localhost:8000/api/retrain/ \
  -H "Authorization: Bearer votre_access_token_ici" \
  -H "Content-Type: application/json" \
  -d '{}'
```
**Attendu**: 200 OK avec message et accuracy

## Admin

### 19. Admin Interface
```bash
curl -X GET http://localhost:8000/admin/
```
**Attendu**: 200 OK ou 302 Redirect

## Instructions d'utilisation

1. **Remplacez** `votre_access_token_ici` par le token obtenu après login
2. **Remplacez** `votre_refresh_token_ici` par le refresh token obtenu après login
3. **Remplacez** `{patient_id}` par l'ID d'un patient existant
4. **Remplacez** `{alert_id}` par l'ID d'une alerte existante
5. **Exécutez** les commandes dans l'ordre recommandé (Register → Login → Create Patient → etc.)

## Séquence de test recommandée

1. Register (créer utilisateur)
2. Login (obtenir tokens)
3. Create Patient (créer profil patient)
4. Create Record (ajouter données médicales)
5. Create Alert (créer alerte)
6. Get Records Stats (vérifier statistiques)
7. Predict (tester ML)
8. Retrain (réentraîner modèle)
9. Test des autres endpoints GET/PUT/DELETE
