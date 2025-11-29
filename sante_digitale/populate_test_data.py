# populate_test_data.py
import os
import sys
import django
from datetime import datetime
import random

# AJOUTE LE DOSSIER RACINE DU PROJET DANS LE PYTHONPATH
# (obligatoire pour que Django trouve Dawini2025.settings)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Active l'environnement Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sante_digitale.settings')
django.setup()

# Maintenant on peut importer les modèles
from django.contrib.auth.models import User
from api.models import PatientProfile, Alert
from pymongo import MongoClient

print("Django configuré – démarrage du remplissage des données...")

# ==================== CONNEXION MONGO ====================
client = MongoClient('mongodb://localhost:27017')
db = client['dawini_db']
col = db['medical_records']

# Optionnel : nettoyer les anciennes données
# col.delete_many({})

print("Création des utilisateurs...")

# ==================== UTILISATEURS DE TEST ====================
users_data = [
    {"username": "patient1", "email": "patient1@dawini.com", "password": "test1234", "first_name": "Ahmed", "last_name": "Benali", "role": "patient"},
    {"username": "patient2", "email": "patient2@dawini.com", "password": "test1234", "first_name": "Fatima", "last_name": "Zahra", "role": "patient"},
    {"username": "doctor1",  "email": "dr.smith@dawini.com", "password": "test1234", "first_name": "Dr Sophie", "last_name": "Martin", "role": "doctor"},
    {"username": "admin",    "email": "admin@dawini.com",    "password": "admin123",  "first_name": "Admin",   "last_name": "Dawini",  "role": "admin"},
]

users = {}
for data in users_data:
    user, created = User.objects.get_or_create(
        username=data["username"],
        defaults={
            "email": data["email"],
            "first_name": data["first_name"],
            "last_name": data["last_name"],
        }
    )
    if created or not user.check_password(data["password"]):
        user.set_password(data["password"])
        user.save()

    profile, _ = PatientProfile.objects.update_or_create(
        user=user,
        defaults={"role": data["role"]}
    )

    if data["role"] in ["doctor", "admin"]:
        user.is_staff = True
        user.is_superuser = data["role"] == "admin"
        user.save()

    users[data["username"]] = user
    print(f"Utilisateur : {user.username} ({profile.role})")

print("Génération de 400 relevés médicaux...")

# ==================== DONNÉES RÉALISTES ====================
def generate_record(username):
    now = datetime.utcnow()
    days_ago = random.randint(0, 89)
    recorded_at = now - datetime.timedelta(
        days=days_ago,
        hours=random.randint(0, 23),
        minutes=random.choice([0, 15, 30, 45])
    )

    # Valeurs normales
    systolic = random.gauss(120, 18)
    diastolic = random.gauss(80, 12)
    glucose = random.gauss(95, 22)
    heart_rate = random.gauss(72, 14)

    # 10% d'anomalies
    if random.random() < 0.10:
        anomaly = random.choice([
            lambda: (systolic + random.uniform(50, 90), diastolic + random.uniform(20, 50)),  # HTA
            lambda: (glucose + random.uniform(100, 250), glucose),  # Hyperglycémie
            lambda: (heart_rate + random.uniform(50, 90), heart_rate),  # Tachycardie
        ])
        if "HTA" in str(anomaly):
            systolic += 70
            diastolic += 30
        elif "Hyper" in str(anomaly):
            glucose += 180
        elif "Tachy" in str(anomaly):
            heart_rate += 70

    return {
        "patient_username": username,
        "systolic": round(max(70, min(220, systolic)), 1),
        "diastolic": round(max(40, min(140, diastolic)), 1),
        "glucose": round(max(50, min(400, glucose)), 1),
        "heart_rate": int(max(40, min(180, heart_rate))),
        "recorded_at": recorded_at,
        "meta": {
            "device": random.choice(["Omron M7", "Accu-Chek", "Apple Watch", "Withings"]),
            "note": random.choice(["", "Après repas", "Matin à jeun", "Après sport"])
        }
    }

records = []
for username in ["patient1", "patient2"]:
    for _ in range(200):
        records.append(generate_record(username))

result = col.insert_many(records)
print(f"400 relevés insérés !")

# ==================== ALERTES ====================
alerts = [
    {"patient": users["patient1"].patientprofile, "level": "high", "message": "Tension très élevée : 185/115 mmHg"},
    {"patient": users["patient1"].patientprofile, "level": "high", "message": "Glycémie critique : 342 mg/dL"},
    {"patient": users["patient2"].patientprofile, "level": "medium", "message": "Fréquence cardiaque élevée sur 3 mesures"},
]

for a in alerts:
    Alert.objects.create(**a)

print(f"{len(alerts)} alertes créées")
print("\nTOUT EST PRÊT !")
print("Connexions :")
print("   → Patient : patient1 / test1234")
print("   → Docteur : doctor1 / test1234")
print("   → Admin Django : admin / admin123")