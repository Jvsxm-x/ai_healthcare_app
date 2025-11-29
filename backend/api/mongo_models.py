# backend/api/mongo_models.py
from pymongo import MongoClient
from django.conf import settings
from datetime import datetime
import bcrypt

client = MongoClient(getattr(settings, 'MONGO_URI', 'mongodb://localhost:27017'))
db = client['dawini_db']

# Collections
users = db['users']
alerts = db['alerts']
records = db['medical_records']

# Index automatiques (avec gestion d'erreurs)
try:
    users.create_index("username", unique=True)
except:
    pass
try:
    users.create_index("email", unique=True)
except:
    pass
try:
    alerts.create_index("patient_username")
except:
    pass
try:
    alerts.create_index("created_at")
except:
    pass
try:
    records.create_index("patient_username")
except:
    pass
try:
    records.create_index("recorded_at")
except:
    pass
try:
    records.create_index([("patient_username", 1), ("recorded_at", -1)])
except:
    pass

class MongoUser:
    @staticmethod
    def create(username, email, password, first_name="", last_name="", role="patient"):
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_doc = {
            "username": username,
            "email": email,
            "password_hash": hashed,
            "first_name": first_name,
            "last_name": last_name,
            "profile": {
                "role": role,
                "birth_date": None,
                "phone": "",
            },
            "is_active": True,
            "is_staff": role in ["doctor", "admin"],
            "date_joined": datetime.utcnow()
        }
        result = users.insert_one(user_doc)
        return str(result.inserted_id)

    @staticmethod
    def get_by_username(username):
        return users.find_one({"username": username})

    @staticmethod
    def verify_password(stored_hash, password):
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)