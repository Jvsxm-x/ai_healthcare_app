# backend/api/mongo.py
from pymongo import MongoClient
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def get_mongo_collection():
    """
    Retourne la collection medical_records
    Crée automatiquement : base de données + collection + index si inexistants
    """
    client = MongoClient(getattr(settings, 'MONGO_URI', 'mongodb://localhost:27017'))
    db = client['dawini_db']                   # ← base auto-créée à la première insertion
    collection = db['medical_records']         # ← collection auto-créée

    # Création des index UNE SEULE FOIS (idempotent = sans erreur si déjà existants)
    try:
        collection.create_index("patient_username", background=True)
        collection.create_index("recorded_at", background=True)
        collection.create_index(
            [("patient_username", 1), ("recorded_at", -1)],
            background=True,
            name="patient_recorded_desc"
        )
        collection.create_index(
            [("recorded_at", -1)],
            background=True,
            name="by_date_desc"
        )
        logger.info("Index MongoDB vérifiés/créés avec succès")
    except Exception as e:
        logger.warning(f"Index déjà existants ou erreur mineure : {e}")

    return collection