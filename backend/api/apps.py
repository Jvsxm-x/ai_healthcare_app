# backend/Dawini2025/apps.py   (ou backend/backend/apps.py selon ton nom de projet)
from django.apps import AppConfig

class DawiniConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'  # ou 'backend' selon ton projet

    def ready(self):
        """
        Cette fonction est exécutée AU DÉMARRAGE du serveur Django
        → On initialise MongoDB automatiquement
        """
        import api.mongo  # ← importe juste pour déclencher la création
        from api.mongo import get_mongo_collection
        try:
            col = get_mongo_collection()
            # Petite insertion de test pour forcer la création physique
            col.update_one(
                {"_id": "DAWINI_INIT_CHECK"},
                {"$setOnInsert": {"init": True, "created_at": __import__('datetime').datetime.utcnow()}},
                upsert=True
            )
            print("MongoDB dawini_db.medical_records prêt et indexé automatiquement !")
        except Exception as e:
            print(f"Attention MongoDB non accessible : {e}")