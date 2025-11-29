# Dawini - Plateforme Santé Digitale (projet complet)

Ce dépôt contient le projet **Dawini** : backend (Django REST + Channels), frontend React (web + PWA mobile-friendly), ML pipeline, et configuration Docker (MongoDB, Redis).

## Contenu
- backend/: Django project
- frontend/: React app (web + mobile-friendly pages)
- infra/: Dockerfiles & docker-compose.yml
- docs/: guides et commandes utiles

## Démarrage (pas à pas)

Prérequis : Docker et Docker Compose (ou Podman) installés.

### 1) Lancer avec Docker (recommandé)
1. Ouvrir un terminal dans le dossier du projet (où se trouve docker-compose.yml).
2. Construire et lancer :
   ```bash
   docker-compose up --build
   ```
3. Services :
   - Backend Django (http://localhost:8000)
   - Frontend React (http://localhost:3000)
   - MongoDB (port 27017)
   - Redis (port 6379) pour WebSockets

### 2) Utilisation sans Docker (optionnel - pour dev local)
#### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Créer la DB Django (auth/admin)
python manage.py migrate
python manage.py createsuperuser
# Lancer (dev)
python manage.py runserver
```
> Note : Assure-toi que MongoDB et Redis tournent (ex : `sudo service mongod start` et `redis-server`).

#### Frontend
```bash
cd frontend
npm install
npm start
```

## MongoDB - configuration rapide
- Docker compose contient un service `mongo` configuré.
- Si tu veux utiliser une instance distante, modifie la variable d'environnement `MONGO_URI` dans `docker-compose.yml` ou `.env` (ex: `mongodb://user:pass@host:27017/dawini`).
- Le backend utilise `pymongo` pour stocker les mesures dans la collection `medical_records`.
- Pour administrer MongoDB localement, tu peux utiliser MongoDB Compass ou `mongosh`.

## Endpoints principaux (backend)
- `POST /api/auth/register/` : inscription (username, email, password, role)
- `POST /api/token/` : obtenir JWT (username, password)
- `GET /api/patients/` : lister profils (doctors/admin)
- `POST /api/records/` : créer mesure (stockée en Mongo)
- `GET /api/records/` : lister mesures (selon rôle)
- `POST /api/predict/` : endpoint d'inférence ML
- `POST /api/retrain/` : relancer l'entraînement du modèle (re-train) — sécurisé

## WebSockets (notifications en temps réel)
- WS endpoint : `ws://localhost:8000/ws/alerts/`
- Lorsqu'une alerte est créée (par ML), le backend crée un objet Alert (Django) **et** envoie un message websocket à tous les clients connectés.

## Remarques importantes
- Ce projet est un démonstrateur. N’utilise pas tel quel pour des données médicales sensibles en production (sécurité, chiffrement, conformité).
- Le modèle ML fourni est un exemple synthétique et doit être ré-entraîné sur des données réelles, avec validation.

---

Si tu veux que j'exécute des ajustements (ex: JWT expirations différentes, ajout d'un provider d'e-mail, génération de certificats HTTPS locaux), dis-moi lesquels ; je peux mettre à jour le dossier et te fournir une nouvelle archive.
"# ai_healthcare_app" 
