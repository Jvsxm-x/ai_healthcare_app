# --- Dockerfile : backend/Dockerfile ---
FROM python:3.11-slim

# Crée le dossier de travail
WORKDIR /app

# Copie les dépendances
COPY requirements.txt /app/

# Installation des dépendances système et Python
RUN apt-get update && apt-get install -y build-essential libpq-dev gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copie le code Django
COPY . /app/

# Expose le port du serveur Django
EXPOSE 8000

# Commande de démarrage
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
