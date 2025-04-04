# Utiliser une image Python officielle comme image de base
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances
# --no-cache-dir réduit la taille de l'image
# --upgrade pip s'assure qu'on a la dernière version de pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source de l'application dans le conteneur
COPY ./src ./src
# Si tu as des fichiers de configuration dans /config que tu veux inclure :
# COPY ./config ./config
# Si tu veux inclure les modèles directement dans l'image (peut la rendre grosse):
# COPY ./models ./models

# Exposer le port sur lequel l'application va tourner
EXPOSE 8000

# Commande pour lancer l'application quand le conteneur démarre
# Utilise 0.0.0.0 pour être accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]