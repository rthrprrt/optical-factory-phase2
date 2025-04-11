# Utiliser une image Python officielle comme image de base
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances
# Utilise --no-cache-dir pour réduire la taille de l'image
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source de l'application
COPY ./src ./src
# Copier la configuration (si le .env n'est pas monté séparément)
COPY ./.env ./.env
# Copier les modèles ! <<< AJOUT IMPORTANT
COPY ./models ./models

# Exposer le port interne
EXPOSE 8000

# Commande pour lancer l'application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]