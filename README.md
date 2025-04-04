# Optical Factory - Backend API (Phase 2)

Backend API pour le projet Optical Factory, fournissant des services d'analyse faciale et de recommandation de lunettes.

## Prérequis

*   Docker

## Lancement Local (via Docker)

1.  **Construire l'image Docker :**
    ```bash
    docker build -t optical-factory-backend .
    ```

2.  **Lancer le conteneur :**
    ```bash
    docker run --rm -p 8000:8000 --name optical-factory-app optical-factory-backend
    ```

3.  L'API sera accessible à l'adresse `http://localhost:8000`.
    *   Documentation interactive (Swagger UI) : `http://localhost:8000/docs`
    *   Health Check : `http://localhost:8000/health`

## Structure du Projet (Backend)

*   `src/`: Code source principal de l'application FastAPI.
    *   `main.py`: Point d'entrée de l'application.
    *   `api/`: Modules définissant les endpoints API.
    *   `core/`: Logique métier principale (analyse, recommandation, etc.).
    *   `schemas/`: Modèles Pydantic pour la validation des données API.
    *   `utils/`: Fonctions utilitaires.
*   `models/`: Contient les modèles de Machine Learning (.task) et les modèles 3D (.obj, .glb). *Note: Ces modèles peuvent être volumineux et ne sont pas toujours inclus dans l'image Docker par défaut.*
*   `tests/`: Tests unitaires et d'intégration.
*   `benchmark/`: Scripts et données pour l'évaluation des performances.
*   `config/`: Fichiers de configuration.
*   `docs/`: Documentation additionnelle.
*   `Dockerfile`: Définit comment construire l'image Docker du backend.
*   `requirements.txt`: Dépendances Python.