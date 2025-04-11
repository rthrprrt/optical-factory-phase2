
**Fichier 2 : `README.md` (Finalisé)**

Remplace le contenu de `README.md` par ceci :

```markdown
# Optical Factory - Backend API (Phase 2)

[![Python Backend CI](https://github.com/rthrprrt/optical-factory-phase2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/rthrprrt/optical-factory-phase2/actions/workflows/python-ci.yml)

API Backend RESTful pour le projet de Master "Optical Factory", développée avec Python et FastAPI. Fournit des fonctionnalités pour l'analyse faciale, la recommandation de lunettes et l'essayage virtuel simplifié.

## Fonctionnalités Principales

*   **Analyse Faciale :** Extrait la pose 3D, les landmarks (470+) et une forme de visage simplifiée (long, proportionné, autre) à partir d'une image.
*   **Recommandation :** Suggère des modèles de lunettes adaptés à une forme de visage donnée (basée sur des règles simples).
*   **Rendu d'Essayage :** Superpose un modèle 3D de lunettes sélectionné sur une image de visage en utilisant la pose détectée.
*   **Flux Combiné :** Permet d'analyser une image et d'obtenir les recommandations associées en un seul appel.

## Architecture

L'application utilise FastAPI et suit une structure modulaire :
*   **API Endpoints (`src/api/endpoints.py`) :** Exposition des routes REST.
*   **Core Logic (`src/core/`) :** Contient la logique métier pour le traitement (`processing.py`), le rendu (`rendering.py`) et la gestion des modèles (`models.py`).
*   **Configuration (`src/core/config.py`) :** Gère les paramètres via un fichier `.env` (Pydantic-Settings).
*   **Schemas (`src/schemas/schemas.py`) :** Modèles Pydantic pour la validation des données API.

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour une description détaillée.

## Technologies Utilisées

*   Python 3.10+
*   FastAPI & Uvicorn
*   Mediapipe (FaceLandmarker)
*   OpenCV-Python
*   Pyrender & Trimesh
*   Pydantic & Pydantic-Settings
*   Pytest & HTTPX
*   Docker
*   Git & Git LFS
*   GitHub Actions

## Installation et Lancement

### Prérequis

*   Python 3.10+ et Pip
*   Git et Git LFS ([Instructions d'installation Git LFS](https://git-lfs.com/))
*   (Optionnel mais recommandé) Docker Desktop

### Lancement Local (Développement avec Uvicorn)

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/rthrprrt/optical-factory-phase2.git
    cd optical-factory-phase2
    ```
2.  **Installer les fichiers LFS :**
    ```bash
    git lfs pull
    ```
3.  **Créer un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    # Activer l'environnement
    # Windows (PowerShell): .\venv\Scripts\Activate.ps1
    # Linux/macOS: source venv/bin/activate
    ```
4.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Créer le fichier de configuration `.env` :**
    *   Copiez `.env.example` (si fourni) ou créez un fichier `.env` à la racine.
    *   Assurez-vous que les chemins (ex: `FACE_MODEL_PATH`) et autres variables sont corrects. Contenu minimal :
        ```dotenv
        # .env
        FACE_MODEL_PATH="./models/face_landmarker_v2_with_blendshapes.task"
        # Ajoutez d'autres variables de src/core/config.py si vous voulez les surcharger
        ```
6.  **Lancer le serveur FastAPI :**
    ```bash
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
    ```
7.  L'API est accessible à `http://localhost:8000`.
    *   Documentation interactive (Swagger UI) : `http://localhost:8000/docs`
    *   Health Check : `http://localhost:8000/health`

### Lancement via Docker

1.  **Assurez-vous que Docker Desktop est lancé.**
2.  **Cloner le dépôt et installer LFS** (voir étapes 1 et 2 ci-dessus).
3.  **Construire l'image Docker :** (Assurez-vous que le fichier `.env` existe pour la configuration par défaut si nécessaire lors du build ou du run)
    ```bash
    docker build -t optical-factory-backend .
    ```
4.  **Lancer le conteneur :**
    ```bash
    # Monte le fichier .env dans le conteneur pour la configuration au runtime
    docker run --rm -p 8000:8000 --env-file .env --name optical-factory-app optical-factory-backend
    # Ou sans --env-file si la config par défaut dans config.py est suffisante
    # docker run --rm -p 8000:8000 --name optical-factory-app optical-factory-backend
    ```
5.  L'API est accessible à `http://localhost:8000`.

## Tests

Les tests unitaires et d'intégration utilisent `pytest`.

1.  **Installer les dépendances de test** (incluses dans `requirements.txt`).
2.  **Créer les données de test :** Certains tests API nécessitent des images dans `tests/test_data/` (ex: `test_face.jpg`, `test_no_face.jpg`). Créez ce dossier et placez-y des images appropriées.
3.  **Exécuter les tests** depuis la racine du projet :
    ```bash
    pytest
    ```

## Benchmark

Un script permet d'évaluer la latence et la précision de détection.

1.  **Assurez-vous que le serveur API est lancé.**
2.  **Placez des images de test** variées dans le dossier `benchmark/test_data/`.
3.  **Exécutez le script** depuis la racine du projet en tant que module :
    ```bash
    python -m benchmark.optical_factory_evaluation
    ```
4.  Les résultats s'affichent et sont sauvegardés dans `benchmark/evaluation_results.json`. Les seuils sont configurés via `.env`/`config.py`.

## Intégration Continue (CI)

Un workflow GitHub Actions (`.github/workflows/python-ci.yml`) est configuré pour :
*   Installer les dépendances.
*   Récupérer les fichiers Git LFS.
*   Exécuter `pytest` à chaque push ou pull request sur les branches principales.