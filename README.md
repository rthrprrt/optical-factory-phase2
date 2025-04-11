# Optical Factory - Backend API (Phase 2)

[![Python Backend CI](https://github.com/rthrprrt/optical-factory-phase2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/rthrprrt/optical-factory-phase2/actions/workflows/python-ci.yml)

API Backend RESTful pour le projet de Master "Optical Factory", développée avec Python et FastAPI. Fournit des fonctionnalités pour l'**analyse faciale (pose 3D, landmarks, forme simplifiée)** et la **recommandation de lunettes** pour une application d'essayage virtuel. **Le rendu 3D est délégué au client (frontend).**

## Fonctionnalités Principales de l'API

*   **Analyse Faciale (`/api/v1/analyze_face`) :** Prend une image en entrée et retourne :
    *   Le succès de la détection.
    *   La matrice de transformation de pose 3D (4x4).
    *   Les coordonnées des landmarks faciaux (468+ points).
    *   Une classification simplifiée de la forme du visage (long, proportionné, autre).
    *   Un message d'erreur le cas échéant.
*   **Recommandation (`/api/v1/recommend_glasses`) :** Prend une forme de visage (string) en entrée et retourne une liste d'IDs de modèles de lunettes suggérés.
*   **Flux Combiné (`/api/v1/analyze_and_recommend`) :** Prend une image en entrée, effectue l'analyse, et retourne à la fois les résultats de l'analyse et les recommandations basées sur la forme détectée.
*   **(Optionnel) Liste des Modèles (`/api/v1/list_models`) :** Peut être ajouté pour lister les IDs des modèles 3D connus du backend.

## Architecture

L'application utilise FastAPI et suit une structure modulaire axée sur l'analyse et la recommandation. Le rendu 3D est exclu du backend.

*   **API Endpoints (`src/api/endpoints.py`) :** Exposition des routes REST (`/analyze_face`, `/recommend_glasses`, etc.).
*   **Core Logic (`src/core/`) :** Contient la logique métier pour le traitement (`processing.py` - analyse, forme, reco) et la gestion des modèles (`models.py` - chargement Mediapipe, liste IDs 3D).
*   **Configuration (`src/core/config.py`) :** Gère les paramètres via un fichier `.env`.
*   **Schemas (`src/schemas/schemas.py`) :** Modèles Pydantic pour la validation API.

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour une description détaillée de l'architecture backend finale.

## Technologies Utilisées (Backend)

*   Python 3.10+
*   FastAPI & Uvicorn
*   Mediapipe (FaceLandmarker)
*   OpenCV-Python (Headless)
*   Pydantic & Pydantic-Settings
*   Pytest & HTTPX
*   Docker
*   Git & Git LFS
*   GitHub Actions
*   NumPy, Pathlib

*(Note: Pyrender, Trimesh, Pillow ont été retirés)*

## Pour l'Équipe Frontend : Utilisation de l'API

1.  **Récupérer l'URL de l'API :** Obtenez l'URL du backend (soit `http://localhost:8000` si lancé localement, soit l'URL publique si déployé sur Render/autre).
2.  **Consulter la Documentation :** Allez sur `<URL_API>/docs` pour voir les détails des endpoints, les formats JSON attendus et les réponses.
3.  **Flux d'Essayage Virtuel Côté Client :**
    *   Capturez une image/frame de la webcam.
    *   Envoyez cette image à l'endpoint `POST /api/v1/analyze_face` (ou `/analyze_and_recommend`).
    *   Récupérez la réponse JSON. Vérifiez `detection_successful`.
    *   Si succès, extrayez :
        *   `facial_transformation_matrix` (matrice 4x4).
        *   `face_landmarks` (liste de {x, y, z}).
        *   `detected_face_shape` (string).
    *   **(Optionnel)** Appelez `POST /api/v1/recommend_glasses` avec la `detected_face_shape` pour obtenir les `recommended_glasses_ids`.
    *   **Chargez le modèle 3D** correspondant à l'ID souhaité (ex: un fichier `.glb` que vous hébergez ou récupérez).
    *   **Utilisez une bibliothèque 3D Web (Three.js, Babylon.js) :**
        *   Affichez le flux vidéo de la webcam en arrière-plan.
        *   Créez une scène 3D et une caméra virtuelle qui correspondent (approximativement) à la caméra réelle.
        *   Chargez le modèle 3D des lunettes.
        *   **Appliquez la `facial_transformation_matrix`** reçue de l'API à la position/rotation du modèle 3D des lunettes dans votre scène 3D. (Attention aux conversions de systèmes de coordonnées si nécessaire entre Mediapipe et votre moteur 3D). Vous devrez peut-être aussi appliquer un offset local au modèle 3D pour l'ajuster parfaitement (nez, oreilles).
        *   Effectuez le rendu de la scène 3D par-dessus la vidéo webcam.

## Installation et Lancement (Backend)

### Prérequis

*   Python 3.10+ et Pip
*   Git et Git LFS
*   (Optionnel) Docker Desktop

### Lancement Local (Développement avec Uvicorn)

1.  `git clone https://github.com/rthrprrt/optical-factory-phase2.git`
2.  `cd optical-factory-phase2`
3.  `git lfs pull`
4.  `python -m venv venv`
5.  `.\venv\Scripts\Activate.ps1` (ou `source venv/bin/activate`)
6.  `pip install -r requirements.txt` (Utilisez le `requirements.txt` **nettoyé**)
7.  Créez `.env` (voir `.env.example`). Assurez-vous que `FACE_MODEL_PATH` est correct.
8.  `uvicorn src.main:app --reload --host 0.0.0.0 --port 8000`
9.  Accédez à `http://localhost:8000/docs`.

### Lancement via Docker

1.  Assurez-vous que Docker est lancé.
2.  Clonez le dépôt et installez LFS (voir ci-dessus).
3.  Construisez l'image (utilisez le `Dockerfile` **nettoyé**) : `docker build -t optical-factory-backend .`
4.  Lancez le conteneur : `docker run --rm -p 8000:8000 --env-file .env --name optical-factory-app optical-factory-backend`
5.  Accédez à `http://localhost:8000`.

## Tests

1.  Installez les dépendances (`pip install -r requirements.txt`).
2.  Placez des images de test valides dans `tests/test_data/` si nécessaire (voir `tests/test_api.py`).
3.  Exécutez depuis la racine : `pytest`

## Benchmark

1.  Assurez-vous que le serveur API est lancé.
2.  Placez des images de test dans `benchmark/test_data/`.
3.  Exécutez depuis la racine : `python -m benchmark.optical_factory_evaluation`
4.  Résultats dans la console et `benchmark/evaluation_results.json`.

## Intégration Continue (CI)

Workflow GitHub Actions (`.github/workflows/python-ci.yml`) exécute `pytest`.