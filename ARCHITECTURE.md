# Architecture Système - Optical Factory API (Phase 2 - Focus Analyse)

Ce document décrit l'architecture du backend API RESTful développé pour le projet Optical Factory, Phase 2, après refocalisation sur les fonctionnalités d'analyse et de recommandation.

## 1. Vue d'Ensemble

Le système backend fournit une interface API pour l'**analyse d'images faciales** et la **recommandation de modèles de lunettes**. Il est développé en **Python** avec le framework **FastAPI**. Le **rendu 3D** pour l'essayage virtuel est **délégué au client** (frontend), qui utilise les données de pose et de landmarks fournies par cette API.

L'architecture vise la modularité et la maintenabilité. Elle est conçue pour être **conteneurisée avec Docker** et intègre un pipeline **CI/CD (GitHub Actions)** pour les tests automatisés.

## 2. Technologies Principales (Backend)

* **Framework API :** FastAPI
* **Serveur ASGI :** Uvicorn
* **Computer Vision / Analyse Faciale :** Mediapipe (FaceLandmarker)
* **Traitement d'Image :** OpenCV-Python (Headless)
* **Validation de Données :** Pydantic
* **Configuration :** Pydantic-Settings (via fichier `.env`)
* **Tests :** Pytest, FastAPI TestClient, HTTPX
* **Conteneurisation :** Docker
* **CI/CD :** GitHub Actions
* **Utilitaires :** NumPy, Pathlib

*(Pyrender, Trimesh, Pillow ont été retirés des dépendances backend)*

## 3. Structure Modulaire (Backend)

Le code source (`src/`) est organisé en modules distincts :

```mermaid
graph TD
    subgraph "Client API (Frontend Web/Mobile)"
        direction LR
        CLIENT[Client API<br>Gère Capture Vidéo & Rendu 3D]
    end

    subgraph "Backend API (Focus Analyse & Reco)"
        direction TB
        MAIN[main.py<br>Point d'Entrée, Startup/Lifespan]
        CONFIG[config.py<br>Gestion Configuration (.env)]
        ROUTER[api/endpoints.py<br>Routes API (/analyze_face, /recommend_glasses)]
        SCHEMAS[schemas/schemas.py<br>Modèles Pydantic (Entrée/Sortie JSON)]

        subgraph "Core Logic"
            PROCESSING[core/processing.py<br>Analyse Image, Forme Visage Simple, Recommandation]
            MODELS_MGMT[core/models.py<br>Chargement Modèle Mediapipe, Liste IDs Modèles 3D]
        end

        subgraph "Dépendances Externes"
            MEDIAPIPE[Mediapipe<br>FaceLandmarker]
            CV[OpenCV]
            MODELS_ASSETS[Fichiers Modèles<br>(models/*.task)]
        end

        MAIN --> ROUTER;
        MAIN -- Initialise --> MODELS_MGMT;
        MAIN -- Lit --> CONFIG;

        ROUTER -- Appelle --> PROCESSING;
        ROUTER -- Utilise --> SCHEMAS;

        PROCESSING -- Appelle --> MODELS_MGMT;
        PROCESSING -- Utilise --> MEDIAPIPE;
        PROCESSING -- Utilise --> CV;

        MODELS_MGMT -- Lit --> CONFIG;
        MODELS_MGMT -- Charge --> MODELS_ASSETS;
    end

    CLIENT -- Appelle --> ROUTER;
    CLIENT -- Utilise Données (Pose, Landmarks) pour Rendu --> CLIENT_RENDU[Moteur 3D Client<br>(Three.js, etc.)];
    CLIENT_RENDU -- Charge --> MODELS_3D[Modèles 3D<br>(.glb, .obj - Hébergés séparément?)]
```

### Composants Principaux :

* **main.py** : Orchestre le démarrage, initialise Mediapipe.
* **config.py** : Centralise la configuration.
* **api/endpoints.py** : Définit les endpoints `/analyze_face`, `/recommend_glasses`, `/analyze_and_recommend`, `/health`. Ne contient plus `/render_glasses`.
* **schemas/schemas.py** : Définit les structures JSON (incluant FaceAnalysisResult avec pose et landmarks).
* **core/models.py** : Charge Mediapipe, fournit la liste des IDs de modèles 3D disponibles. Ne charge plus les modèles 3D eux-mêmes.
* **core/processing.py** : Effectue l'analyse Mediapipe (pose, landmarks, forme simple) et la logique de recommandation. Ne contient plus de logique liée au rendu.
* **Suppression** : Le module core/rendering.py a été supprimé.

## 4. Flux de Données (Backend)

* **Analyse (/analyze_face)** : Image -> endpoints -> processing (Mediapipe -> Pose/Landmarks/Forme) -> endpoints -> Réponse JSON (FaceAnalysisResult).
* **Recommandation (/recommend_glasses)** : Forme (JSON) -> endpoints -> processing (Logique Reco Simple) -> endpoints -> Réponse JSON (RecommendationResult).
* **Flux Combiné (/analyze_and_recommend)** : Image -> endpoints -> processing (Analyse complète) -> endpoints -> processing (Reco) -> endpoints -> Réponse JSON (AnalyzeAndRecommendResult).

Le client utilise ensuite les données de FaceAnalysisResult (principalement facial_transformation_matrix) pour effectuer le rendu 3D de son côté.

## 5. Infrastructure et Tests

* **Conteneurisation** : Dockerfile (simplifié, sans dépendances de rendu).
* **CI/CD** : Workflow GitHub Actions (.github/workflows/python-ci.yml) exécute pytest.
* **Tests** : Suite pytest (tests/) validant la logique d'analyse/recommandation et les endpoints API correspondants. Les tests de rendu ont été supprimés.
* **Benchmarking** : Script (benchmark/) mesure la latence et la précision de l'analyse.

## 6. Limitations et Travail Futur (Backend)

* **Classification de Forme** : Logique simplifiée, améliorable par ML ou heuristiques fines.
* **Latence Analyse** : Mesurée à ~2s sur CPU. Optimisable si nécessaire (quantification, ONNX pour Mediapipe).
* **Robustesse Analyse** : Pas de gestion spécifique éclairage/occlusion.
* **Déploiement PaaS** : À effectuer.
* **Scalabilité** : Monolithe actuel.