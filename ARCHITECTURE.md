# Architecture Système - Optical Factory API (Phase 2)

Ce document décrit l'architecture du backend API RESTful développé pour le projet Optical Factory, Phase 2.

## 1. Vue d'Ensemble

Le système backend fournit une interface API pour l'analyse d'images faciales, la classification simplifiée de la forme du visage, la recommandation de modèles de lunettes et la génération d'images d'essayage virtuel. Il est développé en **Python** avec le framework **FastAPI**.

L'architecture vise la modularité et la maintenabilité. Elle est conçue pour être **conteneurisée avec Docker** et intègre un pipeline **CI/CD (GitHub Actions)** pour l'automatisation des tests unitaires et d'intégration.

## 2. Technologies Principales

* **Framework Backend :** FastAPI
* **Serveur ASGI :** Uvicorn
* **Computer Vision / Analyse Faciale :** Mediapipe (FaceLandmarker)
* **Traitement d'Image :** OpenCV-Python
* **Modélisation & Rendu 3D :** Trimesh (chargement/manipulation), Pyrender (rendu offscreen)
* **Validation de Données :** Pydantic
* **Configuration :** Pydantic-Settings (via fichier `.env`)
* **Tests :** Pytest, FastAPI TestClient, HTTPX
* **Conteneurisation :** Docker
* **CI/CD :** GitHub Actions
* **Utilitaires :** NumPy, Pathlib

## 3. Structure Modulaire

Le code source (`src/`) est organisé en modules distincts :

```mermaid
graph TD
    subgraph "Client (Web/Mobile/Test App)"
        direction LR
        CLIENT[Client API]
    end

    subgraph "Backend API (FastAPI Application)"
        direction TB
        MAIN[main.py<br>Point d'Entrée, Startup/Lifespan]
        CONFIG[config.py<br>Gestion Configuration (.env)]
        ROUTER[api/endpoints.py<br>Routes API, Validation Requêtes/Réponses]
        SCHEMAS[schemas/schemas.py<br>Modèles Pydantic]

        subgraph "Core Logic"
            PROCESSING[core/processing.py<br>Analyse Image, Forme Visage, Recommandation]
            RENDERING[core/rendering.py<br>Rendu 3D, Superposition]
            MODELS_MGMT[core/models.py<br>Chargement Modèles ML & 3D]
        end

        subgraph "Dépendances Externes"
            MEDIAPIPE[Mediapipe<br>FaceLandmarker]
            CV[OpenCV]
            PYRENDER[Pyrender/Trimesh]
            MODELS_ASSETS[Fichiers Modèles<br>(models/*.task, models/**/*.obj)]
        end

        MAIN --> ROUTER;
        MAIN -- Initialise --> MODELS_MGMT;
        MAIN -- Initialise --> RENDERING;
        MAIN -- Lit --> CONFIG;

        ROUTER -- Appelle --> PROCESSING;
        ROUTER -- Appelle --> RENDERING;
        ROUTER -- Utilise --> SCHEMAS;

        PROCESSING -- Appelle --> MODELS_MGMT;
        PROCESSING -- Utilise --> MEDIAPIPE;
        PROCESSING -- Utilise --> CV;

        RENDERING -- Appelle --> MODELS_MGMT;
        RENDERING -- Utilise --> PYRENDER;
        RENDERING -- Utilise --> CV;

        MODELS_MGMT -- Lit --> CONFIG;
        MODELS_MGMT -- Charge --> MODELS_ASSETS;
    end

    CLIENT --> ROUTER;
```

* **main.py** : Orchestre le démarrage, charge la configuration et initialise les composants critiques (Mediapipe via models.py, Pyrender via rendering.py).
* **config.py** : Centralise la configuration (chemins, seuils, etc.) lue depuis un fichier .env via Pydantic-Settings.
* **api/endpoints.py** : Définit tous les endpoints RESTful, gère la réception des requêtes HTTP, la validation des données d'entrée (fichiers, JSON) via Pydantic, appelle les fonctions de logique métier appropriées et formate les réponses HTTP.
* **schemas/schemas.py** : Contient les modèles Pydantic utilisés pour définir la structure attendue des requêtes et des réponses JSON, permettant la validation automatique et la génération de la documentation OpenAPI.
* **core/models.py** : Responsable du chargement et de la mise à disposition des modèles externes : le modèle FaceLandmarker de Mediapipe et les chemins d'accès aux différents modèles 3D de lunettes. Utilise la configuration (config.py) pour trouver les fichiers.
* **core/processing.py** : Cœur de la logique d'analyse. Prend les données d'image brute, utilise Mediapipe pour l'extraction des landmarks et de la matrice de pose, appelle la fonction de classification de forme (actuellement simplifiée), et contient la logique de recommandation basée sur la forme.
* **core/rendering.py** : Gère le rendu 3D. Initialise la scène Pyrender, charge les modèles 3D via Trimesh, positionne le modèle de lunettes sélectionné en fonction de la matrice de pose du visage et d'un offset prédéfini, effectue le rendu offscreen et le compose avec l'image de fond via alpha blending (OpenCV).

## 4. Flux de Données Principaux

* **Analyse (/analyze_face)** : Image -> endpoints -> processing (Mediapipe -> Landmarks/Pose -> Forme Simple) -> endpoints -> Réponse JSON (FaceAnalysisResult).
* **Recommandation Simple (/recommend_glasses)** : Forme (JSON) -> endpoints -> processing (Logique Reco Simple) -> endpoints -> Réponse JSON (RecommendationResult).
* **Rendu (/render_glasses)** : Image + ID Modèle (Form) -> endpoints -> processing (Analyse Pose) -> endpoints -> rendering (Chargement Modèle 3D + Rendu Pyrender + Composition OpenCV) -> endpoints -> Réponse Image Binaire (JPEG).
* **Flux Combiné (/analyze_and_recommend)** : Image -> endpoints -> processing (Analyse Pose/Landmarks/Forme) -> endpoints -> processing (Logique Reco) -> endpoints -> Réponse JSON (AnalyzeAndRecommendResult).

## 5. Infrastructure et Tests

* **Conteneurisation** : Dockerfile pour créer une image isolée de l'application.
* **CI/CD** : Workflow GitHub Actions (.github/workflows/python-ci.yml) déclenché sur push/pull request pour exécuter pytest. Gère Git LFS pour le modèle Mediapipe.
* **Tests** : Suite de tests (tests/) utilisant pytest :
  * Tests unitaires pour la logique de processing.py (classification de forme, recommandation) avec données simulées.
  * Tests d'intégration pour les endpoints API (test_api.py) utilisant FastAPI.TestClient.
* **Benchmarking** : Script (benchmark/optical_factory_evaluation.py) pour mesurer la latence et la précision de détection en appelant l'API sur un jeu de données. Génère un rapport JSON.

## 6. Points d'Amélioration et Limitations Actuelles

* **Classification de Forme** : La logique actuelle est très simplifiée (3 catégories). Une amélioration significative nécessiterait des heuristiques plus fines ou une approche basée sur le Machine Learning (nécessitant des données annotées).
* **Latence** : La latence mesurée pour le flux complet est de l'ordre de 2 secondes sur CPU, loin des performances temps réel (< 50ms). Des optimisations (quantification, ONNX, accélération matérielle) seraient nécessaires pour cet objectif. Le seuil actuel est ajusté pour refléter la performance du prototype.
* **Qualité du Rendu** : Le rendu 3D est fonctionnel mais basique. L'amélioration du photoréalisme nécessiterait un travail sur les matériaux (PBR), l'éclairage de la scène et potentiellement un moteur de rendu différent ou des shaders personnalisés.
* **Robustesse** : Pas de gestion spécifique pour les variations extrêmes d'éclairage ou les occlusions faciales importantes.
* **Déploiement** : L'application n'est pas encore déployée sur une plateforme PaaS.
* **Scalabilité** : L'architecture actuelle est monolithique. Pour une charge très élevée, une migration vers des microservices pourrait être envisagée.
* **Sécurité/RGPD** : Aspects non traités spécifiquement dans cette phase.