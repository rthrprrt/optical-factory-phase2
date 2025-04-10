# Architecture Système - Optical Factory API (Phase 2)

Ce document décrit l'architecture du backend API du projet Optical Factory.

## 1. Vue d'Ensemble

Le backend est une application **API RESTful** développée avec **FastAPI** (Python). Son rôle principal est d'analyser des images de visage pour en extraire des caractéristiques (pose 3D, landmarks, forme) et de fournir des recommandations de lunettes personnalisées. Il propose également un endpoint pour générer une image avec un modèle 3D de lunettes superposé au visage.

L'application est conçue pour être **conteneurisée avec Docker** et intégrée dans un pipeline **CI/CD (GitHub Actions)** pour les tests automatisés.

## 2. Composants Principaux

L'architecture actuelle est basée sur un service API unique (monolithe pour l'instant) structuré de manière modulaire :

```mermaid
graph TD
    A[Client API<br>(Ex: Frontend Web/Mobile, App Test, Benchmark)] --> B{FastAPI Backend API};
    B --> C[API Endpoints<br>(src/api/endpoints.py)];
    C -- Appelle --> D[Logique Métier Core<br>(src/core/processing.py)];
    C -- Appelle --> E[Logique Rendu 3D<br>(src/core/rendering.py)];
    D -- Utilise --> F[Modèles ML<br>(Mediapipe - src/core/models.py)];
    D -- Appelle --> G[Détermination Forme Visage];
    D -- Appelle --> H[Logique Recommandation];
    E -- Utilise --> F;
    E -- Utilise --> I[Modèles 3D<br>(Trimesh/Pyrender - src/core/models.py)];
    B -- Utilise --> J[Schemas Pydantic<br>(src/schemas/schemas.py)];
    F -- Chargé par --> K[Gestion Modèles<br>(src/core/models.py)];
    I -- Chargé par --> K;
    E -- Initialisé par --> L[Startup FastAPI<br>(src/main.py)];
    K -- Initialisé par --> L;
```

- **FastAPI Backend API** (`src/main.py`) : Point d'entrée principal, gère le cycle de vie de l'application (démarrage, arrêt), configure les routes et initialise les composants (modèles, renderer).
- **API Endpoints** (`src/api/endpoints.py`) : Définit les routes REST (`/analyze_face`, `/recommend_glasses`, `/render_glasses`, `/analyze_and_recommend`, `/health`) et gère la validation des requêtes/réponses via les schémas Pydantic.
- **Logique Métier Core** (`src/core/processing.py`) : Contient les fonctions principales pour :
  - L'analyse d'image avec Mediapipe (`analyze_face_from_image_bytes`).
  - La détermination (basique) de la forme du visage (`determine_face_shape`).
  - La génération de recommandations (`get_recommendations_for_face`, `get_recommendations_based_on_analysis`).
- **Logique Rendu 3D** (`src/core/rendering.py`) : Gère l'initialisation de Pyrender/Trimesh, le chargement des modèles 3D, et la superposition du modèle sur l'image de fond (`render_overlay`).
- **Gestion Modèles** (`src/core/models.py`) : Centralise le chargement du modèle Mediapipe (`get_face_landmarker`) et fournit les chemins vers les modèles 3D (`get_3d_model_path`).
- **Schemas Pydantic** (`src/schemas/schemas.py`) : Définit la structure des données pour les requêtes et réponses API, assurant la validation automatique.

## 3. Dépendances Clés

- **Framework API** : FastAPI, Uvicorn
- **Computer Vision** : Mediapipe, OpenCV-Python
- **Rendu 3D** : Pyrender, Trimesh
- **Data Validation** : Pydantic
- **Utilitaires** : NumPy
- **Tests** : Pytest, HTTPX (via TestClient)

## 4. Flux de Données (Exemple : `/analyze_and_recommend`)

1. Le client envoie une requête POST avec une image (multipart/form-data).
2. L'endpoint (`endpoints.py`) reçoit l'image, lit les bytes.
3. Appel à `analyze_face_from_image_bytes` (`processing.py`).
4. `processing.py` décode l'image (OpenCV), appelle Mediapipe pour obtenir pose+landmarks.
5. `processing.py` appelle `determine_face_shape` avec les landmarks.
6. `processing.py` retourne l'objet `FaceAnalysisResult` (contenant pose, landmarks, forme).
7. L'endpoint reçoit `FaceAnalysisResult`.
8. L'endpoint appelle `get_recommendations_based_on_analysis` (`processing.py`) avec le résultat de l'analyse.
9. `processing.py` utilise la forme détectée pour appeler `get_recommendations_for_face`.
10. `processing.py` retourne l'objet `RecommendationResult`.
11. L'endpoint construit l'objet final `AnalyzeAndRecommendResult` et le retourne en JSON.

## 5. Infrastructure et Déploiement (Actuel)

- **Conteneurisation** : Un Dockerfile est fourni pour packager l'application.
- **CI/CD** : Un workflow GitHub Actions (`python-ci.yml`) exécute pytest automatiquement.
- **Déploiement PaaS** : Non implémenté actuellement.

## 6. Points d'Amélioration Futurs

- Amélioration de l'algorithme de détermination de forme de visage.
- Optimisation de la latence d'inférence et de rendu.
- Amélioration de la qualité du rendu 3D (PBR, éclairage).
- Déploiement sur une plateforme PaaS.
- Externalisation de la configuration.
- Extension de la couverture des tests.
- (Optionnel) Migration vers une architecture microservices si la complexité augmente.