# Documentation Technique - Backend API Optical Factory (Phase 2)

**Date :** 11 avril 2025
**Auteur :** Arthur Perrot - Groupe 3
**Version :** 1.0 (Après Refocalisation sur API d'Analyse)

## 1. Introduction et Objectifs

Ce document décrit l'architecture et les fonctionnalités du backend API développé dans le cadre de la Phase 2 du projet de Master "Optical Factory". L'objectif principal de ce backend est de fournir une interface programmatique robuste et testée pour :

* Analyser des images de visages afin d'en extraire des informations géométriques 3D clés (pose, landmarks).
* Effectuer une classification simplifiée de la forme du visage basée sur ces informations.
* Recommander des modèles de lunettes pertinents en fonction de la forme détectée.

Initialement, une fonctionnalité de rendu 3D côté backend avait été implémentée. Cependant, suite aux tests de performance (latence > 2 secondes par frame) et considérant les standards des applications d'essayage virtuel temps réel, la décision stratégique a été prise de déléguer la responsabilité du rendu 3D au client (frontend). Le backend se concentre donc sur la fourniture rapide et fiable des données d'analyse nécessaires au frontend pour effectuer ce rendu.

## 2. Architecture Générale

Le backend est une API RESTful développée en Python avec FastAPI, choisie pour sa performance, sa validation de données intégrée (Pydantic) et sa génération automatique de documentation OpenAPI (Swagger).

L'architecture est modulaire, séparant les préoccupations :

* **API Endpoints** (`src/api/endpoints.py`) : Interface HTTP, validation des entrées/sorties.
* **Core Logic** (`src/core/`) :
  * `processing.py` : Logique métier principale (analyse Mediapipe, classification forme, recommandation).
  * `models.py` : Chargement du modèle Mediapipe et gestion des métadonnées des modèles 3D.
* **Configuration** (`src/core/config.py`) : Gestion centralisée des paramètres via Pydantic-Settings et un fichier .env.
* **Schemas** (`src/schemas/schemas.py`) : Définition des structures de données API avec Pydantic.

L'application est conçue pour être conteneurisée via le Dockerfile fourni.

(Référence : Voir le fichier ARCHITECTURE.md pour un diagramme détaillé).

## 3. Fonctionnalités Principales (Endpoints API)

* **GET /health** : Vérifie la disponibilité de l'API et le chargement du modèle Mediapipe.
* **POST /api/v1/analyze_face** :
  * **Entrée** : Fichier image (multipart/form-data).
  * **Sortie** (JSON - FaceAnalysisResult) : `detection_successful` (bool), `facial_transformation_matrix` (liste 4x4 float), `face_landmarks` (liste d'objets {x, y, z}), `detected_face_shape` (string: "long", "proportionné", "autre", "inconnue", ou "erreur..."), `error_message` (string ou null). Fournit les données essentielles pour le rendu client.
* **POST /api/v1/recommend_glasses** :
  * **Entrée** (JSON - RecommendationRequest) : `{ "face_shape": "..." }`.
  * **Sortie** (JSON - RecommendationResult) : `recommended_glasses_ids` (liste de strings), `analysis_info` (string).
* **POST /api/v1/analyze_and_recommend** :
  * **Entrée** : Fichier image (multipart/form-data).
  * **Sortie** (JSON - AnalyzeAndRecommendResult) : Combine les sorties des deux endpoints précédents en un seul appel.

La documentation interactive et les détails des schémas sont disponibles sur l'endpoint `/docs` de l'API.

## 4. Alignement avec les Livrables Backend (Phase 2)

Ce backend répond aux livrables suivants :

* **Backend Structuré & API Documentée** (Livrable 3.2) : Réalisé via FastAPI, structure modulaire, et documentation OpenAPI auto-générée (`/docs`).
* **Architecture Documentée** (Livrable 3.2, 3.3) : Fournie via ARCHITECTURE.md et ce document.
* **Analyse de Performance (Précision/Latence)** (Livrable 3.1) : Mesurée via le script `benchmark/optical_factory_evaluation.py`. Résultats actuels (sur CPU, 150 images test) : Précision détection ~94%, Latence analyse ~2.1s. Le seuil de latence cible a été justifié et ajusté dans `config/evaluation_criteria.json` en raison des contraintes CPU.
* **Tests Fonctionnels/Intégration** (Livrable 3.1, 3.2) : Implémentés avec pytest (`tests/`), couvrant la logique métier et les endpoints API.
* **Configuration Externalisée & Conteneurisation** (Bonnes Pratiques) : Réalisé via `config.py`, `.env` et Dockerfile.
* **CI/CD** (Livrable 3.2) : Workflow GitHub Actions configuré pour exécuter les tests automatiquement.
* **Utilisation Modèle Vision** (Livrable 3.1) : Intégration de Mediapipe FaceLandmarker. La création d'architectures custom n'a pas été réalisée, privilégiant l'utilisation d'un modèle état de l'art existant.
* **Moteur de Recommandation** (Livrable 3.3) : Implémentation d'une classification de forme simplifiée et d'une logique de recommandation par règles. Les approches avancées (RL, Embeddings) n'ont pas été implémentées.

## 5. Limitations et Perspectives

* La classification de forme est basique et pourrait être améliorée (heuristiques fines ou modèle ML).
* La latence de l'analyse (~2s) peut être un frein pour des applications très réactives mais est acceptable pour une analyse à la demande.
* Le rendu 3D est entièrement délégué au client, qui doit implémenter la logique d'affichage en utilisant la matrice de pose et les landmarks fournis par cette API.

## 6. Conclusion

Le backend API fournit une base solide, testée et documentée pour les fonctionnalités d'analyse faciale et de recommandation du projet Optical Factory. Il expose les données 3D nécessaires pour permettre à l'équipe frontend de construire une expérience d'essayage virtuel interactive et performante.