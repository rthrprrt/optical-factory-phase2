# src/core/processing.py

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarkerResult
from src.core.models import get_face_landmarker
from src.schemas.schemas import FaceAnalysisResult, Landmark, RecommendationResult
from typing import List, Optional, Tuple
import logging
import math

# Utilise le logger configuré au niveau racine (ou via settings si importé)
logger = logging.getLogger(__name__)

# --- Indices des Landmarks (Seulement ceux nécessaires pour L/W) ---
TOP_FOREHEAD = 10
BOTTOM_CHIN = 152
LEFT_TEMPLE = 234 # Point externe pommette gauche
RIGHT_TEMPLE = 454 # Point externe pommette droite

# --- Fonction Distance ---
def distance(p1: Optional[Landmark], p2: Optional[Landmark]) -> float:
    """ Calcule la distance Euclidienne 2D entre deux landmarks (ignore z). """
    if p1 is None or p2 is None:
        logger.warning("Tentative de calcul de distance avec un point None.")
        return 0.0
    if not all(hasattr(p, attr) for p in [p1, p2] for attr in ['x', 'y']):
        logger.warning("Landmark(s) invalide(s) fourni(s) à la fonction distance.")
        return 0.0
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- Logique Simplifiée V6 pour determine_face_shape ---
def determine_face_shape(landmarks: List[Landmark]) -> str:
    """
    Détermine une forme de visage simplifiée (Long, Proportionné, Autre)
    basée principalement sur le ratio Longueur/Largeur.
    """
    shape = "inconnue" # Forme par défaut
    # Vérification basique du nombre de landmarks
    if not landmarks or len(landmarks) < max(TOP_FOREHEAD, BOTTOM_CHIN, LEFT_TEMPLE, RIGHT_TEMPLE) + 1:
        logger.warning(f"Nombre insuffisant de landmarks ({len(landmarks)} fournis) pour les indices requis.")
        return shape

    try:
        # Récupérer seulement les points nécessaires
        p_top_forehead = landmarks[TOP_FOREHEAD]
        p_bottom_chin = landmarks[BOTTOM_CHIN]
        p_left_temple = landmarks[LEFT_TEMPLE]
        p_right_temple = landmarks[RIGHT_TEMPLE]

        # Calculer longueur et largeur principale
        face_length = distance(p_top_forehead, p_bottom_chin)
        cheekbone_width = distance(p_left_temple, p_right_temple) # Largeur max approx

        # Gérer division par zéro ou mesures invalides
        if face_length < 1e-6 or cheekbone_width < 1e-6:
            logger.warning(f"Mesures faciales invalides (L={face_length:.2f}, W={cheekbone_width:.2f}).")
            return "inconnue"

        length_width_ratio = face_length / cheekbone_width
        logger.info(f"Ratio L/W calculé (simple) : {length_width_ratio:.2f}")

        # --- Classification Simplifiée V6 ---
        RATIO_LONG = 1.20 # Seuil pour considérer "long"
        RATIO_PROP_LOW = 0.90 # Borne inférieure pour "proportionné"

        if length_width_ratio > RATIO_LONG:
            shape = "long"
        elif length_width_ratio >= RATIO_PROP_LOW: # Implicitement <= RATIO_LONG
            shape = "proportionné"
        else: # ratio < RATIO_PROP_LOW
            shape = "autre"

        logger.info(f"Forme de visage déterminée (v6 simple) : {shape}")
        return shape.lower()

    except IndexError:
        logger.error(f"Erreur d'indice lors de l'accès aux landmarks (indices requis jusqu'à {max(TOP_FOREHEAD, BOTTOM_CHIN, LEFT_TEMPLE, RIGHT_TEMPLE)}).", exc_info=True)
        return "erreur_indices"
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la détermination de la forme : {e}", exc_info=True)
        return "erreur_calcul"

# --- Analyse Faciale (Utilise la forme simplifiée) ---
def analyze_face_from_image_bytes(image_bytes: bytes) -> FaceAnalysisResult:
    """
    Analyse une image (fournie en bytes) pour détecter la pose du visage,
    les landmarks, et déterminer la forme du visage (simplifiée).
    """
    logger.info("Début de l'analyse faciale (landmarks + pose + forme simple)...")
    landmarker = get_face_landmarker()

    if landmarker is None:
        logger.error("FaceLandmarker non initialisé.")
        return FaceAnalysisResult(detection_successful=False, error_message="Erreur interne: Modèle non disponible.")

    try:
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_cv is None:
            logger.warning("Impossible de décoder l'image.")
            return FaceAnalysisResult(detection_successful=False, error_message="Format d'image invalide ou corrompu.")
    except Exception as e:
        logger.error(f"Erreur lors du décodage de l'image: {e}", exc_info=True)
        return FaceAnalysisResult(detection_successful=False, error_message="Erreur de décodage image.")

    try:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        logger.info("Exécution de la détection FaceLandmarker...")
        detection_result: Optional[FaceLandmarkerResult] = landmarker.detect(mp_image)
        logger.info("Détection terminée.")

        matrix_list: Optional[List[List[float]]] = None
        landmarks_list: Optional[List[Landmark]] = None
        detected_shape: Optional[str] = None
        error_msg: Optional[str] = None
        success: bool = False

        if detection_result and detection_result.facial_transformation_matrixes and len(detection_result.facial_transformation_matrixes) > 0:
            matrix = detection_result.facial_transformation_matrixes[0]
            matrix_list = matrix.tolist()

            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                landmarks_raw = detection_result.face_landmarks[0]
                if landmarks_raw:
                     landmarks_list = [Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks_raw if hasattr(lm, 'x')]
                     # Appelle la fonction de détermination de forme SIMPLIFIÉE V6
                     detected_shape = determine_face_shape(landmarks_list)
                     if "erreur" in detected_shape:
                         error_msg = f"Erreur de calcul de forme ({detected_shape})."
                     success = True
                     logger.info("Visage détecté, landmarks extraits.")
                else:
                    logger.warning("Landmarks vides retournés par Mediapipe.")
                    error_msg = "Données landmarks invalides."
                    success = False
            else:
                logger.warning("Pose détectée mais pas de landmarks extraits.")
                error_msg = "Landmarks non disponibles."
                success = False
        else:
            logger.info("Aucun visage détecté dans l'image.")
            error_msg = "Aucun visage détecté."
            success = False

        analysis_result = FaceAnalysisResult(
            detection_successful=success,
            facial_transformation_matrix=matrix_list,
            face_landmarks=landmarks_list,
            detected_face_shape=detected_shape if success and "erreur" not in (detected_shape or "") else None,
            error_message=error_msg if not success or "erreur" in (detected_shape or "") else None
        )
        return analysis_result

    except Exception as e:
        logger.error(f"Erreur inattendue pendant l'analyse faciale: {e}", exc_info=True)
        return FaceAnalysisResult(detection_successful=False, error_message=f"Erreur serveur inattendue pendant l'analyse.")


# --- Recommandation Basée sur l'Analyse (Simplifiée) ---
def get_recommendations_based_on_analysis(analysis: FaceAnalysisResult) -> Optional[RecommendationResult]:
    """
    Génère des recommandations basées sur les résultats de l'analyse faciale simplifiée.
    """
    if analysis.detection_successful and analysis.detected_face_shape and "erreur" not in analysis.detected_face_shape:
        # Appelle la fonction de recommandation V6
        recommended_ids, analysis_info_str = get_recommendations_for_face(analysis.detected_face_shape)
        return RecommendationResult(
            recommended_glasses_ids=recommended_ids,
            analysis_info=analysis_info_str
        )
    else:
        if not analysis.detection_successful: log_msg = "Analyse faciale échouée."
        elif not analysis.detected_face_shape: log_msg = "Forme de visage non déterminée."
        else: log_msg = f"Erreur lors de la détermination de forme ({analysis.detected_face_shape})."
        logger.info(f"Impossible de générer des recommandations: {log_msg}")
        return None

# --- Recommandation Basée sur Forme Simplifiée (V6) ---
def get_recommendations_for_face(face_shape: str) -> tuple[List[str], str]:
    """
    Génère des recommandations basées sur les formes simplifiées (long, proportionné, autre).
    """
    face_shape_lower = face_shape.lower().strip()
    analysis_info = f"Forme de visage simplifiée utilisée : {face_shape_lower.capitalize()}"
    logger.info(f"Génération de recommandations pour la forme simplifiée : '{face_shape_lower}'")

    model1 = "sunglass_model_1" # Ex: Styles Angulaires/Rectangles
    model2 = "sunglass_model_2" # Ex: Styles Ronds/Ovales
    model3 = "sunglass_model_3" # Ex: Styles Pilote/Variés

    if face_shape_lower == "long": # Ovale, Rectangle -> Styles larges ou contrastants
        recommendations = [model2, model3]
    elif face_shape_lower == "proportionné": # Rond, Carré -> Styles contrastants
        recommendations = [model1, model2, model3]
    elif face_shape_lower == "autre": # Coeur, Diamant, Large -> Styles équilibrants
        recommendations = [model1, model3]
    else: # Inconnue, erreur_indices, erreur_calcul
        recommendations = [model1, model2] # Défaut générique
        analysis_info = f"Forme de visage '{face_shape}' non reconnue ou erreur, recommandations par défaut."

    logger.info(f"Recommandations générées (simple) : {recommendations}")
    return recommendations, analysis_info