# src/core/processing.py

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarkerResult # Importe le type résultat
from src.core.models import get_face_landmarker
# Importe les schémas mis à jour
from src.schemas.schemas import FaceAnalysisResult, Landmark, RecommendationResult
from typing import List, Optional, Tuple
import logging
import math # Pour les calculs de distance

logger = logging.getLogger(__name__)

# --- Indices des Landmarks (selon la documentation Mediapipe Face Mesh) ---
# Ces indices peuvent être utiles pour calculer les ratios pour la forme du visage
# Voir: https://developers.google.com/mediapipe/solutions/vision/face_landmarker#face_geometry
# (Note: Les indices exacts peuvent varier légèrement selon les versions de modèle)
# Exemple d'indices (à vérifier/adapter) :
# Points extrêmes du visage pour la largeur/hauteur
LEFT_MOST_POINT_INDEX = 356 # Environ tempe gauche
RIGHT_MOST_POINT_INDEX = 127 # Environ tempe droite
TOP_MOST_POINT_INDEX = 10   # Environ sommet du front
BOTTOM_MOST_POINT_INDEX = 152 # Environ pointe du menton
# Points pour la largeur de la mâchoire
LEFT_JAW_POINT_INDEX = 172
RIGHT_JAW_POINT_INDEX = 397
# Points pour la largeur des pommettes
LEFT_CHEEK_POINT_INDEX = 234
RIGHT_CHEEK_POINT_INDEX = 454


# --- Détermination de la forme (Implémentation Basique - Logique Ajustée) ---

def distance(p1: Landmark, p2: Landmark) -> float:
    """ Calcule la distance Euclidienne 2D entre deux landmarks (ignore z). """
    # Ajout d'une vérification pour éviter les erreurs si p1 ou p2 est None (peu probable avec List[Landmark])
    if p1 is None or p2 is None:
        return 0.0
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def determine_face_shape(landmarks: List[Landmark]) -> str:
    """
    Détermine la forme du visage à partir des landmarks.
    (Implémentation très basique basée sur des ratios largeur/hauteur - v2 Logique).
    """
    if not landmarks or len(landmarks) < 468: # Assure qu'on a les points attendus
        logger.warning("Nombre insuffisant de landmarks pour déterminer la forme du visage.")
        return "inconnue"

    try:
        # Récupérer les points clés
        top_point = landmarks[TOP_MOST_POINT_INDEX]
        bottom_point = landmarks[BOTTOM_MOST_POINT_INDEX]
        left_point = landmarks[LEFT_MOST_POINT_INDEX]
        right_point = landmarks[RIGHT_MOST_POINT_INDEX]
        left_jaw = landmarks[LEFT_JAW_POINT_INDEX]
        right_jaw = landmarks[RIGHT_JAW_POINT_INDEX]

        # Calculer les dimensions
        face_height = distance(top_point, bottom_point)
        face_width = distance(left_point, right_point)
        jaw_width = distance(left_jaw, right_jaw)

        if face_height <= 1e-6 or face_width <= 1e-6: # Utilise une petite tolérance au lieu de 0
             logger.warning("Dimensions faciales invalides (proches de zéro).")
             return "inconnue"

        # Calculer les ratios
        height_width_ratio = face_height / face_width
        jaw_face_width_ratio = jaw_width / face_width

        logger.info(f"Ratios calculés : H/W={height_width_ratio:.2f}, Jaw/W={jaw_face_width_ratio:.2f}")

        # Classification très simpliste (Logique v2 - Ajustée)
        shape = "inconnue" # Défaut
        if abs(height_width_ratio - 1.0) < 0.15: # Visage plutôt proportionné en H/W (Ratio ~1.0)
            if jaw_face_width_ratio >= 0.75: # Mâchoire large
                 shape = "carrée"
            else: # Mâchoire plus étroite
                 shape = "ronde"
        elif height_width_ratio >= 1.15: # Visage clairement plus haut que large (Ratio > ~1.15)
            if jaw_face_width_ratio >= 0.75: # Mâchoire large
                 shape = "rectangle"
            # Modification ici : La condition originale pour ovale était trop large et attrapait le cas rectangle
            elif jaw_face_width_ratio < 0.72: # Mâchoire significativement plus étroite pour ovale/coeur
                # On pourrait ajouter une logique pour coeur ici si on avait la largeur du front
                shape = "ovale" # Considère ovale comme le cas "long et étroit"
            else: # Cas intermédiaire (mâchoire ni très large ni très étroite)
                shape = "ovale" # Peut aussi être ovale

        # Ajouter des cas si H < W ?
        # elif height_width_ratio < 0.9:
        #    shape = "triangle_inverse"

        logger.info(f"Forme de visage déterminée (basique - v2 ajustée) : {shape}")
        return shape

    except IndexError:
        logger.error("Erreur d'indice lors de l'accès aux landmarks. Le modèle a-t-il changé ?", exc_info=True)
        return "erreur_indices"
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la détermination de la forme : {e}", exc_info=True)
        return "erreur_calcul"


# --- Analyse Faciale (Mise à jour pour mieux gérer les erreurs de format et d'initialisation) ---

def analyze_face_from_image_bytes(image_bytes: bytes) -> FaceAnalysisResult:
    """
    Analyse une image (fournie en bytes) pour détecter la pose du visage,
    les landmarks, et déterminer la forme du visage.
    """
    logger.info("Début de l'analyse faciale (landmarks + pose + forme)...")
    landmarker = get_face_landmarker()

    if landmarker is None:
        logger.error("FaceLandmarker non initialisé.")
        return FaceAnalysisResult(detection_successful=False, error_message="Erreur interne: Modèle non disponible.")

    try:
        # Valider le format de l'image avant tout
        try:
            image_np = np.frombuffer(image_bytes, np.uint8)
            image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image_cv is None:
                logger.warning("Impossible de décoder l'image.")
                return FaceAnalysisResult(detection_successful=False, error_message="Format d'image invalide.")
        except Exception as e:
            logger.warning(f"Erreur de décodage de l'image: {e}")
            return FaceAnalysisResult(detection_successful=False, error_message="Format d'image invalide.")

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
                landmarks_list = [Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks_raw]
                detected_shape = determine_face_shape(landmarks_list)
                if "erreur" in detected_shape:
                    error_msg = f"Erreur lors de la détermination de la forme ({detected_shape})."
                success = True
                logger.info("Visage détecté, landmarks extraits.")
            else:
                logger.warning("Pose détectée mais pas de landmarks extraits.")
                error_msg = "Pose détectée mais landmarks non disponibles."
                success = False
        else:
            logger.info("Aucun visage détecté dans l'image.")
            error_msg = "Aucun visage détecté."
            success = False

        analysis_result = FaceAnalysisResult(
            detection_successful=success,
            facial_transformation_matrix=matrix_list,
            face_landmarks=landmarks_list,
            detected_face_shape=detected_shape if success and "erreur" not in (detected_shape or "") else None, # Ne retourne la forme que si succès et pas d'erreur de calcul
            error_message=error_msg # Retourne le message d'erreur s'il y en a un
        )
        return analysis_result

    except Exception as e:
        logger.error(f"Erreur inattendue pendant l'analyse faciale complète: {e}", exc_info=True)
        return FaceAnalysisResult(detection_successful=False, error_message=f"Erreur interne serveur: {e}")


# --- Recommandation (Mise à jour - Inchangée par rapport à la version précédente) ---

def get_recommendations_based_on_analysis(analysis: FaceAnalysisResult) -> Optional[RecommendationResult]:
    """
    Génère des recommandations basées sur les résultats de l'analyse faciale.
    """
    # Vérifie que l'analyse a réussi ET que la forme a été déterminée ET qu'il n'y a pas eu d'erreur dans sa détermination
    if analysis.detection_successful and analysis.detected_face_shape and "erreur" not in analysis.detected_face_shape:
        recommended_ids, analysis_info_str = get_recommendations_for_face(analysis.detected_face_shape)
        return RecommendationResult(
            recommended_glasses_ids=recommended_ids,
            analysis_info=analysis_info_str
        )
    else:
        logger.info("Impossible de générer des recommandations car la forme du visage n'a pas été déterminée avec succès.")
        return None

# --- Fonction de recommandation par forme (inchangée) ---
def get_recommendations_for_face(face_shape: str) -> tuple[List[str], str]:
    """
    Génère des recommandations de lunettes basées sur la forme du visage fournie.
    """
    face_shape_lower = face_shape.lower()
    analysis_info = f"Forme de visage utilisée pour la recommandation : {face_shape_lower.capitalize()}"
    logger.info(f"Génération de recommandations pour la forme : {face_shape_lower}")

    if face_shape_lower == "ronde":
        recommendations = ["sunglass_model_1", "sunglass_model_3"]
    elif face_shape_lower == "carrée":
        recommendations = ["sunglass_model_2"]
    elif face_shape_lower == "ovale":
         recommendations = ["sunglass_model_1", "sunglass_model_2", "sunglass_model_3"]
    elif face_shape_lower == "rectangle":
        recommendations = ["sunglass_model_2"]
    elif face_shape_lower == "coeur":
        recommendations = ["sunglass_model_1"]
    else:
        recommendations = ["sunglass_model_1", "sunglass_model_2"]
        analysis_info = f"Forme de visage '{face_shape}' non reconnue, recommandations par défaut."

    logger.info(f"Recommandations générées : {recommendations}")
    return recommendations, analysis_info