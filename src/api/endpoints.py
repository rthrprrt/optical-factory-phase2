# src/api/endpoints.py

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body
# Imports simplifiés : plus besoin de Form, Response, cv2, numpy ici
from src.core.processing import analyze_face_from_image_bytes, get_recommendations_for_face, get_recommendations_based_on_analysis
# from src.core.rendering import render_overlay <<< SUPPRIMÉ
# from src.core.models import get_3d_model_path <<< SUPPRIMÉ (sauf si on ajoute /list_models)
from src.schemas.schemas import FaceAnalysisResult, RecommendationResult, RecommendationRequest, AnalyzeAndRecommendResult
import logging
from typing import Optional, List # Ajout List si non présent

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Endpoint d'Analyse (Retourne Pose + Landmarks + Forme) ---
@router.post(
    "/analyze_face",
    response_model=FaceAnalysisResult,
    summary="Analyse une image pour détecter pose, landmarks et forme du visage",
    tags=["Analysis"]
)
async def analyze_face_endpoint(
    image_file: UploadFile = File(..., description="Fichier image à analyser (ex: JPG, PNG)")
):
    """
    Accepte un fichier image, le traite et retourne les détails de l'analyse faciale,
    incluant la matrice de pose, les landmarks, et la forme de visage estimée (simplifiée).
    Ces données sont destinées au client pour le rendu 3D et la logique d'affichage.
    """
    logger.info(f"[analyze_face] Requête reçue pour le fichier: {image_file.filename}")
    try:
        image_bytes = await image_file.read()
        if not image_bytes:
            logger.warning("[analyze_face] Fichier image vide.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier image fourni est vide.")
    except Exception as e:
         logger.error(f"Erreur lecture image uploadée: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Erreur lors de la lecture du fichier image.")

    analysis_result = analyze_face_from_image_bytes(image_bytes)

    if not analysis_result.detection_successful and "interne" in (analysis_result.error_message or "").lower():
         logger.error(f"[analyze_face] Erreur interne: {analysis_result.error_message}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=analysis_result.error_message or "Erreur interne lors de l'analyse")
    elif not analysis_result.detection_successful:
         logger.info(f"[analyze_face] Analyse non réussie: {analysis_result.error_message}")
    else:
        logger.info("[analyze_face] Analyse réussie.")

    return analysis_result # Retourne le JSON FaceAnalysisResult

# --- Endpoint de Recommandation (Basé sur forme fournie) ---
@router.post(
    "/recommend_glasses",
    response_model=RecommendationResult,
    summary="Recommande des lunettes basées sur une forme de visage fournie",
    tags=["Recommendation"]
)
async def recommend_glasses_endpoint(
    request_body: RecommendationRequest = Body(...)
):
    """
    Accepte une forme de visage simplifiée (long, proportionné, autre)
    et retourne une liste d'IDs de modèles de lunettes suggérés.
    """
    logger.info(f"[recommend_glasses] Requête reçue avec forme: {request_body.face_shape}")
    if not request_body.face_shape or not isinstance(request_body.face_shape, str):
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'face_shape' requis (string).")

    # Utilise la fonction de recommandation (qui a été adaptée aux formes simples)
    recommended_ids, analysis_info_str = get_recommendations_for_face(request_body.face_shape)
    response = RecommendationResult(recommended_glasses_ids=recommended_ids, analysis_info=analysis_info_str)
    logger.info(f"[recommend_glasses] Réponse : {response}")
    return response

# --- Endpoint de Rendu <<< SECTION SUPPRIMÉE ---
# @router.post("/render_glasses", ...)
# async def render_glasses_endpoint(...):
#     ...

# --- Endpoint Combiné (Analyse + Recommandation) ---
@router.post(
    "/analyze_and_recommend",
    response_model=AnalyzeAndRecommendResult,
    summary="Analyse une image ET recommande des lunettes basées sur la forme détectée",
    tags=["Combined Workflow"]
)
async def analyze_and_recommend_endpoint(
    image_file: UploadFile = File(..., description="Fichier image à analyser (ex: JPG, PNG)")
):
    """
    Accepte un fichier image, effectue l'analyse faciale complète (pose, landmarks, forme simple),
    puis génère des recommandations de lunettes basées sur cette forme.
    Retourne à la fois les résultats de l'analyse et les recommandations.
    """
    logger.info(f"[analyze_and_recommend] Requête reçue pour: {image_file.filename}")
    try:
        image_bytes = await image_file.read()
        if not image_bytes:
            logger.warning("[analyze_and_recommend] Fichier image vide.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier image fourni est vide.")
    except Exception as e:
         logger.error(f"Erreur lecture image uploadée: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Erreur lecture fichier image.")

    # 1. Effectuer l'analyse complète
    analysis_result = analyze_face_from_image_bytes(image_bytes)

    # Gère les erreurs internes SANS lever d'exception ici
    if not analysis_result.detection_successful and "interne" in (analysis_result.error_message or "").lower():
         logger.error(f"[analyze_and_recommend] Erreur interne durant l'analyse: {analysis_result.error_message}")
         # L'erreur sera dans la partie 'analysis' de la réponse

    # 2. Générer les recommandations si l'analyse a réussi
    recommendation_result: Optional[RecommendationResult] = None
    if analysis_result.detection_successful and analysis_result.detected_face_shape and "erreur" not in analysis_result.detected_face_shape:
        # Appelle la fonction qui utilise get_recommendations_for_face
        recommendation_result = get_recommendations_based_on_analysis(analysis_result)
        if recommendation_result: logger.info("[analyze_and_recommend] Recommandations générées.")
        else: logger.warning("[analyze_and_recommend] Impossible de générer des recommandations."); analysis_result.error_message = (analysis_result.error_message or "") + " Recommandations non générées."
    else:
         log_msg_suffix = "pas de recommandations."
         if not analysis_result.detection_successful: logger.info(f"[analyze_and_recommend] Analyse non réussie, {log_msg_suffix}")
         else: logger.info(f"[analyze_and_recommend] Forme non déterminée, {log_msg_suffix}"); analysis_result.error_message = (analysis_result.error_message or "") + " Forme non déterminée."

    # 3. Construire la réponse combinée
    final_response = AnalyzeAndRecommendResult(
        analysis=analysis_result,
        recommendation=recommendation_result
    )
    return final_response

# --- (Optionnel) Endpoint pour lister les modèles ---
# Décommente et adapte si besoin
# @router.get("/list_models", response_model=List[str], tags=["Models"])
# async def list_available_models():
#     """ Retourne la liste des IDs des modèles de lunettes disponibles. """
#     from src.core.config import settings
#     model_ids = list(settings.MODEL_IDS_TO_PATHS.keys())
#     logger.info(f"Retourne la liste des modèles: {model_ids}")
#     return model_ids