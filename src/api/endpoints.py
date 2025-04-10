# src/api/endpoints.py

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body
# Importe la nouvelle logique et les schémas mis à jour
from src.core.processing import analyze_face_from_image_bytes, get_recommendations_for_face, get_recommendations_based_on_analysis
from src.schemas.schemas import FaceAnalysisResult, RecommendationResult, RecommendationRequest, AnalyzeAndRecommendResult # Ajoute AnalyzeAndRecommendResult
import logging
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Endpoint d'Analyse (Simple - Retourne Pose + Landmarks + Forme) ---
@router.post(
    "/analyze_face",
    response_model=FaceAnalysisResult, # Utilise le schéma mis à jour
    summary="Analyse une image pour détecter pose, landmarks et forme du visage",
    tags=["Analysis"]
)
async def analyze_face_endpoint(
    image_file: UploadFile = File(..., description="Fichier image à analyser (ex: JPG, PNG)")
):
    """
    Accepte un fichier image, le traite et retourne les détails de l'analyse faciale,
    incluant la matrice de pose, les landmarks, et la forme de visage estimée.
    """
    logger.info(f"[analyze_face] Requête reçue pour le fichier: {image_file.filename}")
    image_bytes = await image_file.read()
    if not image_bytes:
        logger.warning("[analyze_face] Fichier image vide.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier image fourni est vide.")

    analysis_result = analyze_face_from_image_bytes(image_bytes)

    # Toujours retourner un objet FaceAnalysisResult, même en cas d'erreur interne
    if not analysis_result.detection_successful and "interne" in (analysis_result.error_message or ""):
         logger.error(f"[analyze_face] Erreur interne: {analysis_result.error_message}")
         # Ne pas lever d'exception en cas d'erreur interne
    elif not analysis_result.detection_successful:
         logger.info(f"[analyze_face] Analyse non réussie: {analysis_result.error_message}")
    else:
        logger.info("[analyze_face] Analyse réussie.")

    return analysis_result


# --- Endpoint de Recommandation (Simple - Basé sur forme fournie) ---
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
    Accepte une forme de visage (ex: "ronde") et retourne une liste d'IDs de lunettes.
    """
    logger.info(f"[recommend_glasses] Requête reçue avec forme: {request_body.face_shape}")
    if not request_body.face_shape or not isinstance(request_body.face_shape, str):
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'face_shape' requis (string).")

    recommended_ids, analysis_info_str = get_recommendations_for_face(request_body.face_shape)
    response = RecommendationResult(recommended_glasses_ids=recommended_ids, analysis_info=analysis_info_str)
    logger.info(f"[recommend_glasses] Réponse : {response}")
    return response


# --- Endpoint Combiné (Analyse + Recommandation) (Nouveau) ---
@router.post(
    "/analyze_and_recommend",
    response_model=AnalyzeAndRecommendResult, # Utilise le nouveau schéma combiné
    summary="Analyse une image et recommande des lunettes basées sur la forme détectée",
    tags=["Combined Workflow"]
)
async def analyze_and_recommend_endpoint(
    image_file: UploadFile = File(..., description="Fichier image à analyser (ex: JPG, PNG)")
):
    """
    Accepte un fichier image, effectue l'analyse faciale complète (pose, landmarks, forme),
    puis génère des recommandations de lunettes basées sur la forme de visage détectée.
    """
    logger.info(f"[analyze_and_recommend] Requête reçue pour: {image_file.filename}")
    image_bytes = await image_file.read()
    if not image_bytes:
        logger.warning("[analyze_and_recommend] Fichier image vide.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier image fourni est vide.")

    # 1. Effectuer l'analyse complète
    analysis_result = analyze_face_from_image_bytes(image_bytes)

    # Ne pas lever d'exception en cas d'erreur interne
    if not analysis_result.detection_successful and "interne" in (analysis_result.error_message or ""):
         logger.error(f"[analyze_and_recommend] Erreur interne durant l'analyse: {analysis_result.error_message}")
         # Ne PAS lever d'exception, continuer avec l'analyse échouée

    # 2. Générer les recommandations si l'analyse a réussi et la forme est trouvée
    recommendation_result: Optional[RecommendationResult] = None
    if analysis_result.detection_successful and analysis_result.detected_face_shape and "erreur" not in analysis_result.detected_face_shape:
        recommendation_result = get_recommendations_based_on_analysis(analysis_result)
        if recommendation_result:
            logger.info("[analyze_and_recommend] Recommandations générées.")
        else:
            logger.warning("[analyze_and_recommend] Analyse réussie mais impossible de générer des recommandations.")
            # On peut ajouter une info à l'analyse
            analysis_result.error_message = (analysis_result.error_message or "") + " Impossible de générer des recommandations."
    else:
         logger.info("[analyze_and_recommend] Analyse non réussie ou forme non déterminée, pas de recommandations.")
         # L'erreur est déjà dans analysis_result.error_message si l'analyse a échoué

    # 3. Construire la réponse combinée
    final_response = AnalyzeAndRecommendResult(
        analysis=analysis_result,
        recommendation=recommendation_result
    )

    return final_response