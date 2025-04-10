# src/api/endpoints.py

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body, Form, Response
# Importe la nouvelle logique et les schémas mis à jour
from src.core.processing import analyze_face_from_image_bytes, get_recommendations_for_face, get_recommendations_based_on_analysis
# Importe la fonction de rendu
from src.core.rendering import render_overlay
# Importe pour obtenir le chemin du modèle
from src.core.models import get_3d_model_path
from src.schemas.schemas import FaceAnalysisResult, RecommendationResult, RecommendationRequest, AnalyzeAndRecommendResult
import logging
from typing import Optional
import cv2 # Pour le décodage/encodage d'image
import numpy as np # Pour manipuler l'image

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Endpoint d'Analyse (Simple - Retourne Pose + Landmarks + Forme) ---
@router.post(
    "/analyze_face", # Nom corrigé ici
    response_model=FaceAnalysisResult,
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

    # Gestion d'erreur: Si une erreur interne survient (ex: modèle non chargé),
    # il est préférable de retourner 500 pour signaler un problème serveur.
    if not analysis_result.detection_successful and "interne" in (analysis_result.error_message or "").lower():
         logger.error(f"[analyze_face] Erreur interne: {analysis_result.error_message}")
         # Lève une exception 500 pour indiquer une erreur serveur
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=analysis_result.error_message)
    elif not analysis_result.detection_successful:
         logger.info(f"[analyze_face] Analyse non réussie: {analysis_result.error_message}")
         # Retourne 200 mais avec le résultat indiquant l'échec
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

# --- Endpoint de Rendu (Utilise la logique de rendering.py) ---
@router.post(
    "/render_glasses",
    summary="Superpose un modèle de lunettes 3D sur une image via analyse faciale",
    tags=["Rendering"],
    response_description="Image au format JPEG avec les lunettes superposées",
    responses={
        200: {"content": {"image/jpeg": {}}},
        400: {"description": "Requête invalide (ex: image vide/invalide, échec détection)"},
        404: {"description": "Modèle 3D (ID) non trouvé"},
        500: {"description": "Erreur interne (modèle ML non chargé, erreur rendu)"}
    }
)
async def render_glasses_endpoint(
    # Reçoit l'image et l'ID via Form car c'est du multipart/form-data
    image_file: UploadFile = File(..., description="Fichier image de fond (visage)"),
    model_id: str = Form(..., description="ID du modèle de lunettes à superposer (ex: sunglass_model_1)")
):
    """
    Accepte une image et un ID de modèle 3D.
    1. Analyse l'image pour trouver la pose du visage.
    2. Rend le modèle 3D spécifié à la pose détectée.
    3. Superpose le rendu sur l'image originale.
    4. Retourne l'image résultante au format JPEG.
    """
    logger.info(f"[render_glasses] Requête reçue pour modèle '{model_id}' sur fichier: {image_file.filename}")

    # 1. Lire l'image d'entrée
    image_bytes = await image_file.read()
    if not image_bytes:
        logger.warning("[render_glasses] Fichier image vide reçu.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier image fourni est vide.")

    # Décoder l'image pour l'analyse et le rendu de fond
    image_np = np.frombuffer(image_bytes, np.uint8)
    background_cv_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if background_cv_image is None:
        logger.warning("[render_glasses] Impossible de décoder l'image d'entrée.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Format d'image invalide ou corrompu.")

    # 2. Analyser l'image pour obtenir la pose
    analysis_result = analyze_face_from_image_bytes(image_bytes)

    # Vérifier les erreurs d'analyse
    if not analysis_result.detection_successful:
         error_detail = analysis_result.error_message or "Échec de la détection faciale"
         if "interne" in error_detail.lower() or "modèle non disponible" in error_detail.lower():
             logger.error(f"[render_glasses] Erreur interne pré-rendu: {error_detail}")
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)
         else: # Erreur liée à l'image
             logger.warning(f"[render_glasses] Échec détection faciale: {error_detail}")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_detail)

    if analysis_result.facial_transformation_matrix is None:
         logger.error("[render_glasses] Matrice de pose manquante malgré détection réussie.")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erreur interne: Matrice de pose non obtenue.")

    # Convertir la matrice de pose en NumPy array
    pose_matrix = np.array(analysis_result.facial_transformation_matrix)

    # 3. Vérifier si l'ID du modèle 3D est valide AVANT d'appeler le rendu
    if get_3d_model_path(model_id) is None:
         logger.warning(f"[render_glasses] ID modèle 3D '{model_id}' non trouvé.")
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Modèle 3D avec ID '{model_id}' non trouvé.")

    # 4. Appeler la fonction de rendu du module 'rendering'
    logger.info(f"[render_glasses] Appel de la fonction render_overlay pour modèle {model_id}...")
    rendered_image = render_overlay(
        background_image=background_cv_image,
        face_pose_matrix=pose_matrix,
        model_id=model_id
    )

    # 5. Gérer le résultat du rendu
    if rendered_image is None:
         logger.error(f"[render_glasses] La fonction render_overlay a retourné None pour le modèle '{model_id}'.")
         # L'erreur spécifique (chargement modèle ou rendu) devrait être logguée dans rendering.py
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erreur lors du rendu de la superposition 3D.")

    # 6. Encoder l'image résultante en JPEG
    logger.info("[render_glasses] Encodage de l'image résultante...")
    success, encoded_image = cv2.imencode(".jpg", rendered_image)
    if not success:
         logger.error("[render_glasses] Échec de l'encodage JPEG.")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erreur lors de l'encodage de l'image de sortie.")

    # 7. Retourner l'image binaire
    logger.info("[render_glasses] Retour de l'image JPEG.")
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


# --- Endpoint Combiné (Analyse + Recommandation) ---
@router.post(
    "/analyze_and_recommend",
    response_model=AnalyzeAndRecommendResult,
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

    analysis_result = analyze_face_from_image_bytes(image_bytes)

    # Ne pas lever d'exception en cas d'erreur interne
    if not analysis_result.detection_successful and "interne" in (analysis_result.error_message or "").lower():
         logger.error(f"[analyze_and_recommend] Erreur interne durant l'analyse: {analysis_result.error_message}")

    recommendation_result: Optional[RecommendationResult] = None
    if analysis_result.detection_successful and analysis_result.detected_face_shape and "erreur" not in analysis_result.detected_face_shape:
        recommendation_result = get_recommendations_based_on_analysis(analysis_result)
        if recommendation_result:
            logger.info("[analyze_and_recommend] Recommandations générées.")
        else:
            logger.warning("[analyze_and_recommend] Analyse réussie mais impossible de générer des recommandations.")
            analysis_result.error_message = (analysis_result.error_message or "") + " Impossible de générer des recommandations."
    else:
         if not analysis_result.detection_successful:
             logger.info("[analyze_and_recommend] Analyse non réussie, pas de recommandations.")
         else: # Analyse OK mais forme non déterminée
            logger.info("[analyze_and_recommend] Forme non déterminée, pas de recommandations.")
            if not analysis_result.error_message:
                analysis_result.error_message = "Forme du visage non déterminée."

    final_response = AnalyzeAndRecommendResult(
        analysis=analysis_result,
        recommendation=recommendation_result
    )
    return final_response