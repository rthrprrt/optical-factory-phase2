# src/main.py

from fastapi import FastAPI
from src.api.endpoints import router as api_router
from src.core.models import get_face_landmarker # Garde l'initialisation Mediapipe
# from src.core.rendering import initialize_renderer <<< LIGNE SUPPRIMÉE
import logging
import os
from src.core.config import settings

logger = logging.getLogger(__name__)

# --- Configuration de l'Application FastAPI ---
app = FastAPI(
    title="Optical Factory API - Analysis & Recommendation", # Titre mis à jour
    description="API backend fournissant l'analyse faciale (pose, landmarks, forme) et la recommandation de lunettes.", # Desc mise à jour
    version="0.2.0" # Version indiquant le changement d'archi
)

# --- Événements de Démarrage/Arrêt ---
@app.on_event("startup")
async def startup_event():
    """ Charge les modèles Mediapipe au démarrage. """
    logger.info("="*10 + " ÉVÉNEMENT DE DÉMARRAGE " + "="*10)
    logger.info(f"Log Level: {settings.LOG_LEVEL}")

    # 1. Charge le modèle Mediapipe
    logger.info("Initialisation du modèle Mediapipe...")
    if not get_face_landmarker():
        logger.error(">>> ÉCHEC de l'initialisation du modèle Mediapipe.")
        # On pourrait vouloir arrêter l'app ici si le modèle est critique
    else:
        logger.info(">>> Modèle Mediapipe OK.")

    # 2. Initialise PyRender <<< SECTION SUPPRIMÉE
    # logger.info("Initialisation du Renderer PyRender...")
    # if not initialize_renderer():
    #      logger.error(">>> ÉCHEC de l'initialisation du Renderer PyRender.")
    # else:
    #     logger.info(">>> Renderer PyRender OK.")

    logger.info("="*10 + " INITIALISATION TERMINÉE " + "="*10)

# --- Inclusion des Routes API ---
app.include_router(api_router, prefix="/api/v1")

# --- Routes de Base ---
@app.get("/", tags=["Root"], include_in_schema=False)
async def read_root():
    return {"message": "Bienvenue sur l'API Optical Factory (Analyse/Reco) - Voir /docs."}

@app.get("/health", tags=["Health Check"])
async def health_check():
    """ Vérifie si le modèle Mediapipe est chargé. """
    landmarker_ok = get_face_landmarker() is not None

    if os.environ.get("TESTING", "false").lower() == "true":
        status = "ok" # En mode test, on dit OK même si modèle absent
        models_loaded = landmarker_ok
        detail = "Running in test mode (model status ignored for health 'ok')" if not models_loaded else "Running in test mode"
        logger.info(f"Health check (TEST MODE): Landmarker loaded = {models_loaded}")
        return {"status": status, "models_loaded": models_loaded, "detail": detail}

    if landmarker_ok:
        logger.info("Health check: OK")
        return {"status": "ok", "models_loaded": True}
    else:
        logger.error("Health check: FAILED - FaceLandmarker non initialisé.")
        return {"status": "error", "models_loaded": False, "detail": "FaceLandmarker failed to initialize."}