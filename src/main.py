from fastapi import FastAPI
from src.api.endpoints import router as api_router
from src.core.models import get_face_landmarker # Importe la fonction
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Optical Factory API",
    description="API backend pour l'essayage virtuel de lunettes et recommandation.",
    version="0.1.0"
)

# Inclure les routes définies dans api/endpoints.py
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """ Charge les modèles au démarrage de l'application. """
    logger.info("Événement de démarrage: Initialisation des modèles...")
    get_face_landmarker() # Appelle la fonction pour pré-charger le modèle
    logger.info("Initialisation terminée.")

@app.get("/", tags=["Root"], include_in_schema=False) # Masque la racine de la doc par défaut
async def read_root():
    return {"message": "Bienvenue sur l'API Optical Factory - Voir /docs pour la documentation."}

@app.get("/health", tags=["Health Check"])
async def health_check():
    """ Vérifie la santé de l'application. """
    # Pourrait être amélioré pour vérifier si le landmarker est chargé
    landmarker = get_face_landmarker()
    if landmarker:
        return {"status": "ok", "models_loaded": True}
    else:
        return {"status": "error", "models_loaded": False, "detail": "FaceLandmarker failed to initialize."}

# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("Événement d'arrêt: Nettoyage...")
    # Peut-être fermer des ressources si nécessaire