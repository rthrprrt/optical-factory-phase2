import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks
import threading
import logging # Pour logger les infos/erreurs
from typing import Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variable globale pour stocker le détecteur initialisé
_face_landmarker_instance = None
_face_landmarker_lock = threading.Lock() # Pour gérer l'accès concurrentiel si nécessaire

# --- Configuration Mediapipe ---
# Adapte le chemin si tu as mis le modèle ailleurs
FACE_MODEL_PATH = "./models/face_landmarker_v2_with_blendshapes.task"

def get_face_landmarker():
    """
    Initialise et retourne une instance unique du FaceLandmarker.
    Utilise un verrou pour être thread-safe au cas où plusieurs requêtes
    tenteraient d'initialiser en même temps au démarrage.
    """
    global _face_landmarker_instance
    with _face_landmarker_lock:
        if _face_landmarker_instance is None:
            logger.info(f"Initialisation du FaceLandmarker depuis : {FACE_MODEL_PATH}")
            try:
                base_options = tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH)
                # Utiliser le mode IMAGE car nous traitons des images uniques reçues par l'API
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE, # IMPORTANT: Mode IMAGE pour l'API
                    output_facial_transformation_matrixes=True,
                    num_faces=1 # On suppose qu'on analyse une seule personne à la fois
                )
                _face_landmarker_instance = vision.FaceLandmarker.create_from_options(options)
                logger.info("FaceLandmarker initialisé avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du FaceLandmarker: {e}", exc_info=True)
                # Gérer l'erreur : lever une exception, retourner None, etc.
                # Ici, on retourne None pour indiquer l'échec.
                return None
        return _face_landmarker_instance

# --- Gestion des modèles 3D (placeholder) ---
# Tu pourras ajouter ici le chargement et la gestion des modèles 3D plus tard
# Par exemple, charger les chemins des fichiers .obj/.glb dans un dictionnaire

def get_3d_model_path(model_id: str) -> Optional[str]:
    """
    Retourne le chemin d'accès à un modèle 3D basé sur son ID.
    (Implémentation basique pour l'instant)
    """
    # Simule une base de données ou une config de modèles
    available_models = {
        "sunglass_model_1": "./models/sunglass/model_normalized.obj", # Utilise les modèles normalisés si tu les as
        "sunglass_model_2": "./models/sunglass/Glasses_normalized.obj",
        "sunglass_model_3": "./models/sunglass/model3_normalized.obj",
        # Ajoute d'autres modèles ici
    }
    return available_models.get(model_id)