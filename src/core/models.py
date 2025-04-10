# src/core/models.py

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks
import threading
import logging
from typing import Optional
from src.core.config import settings # <<< Importe l'objet settings

# Configuration du logging
logging.basicConfig(level=settings.LOG_LEVEL) # Utilise le niveau de log de la config
logger = logging.getLogger(__name__)

# Variable globale pour stocker le détecteur initialisé
_face_landmarker_instance = None
_face_landmarker_lock = threading.Lock()

# --- Configuration Mediapipe (utilise settings) ---
# Le chemin est maintenant dans settings.FACE_MODEL_PATH

def get_face_landmarker():
    """
    Initialise et retourne une instance unique du FaceLandmarker.
    """
    global _face_landmarker_instance
    if _face_landmarker_instance is None: # Vérifie d'abord sans verrou pour la performance
        with _face_landmarker_lock: # Verrouille seulement si on doit initialiser
            if _face_landmarker_instance is None:
                # Utilise le chemin depuis settings
                model_path = settings.FACE_MODEL_PATH
                logger.info(f"Initialisation du FaceLandmarker depuis : {model_path}")
                try:
                    # Vérifie si le chemin est relatif et le rend absolu si besoin
                    # (Bien que Pydantic Settings puisse gérer ça si BASE_DIR est bien défini)
                    from pathlib import Path
                    if not Path(model_path).is_absolute():
                         # Suppose qu'il est relatif à la racine du projet si non absolu
                         model_path = str(settings.BASE_DIR / model_path)
                         logger.info(f"Chemin relatif détecté, utilisation du chemin absolu: {model_path}")

                    # Vérifie l'existence du fichier avant de continuer
                    if not Path(model_path).exists():
                         logger.error(f"ERREUR CRITIQUE: Fichier modèle non trouvé à {model_path}")
                         # Peut-être lever une exception ici pour arrêter le démarrage ?
                         # raise FileNotFoundError(f"Modèle Mediapipe non trouvé: {model_path}")
                         return None # Retourne None pour indiquer l'échec

                    base_options = tasks.BaseOptions(model_asset_path=model_path)
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        output_facial_transformation_matrixes=True,
                        # output_face_landmarks=True est nécessaire (normalement True par défaut)
                        num_faces=1
                    )
                    _face_landmarker_instance = vision.FaceLandmarker.create_from_options(options)
                    logger.info("FaceLandmarker initialisé avec succès.")
                except Exception as e:
                    logger.error(f"Erreur lors de l'initialisation du FaceLandmarker depuis {model_path}: {e}", exc_info=True)
                    # Important : Ne pas laisser _face_landmarker_instance à moitié initialisé
                    _face_landmarker_instance = None
                    return None # Indique clairement l'échec
    return _face_landmarker_instance

# --- Gestion des modèles 3D (utilise settings) ---

def get_3d_model_path(model_id: str) -> Optional[str]:
    """
    Retourne le chemin d'accès à un modèle 3D basé sur son ID,
    en utilisant la configuration centralisée.
    """
    # Utilise le dictionnaire défini dans settings
    path = settings.MODEL_IDS_TO_PATHS.get(model_id)
    if path:
        # Vérifie si le fichier existe réellement
        from pathlib import Path
        if Path(path).exists():
             logger.debug(f"Chemin trouvé pour modèle ID '{model_id}': {path}")
             return path
        else:
             logger.warning(f"Chemin configuré pour ID '{model_id}' mais fichier non trouvé: {path}")
             return None
    else:
        logger.debug(f"Aucun chemin configuré pour modèle ID '{model_id}'.")
        return None