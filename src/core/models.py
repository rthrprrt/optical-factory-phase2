# src/core/models.py

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks
import threading
import logging
from typing import Optional, List # Ajout de List pour le type hint de get_available_model_ids
from src.core.config import settings # Importe l'objet settings
from pathlib import Path # Import Path

# Configure le logging en utilisant le niveau défini dans les settings
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Variable globale pour l'instance unique du landmarker
_face_landmarker_instance: Optional[vision.FaceLandmarker] = None
# Verrou pour gérer l'initialisation concurrente (sécurité)
_face_landmarker_lock = threading.Lock()

def get_face_landmarker() -> Optional[vision.FaceLandmarker]:
    """
    Initialise (si nécessaire) et retourne l'instance unique du FaceLandmarker.
    Thread-safe. Retourne None en cas d'échec d'initialisation.
    """
    global _face_landmarker_instance
    # Optimisation: Vérifie d'abord sans verrou si l'instance existe déjà
    if _face_landmarker_instance is None:
        # Si elle n'existe pas, acquiert le verrou pour l'initialisation
        with _face_landmarker_lock:
            # Revérifie à l'intérieur du verrou (double-checked locking pattern)
            if _face_landmarker_instance is None:
                model_path = settings.FACE_MODEL_PATH
                logger.info(f"Tentative d'initialisation du FaceLandmarker depuis : {model_path}")
                resolved_model_path = None # Pour le logging d'erreur
                try:
                    # Résout le chemin du modèle (relatif ou absolu)
                    resolved_model_path = Path(model_path)
                    if not resolved_model_path.is_absolute():
                         resolved_model_path = settings.BASE_DIR / resolved_model_path
                         logger.info(f"Chemin relatif détecté, utilisation du chemin absolu: {resolved_model_path}")

                    # Vérifie l'existence du fichier avant de continuer
                    if not resolved_model_path.exists():
                         logger.error(f"ERREUR CRITIQUE: Fichier modèle Mediapipe non trouvé à {resolved_model_path}")
                         return None # Echec clair

                    # Prépare les options pour FaceLandmarker
                    base_options = tasks.BaseOptions(model_asset_path=str(resolved_model_path))
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE, # Mode image pour appels API uniques
                        output_facial_transformation_matrixes=True, # Requis pour la pose
                        # output_face_landmarks=True, # <<< LIGNE SUPPRIMÉE (argument invalide)
                        num_faces=1 # Traite un seul visage par image
                    )
                    # Crée l'instance
                    _face_landmarker_instance = vision.FaceLandmarker.create_from_options(options)
                    logger.info("FaceLandmarker initialisé avec succès.")

                except Exception as e:
                    # Loggue l'erreur détaillée en cas d'échec
                    error_path = resolved_model_path if resolved_model_path else model_path
                    logger.error(f"Erreur lors de l'initialisation du FaceLandmarker depuis {error_path}: {e}", exc_info=True)
                    # Assure que l'instance reste None en cas d'erreur
                    _face_landmarker_instance = None
                    return None # Retourne None pour indiquer l'échec
    # Retourne l'instance (qui peut être None si l'initialisation a échoué)
    return _face_landmarker_instance

# --- Gestion des Modèles 3D (Backend ne charge plus, fournit juste les IDs) ---

def get_available_model_ids() -> List[str]:
     """
     Retourne la liste des IDs des modèles de lunettes configurés dans settings.
     Utile pour un endpoint /list_models par exemple.
     """
     # Récupère les clés (IDs) du dictionnaire dans la configuration
     model_ids = list(settings.MODEL_IDS_TO_PATHS.keys())
     logger.debug(f"Retourne les IDs de modèles disponibles: {model_ids}")
     return model_ids

# La fonction get_3d_model_path a été supprimée car le backend ne charge plus les modèles 3D.
# Le frontend utilisera les IDs de get_available_model_ids et les chemins (si fournis par une autre source ou codés en dur).