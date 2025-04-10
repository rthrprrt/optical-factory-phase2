# src/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Dict # Importe Dict pour le type hint

# Calcule BASE_DIR une seule fois au niveau du module
_project_root = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Configuration via .env
    model_config = SettingsConfigDict(env_file=_project_root / '.env', extra='ignore')

    # --- Chemins et Constantes ---
    # Rend BASE_DIR accessible comme attribut de settings
    # Utilise default pour que Pydantic le prenne mais ne cherche pas dans .env
    BASE_DIR: Path = _project_root

    # --- Paramètres configurables via .env ---
    LOG_LEVEL: str = "INFO"
    # Le chemin peut rester relatif dans .env, on le résoudra si besoin
    FACE_MODEL_PATH: str = "./models/face_landmarker_v2_with_blendshapes.task"
    RENDER_WIDTH: int = 640
    RENDER_HEIGHT: int = 480
    TARGET_LATENCY_MS: float = 2500.0
    TARGET_DETECTION_PRECISION: float = 0.95
    # Optionnel : Permet de configurer l'URL de l'API via .env
    API_BASE_URL: str = "http://localhost:8000"
    # Optionnel : Seuil pour la précision de forme
    SHAPE_DETERMINATION_ACCURACY: float = 0.70

    # --- Configuration Statique (non lue depuis .env mais partie des settings) ---
    MODEL_IDS_TO_PATHS: Dict[str, str] = {
        "sunglass_model_1": str(_project_root / "models/sunglass/model_normalized.obj"),
        "sunglass_model_2": str(_project_root / "models/sunglass/Glasses_normalized.obj"),
        "sunglass_model_3": str(_project_root / "models/sunglass/model3_normalized.obj"),
    }


settings = Settings()

# --- Vérifications et Logs au démarrage (optionnel mais utile) ---
# Résout le chemin du modèle ici pour s'assurer qu'il est correct
_resolved_face_model_path = Path(settings.FACE_MODEL_PATH)
if not _resolved_face_model_path.is_absolute():
     _resolved_face_model_path = settings.BASE_DIR / _resolved_face_model_path

print(f"INFO: Using BASE_DIR: {settings.BASE_DIR}")
print(f"INFO: Resolved FACE_MODEL_PATH: {_resolved_face_model_path}")
print(f"INFO: Model Paths Dict: {settings.MODEL_IDS_TO_PATHS}")
# Vérifie l'existence du modèle ici aussi
if not _resolved_face_model_path.exists():
     print(f"ERROR: Face model file not found at resolved path: {_resolved_face_model_path}")

# Vérifie les modèles 3D
for model_id, model_path in settings.MODEL_IDS_TO_PATHS.items():
     if not Path(model_path).exists():
          print(f"WARNING: 3D model file not found for ID '{model_id}': {model_path}")