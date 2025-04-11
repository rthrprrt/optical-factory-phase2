# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from src.main import app # Importe l'application FastAPI (qui a été simplifiée)
import os
from pathlib import Path # Pour créer le fichier

# Crée un client de test
client = TestClient(app)

# --- Préparation des Données de Test ---
TEST_DATA_ROOT = Path(__file__).parent / "test_data"
VALID_FACE_IMAGE_PATH = TEST_DATA_ROOT / "test_face.jpg"
# Crée un faux fichier image valide s'il manque, pour que les tests ne plantent pas
# mais ils seront sautés si skip_if_no_valid_image est utilisé
if not VALID_FACE_IMAGE_PATH.exists():
    TEST_DATA_ROOT.mkdir(exist_ok=True)
    # Crée un fichier minuscule mais potentiellement décodable (ex: 1x1 pixel blanc JPEG)
    # Pour un vrai test, remplace-le par une vraie image !
    try:
        import numpy as np
        import cv2
        placeholder_img = np.full((1, 1, 3), 255, dtype=np.uint8) # 1x1 pixel blanc
        cv2.imwrite(str(VALID_FACE_IMAGE_PATH), placeholder_img)
        print(f"Placeholder créé pour {VALID_FACE_IMAGE_PATH}")
    except ImportError:
         with open(VALID_FACE_IMAGE_PATH, 'w') as f: f.write("fake") # Si OpenCV n'est pas là
         print(f"Fichier texte placeholder créé pour {VALID_FACE_IMAGE_PATH}")

# Définit si on saute les tests qui nécessitent une vraie image valide
# (Saute si le fichier est absent OU s'il fait moins de 1Ko, suggérant un placeholder)
skip_if_no_valid_image = pytest.mark.skipif(
    not VALID_FACE_IMAGE_PATH.exists() or VALID_FACE_IMAGE_PATH.stat().st_size < 1024,
    reason=f"Nécessite une image réelle dans {VALID_FACE_IMAGE_PATH}"
)

# --- Tests ---

def test_health_check():
    """ Vérifie /health (doit être ok même si modèle absent en mode test). """
    # Simule l'environnement de test
    os.environ["TESTING"] = "true"
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    # En mode test, models_loaded peut être True ou False, mais status est ok
    assert "models_loaded" in json_response
    del os.environ["TESTING"] # Nettoie la variable d'env

def test_read_root():
    """ Teste l'endpoint racine. """
    response = client.get("/")
    assert response.status_code == 200
    assert "Bienvenue" in response.json()["message"]

# --- Tests /analyze_face ---
@skip_if_no_valid_image
def test_analyze_face_success():
    """ Teste l'analyse réussie (nécessite modèle chargé et vraie image). """
    with open(VALID_FACE_IMAGE_PATH, "rb") as img_file:
        response = client.post("/api/v1/analyze_face", files={"image_file": ("test_face.jpg", img_file, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["detection_successful"] is True
    assert json_response["facial_transformation_matrix"] is not None
    assert isinstance(json_response["facial_transformation_matrix"], list)
    assert json_response["face_landmarks"] is not None
    assert isinstance(json_response["face_landmarks"], list)
    assert json_response["detected_face_shape"] in ["long", "proportionné", "autre", "inconnue"] # Formes simples
    assert json_response["error_message"] is None

def test_analyze_face_no_face_in_image():
     """ Simule l'échec de détection (difficile sans modèle mocké). On teste avec une image invalide."""
     # Utilise un fichier texte pour simuler une image indécodable -> échec détection
     invalid_file_path = TEST_DATA_ROOT / "invalid.txt"
     with open(invalid_file_path, "w") as f: f.write("not image data")
     with open(invalid_file_path, "rb") as f:
         response = client.post("/api/v1/analyze_face", files={"image_file": ("invalid.txt", f, "text/plain")})
     assert response.status_code == 200 # Retourne 200 mais avec succès=False
     json_response = response.json()
     assert json_response["detection_successful"] is False
     assert "invalide" in json_response["error_message"].lower()

def test_analyze_face_no_file():
     """ Teste l'appel sans fichier. """
     response = client.post("/api/v1/analyze_face")
     assert response.status_code == 422 # Erreur de validation FastAPI

# --- Tests /recommend_glasses ---
@pytest.mark.parametrize("face_shape, expected_status, expected_key", [
    ({"face_shape": "long"}, 200, "sunglass_model_2"),
    ({"face_shape": "proportionné"}, 200, "sunglass_model_1"),
    ({"face_shape": "autre"}, 200, "sunglass_model_3"),
    ({"face_shape": "inconnue"}, 200, "sunglass_model_1"), # Teste cas inconnu
    ({}, 422, None), # Corps vide
    ({"shape": "long"}, 422, None), # Mauvais nom de champ
])
def test_recommend_glasses(face_shape, expected_status, expected_key):
    response = client.post("/api/v1/recommend_glasses", json=face_shape)
    assert response.status_code == expected_status
    if expected_status == 200:
        json_response = response.json()
        assert "recommended_glasses_ids" in json_response
        assert isinstance(json_response["recommended_glasses_ids"], list)
        if expected_key: # Vérifie si au moins une des clés attendues est là
             assert expected_key in json_response["recommended_glasses_ids"]

# --- Tests /analyze_and_recommend ---
@skip_if_no_valid_image
def test_analyze_and_recommend_success():
    """ Teste le flux combiné (nécessite modèle chargé et vraie image). """
    with open(VALID_FACE_IMAGE_PATH, "rb") as img_file:
        response = client.post("/api/v1/analyze_and_recommend", files={"image_file": ("test_face.jpg", img_file, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    # Vérifie la partie analyse
    assert "analysis" in json_response
    analysis = json_response["analysis"]
    assert analysis["detection_successful"] is True
    assert analysis["facial_transformation_matrix"] is not None
    assert analysis["face_landmarks"] is not None
    assert analysis["detected_face_shape"] is not None
    # Vérifie la partie recommandation
    assert "recommendation" in json_response
    recommendation = json_response["recommendation"]
    assert recommendation is not None # Doit avoir une reco si analyse ok
    assert "recommended_glasses_ids" in recommendation
    assert isinstance(recommendation["recommended_glasses_ids"], list)
    assert len(recommendation["recommended_glasses_ids"]) > 0 # Au moins une reco

def test_analyze_and_recommend_invalid_file():
    """ Teste le flux combiné avec un fichier invalide. """
    invalid_file_path = TEST_DATA_ROOT / "invalid.txt"
    if not invalid_file_path.exists(): # Crée si besoin
         with open(invalid_file_path, "w") as f: f.write("not image data")
    with open(invalid_file_path, "rb") as f:
         response = client.post("/api/v1/analyze_and_recommend", files={"image_file": ("invalid.txt", f, "text/plain")})
    assert response.status_code == 200 # API retourne 200
    json_response = response.json()
    assert json_response["analysis"]["detection_successful"] is False
    assert "invalide" in json_response["analysis"]["error_message"].lower()
    assert json_response["recommendation"] is None # Pas de reco si analyse échoue

# Le test pour /render_glasses a été supprimé.