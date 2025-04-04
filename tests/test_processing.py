# tests/test_processing.py

import pytest
from src.core.processing import (
    get_recommendations_for_face,
    determine_face_shape,
    # Importe les constantes d'indices
    TOP_MOST_POINT_INDEX,
    BOTTOM_MOST_POINT_INDEX,
    LEFT_MOST_POINT_INDEX,
    RIGHT_MOST_POINT_INDEX,
    LEFT_JAW_POINT_INDEX,
    RIGHT_JAW_POINT_INDEX
)
from src.schemas.schemas import Landmark

# --- Tests pour get_recommendations_for_face (inchangés) ---
@pytest.mark.parametrize("face_shape, expected_ids, expected_info_part", [
    ("ronde", ["sunglass_model_1", "sunglass_model_3"], "Ronde"),
    ("carrée", ["sunglass_model_2"], "Carrée"),
    ("ovale", ["sunglass_model_1", "sunglass_model_2", "sunglass_model_3"], "Ovale"),
    ("rectangle", ["sunglass_model_2"], "Rectangle"),
    ("coeur", ["sunglass_model_1"], "Coeur"),
    ("inconnue", ["sunglass_model_1", "sunglass_model_2"], "non reconnue"),
    ("TRIANGLE", ["sunglass_model_1", "sunglass_model_2"], "non reconnue"),
])
def test_get_recommendations_for_face_shapes(face_shape, expected_ids, expected_info_part):
    recommendations, info = get_recommendations_for_face(face_shape)
    assert set(recommendations) == set(expected_ids)
    assert expected_info_part in info

def test_get_recommendations_empty_shape():
    recommendations, info = get_recommendations_for_face("")
    assert set(recommendations) == set(["sunglass_model_1", "sunglass_model_2"])
    assert "non reconnue" in info

# --- Tests pour determine_face_shape (maintenant avec les constantes importées) ---
def test_determine_face_shape_insufficient_landmarks():
    short_landmarks = [Landmark(x=0.5, y=0.5, z=0) for _ in range(10)]
    shape = determine_face_shape(short_landmarks)
    assert shape == "inconnue"

def test_determine_face_shape_no_landmarks():
    shape = determine_face_shape([])
    assert shape == "inconnue"

def test_determine_face_shape_simulated_square():
    landmarks = [Landmark(x=0, y=0, z=0)] * 478 # Utilise un nombre suffisant (ex: 478 pour être sûr)
    # Définir les points clés (utilise les constantes importées)
    landmarks[TOP_MOST_POINT_INDEX] = Landmark(x=0.5, y=0.9, z=0)
    landmarks[BOTTOM_MOST_POINT_INDEX] = Landmark(x=0.5, y=0.1, z=0)
    landmarks[LEFT_MOST_POINT_INDEX] = Landmark(x=0.1, y=0.5, z=0)
    landmarks[RIGHT_MOST_POINT_INDEX] = Landmark(x=0.9, y=0.5, z=0)
    landmarks[LEFT_JAW_POINT_INDEX] = Landmark(x=0.15, y=0.2, z=0) # Jaw plus large pour carré
    landmarks[RIGHT_JAW_POINT_INDEX] = Landmark(x=0.85, y=0.2, z=0)
    shape = determine_face_shape(landmarks)
    assert shape == "carrée"

def test_determine_face_shape_simulated_oval():
    landmarks = [Landmark(x=0, y=0, z=0)] * 478
    # Définir les points clés
    landmarks[TOP_MOST_POINT_INDEX] = Landmark(x=0.5, y=0.95, z=0)
    landmarks[BOTTOM_MOST_POINT_INDEX] = Landmark(x=0.5, y=0.05, z=0)
    landmarks[LEFT_MOST_POINT_INDEX] = Landmark(x=0.2, y=0.5, z=0)
    landmarks[RIGHT_MOST_POINT_INDEX] = Landmark(x=0.8, y=0.5, z=0)
    landmarks[LEFT_JAW_POINT_INDEX] = Landmark(x=0.3, y=0.2, z=0) # Jaw plus étroit pour ovale
    landmarks[RIGHT_JAW_POINT_INDEX] = Landmark(x=0.7, y=0.2, z=0)
    shape = determine_face_shape(landmarks)
    assert shape == "ovale"