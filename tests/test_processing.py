# tests/test_processing.py

import pytest
from src.core.processing import (
    get_recommendations_for_face,
    determine_face_shape,
    # Importe les indices nécessaires
    TOP_FOREHEAD, BOTTOM_CHIN, LEFT_TEMPLE, RIGHT_TEMPLE
)
from src.schemas.schemas import Landmark
from typing import List

# --- Fonctions utilitaires pour créer des landmarks simulés ---
def create_mock_landmarks(points_map: dict, num_total_landmarks: int = 478) -> List[Landmark]:
    landmarks = [Landmark(x=0.5, y=0.5, z=0)] * num_total_landmarks
    for index, coords in points_map.items():
        if 0 <= index < num_total_landmarks:
            landmarks[index] = Landmark(x=coords[0], y=coords[1], z=coords[2] if len(coords)>2 else 0)
        else:
             print(f"Attention: Indice {index} hors limites pour {num_total_landmarks} landmarks.")
    return landmarks

# --- Tests pour get_recommendations_for_face (MODIFIÉS pour les formes simples V6) ---
@pytest.mark.parametrize("face_shape, expected_ids, expected_info_part", [
    ("long", ["sunglass_model_2", "sunglass_model_3"], "Long"),
    ("proportionné", ["sunglass_model_1", "sunglass_model_2", "sunglass_model_3"], "Proportionné"),
    ("autre", ["sunglass_model_1", "sunglass_model_3"], "Autre"),
    ("inconnue", ["sunglass_model_1", "sunglass_model_2"], "non reconnue"),
    ("erreur_calcul", ["sunglass_model_1", "sunglass_model_2"], "non reconnue"),
    ("erreur_indices", ["sunglass_model_1", "sunglass_model_2"], "non reconnue"),
])
def test_get_recommendations_for_face_shapes_simple(face_shape, expected_ids, expected_info_part):
    """ Vérifie les recommandations pour les formes simplifiées V6. """
    recommendations, info = get_recommendations_for_face(face_shape)
    assert set(recommendations) == set(expected_ids)
    # Vérifie que l'information retournée contient bien le nom de la forme attendue
    assert expected_info_part.lower() in info.lower()

# --- Tests SIMPLIFIÉS pour determine_face_shape (V6) ---

def test_determine_face_shape_insufficient_landmarks():
    """ Teste avec moins de landmarks que nécessaire. """
    short_landmarks = create_mock_landmarks({}, num_total_landmarks=10)
    shape = determine_face_shape(short_landmarks)
    assert shape == "inconnue"

def test_determine_face_shape_no_landmarks():
    """ Teste avec une liste vide. """
    shape = determine_face_shape([])
    assert shape == "inconnue"

def test_determine_face_shape_invalid_measures():
    """ Teste le cas où les distances calculées sont nulles. """
    points = { TOP_FOREHEAD: (0.5, 0.5, 0), BOTTOM_CHIN: (0.5, 0.5, 0), LEFT_TEMPLE: (0.5, 0.5, 0), RIGHT_TEMPLE: (0.5, 0.5, 0) }
    landmarks = create_mock_landmarks(points)
    shape = determine_face_shape(landmarks)
    assert shape == "inconnue"

def test_determine_face_shape_simulated_long():
    """ L/W > 1.20 """
    # L=1.0, CW=0.6 -> Ratio=1.67
    points = { TOP_FOREHEAD: (0.5, 1.0, 0), BOTTOM_CHIN: (0.5, 0.0, 0), LEFT_TEMPLE: (0.2, 0.5, 0), RIGHT_TEMPLE: (0.8, 0.5, 0) }
    landmarks = create_mock_landmarks(points)
    shape = determine_face_shape(landmarks)
    assert shape == "long"

def test_determine_face_shape_simulated_proportionate():
    """ 0.90 <= L/W <= 1.20 """
    # L=0.8, CW=0.8 -> Ratio=1.0
    points = { TOP_FOREHEAD: (0.5, 0.9, 0), BOTTOM_CHIN: (0.5, 0.1, 0), LEFT_TEMPLE: (0.1, 0.5, 0), RIGHT_TEMPLE: (0.9, 0.5, 0) }
    landmarks = create_mock_landmarks(points)
    shape = determine_face_shape(landmarks)
    assert shape == "proportionné"

def test_determine_face_shape_simulated_other_wide():
    """ L/W < 0.90 """
    # L=0.6, CW=0.8 -> Ratio=0.75
    points = { TOP_FOREHEAD: (0.5, 0.8, 0), BOTTOM_CHIN: (0.5, 0.2, 0), LEFT_TEMPLE: (0.1, 0.5, 0), RIGHT_TEMPLE: (0.9, 0.5, 0) }
    landmarks = create_mock_landmarks(points)
    shape = determine_face_shape(landmarks)
    assert shape == "autre"

# Les anciens tests pour carré, rond, ovale, coeur, diamant ne sont plus pertinents
# car la logique ne distingue plus ces formes spécifiquement.
# On pourrait ajouter des tests aux limites des ratios RATIO_LONG et RATIO_PROP_LOW si besoin.