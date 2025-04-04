# src/schemas/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple

class Landmark(BaseModel):
    x: float
    y: float
    z: float

class FaceAnalysisResult(BaseModel):
    detection_successful: bool = Field(..., description="Indique si un visage a été détecté avec succès.")
    facial_transformation_matrix: Optional[List[List[float]]] = Field(None, description="Matrice de transformation 4x4 représentant la pose du visage détecté.")
    face_landmarks: Optional[List[Landmark]] = Field(None, description="Liste des 468+ landmarks faciaux détectés (coordonnées normalisées).")
    detected_face_shape: Optional[str] = Field(None, description="Forme du visage estimée à partir des landmarks.")
    error_message: Optional[str] = Field(None, description="Message d'erreur en cas d'échec de la détection ou de l'analyse.")
    # Met l'exemple dans json_schema_extra via model_config
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detection_successful": True,
                "facial_transformation_matrix": [
                    [0.99, 0.01, 0.01, 0.1], [0.01, 0.99, 0.01, 0.2], [0.01, 0.01, 0.99, 0.3], [0.0, 0.0, 0.0, 1.0]
                ],
                "face_landmarks": [{"x": 0.5, "y": 0.5, "z": -0.02}, {"x": 0.6, "y": 0.4, "z": -0.01}],
                "detected_face_shape": "ovale",
                "error_message": None
            }
        }
    )

class RecommendationRequest(BaseModel):
    # Met l'exemple dans json_schema_extra via Field directement
    face_shape: str = Field(..., description="Forme du visage détectée ou supposée (ex: 'ronde', 'carrée', 'ovale').", json_schema_extra={'example': "ovale"})

class RecommendationResult(BaseModel):
    recommended_glasses_ids: List[str] = Field(..., description="Liste des identifiants des modèles de lunettes recommandés.")
    analysis_info: Optional[str] = Field(None, description="Informations sur l'analyse ayant mené à la recommandation (ex: forme du visage détectée).")
    # Met l'exemple dans json_schema_extra via model_config
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "recommended_glasses_ids": ["sunglass_model_1", "classic_frame_v2"],
                "analysis_info": "Forme de visage détectée : Ovale"
            }
        }
    )

class AnalyzeAndRecommendResult(BaseModel):
    analysis: FaceAnalysisResult = Field(..., description="Résultats détaillés de l'analyse faciale.")
    recommendation: Optional[RecommendationResult] = Field(None, description="Recommandations de lunettes basées sur l'analyse (si réussie).")
    # Met l'exemple dans json_schema_extra via model_config
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "analysis": { "detection_successful": True, "facial_transformation_matrix": [[0.99,0.01,0.01,0.1],[0.01,0.99,0.01,0.2],[0.01,0.01,0.99,0.3],[0.0,0.0,0.0,1.0]], "face_landmarks": [{"x":0.5,"y":0.5,"z":-0.02},{"x":0.6,"y":0.4,"z":-0.01}], "detected_face_shape": "ovale", "error_message": None },
                "recommendation": { "recommended_glasses_ids": ["sunglass_model_1","sunglass_model_2","sunglass_model_3"], "analysis_info": "Forme de visage utilisée pour la recommandation : Ovale" }
            }
        }
    )