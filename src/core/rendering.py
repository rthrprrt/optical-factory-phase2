# src/core/rendering.py

import pyrender
import trimesh
import numpy as np
import cv2
import logging
import threading
from typing import Optional, Dict, Tuple
from pathlib import Path
from src.core.models import get_3d_model_path
from src.core.config import settings # <<< Importe l'objet settings

# Utilise le logger configuré dans models.py (ou configure-le ici si besoin)
logger = logging.getLogger(__name__)

# --- Variables Globales ---
_renderer: Optional[pyrender.OffscreenRenderer] = None
_scene: Optional[pyrender.Scene] = None
_camera_node: Optional[pyrender.Node] = None
_loaded_3d_nodes: Dict[str, pyrender.Node] = {}
_render_lock = threading.Lock()

# --- Configuration (utilise settings) ---
# RENDER_WIDTH et RENDER_HEIGHT sont maintenant dans settings

# --- Définition de DEFAULT_GLASSES_OFFSET (utilise trimesh) ---
rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
scale_matrix = np.eye(4)
scale_matrix[0, 0] = 1.7
scale_matrix[1, 1] = 1.7
scale_matrix[2, 2] = 1.7
translation_matrix = trimesh.transformations.translation_matrix([0, 0.1, 0])
DEFAULT_GLASSES_OFFSET = translation_matrix @ rotation_matrix @ scale_matrix

# --- Fonctions ---

def initialize_renderer(): # Ne prend plus width/height en argument
    """ Initialise le renderer et la scène pyrender en utilisant les settings. """
    global _renderer, _scene, _camera_node, _render_lock
    # Utilise les valeurs depuis settings
    width = settings.RENDER_WIDTH
    height = settings.RENDER_HEIGHT

    with _render_lock:
        if _renderer is None:
            logger.info(f"Initialisation du Renderer PyRender ({width}x{height})...")
            try:
                _renderer = pyrender.OffscreenRenderer(width, height)
                directional_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
                _scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])
                _scene.add(directional_light, pose=np.eye(4))
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
                camera_pose = np.eye(4)
                _camera_node = _scene.add(camera, pose=camera_pose)
                logger.info("Renderer PyRender initialisé avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de PyRender : {e}", exc_info=True)
                _renderer = None
        return _renderer is not None

def _load_glasses_model(model_id: str) -> Optional[pyrender.Node]:
    """ Charge un modèle 3D si pas déjà en cache, retourne le noeud Pyrender. """
    global _loaded_3d_nodes
    if model_id in _loaded_3d_nodes:
        return _loaded_3d_nodes[model_id]

    # Utilise get_3d_model_path qui lit depuis settings
    model_path_str = get_3d_model_path(model_id)
    if not model_path_str:
        # get_3d_model_path loggue déjà l'avertissement
        return None

    model_path = Path(model_path_str)
    # Pas besoin de revérifier l'existence, get_3d_model_path le fait

    try:
        logger.info(f"Chargement du modèle 3D : {model_path}")
        mesh_or_scene = trimesh.load(model_path_str, force='mesh', process=False)

        if isinstance(mesh_or_scene, trimesh.Scene):
             if len(mesh_or_scene.geometry) > 0:
                 mesh = trimesh.util.concatenate(list(mesh_or_scene.geometry.values()))
                 logger.info(f"Scène {model_id} fusionnée en un seul mesh.")
             else:
                 logger.warning(f"La scène {model_path} ne contient aucune géométrie.")
                 return None
        elif isinstance(mesh_or_scene, trimesh.Trimesh):
             mesh = mesh_or_scene
        else:
            logger.warning(f"Format non géré retourné par trimesh pour {model_path}")
            return None

        render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        node = pyrender.Node(mesh=render_mesh, matrix=np.eye(4))
        _loaded_3d_nodes[model_id] = node
        logger.info(f"Modèle {model_id} chargé avec succès en tant que noeud Pyrender.")
        return node
    except Exception as e:
        logger.error(f"Erreur lors du chargement/conversion du modèle {model_path} : {e}", exc_info=True)
        return None

def render_overlay(background_image: np.ndarray, face_pose_matrix: np.ndarray, model_id: str) -> Optional[np.ndarray]:
    """
    Rend un modèle 3D sur une image de fond en utilisant la pose du visage.
    """
    global _renderer, _scene, _camera_node, _render_lock

    if _renderer is None or _scene is None:
        logger.error("Renderer non initialisé. Impossible de rendre.")
        return None
    if background_image is None or background_image.size == 0:
        logger.error("Image de fond invalide fournie pour le rendu.")
        return None
    if face_pose_matrix is None or face_pose_matrix.shape != (4, 4):
        logger.error("Matrice de pose invalide fournie pour le rendu.")
        return None

    glasses_node = _load_glasses_model(model_id)
    if glasses_node is None:
        logger.warning(f"Impossible de charger le modèle '{model_id}' pour le rendu.")
        return background_image

    with _render_lock:
        node_in_scene = None
        try:
            final_pose = face_pose_matrix @ DEFAULT_GLASSES_OFFSET
            node_in_scene = _scene.add_node(glasses_node)
            _scene.set_pose(node_in_scene, final_pose)

            flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.SKIP_CULL_FACES
            color_rgba, depth = _renderer.render(_scene, flags=flags)

            # --- Composition Alpha ---
            if background_image.shape[2] == 3:
                background_rgba = cv2.cvtColor(background_image, cv2.COLOR_BGR2BGRA)
            elif background_image.shape[2] == 4:
                background_rgba = background_image
            else:
                logger.error(f"Format d'image de fond non supporté (canaux={background_image.shape[2]})")
                return background_image

            # Redimensionne si nécessaire (utilise les dimensions des settings)
            target_height = settings.RENDER_HEIGHT
            target_width = settings.RENDER_WIDTH
            if background_rgba.shape[0] != target_height or background_rgba.shape[1] != target_width:
                logger.warning(f"Redimensionnement de l'image de fond ({background_rgba.shape[:2]}) vers ({target_height}x{target_width}) pour correspondre au renderer.")
                background_rgba = cv2.resize(background_rgba, (target_width, target_height))

            alpha = color_rgba[:, :, 3:4].astype(float) / 255.0
            color_rgb = color_rgba[:, :, :3].astype(float)
            background_rgb_float = background_rgba[:, :, :3][:,:,::-1].astype(float) # BGRA -> RGB float

            composite_rgb_float = (color_rgb * alpha) + (background_rgb_float * (1.0 - alpha))
            composite_rgb_uint8 = np.clip(composite_rgb_float, 0, 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(composite_rgb_uint8, cv2.COLOR_RGB2BGR)
            # --- Fin Composition ---

            return output_bgr

        except Exception as e:
            logger.error(f"Erreur inattendue lors du rendu du modèle '{model_id}' : {e}", exc_info=True)
            return background_image
        finally:
            if node_in_scene is not None and _scene.has_node(node_in_scene):
                _scene.remove_node(node_in_scene)