# benchmark/optical_factory_evaluation.py
import sys
import requests # Pour faire les appels API
import json
import time
import numpy as np
from pathlib import Path
import logging
import os
from typing import List # Assure que List est importé
# Importe l'objet settings depuis la configuration centrale
from src.core.config import settings

# --- Configuration (Utilise l'objet settings) ---
API_BASE_URL = settings.API_BASE_URL if hasattr(settings, 'API_BASE_URL') else "http://localhost:8000"
ANALYZE_RECOMMEND_ENDPOINT = f"{API_BASE_URL}/api/v1/analyze_and_recommend"
# Construit le chemin des données de test à partir de BASE_DIR de settings
TEST_DATA_DIR = settings.BASE_DIR / "benchmark" / "test_data"
# Construit le chemin du rapport de sortie à partir de BASE_DIR
OUTPUT_REPORT_PATH = settings.BASE_DIR / "benchmark" / "evaluation_results.json"
# Récupère les seuils cibles directement depuis settings
TARGET_CRITERIA = {
    "facial_detection_precision": settings.TARGET_DETECTION_PRECISION,
    "inference_latency_ms": settings.TARGET_LATENCY_MS,
    # Ajoute d'autres seuils si définis dans l'objet Settings
    "shape_determination_accuracy": settings.SHAPE_DETERMINATION_ACCURACY if hasattr(settings, 'SHAPE_DETERMINATION_ACCURACY') else 0.70,
}

# Configuration du logging (utilise le niveau de log de settings)
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark")

# La fonction load_target_criteria n'est plus nécessaire car on utilise settings

# --- Fonctions d'Évaluation ---

def evaluate_detection_and_shape(test_image_paths: List[Path]) -> dict:
    """
    Évalue la réussite de la détection et potentiellement la précision de la forme.
    """
    logger.info(f"Évaluation de la détection/forme sur {len(test_image_paths)} images...")
    results = []
    successful_detections = 0
    # correct_shapes = 0 # Pourrait être réintroduit si ground truth
    total_processed = 0

    for image_path in test_image_paths:
        if not image_path.is_file():
            logger.warning(f"Fichier non trouvé ou n'est pas un fichier : {image_path}, ignoré.")
            continue

        total_processed += 1
        try:
            with open(image_path, "rb") as f:
                # Détermine le type MIME de manière simple basé sur l'extension
                extension = image_path.suffix.lower().strip('.')
                mime_type = f"image/{extension}" if extension in ['jpg', 'jpeg', 'png', 'bmp'] else 'application/octet-stream'
                files = {"image_file": (image_path.name, f, mime_type)}
                response = requests.post(ANALYZE_RECOMMEND_ENDPOINT, files=files, timeout=15) # Augmente un peu le timeout

            # Vérifie si le code status est 2xx (succès)
            if 200 <= response.status_code < 300:
                data = response.json()
                analysis_data = data.get("analysis", {})
                detected_shape = analysis_data.get("detected_face_shape")
                is_success = analysis_data.get("detection_successful", False)
                error_msg = analysis_data.get("error_message")

                results.append({
                    "image": image_path.name,
                    "status_code": response.status_code,
                    "detection_successful": is_success,
                    "detected_shape": detected_shape,
                    "error_message": error_msg
                })

                if is_success:
                    successful_detections += 1
                    # --- Comparaison Ground Truth (placeholder) ---
            else:
                 # Gère les erreurs API (4xx, 5xx)
                 logger.error(f"Erreur API {response.status_code} pour {image_path.name}: {response.text[:200]}") # Loggue début de réponse
                 results.append({"image": image_path.name, "status_code": response.status_code, "error": response.text[:200]})

        except requests.exceptions.Timeout:
            logger.error(f"Timeout API pour {image_path.name}")
            results.append({"image": image_path.name, "status_code": "Timeout", "error": "Request timed out"})
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de connexion API pour {image_path.name}: {e}")
            results.append({"image": image_path.name, "status_code": "Connection Error", "error": str(e)})
        except Exception as e:
             logger.error(f"Erreur inattendue lors du traitement de {image_path.name}: {e}", exc_info=True)
             results.append({"image": image_path.name, "status_code": "Script Error", "error": str(e)})

    # Calcul des métriques
    detection_precision = (successful_detections / total_processed) if total_processed > 0 else 0
    threshold = TARGET_CRITERIA.get("facial_detection_precision", 0.95)
    status = "Atteint" if detection_precision >= threshold else "Non atteint"
    logger.info(f"Précision détection: {detection_precision:.2%}")

    return {
        "metric": "facial_detection_precision",
        "value": detection_precision,
        "threshold": threshold,
        "status": status,
        "details": {
            "total_images": total_processed,
            "successful_detections": successful_detections,
            "individual_results": results
        }
    }

def evaluate_inference_latency(test_image_paths: List[Path], num_runs: int = 10) -> dict:
    """
    Évalue la latence moyenne d'inférence en appelant l'API plusieurs fois.
    """
    logger.info(f"Évaluation de la latence sur {num_runs} appels...")
    latencies = []
    valid_image_path = next((p for p in test_image_paths if p.is_file()), None)

    if not valid_image_path:
        logger.error("Aucune image valide trouvée pour le test de latence.")
        return {"metric": "inference_latency_ms", "value": -1, "status": "Erreur", "details": "Pas d'image valide"}

    # Appel de chauffe
    try:
        with open(valid_image_path, "rb") as f:
            extension = valid_image_path.suffix.lower().strip('.')
            mime_type = f"image/{extension}" if extension in ['jpg', 'jpeg', 'png', 'bmp'] else 'application/octet-stream'
            files = {"image_file": (valid_image_path.name, f, mime_type)}
            requests.post(ANALYZE_RECOMMEND_ENDPOINT, files=files, timeout=15)
    except Exception as e:
         logger.warning(f"Erreur lors de l'appel de chauffe pour latence (continuons): {e}")

    # Mesures
    for i in range(num_runs):
        try:
            start_time = time.perf_counter()
            with open(valid_image_path, "rb") as f:
                extension = valid_image_path.suffix.lower().strip('.')
                mime_type = f"image/{extension}" if extension in ['jpg', 'jpeg', 'png', 'bmp'] else 'application/octet-stream'
                files = {"image_file": (valid_image_path.name, f, mime_type)}
                response = requests.post(ANALYZE_RECOMMEND_ENDPOINT, files=files, timeout=15)
                response.raise_for_status() # Lève une exception si status >= 400
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            logger.debug(f"Run {i+1}/{num_runs}: Latence = {latency_ms:.2f} ms")
        except requests.exceptions.Timeout:
            logger.error(f"Timeout API lors du run {i+1} de latence.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur API lors du run {i+1} de latence: {e}")
        except Exception as e:
            logger.error(f"Erreur script lors du run {i+1} de latence: {e}", exc_info=True)

    # Calcul du résultat
    avg_latency = -1
    status = "Erreur"
    if latencies:
        avg_latency = np.mean(latencies)
        threshold = TARGET_CRITERIA.get("inference_latency_ms", 50.0) # Garde seuil par défaut si non défini
        status = "Atteint" if avg_latency <= threshold else "Non atteint"
        logger.info(f"Latence moyenne ({len(latencies)} runs): {avg_latency:.2f} ms")
    else:
        logger.error("Aucune mesure de latence réussie.")

    return {
        "metric": "inference_latency_ms",
        "value": avg_latency,
        "threshold": threshold,
        "status": status,
        "details": {"num_runs_requested": num_runs, "num_runs_successful": len(latencies), "latencies_ms": latencies}
    }

# --- Fonction Principale du Benchmark ---

def generate_evaluation_report(test_data_path: Path):
    """
    Génère un rapport d'évaluation complet en exécutant les différentes métriques.
    """
    logger.info("="*20 + " DÉBUT DE L'ÉVALUATION " + "="*20)
    report = {
        "project": "Optical Factory",
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_url": API_BASE_URL, # Depuis settings
        "test_data_source": str(test_data_path),
        "target_criteria": TARGET_CRITERIA, # Inclut les seuils utilisés
        "metrics": []
    }

    # Vérifie si le dossier de données existe
    if not test_data_path.is_dir():
        logger.error(f"Le dossier de données de test n'existe pas : {test_data_path}")
        report["error"] = f"Dossier de données de test non trouvé: {test_data_path}"
        return report

    # Récupère la liste des fichiers images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    # Utilise rglob pour chercher aussi dans les sous-dossiers si nécessaire, sinon glob
    test_image_paths = [p for p in test_data_path.glob("*") if p.is_file() and p.suffix.lower() in image_extensions]

    if not test_image_paths:
        logger.error(f"Aucun fichier image ({', '.join(image_extensions)}) trouvé dans {test_data_path}")
        report["error"] = "Aucune image de test trouvée."
        return report

    logger.info(f"{len(test_image_paths)} images trouvées pour l'évaluation.")

    # --- Exécute les évaluations ---
    try:
        report["metrics"].append(evaluate_detection_and_shape(test_image_paths))
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation detection/shape: {e}", exc_info=True)
        report["metrics"].append({"metric": "facial_detection_precision", "status": "Erreur Script", "error": str(e)})

    try:
        report["metrics"].append(evaluate_inference_latency(test_image_paths, num_runs=10))
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation de latence: {e}", exc_info=True)
        report["metrics"].append({"metric": "inference_latency_ms", "status": "Erreur Script", "error": str(e)})


    # --- Calcule le résumé global ---
    total_criteria = len(report["metrics"])
    met_criteria = sum(1 for m in report["metrics"] if m.get("status") == "Atteint")
    report["summary"] = {
        "total_criteria_evaluated": total_criteria,
        "criteria_met": met_criteria,
        "success_rate": (met_criteria / total_criteria) if total_criteria > 0 else 0
    }
    logger.info("="*20 + " FIN DE L'ÉVALUATION " + "="*20)

    return report

# --- Exécution du Script ---

if __name__ == "__main__":
    # Vérifie si l'API est accessible
    logger.info(f"Vérification de l'API à {API_BASE_URL}...")
    api_ok = False
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status() # Lève pour 4xx/5xx
        health_data = response.json()
        if health_data.get("status") == "ok":
            api_ok = True
            logger.info(f"API joignable et health check OK (modèles chargés: {health_data.get('models_loaded', 'Inconnu')}).")
        else:
            logger.error(f"API joignable mais health check a échoué: {health_data}")
    except requests.exceptions.Timeout:
        logger.error(f"ERREUR: Timeout en essayant de joindre l'API à {API_BASE_URL}.")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"ERREUR: Impossible de se connecter à l'API à {API_BASE_URL}. Vérifiez qu'elle est lancée et accessible.")
        logger.error(f"Détail: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la requête vers l'API: {e}")
    except Exception as e:
         logger.error(f"Erreur inattendue lors de la vérification de l'API: {e}", exc_info=True)

    if not api_ok:
        logger.error("Le benchmark ne peut pas continuer car l'API n'est pas prête.")
        exit(1) # Arrête le script

    # Génère le rapport
    logger.info(f"Lancement de la génération du rapport avec les données de {TEST_DATA_DIR}...")
    final_report = generate_evaluation_report(TEST_DATA_DIR) # Utilise TEST_DATA_DIR global

    # Affiche le résumé
    print("\n" + "="*15 + " RÉSUMÉ DE L'ÉVALUATION " + "="*15)
    summary = final_report.get("summary", {})
    print(f"Critères Atteints: {summary.get('criteria_met', 0)} / {summary.get('total_criteria_evaluated', 0)}")
    print(f"Taux de Réussite: {summary.get('success_rate', 0):.1%}")
    print("\nDétail par Métrique:")
    for metric_result in final_report.get("metrics", []):
        value = metric_result.get('value', 'N/A')
        value_str = "N/A"
        if isinstance(value, (int, float)) and value != -1 : # Vérifie si c'est un nombre valide
             if metric_result.get('metric') == "facial_detection_precision":
                 value_str = f"{value:.2%}"
             elif metric_result.get('metric') == "inference_latency_ms":
                 value_str = f"{value:.2f} ms"
             else:
                 value_str = f"{value:.2f}" # Format générique pour d'autres métriques
        elif value == -1:
             value_str = "Erreur Calcul"

        print(f"  - {metric_result.get('metric', 'Inconnue'):<30}: {value_str:<15} (Seuil: {metric_result.get('threshold', 'N/A')}) -> {metric_result.get('status', 'N/A')}")

    # Sauvegarde le rapport
    logger.info(f"Sauvegarde du rapport dans {OUTPUT_REPORT_PATH}...")
    try:
        # Crée le dossier parent si nécessaire
        OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_REPORT_PATH, "w") as f:
            json.dump(final_report, f, indent=2)
        logger.info(f"Rapport d'évaluation complet sauvegardé.")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du rapport JSON : {e}", exc_info=True)