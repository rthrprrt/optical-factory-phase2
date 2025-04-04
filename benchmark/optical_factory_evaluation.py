# benchmark/optical_factory_evaluation.py

import requests # Pour faire les appels API
import json
import time
import numpy as np
from pathlib import Path
import logging
import os
from typing import List # <--- AJOUTER CETTE LIGNE

# --- Configuration ---
API_BASE_URL = "http://localhost:8000" # L'URL où tourne ton API FastAPI
ANALYZE_RECOMMEND_ENDPOINT = f"{API_BASE_URL}/api/v1/analyze_and_recommend"
TEST_DATA_DIR = Path(__file__).parent / "test_data" # Chemin vers les images de test
OUTPUT_REPORT_PATH = Path(__file__).parent / "evaluation_results.json"
CONFIG_CRITERIA_PATH = Path(__file__).parent.parent / "config" / "evaluation_criteria.json" # Chemin vers les critères cibles

# Configuration du logging pour le benchmark
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark")

# --- Chargement des Critères Cibles ---
def load_target_criteria():
    """ Charge les seuils cibles depuis le fichier JSON de configuration. """
    default_criteria = {
        "facial_detection_precision": 0.95, # % de détections réussies
        "shape_determination_accuracy": 0.70, # % de formes correctes (si ground truth dispo)
        "inference_latency_ms": 50.0, # ms
        # Ajoute d'autres critères ici si définis dans le JSON
    }
    try:
        # Assure-toi que le dossier config existe
        os.makedirs(CONFIG_CRITERIA_PATH.parent, exist_ok=True)
        if CONFIG_CRITERIA_PATH.exists():
            with open(CONFIG_CRITERIA_PATH, 'r') as f:
                config_data = json.load(f)
                # TODO: Extraire les valeurs du JSON et les fusionner/remplacer les défauts
                # Exemple: default_criteria.update(config_data.get("thresholds", {}))
                logger.info(f"Critères cibles chargés depuis {CONFIG_CRITERIA_PATH}")
                return default_criteria # Retourne les critères chargés/mis à jour
        else:
            logger.warning(f"Fichier {CONFIG_CRITERIA_PATH} non trouvé. Utilisation des seuils par défaut.")
            # Crée un fichier par défaut si absent
            with open(CONFIG_CRITERIA_PATH, 'w') as f:
                 json.dump({"description": "Seuils cibles pour l'évaluation", "thresholds": default_criteria, "justifications": {}}, f, indent=2)
            return default_criteria
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {CONFIG_CRITERIA_PATH}: {e}. Utilisation des seuils par défaut.")
        return default_criteria

TARGET_CRITERIA = load_target_criteria()

# --- Fonctions d'Évaluation ---

def evaluate_detection_and_shape(test_image_paths: List[Path]) -> dict:
    """
    Évalue la réussite de la détection et potentiellement la précision de la forme.

    Critères Grille Évaluation:
        - Précision de détection faciale (>95% de succès sur les images attendues)
        - (Optionnel) Précision de la forme (si ground truth disponible)
    """
    logger.info(f"Évaluation de la détection/forme sur {len(test_image_paths)} images...")
    results = []
    successful_detections = 0
    correct_shapes = 0 # Nécessite une vérité terrain (ground truth)
    total_processed = 0

    for image_path in test_image_paths:
        if not image_path.is_file():
            logger.warning(f"Fichier non trouvé : {image_path}, ignoré.")
            continue

        total_processed += 1
        try:
            with open(image_path, "rb") as f:
                files = {"image_file": (image_path.name, f, f"image/{image_path.suffix.strip('.')}")}
                response = requests.post(ANALYZE_RECOMMEND_ENDPOINT, files=files, timeout=10) # Timeout de 10s

            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP >= 400
            data = response.json()
            analysis_data = data.get("analysis", {})
            detected_shape = analysis_data.get("detected_face_shape")
            is_success = analysis_data.get("detection_successful", False)

            results.append({
                "image": image_path.name,
                "status_code": response.status_code,
                "detection_successful": is_success,
                "detected_shape": detected_shape
            })

            if is_success:
                successful_detections += 1
                # --- Comparaison avec vérité terrain (si disponible) ---
                # Exemple : si le nom du fichier contient la forme attendue
                # expected_shape = image_path.stem.split('_')[-1] # Suppose '..._ovale.jpg'
                # if detected_shape and detected_shape.lower() == expected_shape.lower():
                #    correct_shapes += 1

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur API pour {image_path.name}: {e}")
            results.append({"image": image_path.name, "status_code": "API Error", "error": str(e)})
        except Exception as e:
             logger.error(f"Erreur inattendue pour {image_path.name}: {e}")
             results.append({"image": image_path.name, "status_code": "Script Error", "error": str(e)})

    # Calcul des métriques
    detection_precision = (successful_detections / total_processed) if total_processed > 0 else 0
    # shape_accuracy = (correct_shapes / successful_detections) if successful_detections > 0 else 0 # Si ground truth

    threshold = TARGET_CRITERIA.get("facial_detection_precision", 0.95)
    status = "Atteint" if detection_precision >= threshold else "Non atteint"

    logger.info(f"Précision détection: {detection_precision:.2%}")

    # Retourne les résultats formatés
    return {
        "metric": "facial_detection_precision",
        "value": detection_precision,
        "threshold": threshold,
        "status": status,
        "details": {
            "total_images": total_processed,
            "successful_detections": successful_detections,
            "individual_results": results
            # "shape_accuracy": shape_accuracy # Si calculée
        }
    }

def evaluate_inference_latency(test_image_paths: List[Path], num_runs: int = 10) -> dict:
    """
    Évalue la latence moyenne d'inférence en appelant l'API plusieurs fois.

    Critère Grille Évaluation: Latence < 50 ms (cible très ambitieuse pour ce workflow)
    """
    logger.info(f"Évaluation de la latence sur {num_runs} appels...")
    latencies = []
    # Utilise la première image valide pour les tests de latence
    valid_image_path = next((p for p in test_image_paths if p.is_file()), None)

    if not valid_image_path:
        logger.error("Aucune image valide trouvée pour le test de latence.")
        return {"metric": "inference_latency_ms", "value": -1, "status": "Erreur", "details": "Pas d'image"}

    # Chauffe : effectue un premier appel pour potentiellement charger des caches, etc.
    try:
        with open(valid_image_path, "rb") as f:
            files = {"image_file": (valid_image_path.name, f, f"image/{valid_image_path.suffix.strip('.')}")}
            requests.post(ANALYZE_RECOMMEND_ENDPOINT, files=files, timeout=10)
    except requests.exceptions.RequestException as e:
         logger.error(f"Erreur lors de l'appel de chauffe pour latence : {e}")
         # Ne pas arrêter, essayer quand même les runs chronométrés

    # Mesures réelles
    for i in range(num_runs):
        try:
            start_time = time.perf_counter()
            with open(valid_image_path, "rb") as f:
                files = {"image_file": (valid_image_path.name, f, f"image/{valid_image_path.suffix.strip('.')}")}
                response = requests.post(ANALYZE_RECOMMEND_ENDPOINT, files=files, timeout=10)
                response.raise_for_status() # Vérifie si l'appel a réussi
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            logger.debug(f"Run {i+1}/{num_runs}: Latence = {latency_ms:.2f} ms")
            # Petite pause pour éviter de surcharger ? Optionnel.
            # time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur API lors du run {i+1} de latence: {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue lors du run {i+1} de latence: {e}")


    if not latencies:
        logger.error("Aucune mesure de latence réussie.")
        avg_latency = -1
        status = "Erreur"
    else:
        avg_latency = np.mean(latencies)
        threshold = TARGET_CRITERIA.get("inference_latency_ms", 50.0)
        status = "Atteint" if avg_latency <= threshold else "Non atteint"
        logger.info(f"Latence moyenne: {avg_latency:.2f} ms")


    return {
        "metric": "inference_latency_ms",
        "value": avg_latency,
        "threshold": threshold,
        "status": status,
        "details": {"num_runs": num_runs, "latencies_ms": latencies}
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
        "api_url": API_BASE_URL,
        "test_data_source": str(test_data_path),
        "metrics": []
    }

    # Récupère la liste des fichiers images de test
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    test_image_paths = [p for p in test_data_path.glob("*") if p.suffix.lower() in image_extensions]

    if not test_image_paths:
        logger.error(f"Aucun fichier image trouvé dans {test_data_path}")
        report["error"] = "Aucune image de test trouvée."
        return report

    # --- Exécute les évaluations ---
    report["metrics"].append(evaluate_detection_and_shape(test_image_paths))
    report["metrics"].append(evaluate_inference_latency(test_image_paths, num_runs=10))

    # Ajoute d'autres métriques ici si besoin
    # Ex: evaluate_memory_usage(), evaluate_algorithmic_fairness()

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
    # Vérifie si l'API est accessible avant de lancer
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        if response.json().get("status") != "ok":
            raise ValueError("API health check failed.")
        logger.info("API joignable, lancement du benchmark...")
    except requests.exceptions.RequestException as e:
        logger.error(f"ERREUR: Impossible de joindre l'API à {API_BASE_URL}. Vérifiez qu'elle est lancée.")
        logger.error(f"Détail: {e}")
        exit(1) # Arrête le script si l'API n'est pas là
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de l'API: {e}")
        exit(1)

    # Génère et affiche le rapport
    final_report = generate_evaluation_report(TEST_DATA_DIR)

    # Affiche un résumé dans la console
    print("\n" + "="*15 + " RÉSUMÉ DE L'ÉVALUATION " + "="*15)
    summary = final_report.get("summary", {})
    print(f"Critères Atteints: {summary.get('criteria_met', 0)} / {summary.get('total_criteria_evaluated', 0)}")
    print(f"Taux de Réussite: {summary.get('success_rate', 0):.1%}")
    print("\nDétail par Métrique:")
    for metric_result in final_report.get("metrics", []):
        value = metric_result.get('value', 'N/A')
        if isinstance(value, float) and value != -1:
            value_str = f"{value:.2f}" if metric_result['metric'] != "facial_detection_precision" else f"{value:.2%}"
        else:
            value_str = str(value)
        print(f"  - {metric_result.get('metric', 'Inconnue')}: {value_str} (Seuil: {metric_result.get('threshold', 'N/A')}) -> {metric_result.get('status', 'N/A')}")

    # Sauvegarde le rapport complet en JSON
    try:
        with open(OUTPUT_REPORT_PATH, "w") as f:
            json.dump(final_report, f, indent=2)
        logger.info(f"Rapport d'évaluation complet sauvegardé dans : {OUTPUT_REPORT_PATH}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du rapport JSON : {e}")