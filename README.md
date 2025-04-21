# Optical Factory - Backend API (Phase 2)

[![Python Backend CI](https://github.com/rthrprrt/optical-factory-phase2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/rthrprrt/optical-factory-phase2/actions/workflows/python-ci.yml)

A RESTful backend API for the "Optical Factory" Master's project, developed using Python and FastAPI. It provides robust facial analysis capabilities (3D pose, landmarks, simplified shape classification) and glasses recommendation logic, designed to power a virtual try-on application. **Note:** 3D rendering is handled client-side; this API provides the necessary analysis data.

## Core API Features

*   **Facial Analysis (`POST /api/v1/analyze_face`):**
    *   Accepts an image file (`multipart/form-data`).
    *   Returns a JSON (`FaceAnalysisResult`) containing:
        *   `detection_successful` (boolean).
        *   `facial_transformation_matrix` (4x4 float list): The crucial 3D pose matrix of the detected face relative to the camera. **Essential for client-side rendering.**
        *   `face_landmarks` (list of {x, y, z} objects): Detailed 470+ facial landmark coordinates. **Useful for fine-tuning placement or effects on the client.**
        *   `detected_face_shape` (string: "long", "proportionate", "other", "unknown", or error): Simplified shape classification based on landmarks.
        *   `error_message` (string or null).
*   **Glasses Recommendation (`POST /api/v1/recommend_glasses`):**
    *   Accepts a face shape (string in JSON body, e.g., `{ "face_shape": "long" }`).
    *   Returns a JSON (`RecommendationResult`) containing `recommended_glasses_ids` (list of strings) and `analysis_info` (string).
*   **Combined Workflow (`POST /api/v1/analyze_and_recommend`):**
    *   Accepts an image file.
    *   Performs full analysis and returns both the `FaceAnalysisResult` and `RecommendationResult` in a single JSON response (`AnalyzeAndRecommendResult`).
*   **Health Check (`GET /health`):** Verifies API availability and Mediapipe model load status.

*Detailed API specification and interactive testing available via Swagger UI at the `/docs` endpoint.*

## Architecture Overview

This backend leverages FastAPI for a high-performance, asynchronous API with automatic data validation (Pydantic) and OpenAPI documentation. The modular structure (`src/api`, `src/core`, `src/schemas`) promotes maintainability.

*   **Key Components:** FastAPI application (`main.py`), API route definitions (`endpoints.py`), core processing logic (`processing.py`), Mediapipe model loading (`models.py`), configuration management (`config.py`, `.env`), Pydantic schemas (`schemas.py`).
*   **3D Rendering:** Explicitly **delegated to the client application**. The backend focuses solely on providing accurate analysis data (pose matrix, landmarks).

*Refer to [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed system diagram and description.*

## Technologies (Backend)

*   Python 3.10+
*   FastAPI & Uvicorn
*   Mediapipe (FaceLandmarker)
*   OpenCV-Python (Headless)
*   Pydantic & Pydantic-Settings
*   Pytest & HTTPX
*   Docker
*   Git & Git LFS
*   GitHub Actions
*   NumPy

## For Frontend Developers: Using the API

1.  **API Endpoint:** Get the deployed API URL (e.g., `https://<your-app>.onrender.com`) or use `http://localhost:8000` for local backend instances.
2.  **API Docs:** Access `<API_URL>/docs` for detailed endpoint information.
3.  **Virtual Try-On Flow:**
    *   Capture a webcam frame/image.
    *   Send the image to `POST /api/v1/analyze_face`.
    *   On successful response (`detection_successful: true`), retrieve the `facial_transformation_matrix` (4x4 list) and `face_landmarks` (list of {x,y,z}).
    *   **(Optional)** Get recommendations via `POST /api/v1/recommend_glasses` using the `detected_face_shape`.
    *   **Client-Side Rendering (e.g., using Three.js/WebGL):**
        *   Load the desired 3D glasses model (e.g., a `.glb` file corresponding to a recommended ID).
        *   Display the webcam feed as a background.
        *   Set up your 3D scene and virtual camera.
        *   **Apply the `facial_transformation_matrix`** received from the API to the transform (position and rotation) of your 3D glasses object in the scene. *Note: Coordinate system differences between Mediapipe (often OpenGL-like) and your 3D library (e.g., Three.js/WebGL) might require adjustments or matrix conversions.* A local offset might also be needed for precise fitting on the nose.
        *   Use the `face_landmarks` for optional debugging visualization or advanced fitting/deformation.
        *   Render your 3D scene over the webcam feed.
4.  **3D Models:** Obtain the 3D model files (e.g., `.glb`, `.obj`) corresponding to the `recommended_glasses_ids`. These are stored in the `models/sunglass` directory of this repository but need to be hosted or bundled with the frontend application.

## Getting Started (Backend Development)

### Prerequisites

*   Python 3.10+ and Pip
*   Git client
*   Git LFS ([Install Git LFS](https://git-lfs.com/))
*   (Optional) Docker Desktop

### Local Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/rthrprrt/optical-factory-phase2.git
    cd optical-factory-phase2
    ```
2.  Install LFS files:
    ```bash
    git lfs pull
    ```
3.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    # Activate: .\venv\Scripts\Activate.ps1 (PowerShell) or source venv/bin/activate (Bash)
    ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Create `.env` file:
    *   Copy `.env.example` to `.env`.
    *   Verify `FACE_MODEL_PATH` and other settings if needed.
6.  Run the FastAPI server:
    ```bash
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
    ```
7.  Access the API at `http://localhost:8000` and docs at `http://localhost:8000/docs`.

### Docker Setup

1.  Ensure Docker Desktop is running.
2.  Clone repo and install LFS (steps 1 & 2 above).
3.  Build the Docker image:
    ```bash
    docker build -t optical-factory-backend .
    ```
4.  Run the container (mounting `.env` for runtime configuration):
    ```bash
    docker run --rm -p 8000:8000 --env-file .env --name optical-factory-app optical-factory-backend
    ```
5.  Access the API at `http://localhost:8000`.

## Running Tests

1.  Install dependencies (`pytest` and `httpx` are included).
2.  (Optional) Place test images in `tests/test_data/` (see `tests/test_api.py` for details).
3.  Run from the project root:
    ```bash
    pytest
    ```

## Running Benchmark

1.  Ensure the API server is running.
2.  Place test images in `benchmark/test_data/`.
3.  Run from the project root as a module:
    ```bash
    python -m benchmark.optical_factory_evaluation
    ```
4.  Results are printed and saved to `benchmark/evaluation_results.json`.

## Continuous Integration (CI)

A GitHub Actions workflow (`.github/workflows/python-ci.yml`) automatically runs `pytest` on push/pull_request to main branches, including Git LFS checkout.