# LPR + OCRM Proof of Concept

Real-time Kazakhstan license plate recognition system
with local OCRM vehicle lookup.

## Architecture

Camera
↓
WebRTC
↓
YOLO license plate detection
↓
OpenCV preprocessing
↓
PaddleOCR recognition
↓
OCR normalization and voting
↓
Confirmed license plate
↓
FastAPI OCRM API
↓
Vehicle lookup
↓
Vehicle / borrower / pledge information
↓
Visit event

## Project status

Proof of Concept (PoC).

The current implementation uses a local test OCRM backend
and local test database.

Production integration requires connection to the real
OCRM API/database.

## Main technologies

- Python
- Streamlit
- Streamlit WebRTC
- Ultralytics YOLO
- PaddleOCR / PaddleX
- OpenCV
- FastAPI
- Uvicorn
- Requests
- Regex

## Run backend

python app\ocrm_backend.py

Backend:
http://127.0.0.1:8000

## Run Streamlit

streamlit run app\streamlit_live_ocrm.py

Application:
http://localhost:8501

## Model

models\best.pt

The model is the trained YOLO license plate detector.

## Test data

data\test_videos

## Training results

training

## Evaluation results

results\evaluation
