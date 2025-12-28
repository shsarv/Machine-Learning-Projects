from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
