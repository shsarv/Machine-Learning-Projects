import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess
from config import MODEL_PATH, SCALER_PATH, MODEL_DIR

# Logging setup
logging.basicConfig(
	filename="logs/training.log",
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Training started")

# Load data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

# Train model
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)

# Save artifacts
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(classifier, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

logging.info("Model and scaler saved successfully")

