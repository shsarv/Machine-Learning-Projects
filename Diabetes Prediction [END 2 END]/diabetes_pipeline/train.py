# diabetes_pipeline/train.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess
from config import MODEL_DIR, MODEL_PATH, SCALER_PATH

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

# Train model
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)

# Save model and scaler
joblib.dump(classifier, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Model and scaler saved successfully.")
