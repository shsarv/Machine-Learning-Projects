# diabetes_pipeline/train.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess

# Load and preprocess
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

# Train Random Forest
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)

# Save model & scaler
joblib.dump(classifier, 'model/diabetes_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Model and scaler saved in 'model/' folder.")
