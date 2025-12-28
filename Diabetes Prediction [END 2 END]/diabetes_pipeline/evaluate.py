import joblib
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess
from config import MODEL_PATH

# Load data
X_train, X_test, y_train, y_test, _ = load_and_preprocess()

# Load trained model
model = joblib.load(MODEL_PATH)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
