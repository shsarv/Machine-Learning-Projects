import joblib
import numpy as np
from config import MODEL_PATH, SCALER_PATH

class DiabetesPredictor:
	def __init__(self):
		self.model = joblib.load(MODEL_PATH)
		self.scaler = joblib.load(SCALER_PATH)

	def predict(self, features: list) -> int:
		"""
		features order:
		[Pregnancies, Glucose, BloodPressure, SkinThickness,
		 Insulin, BMI, DPF, Age]
		"""
		features = np.array(features).reshape(1, -1)
		features = self.scaler.transform(features)
		return int(self.model.predict(features)[0])


if __name__ == "__main__":
	predictor = DiabetesPredictor()

	sample_input = [2, 81, 72, 15, 76, 30.1, 0.547, 25]
	result = predictor.predict(sample_input)

	if result == 1:
		print("Oops! You have diabetes.")
	else:
		print("Great! You don't have diabetes.")
