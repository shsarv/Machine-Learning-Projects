import argparse
import joblib
import pandas as pd

MODEL_PATH = "model/diabetes_model.pkl"
SCALER_PATH = "model/scaler.pkl"

parser = argparse.ArgumentParser()
parser.add_argument("--pregnancies", type=int, required=True)
parser.add_argument("--glucose", type=float, required=True)
parser.add_argument("--bp", type=float, required=True)
parser.add_argument("--skin", type=float, required=True)
parser.add_argument("--insulin", type=float, required=True)
parser.add_argument("--bmi", type=float, required=True)
parser.add_argument("--dpf", type=float, required=True)
parser.add_argument("--age", type=int, required=True)

args = parser.parse_args()

# Load model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# IMPORTANT: feature names must match training
input_data = pd.DataFrame([{
	"Pregnancies": args.pregnancies,
	"Glucose": args.glucose,
	"BloodPressure": args.bp,
	"SkinThickness": args.skin,
	"Insulin": args.insulin,
	"BMI": args.bmi,
	"DPF": args.dpf,
	"Age": args.age
}])

# Scale & predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

if prediction == 1:
	print("⚠️ Diabetes detected")
else:
	print("✅ No diabetes detected")


