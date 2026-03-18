# diabetes_pipeline/data_preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_and_preprocess(test_size=0.2, random_state=0):
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / "dataset" / "kaggle_diabetes.csv"
    df = pd.read_csv(csv_path)

    df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
    df['BMI'] = df['BMI'].fillna(df['BMI'].median())

    X = df.drop(columns='Outcome')
    y = df['Outcome']

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
