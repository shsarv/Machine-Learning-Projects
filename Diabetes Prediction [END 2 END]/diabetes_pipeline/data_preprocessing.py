# diabetes_pipeline/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path='dataset/kaggle_diabetes.csv', test_size=0.2, random_state=0):
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Rename column
    df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
    
    # Replace 0s with NaN
    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    
    # Fill NaNs
    df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
    df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
    df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
    df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
    df['BMI'].fillna(df['BMI'].median(), inplace=True)
    
    # Features & Target
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
