import joblib
import numpy as np
import pandas as pd

# Load model and scaler paths
MODEL_PATH = "models/lightgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Load model and scaler
def load_artifacts():
    print("📦 Loading model and scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# Predict using input features
def predict(features: list):
    model, scaler = load_artifacts()
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    label = "Ransomware" if prediction == 1 else "Benign"
    print(f"🧠 Prediction: {label} ({probability:.4f} probability)")
    return label, probability

# Predict from CSV (for multiple samples)
def predict_from_csv(csv_path: str):
    model, scaler = load_artifacts()
    df = pd.read_csv(csv_path)

    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])  # Drop if present

    features_scaled = scaler.transform(df)
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    result_df = df.copy()
    result_df['Prediction'] = ["Ransomware" if pred == 1 else "Benign" for pred in predictions]
    result_df['Probability'] = probabilities
    print("✅ Batch prediction completed.")
    
    return result_df
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
