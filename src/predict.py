import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Paths
MODEL_PATH = "models/lightgbm_model.pkl"  # or "models/random_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_model_and_scaler(model_path, scaler_path):
    print("📦 Loading scaler and model...")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("❌ Model or scaler file not found.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_from_input(data_dict):
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

    # Define the feature columns
    columns = [
        'DebugSize', 'DebugRVA', 'MajorImageVersion', 'MajorOSVersion', 'ExportRVA',
        'ExportSize', 'IatVRA', 'MajorLinkerVersion', 'MinorLinkerVersion',
        'NumberOfSections', 'SizeOfStackReserve', 'DllCharacteristics',
        'ResourceSize', 'BitcoinAddresses'
    ]

    input_df = pd.DataFrame([data_dict], columns=columns)

    # Scale the features
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]  # Probability of being ransomware

    label = "Ransomware" if prediction == 1 else "Benign"
    print(f"\n📊 Prediction: {label}")
    print(f"🧪 Probability: {prob:.4f}")

    return label, prob

if __name__ == "__main__":
    # 🔧 Example input - you can customize this or connect to real-time
    sample_input = {
        'DebugSize': 0, 'DebugRVA': 0, 'MajorImageVersion': 1, 'MajorOSVersion': 5,
        'ExportRVA': 0, 'ExportSize': 0, 'IatVRA': 4096, 'MajorLinkerVersion': 9,
        'MinorLinkerVersion': 0, 'NumberOfSections': 5, 'SizeOfStackReserve': 1048576,
        'DllCharacteristics': 0, 'ResourceSize': 512, 'BitcoinAddresses': 0
    }

    predict_from_input(sample_input)
