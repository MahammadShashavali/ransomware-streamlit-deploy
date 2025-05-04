import os
import joblib
import pandas as pd
from typing import Tuple

# ========================
# Default Paths
# ========================
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ========================
# Feature Columns
# ========================
FEATURE_COLUMNS = [
    'DebugSize', 'DebugRVA', 'MajorImageVersion', 'MajorOSVersion', 'ExportRVA',
    'ExportSize', 'IatVRA', 'MajorLinkerVersion', 'MinorLinkerVersion',
    'NumberOfSections', 'SizeOfStackReserve', 'DllCharacteristics',
    'ResourceSize', 'BitcoinAddresses'
]

# ========================
# Load Model and Scaler
# ========================
def load_model_and_scaler(model_path: str, scaler_path: str):
    print("📦 Loading scaler and model...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Scaler file not found at: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

# ========================
# Predict from Dictionary Input
# ========================
def predict_from_input(data_dict: dict, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH) -> Tuple[str, float]:
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    # Convert to DataFrame
    input_df = pd.DataFrame([data_dict], columns=FEATURE_COLUMNS)

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    label = "Ransomware" if prediction == 1 else "Benign"
    print(f"\n📊 Prediction: {label}")
    print(f"🧪 Probability: {probability:.4f}")

    return label, probability

# ========================
# Example Standalone Run
# ========================
if __name__ == "__main__":
    sample_input = {
        'DebugSize': 0,
        'DebugRVA': 0,
        'MajorImageVersion': 1,
        'MajorOSVersion': 5,
        'ExportRVA': 0,
        'ExportSize': 0,
        'IatVRA': 4096,
        'MajorLinkerVersion': 9,
        'MinorLinkerVersion': 0,
        'NumberOfSections': 5,
        'SizeOfStackReserve': 1048576,
        'DllCharacteristics': 0,
        'ResourceSize': 512,
        'BitcoinAddresses': 0
    }

    predict_from_input(sample_input)
