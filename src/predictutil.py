import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple

# ========================
# Default Paths
# ========================
DEFAULT_MODEL_PATH = "models/best_model.pkl"
DEFAULT_SCALER_PATH = "models/scaler.pkl"

# ========================
# Load model and scaler
# ========================
def load_model_and_scaler(model_path: str = DEFAULT_MODEL_PATH, scaler_path: str = DEFAULT_SCALER_PATH):
    print("📦 Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# ========================
# Predict from a single list of features
# ========================
def predict_single(features: List[float], model_path: str = DEFAULT_MODEL_PATH, scaler_path: str = DEFAULT_SCALER_PATH) -> Tuple[str, float]:
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    features_scaled = scaler.transform([features])
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    label = "Ransomware" if prediction == 1 else "Benign"
    print(f"🧠 Prediction: {label} ({probability:.4f} probability)")
    
    return label, probability

# ========================
# Predict from CSV file
# ========================
def predict_from_csv(csv_path: str, model_path: str = DEFAULT_MODEL_PATH, scaler_path: str = DEFAULT_SCALER_PATH) -> pd.DataFrame:
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    df = pd.read_csv(csv_path)

    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])  # Avoid training leakage

    features_scaled = scaler.transform(df)
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    result_df = df.copy()
    result_df['Prediction'] = ["Ransomware" if pred == 1 else "Benign" for pred in predictions]
    result_df['Probability'] = probabilities

    print("✅ Batch prediction completed.")
    return result_df
if __name__ == "__main__":
    # 🔧 Sample test input (match the expected feature order)
    sample_features = [
        0,        # DebugSize
        0,        # DebugRVA
        1,        # MajorImageVersion
        5,        # MajorOSVersion
        0,        # ExportRVA
        0,        # ExportSize
        4096,     # IatVRA
        9,        # MajorLinkerVersion
        0,        # MinorLinkerVersion
        5,        # NumberOfSections
        1048576,  # SizeOfStackReserve
        0,        # DllCharacteristics
        512,      # ResourceSize
        0         # BitcoinAddresses
    ]

    label, prob = predict_single(sample_features)
    print(f"Result → {label} ({prob:.2%})")
