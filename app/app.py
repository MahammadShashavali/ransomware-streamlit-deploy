import os
import sys
import streamlit as st
import pandas as pd
import joblib
import pefile

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Ransomware Detector", layout="centered")

# Add src/ to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import scale_features
from src.predictutil import load_model_and_scaler

# === Constants ===
MODEL_PATH = "models/lightgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
UPLOAD_DIR = "watch_folder"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Load Model & Scaler ===
@st.cache_resource
def load_artifacts():
    return load_model_and_scaler(MODEL_PATH, SCALER_PATH)

model, scaler = load_artifacts()

# === PE Feature Extractor ===
def extract_pe_features(file_path):
    try:
        pe = pefile.PE(file_path)
        return {
            "DebugSize": pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].Size,
            "DebugRVA": pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].VirtualAddress,
            "MajorImageVersion": pe.OPTIONAL_HEADER.MajorImageVersion,
            "MajorOSVersion": pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            "ExportRVA": pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].VirtualAddress,
            "ExportSize": pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].Size,
            "IatVRA": pe.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress,
            "MajorLinkerVersion": pe.OPTIONAL_HEADER.MajorLinkerVersion,
            "MinorLinkerVersion": pe.OPTIONAL_HEADER.MinorLinkerVersion,
            "NumberOfSections": pe.FILE_HEADER.NumberOfSections,
            "SizeOfStackReserve": pe.OPTIONAL_HEADER.SizeOfStackReserve,
            "DllCharacteristics": pe.OPTIONAL_HEADER.DllCharacteristics,
            "ResourceSize": pe.OPTIONAL_HEADER.DATA_DIRECTORY[2].Size,
            "BitcoinAddresses": 0
        }
    except Exception as e:
        st.error(f"⚠️ Failed to parse PE: {e}")
        return None

# === Streamlit UI ===
st.title("🔐 Ransomware Detection App")
st.markdown("Upload a Windows `.exe` or `.dll` file to detect if it's **Ransomware** or **Benign** using a trained ML model.")

uploaded_file = st.file_uploader("📁 Upload a PE File", type=["exe", "dll"])

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File uploaded: `{uploaded_file.name}`")

    features = extract_pe_features(file_path)
    if features:
        df = pd.DataFrame([features])
        scaled = scale_features(df, scaler)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        label = "🛑 Ransomware" if prediction == 1 else "✅ Benign"
        st.markdown(f"### 🧠 Prediction: **{label}**")
        st.markdown(f"### 🔍 Probability: **{probability:.2%}**")

        st.subheader("📊 Extracted PE Header Features")
        st.dataframe(df)
