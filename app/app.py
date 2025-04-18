import streamlit as st
import os
import pefile
import pandas as pd
import joblib

# Load model and scaler
MODEL_PATH = "models/lightgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Extract features from uploaded PE file
def extract_pe_features(pe):
    try:
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
        st.error(f"Feature extraction failed: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="Ransomware Detector", layout="centered")
st.title("🛡️ Ransomware Detection Web App")
st.write("Upload a `.exe` or `.dll` file to check if it's malicious.")

uploaded_file = st.file_uploader("Upload File", type=["exe", "dll"])

if uploaded_file:
    with open("temp_uploaded_file.exe", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        pe = pefile.PE("temp_uploaded_file.exe", fast_load=True)
        pe.parse_data_directories()

        features = extract_pe_features(pe)
        if features:
            df = pd.DataFrame([features])
            scaled = scaler.transform(df)
            prediction = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1]

            st.success(f"🔍 Prediction: {'Ransomware' if prediction == 1 else 'Benign'}")
            st.info(f"🧠 Probability of being Ransomware: {prob:.4f}")

            st.json(features)
    except Exception as e:
        st.error(f"Failed to analyze file: {e}")

    os.remove("temp_uploaded_file.exe")
