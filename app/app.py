import streamlit as st
import pandas as pd
import pefile
import joblib
import os

# --- Paths ---
MODEL_PATH = "models/lightgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# --- Load model & scaler ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Streamlit Config ---
st.set_page_config(page_title="Ransomware Detector", page_icon="🛡️", layout="centered")

# --- Header Section ---
st.markdown("""
    <h1 style="text-align: center; color: crimson;">🛡️ Ransomware Detection App</h1>
    <p style="text-align: center; font-size:18px;">
        Upload a <code>.exe</code> or <code>.dll</code> file for real-time machine learning analysis.<br>
        This app uses a LightGBM model trained on PE header features.
    </p>
    <hr>
""", unsafe_allow_html=True)

# --- Upload Section ---
uploaded_file = st.file_uploader("📂 Upload an EXE or DLL File", type=["exe", "dll"])

# --- Feature Extraction ---
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
            "BitcoinAddresses": 0  # Placeholder
        }
    except Exception as e:
        st.error(f"❌ Failed to extract features: {e}")
        return None

# --- Prediction ---
if uploaded_file:
    with st.spinner("🔍 Analyzing file..."):
        with open("temp_upload.exe", "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            pe = pefile.PE("temp_upload.exe", fast_load=True)
            pe.parse_data_directories()

            features = extract_pe_features(pe)
            
            if features:
                df = pd.DataFrame([features])
                scaled = scaler.transform(df)
                prediction = model.predict(scaled)[0]
                prob = model.predict_proba(scaled)[0][1]

                st.markdown("### 🔎 Prediction Result:")
                if prediction == 1:
                    st.error("🛑 **Ransomware Detected!**")
                else:
                    st.success("✅ **Benign File**")

                st.markdown(f"### 📊 Probability of Ransomware: `{prob:.4f}`")
                st.progress(min(prob, 1.0))

                with st.expander("📋 View Extracted Features"):
                    st.json(features)
        except Exception as e:
            st.error(f"❌ File analysis failed: {e}")

        os.remove("temp_upload.exe")

# --- Footer ---
st.markdown("""
<hr>
<p style="text-align:center; font-size:14px;">
    Built with ❤️ by <a href="https://github.com/MahammadShashavali" target="_blank">Mahammad Shashavali</a> ·
    Powered by LightGBM · Deployed on Streamlit
</p>
""", unsafe_allow_html=True)
