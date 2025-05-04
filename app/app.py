import os
import sys
import streamlit as st
import pandas as pd
import joblib
import pefile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ✅ Streamlit configuration (must be first)
st.set_page_config(page_title="Ransomware Detector", layout="centered")

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import scale_features
from src.predictutil import load_model_and_scaler

# === Email Alert Function ===
def send_email_alert(subject, body, to_email):
    sender_email = "mahammadshashavali5@gmail.com"
    sender_password = "Mahammad@123" \
    ""  # Use Gmail App Password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("✅ Email alert sent.")
    except Exception as e:
        print(f"❌ Email error: {e}")

# === PE Header Feature Extractor ===
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
        st.error(f"⚠️ Could not extract features: {e}")
        return None

# === Load model and scaler once
@st.cache_resource
def load_artifacts():
    return load_model_and_scaler("models/lightgbm_model.pkl", "models/scaler.pkl")

model, scaler = load_artifacts()

# === Streamlit UI ===
st.title("🔐 Ransomware Detection System")
st.markdown("Upload a Windows `.exe` or `.dll` file to predict if it's **Benign** or **Ransomware** using a trained LightGBM model.")

uploaded_file = st.file_uploader("📁 Upload Portable Executable File", type=["exe", "dll"])

if uploaded_file:
    os.makedirs("watch_folder", exist_ok=True)
    save_path = os.path.join("watch_folder", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded: `{uploaded_file.name}`")

    features = extract_pe_features(save_path)
    if features:
        df = pd.DataFrame([features])
        scaled = scale_features(df, scaler)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]
        label = "🛑 Ransomware" if prediction == 1 else "✅ Benign"

        st.markdown(f"### 🧠 Prediction: **{label}**")
        st.markdown(f"### 🔍 Confidence: **{probability:.2%}**")

        # 🔔 Email alert if ransomware detected
        if prediction == 1:
            send_email_alert(
                subject="🚨 Ransomware Alert Triggered",
                body=f"File `{uploaded_file.name}` classified as RANSOMWARE.\nDetection Probability: {probability:.2%}",
                to_email="receiver_email@gmail.com"
            )

        # Display feature breakdown
        st.subheader("📊 Extracted PE Features")
        st.dataframe(df)
