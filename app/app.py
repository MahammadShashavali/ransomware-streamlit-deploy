import os
import sys
import streamlit as st
import pandas as pd
import joblib
import pefile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Streamlit config (must be first)
st.set_page_config(page_title="Ransomware Detector", layout="centered")

# Add root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import scale_features
from src.predictutil import load_model_and_scaler

# 🔐 Load ML model and scaler
@st.cache_resource
def load_artifacts():
    return load_model_and_scaler("models/lightgbm_model.pkl", "models/scaler.pkl")

model, scaler = load_artifacts()

# === Email alert function ===
def send_email_alert(subject, body, to_email):
    sender_email = "mahammadshashavali5@gmail.com"              # Your Gmail
    sender_password = "zhjdrqcgltocewyq"     # App Password from Google

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
        st.success("📧 Email alert sent!")
    except Exception as e:
        st.error(f"❌ Email sending failed: {e}")

# === Feature extraction from PE ===
def extract_or_inject_features(file_path):
    if "fake_ransomware" in os.path.basename(file_path).lower():
        st.warning("⚠️ Injecting fake ransomware features for demo.")
        return {
            "DebugSize": 5000, "DebugRVA": 4096, "MajorImageVersion": 7,
            "MajorOSVersion": 10, "ExportRVA": 1024, "ExportSize": 512,
            "IatVRA": 8192, "MajorLinkerVersion": 12, "MinorLinkerVersion": 5,
            "NumberOfSections": 10, "SizeOfStackReserve": 4194304,
            "DllCharacteristics": 6000, "ResourceSize": 50000,
            "BitcoinAddresses": 1
        }, True

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
        }, False
    except Exception as e:
        st.error(f"❌ PE feature extraction failed: {e}")
        return None, False

# === Streamlit UI ===
st.title("🔐 Ransomware Detection System")
st.markdown("Upload a `.exe`, `.dll`, or `.csv` file to check for ransomware using machine learning.")

uploaded_file = st.file_uploader("📁 Upload File", type=["exe", "dll", "csv"])

if uploaded_file:
    os.makedirs("watch_folder", exist_ok=True)
    file_path = os.path.join("watch_folder", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded: `{uploaded_file.name}`")

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(file_path)
        if "Label" in df.columns:
            df.drop(columns=["Label"], inplace=True)
        scaled = scale_features(df, scaler)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1]

        df["Prediction"] = ["Ransomware" if p == 1 else "Benign" for p in preds]
        df["Confidence"] = [f"{p*100:.2f}%" for p in probs]

        st.dataframe(df)
        st.download_button("⬇️ Download Results", df.to_csv(index=False), file_name="predictions.csv")
    else:
        features, is_fake = extract_or_inject_features(file_path)
        if features:
            df = pd.DataFrame([features])
            scaled = scale_features(df, scaler)

            if is_fake:
                prediction, probability = 1, 0.9876
            else:
                prediction = model.predict(scaled)[0]
                probability = model.predict_proba(scaled)[0][1]

            label = "🛑 Ransomware" if prediction == 1 else "✅ Benign"
            st.markdown(f"### 🧠 Prediction: **{label}**")
            st.markdown(f"### 🔍 Confidence: **{probability:.2%}**")

            if prediction == 1:
                send_email_alert(
                    subject="🚨 Ransomware Detected!",
                    body=f"Alert: File `{uploaded_file.name}` was classified as RANSOMWARE with confidence {probability:.2%}.",
                    to_email="mahammadshashavali49@gmail.com"
                )

            st.subheader("📊 Feature Summary")
            st.dataframe(df)
