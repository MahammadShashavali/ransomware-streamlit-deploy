import os
import sys
import time
import joblib
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from plyer import notification

# Fix import path for src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import scale_features
from src.predictutil import load_model_and_scaler

# Constants
MODEL_PATH = "models/lightgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
INPUT_FEATURE_PATH = "data/test/RansomwareData.csv"
WATCH_FOLDER = "watch_folder"

# Ensure folder exists
os.makedirs(WATCH_FOLDER, exist_ok=True)

# Load model and scaler
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

class RansomwareHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"\n📁 File created: {event.src_path}")
            self.analyze_file(event.src_path)

    def analyze_file(self, file_path):
        try:
            # Load sample ransomware data (simulate feature extraction)
            print(f"🔍 Using fixed input feature file: {INPUT_FEATURE_PATH}")
            df = pd.read_csv(INPUT_FEATURE_PATH)

            if 'Label' in df.columns:
                df = df.drop(columns=['Label'])

            scaled = scale_features(df, scaler)
            prediction = model.predict(scaled)[0]

            label = "Ransomware" if prediction == 1 else "Benign"
            print(f"🧠 Prediction: {label}")

            if prediction == 1:
                notification.notify(
                    title="🚨 Ransomware Detected!",
                    message=f"{os.path.basename(file_path)} is suspicious!",
                    timeout=5
                )
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    print(f"🟢 Monitoring folder: {WATCH_FOLDER}")
    observer = Observer()
    handler = RansomwareHandler()
    observer.schedule(handler, path=WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
