import os
import sys
import time
import threading
import joblib
import pandas as pd
import pefile
from plyer import notification
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Fix path to import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import scale_features
from src.predictutil import load_model_and_scaler

WATCH_FOLDER = "watch_folder"
MODEL_PATH = "models/lightgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Ensure folder exists
os.makedirs(WATCH_FOLDER, exist_ok=True)

model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

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
        print(f"⚠️ Failed to extract PE features: {e}")
        return None

class CombinedHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".exe"):
            print(f"\n📁 New file detected: {event.src_path}")
            features = extract_pe_features(event.src_path)
            if features:
                df = pd.DataFrame([features])
                scaled = scale_features(df, scaler)
                prediction = model.predict(scaled)[0]
                label = "Ransomware" if prediction == 1 else "Benign"
                print(f"🧠 Prediction: {label}")
                if prediction == 1:
                    notification.notify(
                        title="🚨 Ransomware Detected!",
                        message=os.path.basename(event.src_path),
                        timeout=5
                    )

def drop_simulated_exe():
    time.sleep(5)  # Give the monitor time to start
    fake_pe = bytearray([0x4D, 0x5A, 0x90, 0x00, 0x03, 0x00] + [0x00]*1024)
    file_path = os.path.join(WATCH_FOLDER, "auto_simulated.exe")
    with open(file_path, "wb") as f:
        f.write(fake_pe)
    print(f"✅ Simulated .exe dropped: {file_path}")

if __name__ == "__main__":
    print("🛡️ Starting combined real-time monitor with auto drop...")

    observer = Observer()
    observer.schedule(CombinedHandler(), path=WATCH_FOLDER, recursive=False)
    observer.start()

    # Start dropper in separate thread
    threading.Thread(target=drop_simulated_exe, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
