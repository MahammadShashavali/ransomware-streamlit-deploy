import os
import sys
import time
import shutil
import threading
import datetime
import joblib
import pandas as pd
import pefile
from plyer import notification
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import scale_features
from src.predictutil import load_model_and_scaler

# === Configuration ===
WATCH_FOLDER = "watch_folder"
MODEL_PATH = "models/lightgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
REAL_EXE_SOURCE = r"C:\Windows\System32\cmd.exe"
REAL_DLL_SOURCE = r"C:\Windows\System32\kernel32.dll"

# Create watch folder if it doesn't exist
os.makedirs(WATCH_FOLDER, exist_ok=True)

# Load model and scaler
print("📦 Loading model and scaler...")
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

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
        print(f"⚠️ Failed to extract PE features from {file_path}: {e}")
        return None

# === Monitor Handler ===
class RansomwareMonitor(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".exe", ".dll")):
            print(f"\n📁 Detected: {event.src_path}")
            features = extract_pe_features(event.src_path)
            if features:
                df = pd.DataFrame([features])
                scaled = scale_features(df, scaler)
                prediction = model.predict(scaled)[0]
                label = "Ransomware" if prediction == 1 else "Benign"
                print(f"🧠 Prediction: {label}")

                if prediction == 1:
                    notification.notify(
                        title="🚨 Ransomware Detected",
                        message=f"{os.path.basename(event.src_path)} is suspicious!",
                        timeout=5
                    )

    def on_modified(self, event):
        self.on_created(event)  # Trigger prediction on overwrite too

# === Auto-drop EXE and DLL ===
def drop_test_files():
    time.sleep(3)  # Ensure monitor is running
    timestamp = datetime.datetime.now().strftime("%H%M%S")

    # EXE
    exe_dst = os.path.join(WATCH_FOLDER, f"auto_valid_{timestamp}.exe")
    try:
        shutil.copy(REAL_EXE_SOURCE, exe_dst)
        print(f"✅ EXE file copied: {exe_dst}")
    except Exception as e:
        print(f"❌ Failed to copy EXE: {e}")

    # DLL
    dll_dst = os.path.join(WATCH_FOLDER, f"auto_valid_{timestamp}.dll")
    try:
        shutil.copy(REAL_DLL_SOURCE, dll_dst)
        print(f"✅ DLL file copied: {dll_dst}")
    except Exception as e:
        print(f"❌ Failed to copy DLL: {e}")

# === Main Runner ===
if __name__ == "__main__":
    print("🛡️ Advanced monitoring with EXE & DLL support started...")

    observer = Observer()
    observer.schedule(RansomwareMonitor(), path=WATCH_FOLDER, recursive=False)
    observer.start()

    threading.Thread(target=drop_test_files, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Monitoring stopped.")
    observer.join()
