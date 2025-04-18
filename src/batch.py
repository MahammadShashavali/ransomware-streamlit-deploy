import os
import pandas as pd
from predictutil import predict_from_csv  # Make sure predict_util.py is in the same folder

# 👇 Define input CSV path (update this if needed)
INPUT_CSV_PATH = os.path.join("data", "test", "RansomwareData.csv")  # e.g., ransomware features without labels

# 👇 Check if file exists
if not os.path.exists(INPUT_CSV_PATH):
    print(f"❌ File not found: {INPUT_CSV_PATH}")
else:
    print("📂 Found input file. Starting batch prediction...")

    # ✅ Get predictions
    try:
        predictions_df = predict_from_csv(INPUT_CSV_PATH)

        # ✅ Show results
        print("\n🔍 Prediction Results:")
        print(predictions_df.head())

        # 💾 Optional: Save to CSV
        output_path = os.path.join("results", "batch_predictions.csv")
        os.makedirs("results", exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during batch prediction: {e}")
