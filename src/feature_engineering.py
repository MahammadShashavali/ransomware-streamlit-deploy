import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

def engineer_features(input_path, output_path, scaler_path):
    print("📥 Reading preprocessed data...")
    df = pd.read_csv(input_path)

    # Separate features and label
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Scaling
    print("⚙️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reconstruct DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df["Label"] = y

    # Save engineered dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_scaled_df.to_csv(output_path, index=False)

    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print("✅ Feature engineering complete. Saved:")
    print(f"   • Engineered data -> {output_path}")
    print(f"   • Scaler object   -> {scaler_path}")
def scale_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(df), columns=df.columns)


engineer_features(
    input_path="data/processed/preprocessed_data.csv",
    output_path="data/processed/engineered_data.csv",
    scaler_path="models/scaler.pkl"
)
