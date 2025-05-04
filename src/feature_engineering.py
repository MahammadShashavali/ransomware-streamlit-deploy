import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

def engineer_features(input_path: str, output_path: str, scaler_path: str) -> pd.DataFrame:
    print("📥 Reading preprocessed data...")
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded data shape: {df.shape}")

    # Separate features and label
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Feature scaling
    print("⚙️ Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reconstruct DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df["Label"] = y

    # Save outputs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_scaled_df.to_csv(output_path, index=False)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print("✅ Feature engineering complete. Saved:")
    print(f"   • Engineered data -> {output_path}")
    print(f"   • Scaler object   -> {scaler_path}")

    return X_scaled_df


def scale_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Used during prediction to scale new input data."""
    return pd.DataFrame(scaler.transform(df), columns=df.columns)


# Standalone run
if __name__ == "__main__":
    engineer_features(
        input_path="data/processed/preprocessed_data.csv",
        output_path="data/processed/engineered_data.csv",
        scaler_path="models/scaler.pkl"
    )
