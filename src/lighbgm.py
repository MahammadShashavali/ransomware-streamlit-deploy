# src/lgbm_model.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def main():
    # Load the processed data
    df = pd.read_csv("data/processed/ransomware_data.csv")

    # Drop non-numeric/categorical columns
    non_numeric_cols = ['FileName', 'md5Hash']
    for col in non_numeric_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Split features and target
    X = df.drop("Benign", axis=1)
    y = df["Benign"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize LightGBM classifier
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    # Train model
    lgb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = lgb_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(lgb_model, "models/lgbm_model.pkl")
    print("✅ Model saved as models/lgbm_model.pkl")

if __name__ == "__main__":
    main()
