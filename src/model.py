import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import os
import json

def train_and_evaluate_models(data_path="data/processed/engineered_data.csv", model_dir="models", result_dir="results"):
    print("📦 Loading engineered data...")
    df = pd.read_csv(data_path)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42)
    }

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    evaluation_results = {}

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Save model
        model_path = os.path.join(model_dir, f"{name.lower()}_model.pkl")
        joblib.dump(model, model_path)

        # Plot Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(result_dir, f"{name}_confusion_matrix.png"))
        plt.close()

        # Plot ROC Curve
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(result_dir, f"{name}_roc_curve.png"))
        plt.close()

        # Save metrics
        evaluation_results[name] = {
            "accuracy": acc,
            "roc_auc": roc_auc
        }

    # Save all evaluation metrics
    with open(os.path.join(result_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)

    # Pick best model based on AUC
    best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k]["roc_auc"])
    print(f"\n✅ Best model based on AUC: {best_model_name}")
    print(f"📂 Model saved at: {os.path.join(model_dir, best_model_name.lower() + '_model.pkl')}")

if __name__ == "__main__":
    train_and_evaluate_models()