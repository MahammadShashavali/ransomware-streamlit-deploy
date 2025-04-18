import os
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import json
from colorama import Fore, Style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import os
import json

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{Fore.CYAN}📊 Evaluation for {name}:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{Style.RESET_ALL}")
    print(classification_report(y_test, y_pred))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Confusion_Matrix": cm.tolist(),
        "Classification_Report": report
    }

# After training models
results = []
results = []

# Train and evaluate Random Forest
print("\n🚀 Training RandomForest...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
results.append(evaluate_model("Random Forest", rf_model, X_test, y_test))

# Train and evaluate LightGBM
print("\n🚀 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)
results.append(evaluate_model("LightGBM", lgb_model, X_test, y_test))

# Save evaluation results to file
os.makedirs("results", exist_ok=True)
with open("results/evaluation_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"{Fore.GREEN}✅ Evaluation metrics saved to 'results/evaluation_metrics.json'{Style.RESET_ALL}")

# Save metrics

