import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import pandas as pd

# Load model and test data
model = joblib.load("models/lightgbm_model.pkl")  # or random_forest_model.pkl
df = pd.read_csv("data/processed/engineered_data.csv")

# Prepare data
X = df.drop("Label", axis=1)
y = df["Label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LightGBM (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Ransomware Detection")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/roc_curve_lightgbm.png")  # Save to file
plt.show()
