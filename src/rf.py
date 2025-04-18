import pandas as pd

df = pd.read_csv("data/processed/preprocessed_data.csv")
print("✅ Columns in your processed CSV:\n")
print(df.columns.tolist())
X = df.drop["label"]
y = df["label"]

