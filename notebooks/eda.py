import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load preprocessed data
data_path = "data/processed/preprocessed_data.csv"
df = pd.read_csv(data_path)

# Basic Info
print("Shape of dataset:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Label', data=df)
plt.title("Class Distribution (Labelvs Ransomware)")
plt.xticks([0, 1], ['Ransomware', 'Label'])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(16, 6))
df.drop(columns=['Label']).boxplot()
plt.xticks(rotation=90)
plt.title("Feature Distributions")
plt.show()