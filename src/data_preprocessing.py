import pandas as pd
import os

def load_and_preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    # Load raw data
    df = pd.read_csv(input_path)
    print(f"[INFO] Raw data loaded from {input_path}. Shape: {df.shape}")

    # Drop unnecessary columns
    drop_cols = ["FileName", "md5Hash", "Machine"]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Convert target column to binary if not already
    if 'Benign' in df.columns:
        df['Label'] = df['Benign'].apply(lambda x: 0 if x == 1 else 1)
        df.drop(columns=['Benign'], inplace=True)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the cleaned data
    df.to_csv(output_path, index=False)

    # Confirm file saved
    if os.path.exists(output_path):
        print(f"[SUCCESS] Preprocessed data saved to: {output_path}")
    else:
        print(f"[ERROR] Failed to save data at: {output_path}")

    return df

# Standalone execution
if __name__ == "__main__":
    input_path = "data/raw/data_file.csv"
    output_path = "data/processed/preprocessed_data.csv"
    load_and_preprocess_data(input_path, output_path)