import pandas as pd
import os

def load_and_preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    # Load raw data
    df = pd.read_csv(input_path)
    print(f"[INFO] Raw data loaded from {input_path}. Shape: {df.shape}")

    # Drop columns if they exist
    drop_cols = ["FileName", "md5Hash", "Machine"]
    existing_drops = [col for col in drop_cols if col in df.columns]
    if existing_drops:
        df.drop(columns=existing_drops, inplace=True)
        print(f"[INFO] Dropped columns: {existing_drops}")

    # Remove duplicates
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    print(f"[INFO] Removed {before - after} duplicate rows")

    # Fill missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"[INFO] Filling {missing} missing values with 0")
    df.fillna(0, inplace=True)

    # Convert Benign column to Label (0 = benign, 1 = ransomware)
    if 'Benign' in df.columns:
        df['Label'] = df['Benign'].apply(lambda x: 0 if x == 1 else 1)
        df.drop(columns=['Benign'], inplace=True)
        print("[INFO] Converted 'Benign' to binary 'Label'")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Preprocessed data saved to: {output_path}")

    return df

# Standalone run
if __name__ == "__main__":
    input_path = "data/raw/data_file.csv"
    output_path = "data/processed/preprocessed_data.csv"
    load_and_preprocess_data(input_path, output_path)
