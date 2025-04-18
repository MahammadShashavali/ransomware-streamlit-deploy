import os
import pandas as pd

# Define correct columns (excluding the target 'Label')
columns = [
    'DebugSize', 'DebugRVA', 'MajorImageVersion', 'MajorOSVersion', 'ExportRVA',
    'ExportSize', 'IatVRA', 'MajorLinkerVersion', 'MinorLinkerVersion',
    'NumberOfSections', 'SizeOfStackReserve', 'DllCharacteristics',
    'ResourceSize', 'BitcoinAddresses'
]

# Create sample rows (list of dicts or list of lists)
data = [
    [0, 0, 1, 6, 0, 0, 4096, 2, 25, 4, 1048576, 8464, 212992, 0],
    [0, 0, 1, 6, 0, 0, 4096, 2, 25, 3, 1048576, 8464, 204800, 1],
]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Define output path
output_path = os.path.join("data", "test", "RansomwareData.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save CSV
df.to_csv(output_path, index=False)
print(f"✅ Sample input CSV saved to: {output_path}")
