import pandas as pd

ransom_sample = {
    "DebugSize": 1024,
    "DebugRVA": 2048,
    "MajorImageVersion": 5,
    "MajorOSVersion": 6,
    "ExportRVA": 512,
    "ExportSize": 256,
    "IatVRA": 4096,
    "MajorLinkerVersion": 9,
    "MinorLinkerVersion": 0,
    "NumberOfSections": 8,
    "SizeOfStackReserve": 2097152,
    "DllCharacteristics": 5120,
    "ResourceSize": 16384,
    "BitcoinAddresses": 1
}

df = pd.DataFrame([ransom_sample])
df.to_csv("test_ransomware_sample.csv", index=False)
print("✅ test_ransomware_sample.csv created.")
