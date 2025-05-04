# generate_valid_pe.py
import os

def create_valid_pe_file(filepath="watch_folder/valid_dummy.exe"):
    # Minimal valid MZ + PE headers
    mz_header = b'MZ' + bytes(58) + b'\x80\x00'  # MZ magic + padding + offset to PE
    pe_header = b'PE\x00\x00' + bytes(248)       # PE signature + minimal COFF header

    with open(filepath, "wb") as f:
        f.write(mz_header + pe_header)
    print(f"✅ Created valid PE dummy at: {filepath}")

if __name__ == "__main__":
    os.makedirs("watch_folder", exist_ok=True)
    create_valid_pe_file()
