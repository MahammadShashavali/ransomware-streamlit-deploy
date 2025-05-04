import struct
import os

# Define PE header values closer to typical ransomware signatures
fake_pe_data = bytearray()

# MZ Header (first two bytes for a valid PE)
fake_pe_data += b'MZ'
fake_pe_data += bytearray([0x00] * 58)  # Padding
fake_pe_data += struct.pack("<I", 0x80)  # e_lfanew offset (PE header starts)

# PE Signature + File Header + Optional Header (simplified and padded)
fake_pe_data += b'PE\x00\x00'  # PE Signature
fake_pe_data += bytearray([0x00] * 248)  # Simplified header padding

# Save as .dll file
dll_path = "watch_folder/fake_ransomware_payload.dll"
os.makedirs("watch_folder", exist_ok=True)

with open(dll_path, "wb") as f:
    f.write(fake_pe_data)

print(f"✅ Fake ransomware-style .dll created at: {dll_path}")
