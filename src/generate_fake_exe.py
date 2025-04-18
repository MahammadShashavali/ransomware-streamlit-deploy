import os
import time

# Delay to ensure monitor is running
print("⏳ Waiting for monitor to start...")
time.sleep(5)

fake_pe_data = bytearray([
    0x4D, 0x5A, 0x90, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0x00, 0x00, 0xB8, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
    0x00, 0x00
])
fake_pe_data.extend(b'\x00' * 1024)

output_path = os.path.join("watch_folder", "test_simulated.exe")
os.makedirs("watch_folder", exist_ok=True)

with open(output_path, "wb") as f:
    f.write(fake_pe_data)

print(f"✅ Simulated PE file created: {output_path}")
