import sqlite3
from pathlib import Path
import shutil
import base64
import json

profile_dir = Path(r"d:\realtime-trans\realtime-translation-new\profiles\Gemine")
cookie_db_path = profile_dir / "Default" / "Network" / "Cookies"
local_state_path = profile_dir / "Local State"

# Decrypt key
try:
    import win32crypt
    with open(local_state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    encrypted_key_b64 = data["os_crypt"]["encrypted_key"]
    encrypted_key = base64.b64decode(encrypted_key_b64)[5:]
    key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]
except Exception as e:
    print(f"Error decrypting master key: {e}")
    exit(1)

# Copy to temporary file
tmp_db = Path("scratch_raw_test.db")
shutil.copy2(cookie_db_path, tmp_db)

try:
    conn = sqlite3.connect(str(tmp_db))
    cur = conn.cursor()
    cur.execute("SELECT name, encrypted_value FROM cookies WHERE host_key = '.google.com' AND name = '__Secure-1PSID'")
    row = cur.fetchone()
    if row:
        name, enc_val = row
        if enc_val[:3] in (b"v10", b"v11"):
            from Cryptodome.Cipher import AES
            iv = enc_val[3:15]
            cipher_text = enc_val[15:-16]
            tag = enc_val[-16:]
            cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
            decrypted = cipher.decrypt_and_verify(cipher_text, tag)
            print("=== RAW DECRYPTED BYTES ===")
            print(f"Length: {len(decrypted)}")
            print(f"Hex: {decrypted.hex()}")
            print(f"Bytes: {decrypted}")
            
            # Check if it has readable ASCII characters
            try:
                print(f"ASCII portion: {decrypted.decode('ascii', errors='ignore')}")
            except Exception:
                pass
    conn.close()
except Exception as e:
    print(f"Error: {e}")
finally:
    tmp_db.unlink(missing_ok=True)
