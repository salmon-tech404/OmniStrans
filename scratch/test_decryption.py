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
    print(f"Master key decrypted successfully, length: {len(key)} bytes")
except Exception as e:
    print(f"Error decrypting master key: {e}")
    exit(1)

# Copy to temporary file
tmp_db = Path("scratch_decryption_test.db")
shutil.copy2(cookie_db_path, tmp_db)

try:
    conn = sqlite3.connect(str(tmp_db))
    cur = conn.cursor()
    cur.execute("SELECT name, encrypted_value FROM cookies WHERE host_key = '.google.com' AND name = '__Secure-1PSID'")
    row = cur.fetchone()
    if not row:
        print("Error: __Secure-1PSID cookie row not found!")
        exit(1)
        
    name, enc_val = row
    print(f"Cookie name: {name}, Encrypted value length: {len(enc_val)} bytes, Prefix: {enc_val[:3]}")
    
    # Try decrypting
    if enc_val[:3] in (b"v10", b"v11"):
        try:
            from Cryptodome.Cipher import AES
            iv = enc_val[3:15]
            cipher_text = enc_val[15:-16]
            tag = enc_val[-16:]
            cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
            decrypted = cipher.decrypt_and_verify(cipher_text, tag).decode("utf-8")
            print(f"Success! Decrypted value: {decrypted[:10]}...")
        except Exception as e:
            print(f"AES decryption error: {e}")
    else:
        try:
            result = win32crypt.CryptUnprotectData(enc_val, None, None, None, 0)
            decrypted = result[1].decode("utf-8")
            print(f"Legacy DPAPI Success! Decrypted: {decrypted[:10]}...")
        except Exception as e:
            print(f"Legacy DPAPI decryption error: {e}")
            
    conn.close()
except Exception as e:
    print(f"Error during query/decryption: {e}")
finally:
    tmp_db.unlink(missing_ok=True)
