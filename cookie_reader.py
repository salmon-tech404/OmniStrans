"""
cookie_reader.py – Đọc cookie từ Chrome profile tuỳ chỉnh trên Windows.
Hỗ trợ cả v10/v11 (AES-256-GCM) và legacy DPAPI.
"""

import json
import shutil
import sqlite3
import base64
from pathlib import Path


def _get_chrome_key(local_state_path: Path) -> bytes:
    """Giải mã master key trong Local State bằng DPAPI."""
    import win32crypt  # pywin32

    with open(local_state_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    encrypted_key_b64 = data["os_crypt"]["encrypted_key"]
    encrypted_key = base64.b64decode(encrypted_key_b64)[5:]  # bỏ "DPAPI"
    key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]
    return key


def _decrypt_cookie_value(encrypted_value: bytes, key: bytes) -> str:
    """Giải mã giá trị cookie (v10/v11 AES-GCM hoặc DPAPI legacy)."""
    if encrypted_value[:3] in (b"v10", b"v11"):
        from Cryptodome.Cipher import AES  # pycryptodomex

        iv = encrypted_value[3:15]
        cipher_text = encrypted_value[15:-16]
        tag = encrypted_value[-16:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
        decrypted = cipher.decrypt_and_verify(cipher_text, tag)
        try:
            return decrypted.decode("utf-8")
        except UnicodeDecodeError:
            # Thử bỏ đi 32 byte đầu (header nhị phân mới của Chrome v114+)
            try:
                return decrypted[32:].decode("utf-8")
            except UnicodeDecodeError:
                return ""
    else:
        # Legacy DPAPI (Chrome < 80)
        import win32crypt

        result = win32crypt.CryptUnprotectData(encrypted_value, None, None, None, 0)
        return result[1].decode("utf-8")


def extract_google_cookies(profile_dir: Path) -> dict[str, str]:
    """
    Trích xuất tất cả cookie domain *.google.com từ Chrome profile.
    Trả về dict {name: value}.
    """
    local_state_path = profile_dir / "Local State"
    cookie_db_path = profile_dir / "Default" / "Network" / "Cookies"

    if not local_state_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {local_state_path}")
    if not cookie_db_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {cookie_db_path}")

    key = _get_chrome_key(local_state_path)

    # Copy DB vì Chrome có thể đang lock
    tmp_db = profile_dir / "_cookies_read_tmp.db"
    shutil.copy2(cookie_db_path, tmp_db)

    cookies: dict[str, str] = {}
    try:
        conn = sqlite3.connect(str(tmp_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT name, encrypted_value FROM cookies WHERE host_key LIKE '%google.com%'"
        )
        for name, enc_val in cur.fetchall():
            try:
                cookies[name] = _decrypt_cookie_value(enc_val, key)
            except Exception:
                pass
        conn.close()
    finally:
        tmp_db.unlink(missing_ok=True)

    return cookies


def get_gemini_cookies(profile_dir: Path) -> tuple[str | None, str | None]:
    """Trả về (PSID, PSIDTS) hoặc (None, None) nếu chưa login."""
    try:
        cookies = extract_google_cookies(profile_dir)
        psid   = cookies.get("__Secure-1PSID")
        psidts = cookies.get("__Secure-1PSIDTS")
        return psid, psidts
    except Exception as e:
        print(f"[cookie_reader] {e}")
        return None, None
