import sqlite3
from pathlib import Path
import shutil

profile_dir = Path(r"d:\realtime-trans\realtime-translation-new\profiles\Gemine")
cookie_db_path = profile_dir / "Default" / "Network" / "Cookies"

print(f"Checking path: {cookie_db_path}")
if not cookie_db_path.exists():
    print("Error: Cookie file does not exist!")
    exit(1)

# Copy to temporary file to avoid locking
tmp_db = Path("scratch_cookies_test.db")
shutil.copy2(cookie_db_path, tmp_db)

try:
    conn = sqlite3.connect(str(tmp_db))
    cur = conn.cursor()
    
    # Check total cookies
    cur.execute("SELECT COUNT(*) FROM cookies")
    count = cur.fetchone()[0]
    print(f"Total cookies in database: {count}")
    
    # Check google cookies
    cur.execute("SELECT host_key, name FROM cookies WHERE host_key LIKE '%google.com%'")
    google_cookies = cur.fetchall()
    print(f"Google cookies count: {len(google_cookies)}")
    print("\n--- Google Cookies List ---")
    for host, name in google_cookies:
        print(f"Host: {host} | Name: {name}")
        
    conn.close()
except Exception as e:
    print(f"Database error: {e}")
finally:
    tmp_db.unlink(missing_ok=True)
