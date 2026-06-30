@echo off
echo ====================================================
echo   KICH HOAT VIRTUAL ENVIRONMENT VA CAI DAT THU VIEN
echo ====================================================
echo.
cd /d "%~dp0"
if not exist venv (
    echo [INFO] Khong tim thay thu muc venv. Dang tu dong tao venv moi...
    python -m venv venv
    if not exist venv (
        echo [ERROR] Khong the tu dong tao venv. Vui long kiem tra xem Python da duoc cai dat va them vao PATH chua!
        pause
        exit /b
    )
)
echo Kich hoat venv...
call venv\Scripts\activate
echo.
echo Cap nhat pip...
python -m pip install --upgrade pip
echo.
echo Dang cai dat cac thu vien can thiet...
pip install faster-whisper sounddevice numpy deep-translator PyQt6 pywin32 pyaudiowpatch pycryptodomex gemini-webapi pyinstaller yt-dlp imageio-ffmpeg easyocr
echo.
echo ====================================================
echo   CAI DAT HOAN TAT!
echo ====================================================
pause
