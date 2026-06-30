@echo off
echo ====================================================
echo   DONG GOI UNG DUNG REALTIME TRANSLATION THANH EXE
echo ====================================================
echo.
cd /d "%~dp0"

if not exist venv (
    echo [ERROR] Khong tim thay thu muc venv. Vui long chay install_libs.bat truoc!
    pause
    exit /b
)

echo Bat dau dong goi bang PyInstaller...
venv\Scripts\python.exe -m PyInstaller --name="HT-Tool-Pro" --onedir --windowed --noconfirm --icon="icon.ico" --add-data "icon.png;." main.py

echo.
echo ====================================================
echo   DONG GOI HOAN TAT!
echo   File EXE nam trong thu muc: dist\HT-Tool-Pro\HT-Tool-Pro.exe
echo ====================================================
pause
