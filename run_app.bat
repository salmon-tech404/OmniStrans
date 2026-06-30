@echo off
echo ====================================================
echo   KHOI CHAY UNG DUNG REALTIME TRANSLATION (HT TOOL)
echo ====================================================
echo.
cd /d "%~dp0"

if not exist venv (
    echo [WARNING] Khong tim thay thu muc venv. Dang tu dong chay install_libs.bat de cai dat...
    call install_libs.bat
    if not exist venv (
        echo [ERROR] Cai dat that bai! Vui long kiem tra lai.
        pause
        exit /b
    )
)

echo Kich hoat moi truong ao va mo ung dung...
venv\Scripts\python.exe main.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Ung dung bi dong voi ma loi: %errorlevel%
    pause
)
