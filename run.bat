@echo off
title SitSense - Posture Detection App

echo.
echo ========================================
echo    SitSense - Posture Detection App
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found
    echo Please run: python -m venv .venv
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

echo [INFO] Using Python 3.10 virtual environment

REM Check if required files exist
if not exist "sitsense.py" (
    echo [ERROR] sitsense.py not found
    echo Please ensure you're running this from the SitSense directory
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found
    echo Please ensure you're running this from the SitSense directory
    pause
    exit /b 1
)

echo [INFO] Checking dependencies...

REM Try to import required modules
.venv\Scripts\python.exe -c "import customtkinter, cv2, mediapipe, plyer, PIL" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Some dependencies are missing
    echo [INFO] Installing dependencies...
    .venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        echo Please run: .venv\Scripts\pip.exe install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [INFO] Starting SitSense...
echo [INFO] Press Ctrl+C to stop
echo.

.venv\Scripts\python.exe sitsense.py

if errorlevel 1 (
    echo.
    echo [ERROR] SitSense encountered an error
    pause
)

echo.
echo [INFO] SitSense has been closed
pause
