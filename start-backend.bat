@echo off
echo ================================================
echo   AI Training Platform - Backend Server
echo ================================================
echo.

cd backend

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo.
echo Starting backend server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python server.py

pause
