@echo off
echo ================================================
echo   AI Training Platform - Setup Script
echo ================================================
echo.
echo This script will install all required dependencies
echo.

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Checking Node.js...
node --version
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16 or higher from https://nodejs.org/
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Installing Backend Dependencies
echo ================================================
echo.

cd backend
echo Installing Python packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

cd ..

echo.
echo ================================================
echo   Installing Frontend Dependencies
echo ================================================
echo.

cd frontend
echo Installing Node packages (this may take a few minutes)...
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install Node dependencies
    pause
    exit /b 1
)

cd ..

echo.
echo ================================================
echo   Setup Complete!
echo ================================================
echo.
echo To start the application:
echo   1. Run start-backend.bat
echo   2. Run start-frontend.bat (in a new terminal)
echo   3. Open http://localhost:2121 in your browser
echo.

pause
