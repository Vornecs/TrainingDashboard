@echo off
echo ================================================
echo   AI Training Platform - Frontend
echo ================================================
echo.

cd frontend

echo Checking Node.js installation...
node --version
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16 or higher
    pause
    exit /b 1
)

echo.
echo Starting frontend development server on http://localhost:2121
echo Press Ctrl+C to stop the server
echo.

npm run dev

pause
