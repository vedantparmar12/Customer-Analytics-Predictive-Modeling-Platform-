@echo off
echo ========================================
echo Starting E-commerce Analytics Frontend
echo ========================================
echo.

echo Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed\!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js found\!
echo.

echo Navigating to frontend directory...
cd frontend

echo.
echo Installing dependencies (this may take a few minutes on first run)...
call npm install

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Try running: npm cache clean --force
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting React Development Server...
echo ========================================
echo.
echo The browser will open automatically at http://localhost:3000
echo Press Ctrl+C to stop the server
echo.

call npm start

pause
