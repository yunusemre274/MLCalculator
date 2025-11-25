@echo off
echo ========================================
echo  ML Analytics - Quick Start
echo ========================================
echo.

REM Start Backend Server
echo [1/2] Starting Backend Server...
start "Backend Server" cmd /k "cd backend && ..\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul

REM Start Frontend Server
echo [2/2] Starting Frontend Server...
start "Frontend Server" cmd /k "npm run dev"
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo  Servers are starting...
echo ========================================
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:8080
echo  API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Press any key to exit this window...
pause >nul
