@echo off
echo ========================================
echo  ML Analytics - Stopping Servers
echo ========================================
echo.

echo Stopping Backend Server...
taskkill /FI "WINDOWTITLE eq Backend Server*" /T /F 2>nul

echo Stopping Frontend Server...
taskkill /FI "WINDOWTITLE eq Frontend Server*" /T /F 2>nul

echo.
echo All servers stopped.
echo.
pause
