@echo off
setlocal
cd /d "%~dp0"

title Football AI Lab
echo ========================================
echo          Football AI Lab MVP
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "import streamlit, pandas, numpy" >nul 2>&1
if errorlevel 1 goto setup
goto launch

:setup
echo Virtual environment or packages are missing.
echo Running install_windows.cmd...
call install_windows.cmd
if errorlevel 1 (
    echo.
    echo Setup failed. Review the messages above.
    pause
    exit /b 1
)

:launch
set "START_PORT=%~1"
if not defined START_PORT set "START_PORT=8501"

echo.
echo Checking for an available port starting at %START_PORT%...
echo Press Ctrl+C to stop the server.
echo.

call run_windows.cmd %START_PORT%
exit /b %errorlevel%
