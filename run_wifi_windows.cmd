@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run install_windows.cmd first.
    exit /b 1
)

set "START_PORT=%~1"
if not defined START_PORT set "START_PORT=8501"
set /a APP_PORT=START_PORT >nul 2>&1
if %APP_PORT% LSS 1024 goto invalid_port

:check_port
if %APP_PORT% GTR 65535 goto no_port
powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort %APP_PORT% -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 goto port_ready
echo Port %APP_PORT% is already in use. Checking the next port...
set /a APP_PORT+=1
goto check_port

:port_ready

echo Football AI Lab will listen on this PC's network interfaces.
echo Connect the Galaxy phone to the same Wi-Fi, then open http://PC_IP_ADDRESS:%APP_PORT%
echo Find the PC's IPv4 address below. Windows Firewall may ask for permission.
ipconfig | findstr /i "IPv4"
echo.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 0.0.0.0 --server.port %APP_PORT%
exit /b %errorlevel%

:invalid_port
echo Invalid port. Enter a number from 1024 to 65535.
exit /b 1

:no_port
echo No available port was found.
exit /b 1
