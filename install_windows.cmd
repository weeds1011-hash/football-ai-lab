@echo off
setlocal
cd /d "%~dp0"

echo [Football AI Lab] Python 3.12 virtual environment setup

if exist ".venv\Scripts\python.exe" goto install

where uv >nul 2>&1
if %errorlevel%==0 (
    uv venv --seed --python 3.12.13 .venv
    if errorlevel 1 goto failed
    goto install
)

py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
    goto install
)

echo Python 3.12 was not found. Install Python 3.12 or uv, then retry.
goto failed

:install
".venv\Scripts\python.exe" -m pip --version >nul 2>&1
if errorlevel 1 (
    where uv >nul 2>&1
    if %errorlevel%==0 (
        uv pip install --python ".venv\Scripts\python.exe" pip
    ) else (
        ".venv\Scripts\python.exe" -m ensurepip --upgrade
    )
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Installation completed. Run run_windows.cmd.
exit /b 0

:failed
echo.
echo Installation failed.
exit /b 1
