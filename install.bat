@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Krea 2 Merge Tool - install

rem ============================================================
rem  install.bat - creates the venv and installs dependencies.
rem  torch comes from the PyTorch CUDA index, never from PyPI:
rem  a PyPI torch on Windows is CPU only.
rem ============================================================

cd /d "%~dp0"

set "TORCH_SPEC=torch==2.13.0 torchvision"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu132"

set "PYTHON="
where python >nul 2>nul && set "PYTHON=python"
if not defined PYTHON (
    where py >nul 2>nul && set "PYTHON=py -3.12"
)
if not defined PYTHON (
    echo [ERROR] Python was not found. Install Python 3.12 from https://www.python.org/downloads/
    echo         and tick "Add Python to PATH".
    pause
    exit /b 1
)

echo ============================================================
echo  Krea 2 Merge Tool - installation
echo ============================================================
echo.

%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    pause
    exit /b 1
)

%PYTHON% -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] tkinter is missing from this Python. Re-run the python.org installer
    echo         and enable "tcl/tk and IDLE" under Optional Features.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating the virtual environment...
    %PYTHON% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)

set "VPY=venv\Scripts\python.exe"

echo Upgrading pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo Installing torch from %TORCH_INDEX% ...
"%VPY%" -m pip install %TORCH_SPEC% --index-url %TORCH_INDEX%
if errorlevel 1 goto :fail

echo.
echo Installing the remaining dependencies...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Self check...
"%VPY%" -m k2merge.selfcheck
if errorlevel 1 (
    echo.
    echo [WARNING] The self check reported a problem. See the messages above.
    pause
    exit /b 1
)

echo.
echo Installation finished. Start the tool with run.bat
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
