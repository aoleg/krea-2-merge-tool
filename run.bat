@echo off
setlocal
chcp 65001 >nul
title Krea 2 Merge Tool

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

rem Arguments are passed through. With none, the GUI starts.
rem Example headless run:  run.bat --recipe my_merge.json
"venv\Scripts\python.exe" krea2_merge_tool.py %*
if errorlevel 1 (
    echo.
    echo The program exited with an error.
    pause
)
endlocal
