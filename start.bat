@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    python run.py
) else (
    py -3.11 -c "import sys" >nul 2>nul
    if errorlevel 1 (
        py -3.12 run.py
    ) else (
        py -3.11 run.py
    )
)
if errorlevel 1 pause
