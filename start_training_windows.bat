@echo off
cd /d "%~dp0"
py -3.11 -m venv .ppe-venv || exit /b 1
call .ppe-venv\Scripts\activate.bat
python -m pip install --upgrade pip || exit /b 1
python -m pip install -r training\requirements-train.txt || exit /b 1
python training\launch_ppe.py --device 0 --epochs 40 --batch 12
pause
