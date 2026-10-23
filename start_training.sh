#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v python3.11 >/dev/null || { echo 'Install Python 3.11 first'; exit 1; }
python3.11 -m venv .ppe-venv
source .ppe-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements-train.txt
python training/launch_ppe.py --device "${PPE_DEVICE:-0}" --epochs "${PPE_EPOCHS:-40}" --batch "${PPE_BATCH:-12}"
