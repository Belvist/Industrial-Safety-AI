#!/bin/bash
cd "$(dirname "$0")" || exit 1
if command -v python3.11 >/dev/null 2>&1; then
  python3.11 run.py
elif command -v python3.12 >/dev/null 2>&1; then
  python3.12 run.py
else
  python3 run.py
fi
if [ $? -ne 0 ]; then
  read -r -p "Press Enter to close..."
fi
