#!/bin/bash
# setup.sh - Setup virtual environment and install dependencies
# Works in Git Bash on Windows or standard Bash on Linux/macOS

echo "==> Creating virtual environment..."
py -3 -m venv venv || python3 -m venv venv || python -m venv venv

echo "==> Activating virtual environment..."
if [ -d "venv/Scripts" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "==> Environment setup complete! Run 'source venv/Scripts/activate' (or 'venv/bin/activate' on Linux/macOS) to start."
