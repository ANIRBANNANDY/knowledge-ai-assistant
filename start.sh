#!/usr/bin/env bash
# macOS / Linux One-Click Launcher for Knowledge AI Chatbot
set -e

echo "==================================================="
echo "  Starting Knowledge AI Documentation Chatbot..."
echo "==================================================="

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "[ERROR] Python 3 not found in PATH! Please install Python 3.10+."
        exit 1
    fi
fi

$PYTHON_CMD -m pip install -r requirements.txt --quiet
$PYTHON_CMD run_server.py
