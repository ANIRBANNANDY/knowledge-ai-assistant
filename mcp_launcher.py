#!/usr/bin/env python3
"""
Portable MCP server launcher for the Knowledge Base system.
Resolves the project root dynamically from this script's location,
making the system runnable on any machine without hardcoded paths.

Usage in mcp_config.json:
  "command": "python",
  "args": ["path/to/mcp_launcher.py"]
"""
import os
import sys

# Project root = parent directory of this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path for src.* imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set UTF-8 encoding on Windows
if sys.platform == "win32":
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

# Run the MCP server
from src.mcp_server import KnowledgeMCPServer

if __name__ == "__main__":
    server = KnowledgeMCPServer(
        db_path=os.path.join(PROJECT_ROOT, "data", "knowledge_base.db")
    )
    server.run()
