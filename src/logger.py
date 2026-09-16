"""
Structured Activity Logger for Knowledge AI Chatbot and MCP Server.
Records all Gemini <-> MCP <-> Web interactions with timestamps, latency, and payloads.
"""

from __future__ import annotations
import datetime
import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

# Configure UTF-8 encoding for stdio on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "activity.log")

_log_lock = threading.Lock()


def _ensure_log_dir():
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)


def log_event(
    category: str,
    event_type: str,
    action: str,
    details: Dict[str, Any],
    payload: Optional[Any] = None,
    latency_ms: Optional[float] = None,
    status: str = "success",
) -> Dict[str, Any]:
    """
    Log an event to data/activity.log and sys.stderr.
    
    Categories:
      - 'MCP': Communication between Gemini (LLM) and MCP server
      - 'WEB': User activity in the Web Dashboard
      - 'RAG': Hybrid search, ranking, and context assembly
      - 'SCRAPER': URL ingestion and parsing
    """
    _ensure_log_dir()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    entry = {
        "timestamp": now,
        "category": category,
        "event_type": event_type,
        "action": action,
        "status": status,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "details": details,
        "payload_summary": str(payload)[:500] if payload is not None else None,
    }

    line = json.dumps(entry, ensure_ascii=False)

    with _log_lock:
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            try:
                sys.stderr.write(f"Failed to write to log file: {e}\n")
            except Exception:
                pass

    # Mirror to stderr for console visibility without polluting stdout (needed for MCP JSON-RPC)
    cat_badge = f"[{category}:{event_type}]"
    lat_str = f" ({latency_ms:.1f}ms)" if latency_ms is not None else ""
    log_line = f"LOG {now[:19]} {cat_badge} {action}{lat_str} - Status: {status}\n"
    try:
        sys.stderr.write(log_line)
        sys.stderr.flush()
    except UnicodeEncodeError:
        try:
            sys.stderr.write(log_line.encode("ascii", errors="backslashreplace").decode("ascii"))
            sys.stderr.flush()
        except Exception:
            pass
    except Exception:
        pass

    return entry


def get_recent_logs(limit: int = 100, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve the most recent log entries in reverse chronological order."""
    _ensure_log_dir()
    if not os.path.exists(LOG_FILE_PATH):
        return []

    entries = []
    with _log_lock:
        try:
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if category and data.get("category") != category:
                            continue
                        entries.append(data)
                        if len(entries) >= limit:
                            break
                    except Exception:
                        pass
        except Exception as e:
            sys.stderr.write(f"Error reading activity log: {e}\n")

    return entries


def clear_logs() -> bool:
    """Clear all historical activity logs."""
    _ensure_log_dir()
    with _log_lock:
        try:
            with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                f.write("")
            return True
        except Exception:
            return False
