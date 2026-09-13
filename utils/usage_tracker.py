"""
Usage tracker: logs every successful prediction call to Supabase
(table `usage_log`), and falls back to a local JSON file if Supabase
is not configured or the request fails, so the app never breaks.
"""

import json
import os
import threading
from datetime import datetime, timezone

from .config import supabase_client

_LOCK = threading.Lock()
_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "usage_log.json"
)
_MAX_TIMESTAMPS_PER_MODEL = 1000
_TABLE_NAME = "usage_log"

_DEFAULT = {
    "forest": {"count": 0, "timestamps": []},
    "xgboost": {"count": 0, "timestamps": []},
}


# ---------- local file fallback ----------

def _load_local() -> dict:
    if not os.path.exists(_LOG_PATH):
        return {k: {"count": 0, "timestamps": []} for k in _DEFAULT}
    try:
        with open(_LOG_PATH, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {}
    for k in _DEFAULT:
        data.setdefault(k, {"count": 0, "timestamps": []})
    return data


def _save_local(data: dict) -> None:
    with open(_LOG_PATH, "w") as f:
        json.dump(data, f)


def _log_local(model_name: str) -> None:
    with _LOCK:
        data = _load_local()
        if model_name not in data:
            data[model_name] = {"count": 0, "timestamps": []}
        data[model_name]["count"] += 1
        data[model_name]["timestamps"].append(datetime.now(timezone.utc).isoformat())
        data[model_name]["timestamps"] = data[model_name]["timestamps"][-_MAX_TIMESTAMPS_PER_MODEL:]
        _save_local(data)


# ---------- public API ----------

def log_usage(model_name: str) -> None:
    """Record one successful prediction call for the given model."""
    if supabase_client is not None:
        try:
            supabase_client.table(_TABLE_NAME).insert({
                "model_name": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
            return
        except Exception as e:
            print(f"[usage_tracker] Supabase insert failed, falling back to local file: {e}")

    # Fallback: Supabase not configured or the insert failed
    _log_local(model_name)


def get_stats() -> dict:
    """Return usage counts and timestamps for every model."""
    if supabase_client is not None:
        try:
            response = supabase_client.table(_TABLE_NAME).select("model_name, created_at").execute()
            rows = response.data or []
            stats = {k: {"count": 0, "timestamps": []} for k in _DEFAULT}
            for row in rows:
                name = row.get("model_name")
                if name not in stats:
                    stats[name] = {"count": 0, "timestamps": []}
                stats[name]["count"] += 1
                stats[name]["timestamps"].append(row.get("created_at"))
            return stats
        except Exception as e:
            print(f"[usage_tracker] Supabase read failed, falling back to local file: {e}")

    with _LOCK:
        return _load_local()