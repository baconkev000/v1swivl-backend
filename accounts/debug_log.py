# Minimal NDJSON debug logger for agent instrumentation (session dcfc8b).
from __future__ import annotations

import json
import os
import time

_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "debug-dcfc8b.log")


def log(location: str, message: str, data: dict | None = None, hypothesis_id: str = "") -> None:
    try:
        payload = {
            "sessionId": "dcfc8b",
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        if hypothesis_id:
            payload["hypothesisId"] = hypothesis_id
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
