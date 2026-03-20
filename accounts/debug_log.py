# Minimal NDJSON debug logger for agent instrumentation (session 442421).
from __future__ import annotations

import json
import os
import time
import uuid
import logging

_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "debug-442421.log")
_logger = logging.getLogger(__name__)


def log(
    location: str,
    message: str,
    data: dict | None = None,
    hypothesis_id: str = "",
    run_id: str = "pre-fix",
) -> None:
    try:
        ts = int(time.time() * 1000)
        payload = {
            "sessionId": "442421",
            "id": f"log_{ts}_{uuid.uuid4().hex[:8]}",
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": ts,
            "runId": run_id,
        }
        if hypothesis_id:
            payload["hypothesisId"] = hypothesis_id
        _logger.info(
            "[442421_debug] run=%s hyp=%s location=%s message=%s",
            run_id,
            hypothesis_id or "-",
            location,
            message,
        )
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        _logger.exception("[442421_debug] failed to write debug line to %s", _LOG_PATH)
