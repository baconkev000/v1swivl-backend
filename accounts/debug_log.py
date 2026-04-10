# ruff: noqa: PTH100, PTH111, PTH112, PTH118, PTH120, PTH123, INP001
# Minimal NDJSON debug logger for agent instrumentation (session 442421).
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid

_logger = logging.getLogger(__name__)

# Single-element list avoids ``global`` for lazy-resolved path (see PLW0603).
_log_path_cache: list[str | None] = [None]


def _log_path() -> str:
    """
    Writable path for NDJSON lines.

    Prefer env, then project root if writable, else system temp
    (Celery may not own /app).
    """
    if _log_path_cache[0] is not None:
        return _log_path_cache[0]
    explicit = (
        os.environ.get("DEBUG_442421_LOG_PATH")
        or os.environ.get("DEBUG_LOG_PATH")
        or ""
    ).strip()
    if explicit:
        _log_path_cache[0] = os.path.abspath(os.path.expanduser(explicit))
        return _log_path_cache[0]
    project_log = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "debug-442421.log"),
    )
    parent = os.path.dirname(project_log)
    try:
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            _log_path_cache[0] = project_log
            return _log_path_cache[0]
    except OSError:
        pass
    _log_path_cache[0] = os.path.join(tempfile.gettempdir(), "debug-442421.log")
    return _log_path_cache[0]


def _set_log_path(path: str) -> None:
    _log_path_cache[0] = path


def log(
    location: str,
    message: str,
    data: dict | None = None,
    hypothesis_id: str = "",
    run_id: str = "pre-fix",
) -> None:
    path = ""
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
        path = _log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except PermissionError:
        fallback = os.path.join(tempfile.gettempdir(), "debug-442421.log")
        if _log_path_cache[0] != fallback:
            _set_log_path(fallback)
            _logger.warning(
                "[442421_debug] no write access; using %s "
                "(set DEBUG_442421_LOG_PATH to override)",
                fallback,
            )
            try:
                with open(fallback, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, default=str) + "\n")
            except OSError:
                _logger.debug(
                    "[442421_debug] skip file write to %s",
                    fallback,
                    exc_info=True,
                )
        else:
            _logger.debug(
                "[442421_debug] skip file write to %s",
                path or fallback,
                exc_info=True,
            )
    except Exception:
        _logger.exception(
            "[442421_debug] failed to write debug line to %s",
            _log_path(),
        )
