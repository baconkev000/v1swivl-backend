# Request/response logging: every request and 4xx/5xx to stderr (docker logs)
import json
import sys
import traceback
from django.conf import settings


def _log(payload):
    path = getattr(settings, "DEBUG_LOG_PATH", None)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _stderr(line):
    """Print to stderr so it shows in docker compose logs."""
    print(line, file=sys.stderr, flush=True)


class DebugLogMiddleware:
    """Log every request+response status; on 500 print full traceback to stderr."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            _log({
                "sessionId": "e47e3c",
                "runId": "request",
                "hypothesisId": "H1",
                "location": "config/debug_middleware.py:request_start",
                "message": "Request start",
                "data": {
                    "path": request.path,
                    "host": request.get_host(),
                    "allowed_hosts": getattr(settings, "ALLOWED_HOSTS", []),
                },
                "timestamp": __import__("time").time() * 1000,
            })
        except Exception:
            pass

        try:
            response = self.get_response(request)
            status = response.status_code
            # Log every request so 404/500 show up in docker logs
            _stderr(f"[SWIVL] {request.method} {request.path} -> {status}")
            if status >= 400:
                _stderr(f"[SWIVL] {status} {request.method} {request.path}")
            return response
        except Exception as exc:
            tb = traceback.format_exc()
            msg = f"{type(exc).__name__}: {exc}"
            try:
                _log({
                    "sessionId": "e47e3c",
                    "runId": "exception",
                    "hypothesisId": "H2_H3_H4_H5",
                    "location": "config/debug_middleware.py:exception",
                    "message": "Uncaught exception",
                    "data": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "path": getattr(request, "path", None),
                        "host": getattr(request, "get_host", lambda: None)(),
                        "traceback": tb,
                    },
                    "timestamp": __import__("time").time() * 1000,
                })
            except Exception:
                pass
            _stderr(f"\n[SWIVL 500] {request.method} {request.path}")
            _stderr(f"[SWIVL 500] {msg}")
            _stderr(tb)
            raise
