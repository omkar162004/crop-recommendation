import time
from collections import deque

_LOG = deque(maxlen=500)


def record(service: str, ok: bool, latency_ms: float, detail: str = "") -> None:
    _LOG.append(
        {"service": service, "ok": ok, "latency_ms": round(latency_ms, 1),
         "detail": detail, "ts": time.time()}
    )


def summary() -> dict:
    out = {}
    for service in sorted({e["service"] for e in _LOG}):
        events = [e for e in _LOG if e["service"] == service]
        ok = [e for e in events if e["ok"]]
        failures = [e for e in events if not e["ok"]]
        out[service] = {
            "calls": len(events),
            "success_rate": round(len(ok) / len(events), 3),
            "avg_latency_ms": round(sum(e["latency_ms"] for e in ok) / len(ok), 1) if ok else None,
            "last_error": failures[-1]["detail"] if failures else None,
        }
    return out