"""Structured (JSON) rotating logs with secret redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FIELDS = (
    "component",
    "event",
    "symbol",
    "mode",
    "signal_id",
    "order_id",
    "contract_id",
    "latency_ms",
)
_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(otp=)[^&\s\"']+"),
    re.compile(r"(?i)((?:token|secret|password|api[_-]?key)\"?\s*[:=]\s*\"?)[^\s\"',}]+"),
    re.compile(r"(bot)\d{6,}:[A-Za-z0-9_\-]{20,}"),
]
_SECRETS: set[str] = set()


def register_secrets(*values: str) -> None:
    """Exact secret values to scrub wherever they appear."""
    _SECRETS.update(v for v in values if v and len(v) >= 6)


def redact(text: str) -> str:
    for secret in _SECRETS:
        text = text.replace(secret, "***")
    for pat in _PATTERNS:
        text = pat.sub(lambda m: f"{m.group(1)}***", text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "component": getattr(record, "component", record.name.removeprefix("derivbot.")),
            "event": getattr(record, "event", record.getMessage()),
        }
        for key in _FIELDS[2:]:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in ("delay_s", "error", "latency_ms") and key not in payload:
                payload[key] = value
        return redact(json.dumps(payload, default=str))


def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("derivbot")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = JsonFormatter()
    file_handler = RotatingFileHandler(
        Path(log_dir) / "derivbot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(stream)
    root.propagate = False
    # Third-party libraries can echo URLs (with OTPs) at DEBUG/INFO: keep them quiet.
    for noisy in ("websockets", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
