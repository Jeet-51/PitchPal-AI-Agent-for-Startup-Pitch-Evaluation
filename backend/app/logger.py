"""
PitchPal v2 - Structured JSON Logging

Replaces Python's default text logs with machine-readable JSON (NDJSON format).
Every log line is a valid JSON object — easy to pipe into log aggregators
like Datadog, Grafana Loki, AWS CloudWatch, or just grep/jq locally.

Example output:
  {"timestamp":"2026-03-08T22:28:00Z","level":"INFO","logger":"app.main",
   "message":"Evaluation complete","event":"evaluation_complete",
   "startup":"ClearLend","role":"startup","processing_time_s":32.7,
   "overall_score":6.4,"from_cache":false,"tool_calls":6,"contradictions":1}
"""

import json
import logging
from datetime import datetime, timezone

# Fields that are part of LogRecord's internal state — we exclude these
# so we don't pollute the JSON output with Python internals
_INTERNAL_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
})


class JSONFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.
    Extra structured fields can be passed via the extra={} kwarg on any
    logger call, e.g.:
        logger.info("Eval done", extra={"startup": "ClearLend", "score": 6.4})
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra structured fields the caller passed in
        for key, val in record.__dict__.items():
            if key not in _INTERNAL_FIELDS and not key.startswith("_"):
                log_obj[key] = val

        # Append exception traceback if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def setup_json_logging(level: int = logging.INFO) -> None:
    """
    Install the JSON formatter on the root logger.
    Call ONCE at application startup — before any other imports log anything.

    After calling this, ALL loggers (including uvicorn, fastapi, app.*)
    will output structured JSON.
    """
    formatter = JSONFormatter()

    root = logging.getLogger()
    root.setLevel(level)

    # Replace ALL existing handlers so we don't get duplicate/mixed output
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Suppress uvicorn's noisy per-request access logs (keep errors)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "JSON structured logging initialized",
        extra={"event": "logging_setup"},
    )
