"""Structured logging helpers.

A small JSON-lines formatter keeps the stream of pipeline events machine
readable (the web cockpit consumes exactly this format) while still being
pleasant to read on a terminal.  Package loggers accept a ``context=`` keyword
so every call site can attach structured metadata without string formatting.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

_CONFIGURED = False
_LEVEL_PINNED = False


def _resolve_level(level: Optional[str]) -> str:
    """Normalise ``level`` and mark it as explicitly chosen by the caller."""
    global _LEVEL_PINNED
    if level is None:
        return _LEVEL
    _LEVEL_PINNED = True
    return str(level).upper()


# The environment variable is an explicit choice too, so honour it above the
# configuration file (a CLI ``--log-level`` flag wins over both).
_env_level = os.environ.get("INTERVIEWGENIE_LOG_LEVEL")
if _env_level:
    _LEVEL_PINNED = True
_LEVEL = (_env_level or "INFO").upper()


def level_pinned() -> bool:
    """True once a caller has pinned the log level explicitly.

    Used by the factory so that a CLI flag or environment variable is not
    silently overridden by ``system.log_level`` from the configuration file.
    """
    return _LEVEL_PINNED


# --------------------------------------------------------------------------- #
# Logger that understands ``context=``
# --------------------------------------------------------------------------- #
class ContextLogger(logging.Logger):
    """Logger whose methods accept an optional ``context`` mapping."""

    def _log(  # type: ignore[override]
        self,
        level: int,
        msg: object,
        args: tuple,
        exc_info: Any = None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ``Logger.debug``/``info``/... normally perform this check; because we
        # override them we have to do it here or every record would be emitted.
        if not self.isEnabledFor(level):
            return
        if context is not None:
            extra = dict(extra or {})
            extra["context"] = context
        super()._log(level, msg, args, exc_info=exc_info, extra=extra,
                     stack_info=stack_info, stacklevel=stacklevel)

    def debug(self, msg: object, *args: Any, context: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        self._log(logging.DEBUG, msg, args, context=context, **kw)

    def info(self, msg: object, *args: Any, context: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        self._log(logging.INFO, msg, args, context=context, **kw)

    def warning(self, msg: object, *args: Any, context: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        self._log(logging.WARNING, msg, args, context=context, **kw)

    def error(self, msg: object, *args: Any, context: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        self._log(logging.ERROR, msg, args, context=context, **kw)

    def critical(self, msg: object, *args: Any, context: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        self._log(logging.CRITICAL, msg, args, context=context, **kw)

    def exception(self, msg: object, *args: Any, context: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        self._log(logging.ERROR, msg, args, exc_info=True, context=context, **kw)


logging.setLoggerClass(ContextLogger)


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #
class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: Dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if isinstance(extra, dict) and extra:
            payload["ctx"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PrettyFormatter(logging.Formatter):
    """Human friendly formatter used when stderr is a TTY."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;41m",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        color = self.COLORS.get(record.levelname, "")
        reset = "\033[0m" if color else ""
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        msg = record.getMessage()
        extra = getattr(record, "context", None)
        if isinstance(extra, dict) and extra:
            detail = " ".join(f"{k}={v}" for k, v in list(extra.items())[:6])
            msg = f"{msg} | {detail}"
        return f"{color}{stamp} {record.levelname:<7}{reset} {record.name:<22} {msg}"


def configure_logging(level: Optional[str] = None, json_logs: Optional[bool] = None,
                      fmt: Optional[str] = None, force: bool = False) -> None:
    """Install the package logger handler exactly once.

    ``fmt`` accepts ``"json"``, ``"pretty"`` or ``"auto"`` (the default, which
    picks JSON when stderr is not a terminal).  Pass ``force=True`` to reinstall
    the handler after the configuration changes.
    """
    global _CONFIGURED, _LEVEL
    if level:
        _LEVEL = _resolve_level(level)
    if json_logs is None:
        if fmt:
            json_logs = str(fmt).lower().startswith("json")
        else:
            json_logs = not sys.stderr.isatty()
    if _CONFIGURED and not force:
        return
    root = logging.getLogger("interviewgenie")
    root.setLevel(_LEVEL)
    root.propagate = False
    if not isinstance(root, ContextLogger):  # pragma: no cover - defensive
        root.__class__ = ContextLogger
    for existing in list(root.handlers):
        root.removeHandler(existing)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else PrettyFormatter())
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> ContextLogger:
    """Return a child logger of the package root, configuring on first use."""
    configure_logging()
    logger = logging.getLogger(f"interviewgenie.{name}")
    if not isinstance(logger, ContextLogger):  # pragma: no cover - defensive
        logger.__class__ = ContextLogger
    return logger  # type: ignore[return-value]


class log_context:  # noqa: N801 - used as a context manager
    """Temporarily attach structured context to every record in the block."""

    def __init__(self, logger: logging.Logger, **context: Any) -> None:
        self.logger = logger
        self.context = context

    def __enter__(self) -> "log_context":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None
