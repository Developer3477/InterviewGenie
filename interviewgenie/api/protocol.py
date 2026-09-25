"""Wire protocol for the live cockpit.

Messages are JSON objects with a ``type`` field.  The protocol is deliberately
small and symmetric so a browser, a CLI or another service can drive the same
session.

Client -> server
    ``{"type": "start", "profile": {...}}``
    ``{"type": "question", "text": "...", "confidence": 0.9}``
    ``{"type": "audio", "encoding": "pcm16le", "sample_rate": 16000,
      "data": "<base64>"}``
    ``{"type": "feedback", "text": "too long"}``
    ``{"type": "accept", "edited": false, "rejected": false}``
    ``{"type": "report"}``
    ``{"type": "stop"}``

Server -> client
    ``{"type": "session", ...}``
    ``{"type": "partial", "text": "..."}``
    ``{"type": "analysis", ...}``
    ``{"type": "evidence", ...}``
    ``{"type": "response", "text": "...", "scorecard": {...}, ...}``
    ``{"type": "event", ...}``
    ``{"type": "report", ...}``
    ``{"type": "error", "message": "..."}``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging import get_logger

LOG = get_logger("api.protocol")

PROTOCOL_VERSION = 1

#: message types the server understands
CLIENT_MESSAGES = ("start", "question", "audio", "feedback", "accept", "report",
                   "stop", "ping", "describe")
#: message types the server emits
SERVER_MESSAGES = ("session", "partial", "analysis", "evidence", "response",
                   "event", "report", "error", "pong", "describe", "stopped")


def encode(message: Dict[str, Any]) -> str:
    """Serialise a message, guaranteeing a ``version`` field."""
    payload = {"version": PROTOCOL_VERSION, **message}
    return json.dumps(payload, ensure_ascii=False, default=_default)


def decode(raw: str) -> Dict[str, Any]:
    """Parse a client message, raising ``ValueError`` on malformed input."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("message must be a JSON object")
    if "type" not in data:
        raise ValueError("message is missing a 'type' field")
    return data


def _default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


# --------------------------------------------------------------------------- #
# Typed message helpers
# --------------------------------------------------------------------------- #
@dataclass
class ClientMessage:
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: str) -> "ClientMessage":
        data = decode(raw)
        return cls(type=str(data["type"]), payload={k: v for k, v in data.items()
                                                    if k != "type"})

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def error_message(message: str, *, code: str = "bad_request",
                  recoverable: bool = True) -> Dict[str, Any]:
    return {"type": "error", "code": code, "message": message,
            "recoverable": recoverable}


def response_message(response: Any) -> Dict[str, Any]:
    payload = {"type": "response"}
    if hasattr(response, "to_dict"):
        payload.update(response.to_dict())
    else:  # pragma: no cover - defensive
        payload["text"] = str(response)
    return payload
