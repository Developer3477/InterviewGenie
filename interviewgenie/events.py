"""A tiny typed publish/subscribe event bus.

The pipeline is fully decoupled from its consumers: stages emit
:class:`~interviewgenie.types.PipelineEvent` objects, and any number of sinks
(web cockpit, metrics collector, logger, transcript recorder) subscribe.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional

from .logging import get_logger
from .types import PipelineEvent

LOG = get_logger("events")

Handler = Callable[[PipelineEvent], None]


class EventBus:
    """Thread-safe, wildcard-capable event bus."""

    def __init__(self, history_size: int = 1000) -> None:
        self._handlers: Dict[str, List[Handler]] = defaultdict(list)
        self._wildcard: List[Handler] = []
        self._lock = threading.RLock()
        self._history: Deque[PipelineEvent] = deque(maxlen=history_size)
        self._dropped = 0

    # -- subscription ------------------------------------------------------ #
    def on(self, event_type: str, handler: Handler) -> Handler:
        """Subscribe ``handler`` to a concrete event type (or ``'*'``)."""
        with self._lock:
            if event_type == "*":
                self._wildcard.append(handler)
            else:
                self._handlers[event_type].append(handler)
        return handler

    def off(self, event_type: str, handler: Handler) -> None:
        with self._lock:
            if event_type == "*":
                if handler in self._wildcard:
                    self._wildcard.remove(handler)
            elif handler in self._handlers.get(event_type, []):
                self._handlers[event_type].remove(handler)

    def once(self, event_type: str, handler: Handler) -> Handler:
        def _wrapper(event: PipelineEvent) -> None:
            self.off(event_type, _wrapper)
            handler(event)

        return self.on(event_type, _wrapper)

    # -- emission ---------------------------------------------------------- #
    def emit(self, event: PipelineEvent) -> PipelineEvent:
        with self._lock:
            self._history.append(event)
            handlers = list(self._handlers.get(event.type, ())) + list(self._wildcard)
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 - a bad sink must not kill the pipeline
                self._dropped += 1
                LOG.debug("event handler failed: %s", exc, context={"type": event.type})
        return event

    def publish(self, event_type: str, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> PipelineEvent:
        return self.emit(PipelineEvent(type=event_type, payload=payload or {}, **kwargs))

    # -- introspection ----------------------------------------------------- #
    @property
    def history(self) -> List[PipelineEvent]:
        return list(self._history)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = defaultdict(int)
            for evt in self._history:
                counts[evt.type] += 1
            return {
                "events": len(self._history),
                "handlers": sum(len(v) for v in self._handlers.values()) + len(self._wildcard),
                "handler_failures": self._dropped,
                "by_type": dict(counts),
            }


class ReplayableBus(EventBus):
    """Event bus that can replay its history to a late subscriber."""

    def on(self, event_type: str, handler: Handler, replay: bool = False) -> Handler:  # noqa: D102
        registered = super().on(event_type, handler)
        if replay:
            for evt in self.history:
                if event_type == "*" or evt.type == event_type:
                    try:
                        handler(evt)
                    except Exception:  # noqa: BLE001
                        pass
        return registered


#: process-wide default bus
DEFAULT_BUS = EventBus()


def emit(event_type: str, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> PipelineEvent:
    return DEFAULT_BUS.publish(event_type, payload, **kwargs)
