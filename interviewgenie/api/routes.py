"""HTTP routes for the InterviewGenie API server.

Endpoints
---------
``GET  /``                     the cockpit (``web/index.html``)
``GET  /static/<file>``        cockpit assets
``GET  /api/health``           liveness + subsystem summary
``GET  /api/describe``         full system description
``POST /api/session``          start a session (JSON profile)
``GET  /api/session``          current session state
``POST /api/question``         ask a question, get a scored answer
``POST /api/feedback``         send feedback
``POST /api/consolidate``      apply pending learning updates
``GET  /api/report``           session report
``DELETE /api/session``        stop the session
``GET  /api/knowledge``        graph statistics
``GET  /api/knowledge/search`` fuzzy entity search
``GET  /api/benchmark``        run the evaluation benchmark
``WS   /ws``                   the live duplex channel
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from ..errors import InterviewGenieError
from ..logging import get_logger
from .protocol import (CLIENT_MESSAGES, PROTOCOL_VERSION, ClientMessage,
                       encode, error_message)

LOG = get_logger("api.routes")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".map": "application/json",
}


def json_response(payload: Any, status: int = 200) -> Tuple[int, Dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False, default=_default).encode("utf-8")
    return status, {"Content-Type": "application/json; charset=utf-8",
                    "Content-Length": str(len(body)),
                    "Cache-Control": "no-store",
                    "Access-Control-Allow-Origin": "*"}, body


def text_response(body: str, content_type: str, status: int = 200
                  ) -> Tuple[int, Dict[str, str], bytes]:
    data = body.encode("utf-8") if isinstance(body, str) else body
    return status, {"Content-Type": content_type,
                    "Content-Length": str(len(data)),
                    "Access-Control-Allow-Origin": "*"}, data


def error_response(message: str, status: int = 400,
                   code: str = "bad_request") -> Tuple[int, Dict[str, str], bytes]:
    return json_response({"error": message, "code": code}, status=status)


def _default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


# --------------------------------------------------------------------------- #
# Session management
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    """One live interview session."""

    id: str
    genie: Any
    created_at: float = field(default_factory=time.time)
    started: bool = False
    turn_count: int = 0
    last_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "started": self.started,
            "turns": self.turn_count,
            "created_at": self.created_at,
            "phase": getattr(self.genie, "phase", "idle"),
            "has_last_response": self.last_response is not None,
        }


class SessionStore:
    """In-memory session registry (one active session is typical)."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._sessions: Dict[str, Session] = {}
        self._counter = 0

    def create(self) -> Session:
        self._counter += 1
        session = Session(id=f"session-{self._counter}", genie=self._factory())
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def default(self) -> Session:
        for session in self._sessions.values():
            return session
        return self.create()

    def remove(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def all(self) -> List[Session]:
        return list(self._sessions.values())


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
@dataclass
class Router:
    """Maps ``(method, path)`` onto handlers bound to a session store."""

    store: SessionStore
    web_root: str = "web"
    protocol_version: int = PROTOCOL_VERSION

    # -- dispatch --------------------------------------------------------- #
    def handle(self, method: str, path: str, query: str,
               body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        parsed = urlparse(path)
        route = unquote(parsed.path)
        params = parse_qs(query or "")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return error_response("request body is not valid JSON")
        if not isinstance(payload, dict):
            payload = {}

        try:
            return self._dispatch(method.upper(), route, params, payload)
        except InterviewGenieError as exc:
            LOG.warning("request failed", context={"path": route, "error": str(exc)})
            return error_response(str(exc), status=422, code=str(exc.code))
        except Exception as exc:  # noqa: BLE001 - never leak a traceback
            LOG.warning("unhandled request error", context={"path": route, "error": str(exc)})
            return error_response("internal error", status=500, code="internal")

    def _dispatch(self, method: str, route: str, params: Dict[str, List[str]],
                  payload: Dict[str, Any]) -> Tuple[int, Dict[str, str], bytes]:
        # static + cockpit
        if method == "GET" and route in {"/", "/index.html"}:
            return self._static("index.html")
        if method == "GET" and route.startswith("/static/"):
            return self._static(route[len("/static/"):])

        # health / describe
        if method == "GET" and route == "/api/health":
            return json_response({"status": "ok", "version": self.protocol_version,
                                  "time": time.time()})
        if method == "GET" and route == "/api/describe":
            return json_response(self.store.default().genie.describe())

        # sessions
        if route == "/api/session":
            if method == "POST":
                return self._start_session(payload)
            if method == "GET":
                return json_response(self.store.default().to_dict())
            if method == "DELETE":
                return self._stop_session()
        if method == "POST" and route == "/api/question":
            return self._question(payload)
        if method == "POST" and route == "/api/feedback":
            return self._feedback(payload)
        if method == "POST" and route == "/api/accept":
            return self._accept(payload)
        if method == "POST" and route == "/api/consolidate":
            return json_response(self.store.default().genie.consolidate())
        if method == "GET" and route == "/api/report":
            return json_response(self.store.default().genie.report())
        if method == "GET" and route == "/api/knowledge":
            return json_response(self.store.default().genie.graph.stats())
        if method == "GET" and route == "/api/knowledge/search":
            term = (params.get("q") or [""])[0]
            graph = self.store.default().genie.graph
            return json_response({
                "query": term,
                "results": [{"id": n.id, "name": n.get("name", n.id),
                             "label": n.label(),
                             "description": n.get("description", "")}
                            for n in graph.search(term, limit=10)],
            })
        if method == "GET" and route == "/api/benchmark":
            return self._benchmark(params)
        if method == "GET" and route == "/api/protocol":
            return json_response({"version": self.protocol_version,
                                  "client_messages": list(CLIENT_MESSAGES)})

        return error_response(f"no route for {method} {route}", status=404,
                              code="not_found")

    # -- handlers --------------------------------------------------------- #
    def _static(self, name: str) -> Tuple[int, Dict[str, str], bytes]:
        safe = os.path.normpath(name).lstrip("/\\")
        if safe.startswith(".."):
            return error_response("forbidden", status=403, code="forbidden")
        candidates = [
            os.path.join(self.web_root, safe),
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), self.web_root, safe),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                extension = os.path.splitext(candidate)[1].lower()
                with open(candidate, "rb") as fh:
                    return text_response(fh.read(),
                                         CONTENT_TYPES.get(extension, "application/octet-stream"))
        return error_response("not found", status=404, code="not_found")

    def _start_session(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, str], bytes]:
        session = self.store.create()
        profile = payload.get("profile") or {}
        summary = session.genie.start(**profile)
        session.started = True
        return json_response({"session": session.to_dict(), "summary": summary}, status=201)

    def _stop_session(self) -> Tuple[int, Dict[str, str], bytes]:
        session = self.store.default()
        report = session.genie.stop()
        session.started = False
        return json_response({"session": session.to_dict(), "report": report})

    def _question(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, str], bytes]:
        session = self.store.default()
        if not session.started:
            session.genie.start()
            session.started = True
        text = str(payload.get("text", "")).strip()
        if not text:
            return error_response("'text' is required")
        confidence = float(payload.get("confidence", 1.0))
        response = session.genie.ask(text, confidence=confidence)
        session.turn_count += 1
        session.last_response = response.to_dict()
        return json_response({"session": session.to_dict(),
                              "response": response.to_dict()})

    def _feedback(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, str], bytes]:
        session = self.store.default()
        text = str(payload.get("text", "")).strip()
        if not text:
            return error_response("'text' is required")
        return json_response(session.genie.feedback(text))

    def _accept(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, str], bytes]:
        session = self.store.default()
        return json_response(session.genie.accept(
            edited=bool(payload.get("edited", False)),
            rejected=bool(payload.get("rejected", False))))

    def _benchmark(self, params: Dict[str, List[str]]) -> Tuple[int, Dict[str, str], bytes]:
        from ..evaluation.dataset import default_cases
        from ..evaluation.metrics import Benchmark

        session = self.store.default()
        genie = session.genie
        limit = int((params.get("limit") or ["0"])[0] or 0) or None
        benchmark = Benchmark(cases=default_cases())
        result = benchmark.run(genie.nlp, genie.composer, genie.retriever, limit=limit)
        return json_response(result.to_dict())


# --------------------------------------------------------------------------- #
# WebSocket handler
# --------------------------------------------------------------------------- #
async def websocket_handler(connection: Any, router: Router) -> None:
    """Drive one WebSocket session."""
    from .ws import WebSocketConnection  # noqa: F401 - re-exported for clarity

    session = router.store.create()
    await connection.send_text(encode({
        "type": "session",
        "session": session.to_dict(),
        "protocol_version": router.protocol_version,
    }))
    try:
        while not connection.closed:
            raw = await connection.recv()
            if raw is None:
                break
            try:
                message = ClientMessage.parse(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                await connection.send_text(encode(
                    error_message(f"malformed message: {exc}")))
                continue
            await _handle_ws_message(connection, session, router, message)
    except Exception as exc:  # noqa: BLE001 - log and drop the socket
        LOG.warning("websocket session ended", context={"error": str(exc)})
    finally:
        await connection.close()


async def _handle_ws_message(connection: Any, session: Session, router: Router,
                             message: ClientMessage) -> None:
    kind = message.type
    genie = session.genie
    if kind == "ping":
        await connection.send_text(encode({"type": "pong", "time": time.time()}))
        return
    if kind == "start":
        profile = message.get("profile") or {}
        summary = genie.start(**profile)
        session.started = True
        await connection.send_text(encode({"type": "session", "session": session.to_dict(),
                                           "summary": summary}))
        return
    if kind == "question":
        text = str(message.get("text", "")).strip()
        if not text:
            await connection.send_text(encode(error_message("'text' is required")))
            return
        if not session.started:
            genie.start()
            session.started = True
        await _emit_analysis(connection, genie, text)
        response = genie.ask(text, confidence=float(message.get("confidence", 1.0)))
        session.turn_count += 1
        session.last_response = response.to_dict()
        await connection.send_text(encode({
            "type": "response", "session": session.to_dict(), **response.to_dict()}))
        return
    if kind == "audio":
        data = message.get("data", "")
        try:
            samples = _decode_audio(data, str(message.get("encoding", "pcm16le")))
        except Exception as exc:  # noqa: BLE001
            await connection.send_text(encode(error_message(f"bad audio: {exc}")))
            return
        if not session.started:
            genie.start()
            session.started = True
        for chunk in genie.ingest(samples, int(message.get("sample_rate", 16000))):
            if chunk.is_final:
                await _emit_analysis(connection, genie, chunk.text)
                response = genie.handle_transcript(chunk)
                if response is not None:
                    await connection.send_text(encode({
                        "type": "response", "session": session.to_dict(),
                        **response.to_dict()}))
            else:
                await connection.send_text(encode({
                    "type": "partial", "text": chunk.text,
                    "confidence": chunk.confidence}))
        return
    if kind == "feedback":
        result = genie.feedback(str(message.get("text", "")))
        await connection.send_text(encode({"type": "event", "event": "learning.feedback",
                                           "payload": result}))
        return
    if kind == "accept":
        result = genie.accept(edited=bool(message.get("edited", False)),
                              rejected=bool(message.get("rejected", False)))
        await connection.send_text(encode({"type": "event", "event": "learning.outcome",
                                           "payload": result}))
        return
    if kind == "report":
        await connection.send_text(encode({"type": "report",
                                           "report": genie.report()}))
        return
    if kind == "describe":
        await connection.send_text(encode({"type": "describe",
                                           "describe": genie.describe()}))
        return
    if kind == "stop":
        report = genie.stop()
        session.started = False
        await connection.send_text(encode({"type": "stopped", "report": report}))
        await connection.close()
        return
    await connection.send_text(encode(error_message(f"unknown message type {kind!r}")))


async def _emit_analysis(connection: Any, genie: Any, text: str) -> None:
    """Push the NLP analysis and the retrieved evidence to the cockpit."""
    try:
        analysis = genie.nlp.analyze(text, update_history=False)
        retrieval = genie.retriever.retrieve(analysis)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("analysis emission failed: %s", exc)
        return
    await connection.send_text(encode({
        "type": "analysis",
        "text": text,
        "intent": analysis.intent.name if analysis.intent else None,
        "topic": analysis.intent.topic if analysis.intent else None,
        "confidence": analysis.intent.confidence if analysis.intent else 0.0,
        "entities": [{"text": e.text, "label": e.label,
                      "canonical": e.canonical} for e in analysis.entities],
        "keywords": [[k, round(v, 4)] for k, v in analysis.keywords[:8]],
        "sentiment": analysis.sentiment.to_dict() if analysis.sentiment else None,
        "emotion": analysis.emotion.to_dict() if analysis.emotion else None,
    }))
    await connection.send_text(encode({
        "type": "evidence",
        "text": text,
        "entities": [str(n.get("name", n.id)) for n in retrieval.entities],
        "evidence": [e.to_dict() for e in retrieval.evidence],
        "paths": [p.to_dict() for p in retrieval.paths[:5]],
    }))


def _decode_audio(data: str, encoding: str) -> List[float]:
    """Decode base64 audio into floats."""
    from ..asr.dsp import pcm16_to_floats

    raw = base64.b64decode(data)
    if encoding in {"pcm16le", "pcm16", "linear16", ""}:
        return [v / 32768.0 for v in pcm16_to_floats(raw)]
    if encoding in {"float32", "f32"}:
        import struct

        count = len(raw) // 4
        return list(struct.unpack(f"<{count}f", raw[: count * 4]))
    raise ValueError(f"unsupported audio encoding {encoding!r}")
