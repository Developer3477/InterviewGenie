"""The API server: pure-stdlib asyncio HTTP + WebSocket.

No third-party web framework is required, so the live cockpit runs on a bare
Python interpreter::

    python -m interviewgenie serve --port 8420

and then open ``http://localhost:8420``.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..errors import InterviewGenieError
from ..events import DEFAULT_BUS, EventBus
from ..logging import get_logger
from .routes import Router, SessionStore, websocket_handler
from .ws import handshake

LOG = get_logger("api.server")

MAX_BODY = 32 << 20          # 32 MiB


@dataclass
class APIServer:
    """A tiny asyncio HTTP server exposing the InterviewGenie API."""

    factory: Callable[[], Any] = None  # type: ignore[assignment]
    host: str = "0.0.0.0"
    port: int = 8420
    web_root: str = "web"
    bus: EventBus = field(default_factory=EventBus)

    def __post_init__(self) -> None:
        if self.factory is None:
            from ..factory import build_default_genie

            self.factory = build_default_genie
        self.store = SessionStore(self.factory)
        self.router = Router(store=self.store, web_root=self.web_root)
        self._server: Optional[asyncio.AbstractServer] = None
        self._connections = 0
        self.started_at: Optional[float] = None

    # -- lifecycle -------------------------------------------------------- #
    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        self.started_at = time.time()
        sockets = self._server.sockets or []
        bound = ", ".join(f"{s.getsockname()[0]}:{s.getsockname()[1]}" for s in sockets)
        LOG.info("api server listening", context={"address": bound,
                                                  "web_root": self.web_root})
        return bound

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
        LOG.info("api server stopped")

    def run(self) -> None:  # pragma: no cover - interactive entry point
        """Blocking entry point used by ``python -m interviewgenie serve``.

        Shutdown is driven by an ``asyncio.Event`` rather than ``loop.stop()``:
        stopping the loop outright aborts in-flight requests and leaves
        "Event loop stopped before Future completed" / "Task was destroyed but
        it is pending" noise on every Ctrl-C.  Setting the event lets the
        server finish cleanly instead.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        shutdown = asyncio.Event()

        async def _main() -> None:
            await self.start()
            serving = asyncio.ensure_future(self.serve_forever())
            # Block until a signal arrives, then unwind the serving task so the
            # "async with self._server" block closes the listening socket.
            await shutdown.wait()
            serving.cancel()
            try:
                await serving
            except asyncio.CancelledError:
                pass

        def _request_shutdown() -> None:
            if not shutdown.is_set():
                shutdown.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_shutdown)
            except (NotImplementedError, ValueError):  # pragma: no cover
                pass
        try:
            loop.run_until_complete(_main())
        except KeyboardInterrupt:  # pragma: no cover
            pass
        finally:
            try:
                loop.run_until_complete(self.stop())
            finally:
                loop.close()

    # -- connection handling ---------------------------------------------- #
    async def _handle_client(self, reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter) -> None:
        self._connections += 1
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            # peek at the request line to decide HTTP vs WebSocket upgrade
            parts = request_line.decode("latin-1", "replace").strip().split()
            if len(parts) < 3:
                return
            method, target, _version = parts[0], parts[1], parts[2]

            headers: Dict[str, str] = {}
            while True:
                line = await reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                decoded = line.decode("latin-1", "replace")
                if ":" in decoded:
                    key, value = decoded.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            if "websocket" in headers.get("upgrade", "").lower():
                await self._handle_websocket(reader, writer, request_line, headers)
                return

            body = await self._read_body(reader, headers)
            status, response_headers, payload = self.router.handle(
                method, target, _split_query(target), body)
            await self._send_http(writer, status, response_headers, payload)
        except (ConnectionError, asyncio.IncompleteReadError):  # pragma: no cover
            pass
        except Exception as exc:  # noqa: BLE001 - never kill the accept loop
            LOG.warning("request handling failed", context={"error": str(exc)})
            try:
                await self._send_http(writer, 500, {"Content-Type": "application/json"},
                                      b'{"error":"internal error"}')
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._connections -= 1
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def _handle_websocket(self, reader: asyncio.StreamReader,
                                writer: asyncio.StreamWriter,
                                request_line: bytes, headers: Dict[str, str]) -> None:
        """Upgrade the connection and hand it to the WebSocket session handler.

        The request line and headers have already been consumed by
        :meth:`_handle_client`, so they are replayed into a buffered reader that
        the handshake can consume before it continues on the live socket.
        """
        replay = asyncio.StreamReader()
        replay.feed_data(request_line)
        for key, value in headers.items():
            replay.feed_data(f"{key}: {value}\r\n".encode("latin-1"))
        replay.feed_data(b"\r\n")

        async def _pump() -> None:
            try:
                while True:
                    chunk = await reader.read(4096)
                    if not chunk:
                        replay.feed_eof()
                        return
                    replay.feed_data(chunk)
            except (ConnectionError, OSError):  # pragma: no cover - peer gone
                replay.feed_eof()

        pump_task = asyncio.ensure_future(_pump())
        try:
            connection = await handshake(replay, writer)
            if connection is None:
                return
            await websocket_handler(connection, self.router)
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _read_body(self, reader: asyncio.StreamReader,
                         headers: Dict[str, str]) -> bytes:
        length = int(headers.get("content-length", "0") or 0)
        if length <= 0:
            return b""
        if length > MAX_BODY:
            raise InterviewGenieError("request body too large", recoverable=False)
        return await reader.readexactly(length)

    async def _send_http(self, writer: asyncio.StreamWriter, status: int,
                         headers: Dict[str, str], payload: bytes) -> None:
        reason = {200: "OK", 201: "Created", 204: "No Content", 400: "Bad Request",
                  403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                  422: "Unprocessable Entity", 500: "Internal Server Error"}.get(
                      status, "OK")
        lines = [f"HTTP/1.1 {status} {reason}"]
        merged = {"Server": "InterviewGenie/1.0", "Date": _http_date(), **headers}
        merged.setdefault("Content-Length", str(len(payload)))
        for key, value in merged.items():
            lines.append(f"{key}: {value}")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + payload)
        try:
            await writer.drain()
        except (ConnectionError, OSError):  # pragma: no cover
            pass

    # -- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        return {"connections": self._connections, "sessions": len(self.store.all()),
                "uptime_s": round(time.time() - self.started_at, 2) if self.started_at else 0.0,
                "bus": self.bus.stats()}


class _LiveReader:
    """Adapter exposing the live socket as a StreamReader-like object."""

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader

    async def readline(self) -> bytes:
        return await self._reader.readline()

    async def readexactly(self, count: int) -> bytes:
        return await self._reader.readexactly(count)


def _split_query(target: str) -> str:
    if "?" in target:
        return target.split("?", 1)[1]
    return ""


def _http_date() -> str:
    from email.utils import formatdate

    return formatdate(timeval=None, localtime=False, usegmt=True)


def create_server(host: str = "0.0.0.0", port: int = 8420,
                  web_root: Optional[str] = None, **overrides: Any) -> APIServer:
    """Build a server bound to the repository's default factory."""
    from ..factory import build_demo_genie

    root = web_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "web")
    return APIServer(factory=lambda: build_demo_genie(**overrides),
                     host=host, port=port, web_root=root, bus=DEFAULT_BUS)
