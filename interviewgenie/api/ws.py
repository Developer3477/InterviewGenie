"""A minimal, dependency-free WebSocket (RFC 6455) implementation.

The live cockpit needs a duplex channel so partial hypotheses can stream out
while audio streams in.  Rather than pull in a websockets library, this module
implements the subset the server actually uses:

* the ``Sec-WebSocket-Accept`` handshake,
* server-side frame parsing (text, binary, close, ping, pong),
* fragmented-message reassembly,
* masked client frames and unmasked server frames,
* close handshake and ping/pong keep-alive.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import struct
from typing import Awaitable, Callable, Optional

from ..logging import get_logger

LOG = get_logger("api.ws")

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

#: opcodes
OPCODE_CONTINUATION = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA

_MAX_FRAME = 1 << 20          # 1 MiB per frame
_MAX_MESSAGE = 8 << 20        # 8 MiB per message


class WebSocketError(Exception):
    """Protocol level failure."""


def accept_key(client_key: str) -> str:
    """Compute the ``Sec-WebSocket-Accept`` value for a client key."""
    digest = hashlib.sha1((client_key.strip() + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


class WebSocketConnection:
    """Server side of one WebSocket connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 path: str = "/") -> None:
        self.reader = reader
        self.writer = writer
        self.path = path
        self.closed = False
        self._fragments: list[bytes] = []
        self._fragment_opcode: Optional[int] = None

    # -- lifecycle -------------------------------------------------------- #
    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        try:
            payload = struct.pack("!H", code) + reason.encode("utf-8")[:123]
            await self._send_frame(OPCODE_CLOSE, payload)
            await self.writer.drain()
        except (ConnectionError, OSError):  # pragma: no cover - peer gone
            pass
        finally:
            try:
                self.writer.close()
            except Exception:  # noqa: BLE001
                pass

    # -- reading ---------------------------------------------------------- #
    async def recv(self) -> Optional[bytes]:
        """Return the next complete message, or ``None`` when the peer closes."""
        while True:
            frame = await self._read_frame()
            if frame is None:
                return None
            opcode, payload = frame
            if opcode == OPCODE_CLOSE:
                await self.close(1000, "")
                return None
            if opcode == OPCODE_PING:
                await self._send_frame(OPCODE_PONG, payload)
                continue
            if opcode == OPCODE_PONG:
                continue
            if opcode == OPCODE_CONTINUATION:
                if self._fragment_opcode is None:
                    raise WebSocketError("continuation frame without a start frame")
                self._fragments.append(payload)
            else:
                if self._fragment_opcode is not None:
                    raise WebSocketError("new data frame during a fragmented message")
                self._fragment_opcode = opcode
                self._fragments.append(payload)
            if not self._final:
                continue
            message = b"".join(self._fragments)
            opcode = self._fragment_opcode or OPCODE_TEXT
            self._fragments = []
            self._fragment_opcode = None
            if len(message) > _MAX_MESSAGE:
                raise WebSocketError("message too large")
            return message

    async def _read_frame(self) -> Optional[tuple[int, bytes]]:
        header = await self._read_exact(2)
        if header is None:
            return None
        first, second = header[0], header[1]
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        self._final = fin
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            extended = await self._read_exact(2)
            if extended is None:
                return None
            length = struct.unpack("!H", extended)[0]
        elif length == 127:
            extended = await self._read_exact(8)
            if extended is None:
                return None
            length = struct.unpack("!Q", extended)[0]
        if length > _MAX_FRAME:
            raise WebSocketError("frame too large")
        mask = b""
        if masked:
            mask = await self._read_exact(4) or b""
        payload = await self._read_exact(length) if length else b""
        if payload is None:
            payload = b""
        if masked and mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    async def _read_exact(self, count: int) -> Optional[bytes]:
        if count <= 0:
            return b""
        try:
            return await self.reader.readexactly(count)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

    # -- writing ---------------------------------------------------------- #
    async def send_text(self, text: str) -> None:
        await self._send_frame(OPCODE_TEXT, text.encode("utf-8"))

    async def send_binary(self, data: bytes) -> None:
        await self._send_frame(OPCODE_BINARY, data)

    async def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self.closed and opcode != OPCODE_CLOSE:
            return
        header = bytearray()
        header.append(0x80 | opcode)
        length = len(payload)
        if length < 126:
            header.append(length)
        elif length < (1 << 16):
            header.append(126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", length))
        self.writer.write(bytes(header) + payload)
        try:
            await self.writer.drain()
        except (ConnectionError, OSError):  # pragma: no cover - peer gone
            self.closed = True


async def handshake(reader: asyncio.StreamReader,
                    writer: asyncio.StreamWriter) -> Optional[WebSocketConnection]:
    """Perform the HTTP upgrade handshake; returns ``None`` if it is not one."""
    request_line = await reader.readline()
    if not request_line:
        return None
    parts = request_line.decode("latin-1").strip().split()
    if len(parts) < 3:
        return None
    method, path, _version = parts[0], parts[1], parts[2]

    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("latin-1")
        if ":" in decoded:
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    upgrade = headers.get("upgrade", "").lower()
    key = headers.get("sec-websocket-key")
    if "websocket" not in upgrade or not key:
        body = b'{"error":"expected a websocket upgrade"}'
        writer.write(b"HTTP/1.1 400 Bad Request\r\n"
                     b"Content-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                     b"Connection: close\r\n\r\n" + body)
        await writer.drain()
        writer.close()
        return None

    accept = accept_key(key)
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    writer.write(response.encode("ascii"))
    await writer.drain()
    LOG.debug("websocket handshake complete", context={"path": path})
    return WebSocketConnection(reader, writer, path=path)
