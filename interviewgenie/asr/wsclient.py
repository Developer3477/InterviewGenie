"""A minimal WebSocket client built only from the standard library.

Third-party SDKs are not available in this environment, and ``urllib`` cannot
hold a socket open long enough for real-time speech.  This module implements
just enough of RFC 6455 to talk to a streaming transcription service: the
opening handshake, client-side masking, and text/binary frame decoding.

Only the pieces a speech stream needs are implemented — no extensions, no
subprotocols, no fragmentation reassembly beyond the trivial case, because
every vendor used here sends small unfragmented frames.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocketError(RuntimeError):
    """Raised for handshake, framing or connection failures."""


class WebSocketClient:
    """A blocking, single-connection WebSocket client.

    ``send_text``/``send_binary`` push frames out; ``recv`` returns one frame at
    a time (or ``None`` when the peer closes).  All reads are bounded by
    ``timeout`` so a stalled transcription service cannot hang a turn.
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None,
                 timeout: float = 20.0, extra_headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.timeout = timeout
        self.closed = False
        self._sock: Optional[socket.socket] = None
        self._buffer = b""
        parsed = urlparse(url)
        if parsed.scheme not in ("ws", "wss"):
            raise WebSocketError(f"unsupported scheme: {parsed.scheme}")
        self._secure = parsed.scheme == "wss"
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or (443 if self._secure else 80)
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += "?" + parsed.query
        self._connect(headers or {}, extra_headers or {})

    # -- handshake --------------------------------------------------------
    def _connect(self, headers: Dict[str, str], extra: Dict[str, str]) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        lines = [
            f"GET {self.path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        for name, value in {**headers, **extra}.items():
            lines.append(f"{name}: {value}")
        request = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")

        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise WebSocketError(f"connect failed: {exc}") from exc
        if self._secure:
            context = ssl.create_default_context()
            sock = context.wrap_socket(sock, server_hostname=self.host)
        self._sock = sock
        sock.sendall(request)
        sock.settimeout(self.timeout)

        response = self._read_http_response()
        status = response.get("status", "")
        if "101" not in status:
            raise WebSocketError(f"handshake rejected: {status} {response.get('body', '')[:200]}")
        expected = base64.b64encode(
            hashlib.sha1((key + _GUID).encode("ascii")).digest()).decode("ascii")
        if response["headers"].get("sec-websocket-accept", "").strip() != expected:
            raise WebSocketError("handshake accept key mismatch")

    def _read_http_response(self) -> Dict[str, Any]:
        assert self._sock is not None
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise WebSocketError("connection closed during handshake")
            data += chunk
            if len(data) > 65536:
                raise WebSocketError("handshake response too large")
        head, _, rest = data.partition(b"\r\n\r\n")
        self._buffer = rest
        lines = head.decode("iso-8859-1").split("\r\n")
        out: Dict[str, Any] = {"status": lines[0], "headers": {}, "body": ""}
        for line in lines[1:]:
            if ":" in line:
                name, _, value = line.partition(":")
                out["headers"][name.strip().lower()] = value.strip()
        return out

    # -- framing ----------------------------------------------------------
    def _read_exact(self, count: int) -> bytes:
        assert self._sock is not None
        while len(self._buffer) < count:
            chunk = self._sock.recv(max(4096, count - len(self._buffer)))
            if not chunk:
                raise WebSocketError("connection closed")
            self._buffer += chunk
        out, self._buffer = self._buffer[:count], self._buffer[count:]
        return out

    def send_binary(self, payload: bytes) -> None:
        self._send(0x2, payload)

    def send_text(self, text: str) -> None:
        self._send(0x1, text.encode("utf-8"))

    def _send(self, opcode: int, payload: bytes) -> None:
        assert self._sock is not None
        mask = os.urandom(4)
        header = bytes([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header += bytes([0x80 | length])
        elif length < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", length)
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", length)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        try:
            self._sock.sendall(header + mask + masked)
        except OSError as exc:
            self.closed = True
            raise WebSocketError(f"send failed: {exc}") from exc

    def recv(self) -> Optional[Tuple[int, bytes]]:
        """Return ``(opcode, payload)`` or ``None`` on a close frame."""
        if self.closed:
            return None
        try:
            first = self._read_exact(2)
            opcode = first[0] & 0x0F
            length = first[1] & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read_exact(8))[0]
            payload = self._read_exact(length) if length else b""
            if opcode == 0x8:          # close
                self.close()
                return None
            if opcode == 0x9:          # ping -> pong
                self._send(0xA, payload)
                return self.recv()
            return opcode, payload
        except (WebSocketError, OSError):
            self.closed = True
            return None

    def recv_json(self) -> Optional[Dict[str, Any]]:
        frame = self.recv()
        if frame is None:
            return None
        try:
            return json.loads(frame[1].decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    # -- teardown ---------------------------------------------------------
    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._sock is None:
            return
        try:
            self._sock.sendall(b"\x88\x80" + os.urandom(4))
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "WebSocketClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
