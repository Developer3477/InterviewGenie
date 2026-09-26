"""Tests for the stdlib WebSocket client and the streaming STT adapters.

The client is exercised against a real local WebSocket server, because the
failure mode that matters -- a framing bug that silently truncates audio --
only shows up against an actual socket.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from tests.base import GenieTestCase

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class _EchoWSServer:
    """A minimal WebSocket server that echoes audio and replies with JSON."""

    def __init__(self, reply_factory=None):
        seen = {"audio": b"", "closed": False, "path": "", "auth": None,
                "text": []}
        self.seen = seen
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.headers.get("Upgrade", "").lower() != "websocket":
                    self.send_response(400)
                    self.end_headers()
                    return
                seen["path"] = self.path
                seen["auth"] = self.headers.get("Authorization")
                accept = base64.b64encode(hashlib.sha1(
                    (self.headers["Sec-WebSocket-Key"] + _GUID).encode()).digest()).decode()
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                sock = self.connection
                sock.settimeout(10)
                buf = b""

                def read(count):
                    nonlocal buf
                    while len(buf) < count:
                        chunk = sock.recv(65536)
                        if not chunk:
                            raise ConnectionError("closed")
                        buf += chunk
                    out, buf = buf[:count], buf[count:]
                    return out

                def send_text(obj):
                    payload = json.dumps(obj).encode()
                    if len(payload) < 126:
                        head = b"\x81" + bytes([len(payload)])
                    else:
                        head = b"\x81\x7e" + struct.pack(">H", len(payload))
                    sock.sendall(head + payload)

                try:
                    while True:
                        first, second = read(2)
                        length = second & 0x7F
                        if length == 126:
                            length = struct.unpack(">H", read(2))[0]
                        elif length == 127:
                            length = struct.unpack(">Q", read(8))[0]
                        mask = read(4) if second & 0x80 else b"\x00\x00\x00\x00"
                        data = read(length) if length else b""
                        if second & 0x80:
                            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
                        opcode = first & 0x0F
                        if opcode == 0x8:
                            seen["closed"] = True
                            break
                        if opcode == 0x1:
                            seen["text"].append(json.loads(data.decode()))
                            if reply_factory:
                                send_text(reply_factory(seen))
                            continue
                        if opcode == 0x2:
                            seen["audio"] += data
                            if reply_factory:
                                send_text(reply_factory(seen))
                except Exception:  # noqa: BLE001 - teardown
                    pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}/v1/listen"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class WebSocketClientTests(GenieTestCase):
    def setUp(self):
        super().setUp()
        self.server = _EchoWSServer()
        self.addCleanup(self.server.stop)

    def test_handshake_and_text_frame(self):
        from interviewgenie.asr.wsclient import WebSocketClient

        with WebSocketClient(self.server.url, headers={"Authorization": "Token k"},
                             timeout=10) as ws:
            ws.send_text(json.dumps({"hello": "world"}))
            time.sleep(0.2)
            self.assertEqual(self.server.seen["auth"], "Token k")
            self.assertIn("hello", self.server.seen["text"][0])

    def test_binary_frames_arrive_intact(self):
        from interviewgenie.asr.wsclient import WebSocketClient

        with WebSocketClient(self.server.url, timeout=10) as ws:
            ws.send_binary(b"\x01\x02" * 100)
            time.sleep(0.2)
            self.assertEqual(self.server.seen["audio"], b"\x01\x02" * 100)

    def test_extended_length_frames(self):
        from interviewgenie.asr.wsclient import WebSocketClient

        with WebSocketClient(self.server.url, timeout=20) as ws:
            big = bytes(range(256)) * 500        # 128 000 bytes
            ws.send_binary(big)
            time.sleep(0.3)
            self.assertEqual(len(self.server.seen["audio"]), len(big))

    def test_json_frames_are_decoded(self):
        from interviewgenie.asr.wsclient import WebSocketClient

        reply = {"type": "Results", "is_final": True,
                 "channel": {"alternatives": [{"transcript": "hi there",
                                               "confidence": 0.9}]}}
        server = _EchoWSServer(reply_factory=lambda seen: reply)
        self.addCleanup(server.stop)
        with WebSocketClient(server.url, timeout=10) as ws:
            ws.send_binary(b"\x00" * 32)
            message = ws.recv_json()
            self.assertIsNotNone(message)
            self.assertEqual(message["channel"]["alternatives"][0]["transcript"],
                             "hi there")

    def test_close_frame_is_honoured(self):
        from interviewgenie.asr.wsclient import WebSocketClient

        ws = WebSocketClient(self.server.url, timeout=10)
        ws.close()
        self.assertTrue(ws.closed)
        time.sleep(0.2)
        self.assertTrue(self.server.seen["closed"])

    def test_bad_scheme_raises_a_typed_error(self):
        from interviewgenie.asr.wsclient import WebSocketClient, WebSocketError

        with self.assertRaises(WebSocketError):
            WebSocketClient("http://example.com/", timeout=2)

    def test_refused_connection_raises_a_typed_error(self):
        from interviewgenie.asr.wsclient import WebSocketClient, WebSocketError

        with self.assertRaises(WebSocketError):
            WebSocketClient("ws://127.0.0.1:1/x", timeout=2)

    def test_recv_after_close_returns_none(self):
        from interviewgenie.asr.wsclient import WebSocketClient

        ws = WebSocketClient(self.server.url, timeout=10)
        ws.close()
        self.assertIsNone(ws.recv())


class DeepgramStreamProviderTests(GenieTestCase):
    """The Deepgram adapter is driven against a local stand-in socket."""

    def _provider(self):
        import interviewgenie.asr.wsclient as wsclient
        from interviewgenie.asr.cloud import DeepgramStreamProvider

        real = wsclient.WebSocketClient
        outer = self

        class Redirected(real):
            def __init__(self, url, headers=None, timeout=20.0):
                path = url.split("api.deepgram.com", 1)[-1]
                super().__init__(f"ws://127.0.0.1:{outer.port}{path}",
                                 headers=headers, timeout=timeout)

        self.patcher = mock.patch.object(wsclient, "WebSocketClient", Redirected)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        provider = DeepgramStreamProvider(api_key="dg-test")
        self.addCleanup(provider.close)
        return provider

    def setUp(self):
        super().setUp()
        replies = {"index": 0}
        # the provider opens the socket before sending audio, so the first
        # reply the reader thread sees is the one for the first PCM frame
        sequence = [
            {"type": "Results", "is_final": False, "start": 0.0, "end": 1.2,
             "channel": {"alternatives": [{"transcript": "tell me about a",
                                           "confidence": 0.6}]}},
            {"type": "Results", "is_final": True, "speech_final": True,
             "start": 0.0, "end": 3.4,
             "channel": {"alternatives": [{"transcript": "Tell me about a time you led a team.",
                                           "confidence": 0.97}]}},
        ]

        def factory(seen):
            frame = sequence[min(replies["index"], len(sequence) - 1)]
            replies["index"] += 1
            return frame

        self.server = _EchoWSServer(reply_factory=factory)
        self.port = self.server.port
        self.addCleanup(self.server.stop)

    def test_query_string_and_auth_header(self):
        provider = self._provider()
        provider.accept_audio([0.0] * 3200, 16000)
        time.sleep(0.3)
        self.assertIn("encoding=linear16", self.server.seen["path"])
        self.assertIn("punctuate=true", self.server.seen["path"])
        self.assertIn("smart_format=true", self.server.seen["path"])
        self.assertIn("endpointing=300", self.server.seen["path"])
        self.assertIn("sample_rate=16000", self.server.seen["path"])
        self.assertEqual(self.server.seen["auth"], "Token dg-test")

    def test_audio_is_streamed_as_pcm(self):
        provider = self._provider()
        provider.accept_audio([0.0] * 3200, 16000)
        time.sleep(0.3)
        # 3200 float samples -> 6400 bytes of little-endian int16
        self.assertEqual(len(self.server.seen["audio"]), 6400)

    def test_interim_then_final_transcripts(self):
        provider = self._provider()
        provider.accept_audio([0.0] * 3200, 16000)
        time.sleep(0.4)
        chunks = provider.poll()
        self.assertTrue(any(not c.is_final for c in chunks))
        self.assertEqual(chunks[0].provider, "deepgram")

        provider.flush()
        time.sleep(0.4)
        finals = [c for c in provider.poll() if c.is_final]
        self.assertTrue(finals)
        self.assertGreater(finals[0].confidence, 0.9)
        self.assertIn("led a team", finals[0].text)

    def test_missing_key_raises_a_typed_error(self):
        from interviewgenie.errors import ASRError

        provider = self._provider()
        provider.api_key = ""
        with self.assertRaises(ASRError):
            provider.accept_audio([0.0] * 1600, 16000)

    def test_capabilities_report_streaming(self):
        provider = self._provider()
        caps = provider.capabilities()
        self.assertTrue(caps.streaming)
        self.assertTrue(caps.partial_results)
        self.assertTrue(caps.punctuation)
        self.assertTrue(caps.noise_robust)

    def test_reset_clears_pending_chunks(self):
        provider = self._provider()
        provider.accept_audio([0.0] * 3200, 16000)
        time.sleep(0.4)
        provider.reset()
        self.assertEqual(provider.poll(), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
