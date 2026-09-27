"""Tests for the API server, protocol and the orchestrator end-to-end."""

from __future__ import annotations

import asyncio
import base64
import json
import unittest

from tests.base import GenieTestCase


class ProtocolTests(GenieTestCase):
    def test_encode_adds_a_version(self):
        from interviewgenie.api.protocol import decode, encode

        raw = encode({"type": "ping"})
        self.assertIn('"version"', raw)
        self.assertEqual(decode(raw)["type"], "ping")

    def test_decode_rejects_malformed_messages(self):
        from interviewgenie.api.protocol import decode

        with self.assertRaises(ValueError):
            decode("not json")
        with self.assertRaises(ValueError):
            decode(json.dumps({"no": "type"}))

    def test_client_message_parsing(self):
        from interviewgenie.api.protocol import ClientMessage

        message = ClientMessage.parse('{"type":"question","text":"hello"}')
        self.assertEqual(message.type, "question")
        self.assertEqual(message.get("text"), "hello")
        self.assertIsNone(message.get("missing"))

    def test_error_message_shape(self):
        from interviewgenie.api.protocol import error_message

        message = error_message("boom")
        self.assertEqual(message["type"], "error")
        self.assertEqual(message["message"], "boom")
        self.assertTrue(message["recoverable"])


class RouterTests(GenieTestCase):
    def _router(self):
        from interviewgenie.api.routes import Router, SessionStore

        return Router(store=SessionStore(lambda: self.genie), web_root="web")

    def test_health_endpoint(self):
        status, headers, body = self._router().handle("GET", "/api/health", "", b"")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_static_index_is_served(self):
        status, _headers, body = self._router().handle("GET", "/", "", b"")
        self.assertEqual(status, 200)
        self.assertIn(b"InterviewGenie", body)

    def test_static_assets_are_served(self):
        for asset in ("/static/capture.html",):
            status, _headers, body = self._router().handle("GET", asset, "", b"")
            self.assertEqual(status, 200, msg=asset)
            self.assertTrue(body)

    def test_path_traversal_is_blocked(self):
        status, _headers, _body = self._router().handle(
            "GET", "/static/../config/intent_dataset.json", "", b"")
        self.assertEqual(status, 403)

    def test_question_round_trip(self):
        status, _headers, body = self._router().handle(
            "POST", "/api/question", "", json.dumps(
                {"text": "Tell me about yourself."}).encode())
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["response"]["text"])
        self.assertIn("scorecard", payload["response"])

    def test_question_requires_text(self):
        status, _headers, body = self._router().handle(
            "POST", "/api/question", "", json.dumps({}).encode())
        self.assertEqual(status, 400)

    def test_malformed_json_is_rejected(self):
        status, _headers, _body = self._router().handle(
            "POST", "/api/question", "", b"{not json")
        self.assertEqual(status, 400)

    def test_unknown_route_returns_404(self):
        status, _headers, _body = self._router().handle("GET", "/api/nope", "", b"")
        self.assertEqual(status, 404)

    def test_knowledge_search(self):
        status, _headers, body = self._router().handle(
            "GET", "/api/knowledge/search", "q=redis", b"")
        self.assertEqual(status, 200)
        results = json.loads(body)["results"]
        self.assertTrue(any(r["name"] == "Redis" for r in results))

    def test_report_endpoint(self):
        status, _headers, body = self._router().handle("GET", "/api/report", "", b"")
        self.assertEqual(status, 200)
        self.assertIn("metrics", json.loads(body))

    def test_benchmark_endpoint(self):
        status, _headers, body = self._router().handle(
            "GET", "/api/benchmark", "limit=3", b"")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["cases"], 3)


class WebSocketTests(GenieTestCase):
    def test_handshake_key_matches_rfc6455(self):
        from interviewgenie.api.ws import accept_key

        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
                         "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_frame_codec_round_trip(self):
        async def scenario():
            from interviewgenie.api.ws import WebSocketConnection

            class FakeWriter:
                def __init__(self):
                    self.buffer = bytearray()
                    self.closed = False

                def write(self, data):
                    self.buffer.extend(data)

                async def drain(self):
                    return None

                def close(self):
                    self.closed = True

            reader = asyncio.StreamReader()
            # a masked text frame carrying "ping"
            payload = b"ping"
            mask = b"\x01\x02\x03\x04"
            masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            header = bytes([0x81, 0x80 | len(payload)])
            reader.feed_data(header + mask + masked)
            reader.feed_eof()

            writer = FakeWriter()
            connection = WebSocketConnection(reader, writer)
            message = await connection.recv()
            self.assertEqual(message, b"ping")

            await connection.send_text("pong")
            self.assertTrue(bytes(writer.buffer).startswith(b"\x81"))

        asyncio.run(scenario())

    def test_close_frame_ends_the_stream(self):
        async def scenario():
            from interviewgenie.api.ws import WebSocketConnection

            class FakeWriter:
                def __init__(self):
                    self.buffer = bytearray()

                def write(self, data):
                    self.buffer.extend(data)

                async def drain(self):
                    return None

                def close(self):
                    pass

            reader = asyncio.StreamReader()
            reader.feed_data(bytes([0x88, 0x00]))
            reader.feed_eof()
            connection = WebSocketConnection(reader, FakeWriter())
            self.assertIsNone(await connection.recv())
            self.assertTrue(connection.closed)

        asyncio.run(scenario())

    def test_ping_is_answered_with_pong(self):
        async def scenario():
            from interviewgenie.api.ws import WebSocketConnection

            class FakeWriter:
                def __init__(self):
                    self.buffer = bytearray()

                def write(self, data):
                    self.buffer.extend(data)

                async def drain(self):
                    return None

                def close(self):
                    pass

            reader = asyncio.StreamReader()
            reader.feed_data(bytes([0x89, 0x80, 0, 0, 0, 0]))   # masked empty ping
            reader.feed_data(bytes([0x81, 0x80, 0, 0, 0, 0]) + b"")  # masked empty text
            reader.feed_eof()
            writer = FakeWriter()
            connection = WebSocketConnection(reader, writer)
            await connection.recv()
            self.assertTrue(bytes(writer.buffer).startswith(b"\x8a"))

        asyncio.run(scenario())

    def test_audio_decoding(self):
        from interviewgenie.api.routes import _decode_audio

        raw = base64.b64encode(b"\x00\x10\x00\x20\xff\xf0").decode()
        floats = _decode_audio(raw, "pcm16le")
        self.assertEqual(len(floats), 3)
        self.assertAlmostEqual(floats[1], 0x2000 / 32768.0, places=5)
        with self.assertRaises(ValueError):
            _decode_audio(raw, "mulaw")


class ServerTests(GenieTestCase):
    def test_http_server_serves_the_api(self):
        async def scenario():
            from interviewgenie.api.routes import Router, SessionStore
            from interviewgenie.api.server import APIServer

            server = APIServer(factory=lambda: self.genie, host="127.0.0.1", port=0)
            address = await server.start()
            port = address.split(":")[1]
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", int(port))
                writer.write(b"GET /api/health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                await writer.drain()
                response = await reader.read(4096)
                writer.close()
                self.assertIn(b"200 OK", response)
                self.assertIn(b'"status"', response)
            finally:
                await server.stop()

        asyncio.run(scenario())

    def test_websocket_server_round_trip(self):
        async def scenario():
            from interviewgenie.api.server import APIServer

            server = APIServer(factory=lambda: self.genie, host="127.0.0.1", port=0)
            address = await server.start()
            port = int(address.split(":")[1])

            async def read_frame(reader):
                """Read one server frame, returning ``(opcode, payload)``."""
                header = await reader.readexactly(2)
                opcode = header[0] & 0x0F
                length = header[1] & 0x7F
                if length == 126:
                    length = int.from_bytes(await reader.readexactly(2), "big")
                elif length == 127:
                    length = int.from_bytes(await reader.readexactly(8), "big")
                payload = await reader.readexactly(length) if length else b""
                return opcode, payload

            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                key = "dGhlIHNhbXBsZSBub25jZQ=="
                writer.write(
                    b"GET /ws HTTP/1.1\r\nHost: localhost\r\n"
                    b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    b"Sec-WebSocket-Key: " + key.encode() + b"\r\n"
                    b"Sec-WebSocket-Version: 13\r\n\r\n")
                await writer.drain()
                handshake = await reader.readuntil(b"\r\n\r\n")
                self.assertIn(b"101 Switching Protocols", handshake)
                self.assertIn(b"s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", handshake)

                payload = json.dumps({"type": "question",
                                      "text": "Tell me about yourself."}).encode()
                mask = b"\x11\x22\x33\x44"
                masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
                writer.write(bytes([0x81, 0x80 | len(payload)]) + mask + masked)
                await writer.drain()

                deadline = asyncio.get_event_loop().time() + 60
                seen = []
                while asyncio.get_event_loop().time() < deadline:
                    opcode, body = await read_frame(reader)
                    if opcode != 0x1:            # ignore ping/pong/close
                        continue
                    message = json.loads(body)
                    seen.append(message.get("type"))
                    if message.get("type") == "response":
                        self.assertTrue(message["text"])
                        break
                self.assertIn("response", seen)
                writer.close()
            finally:
                await server.stop()

        asyncio.run(scenario())


class OrchestratorTests(GenieTestCase):
    def test_full_lifecycle(self):
        summary = self.genie.start()
        self.assertIn("graph", summary)
        self.assertIn("asr", summary)
        self.assertIn("nlp", summary)

        response = self.genie.ask("How would you design a rate limiter?")
        self.assertTrue(response.text)

        report = self.genie.stop()
        self.assertIn("metrics", report)
        self.assertIn("dialogue", report)
        self.assertIn("recovery", report)

    def test_low_confidence_transcript_asks_for_clarification(self):
        from interviewgenie.types import TranscriptChunk

        chunk = TranscriptChunk(text="something", is_final=True, confidence=0.1)
        response = self.genie.handle_transcript(chunk)
        self.assertIsNotNone(response)
        self.assertIn("repeat", response.text.lower())

    def test_interim_transcripts_are_ignored(self):
        from interviewgenie.types import TranscriptChunk

        chunk = TranscriptChunk(text="partial", is_final=False, confidence=0.9)
        self.assertIsNone(self.genie.handle_transcript(chunk))

    def test_empty_transcript_is_ignored(self):
        from interviewgenie.types import TranscriptChunk

        self.assertIsNone(self.genie.handle_transcript(
            TranscriptChunk(text="   ", is_final=True, confidence=0.9)))

    def test_audio_ingestion_through_the_mock_engine(self):
        chunks = self.genie.ingest([0.0] * 16000, 16000)
        self.assertIsInstance(chunks, list)

    def test_phase_transitions(self):
        self.genie.start()
        self.assertEqual(self.genie.phase, "listening")
        self.genie.ask("Tell me about yourself.")
        self.assertEqual(self.genie.phase, "listening")

    def test_describe_reports_the_subsystems(self):
        description = self.genie.describe()
        self.assertIn("graph", description)
        self.assertIn("asr", description)
        self.assertIn("profile", description)
        self.assertIn("llm", description)

    def test_events_are_emitted(self):
        seen = []
        self.genie.bus.on("response.generated", lambda event: seen.append(event))
        self.genie.ask("Tell me about yourself.")
        self.assertTrue(seen)
        self.assertEqual(seen[-1].type, "response.generated")

    def test_report_metrics_are_consistent(self):
        before = self.genie.report()
        for question in ("Tell me about yourself.",
                         "How would you design a cache?",
                         "What is your greatest weakness?"):
            self.genie.ask(question)
        after = self.genie.report()
        self.assertEqual(after["responses"], before["responses"] + 3)
        self.assertEqual(after["metrics"]["samples"], before["metrics"]["samples"] + 3)

    def test_error_recovery_produces_a_usable_answer(self):
        from interviewgenie.errors import KnowledgeError

        class Boom:
            def retrieve(self, *_args, **_kwargs):
                raise KnowledgeError("simulated KG failure")

        original = self.genie.retriever
        self.genie.retriever = Boom()
        try:
            response = self.genie.ask("Tell me about yourself.")
            self.assertTrue(response.text)
            self.assertTrue(response.validation.get("recovered"))
        finally:
            self.genie.retriever = original

    def test_generation_failure_is_recovered(self):
        class Boom:
            def compose(self, *_args, **_kwargs):
                raise RuntimeError("simulated composer failure")

            profile = None

        original = self.genie.composer
        self.genie.composer = Boom()
        try:
            response = self.genie.ask("Tell me about yourself.")
            self.assertTrue(response.text)
        finally:
            self.genie.composer = original


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
