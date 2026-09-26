"""Tests for the streaming, multi-provider LLM layer.

Each provider's server-sent-event parser is exercised against a real local
HTTP endpoint that speaks that provider's dialect, so a change to the framing
logic fails here rather than during a live interview.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from tests.base import GenieTestCase


class _SSEServer:
    """A one-shot HTTP server that streams a fixed list of SSE frames."""

    def __init__(self, handler_cls):
        self.server = HTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def _sse_handler(frames, done=b"data: [DONE]\n\n"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.body = json.loads(self.rfile.read(length)) if length else {}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in frames:
                self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.write(done)
            self.wfile.flush()

    return Handler


class OpenAIDialectStreamTests(GenieTestCase):
    def test_streaming_parses_delta_frames(self):
        frames = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "I led the "}}]},
            {"choices": [{"delta": {"content": "migration."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        server = _SSEServer(_sse_handler(frames))
        try:
            from interviewgenie.generation.llm import OpenAICompatibleClient

            client = OpenAICompatibleClient(
                api_key="k", endpoint=f"{server.url}/v1/chat/completions",
                model="test-model")
            chunks = list(client.stream("Question: tell me about a migration"))
            self.assertEqual(chunks, ["I led the ", "migration."])
            self.assertEqual(client.complete("Question: hi"),
                             "I led the migration.")
        finally:
            server.stop()

    def test_stream_requests_streaming_and_a_system_prompt(self):
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                captured["body"] = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"])))
                captured["auth"] = self.headers.get("Authorization")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: [DONE]\n\n")

        server = _SSEServer(Handler)
        try:
            from interviewgenie.generation.llm import OpenAICompatibleClient

            client = OpenAICompatibleClient(
                api_key="secret", endpoint=f"{server.url}/v1/chat/completions")
            list(client.stream("Question: hi", system="be brief", max_tokens=42))
            self.assertTrue(captured["body"]["stream"])
            self.assertEqual(captured["body"]["messages"][0]["role"], "system")
            self.assertEqual(captured["body"]["messages"][0]["content"], "be brief")
            self.assertEqual(captured["body"]["messages"][1]["content"], "Question: hi")
            self.assertEqual(captured["body"]["max_tokens"], 42)
            self.assertEqual(captured["auth"], "Bearer secret")
        finally:
            server.stop()

    def test_missing_key_raises_rather_than_returning_garbage(self):
        from interviewgenie.errors import LLMUnavailableError
        from interviewgenie.generation.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="")
        with self.assertRaises(LLMUnavailableError):
            list(client.stream("Question: hi"))

    def test_connection_failure_is_a_typed_error(self):
        from interviewgenie.errors import LLMUnavailableError
        from interviewgenie.generation.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key="k", endpoint="http://127.0.0.1:1/none", timeout=2.0)
        with self.assertRaises(LLMUnavailableError):
            list(client.stream("Question: hi"))


class AnthropicDialectStreamTests(GenieTestCase):
    def test_content_block_delta_frames(self):
        frames = [
            {"type": "message_start", "message": {}},
            {"type": "content_block_delta", "delta": {"text": "Claude "}},
            {"type": "content_block_delta", "delta": {"text": "says hi."}},
            {"type": "message_stop"},
        ]
        server = _SSEServer(_sse_handler(frames))
        try:
            from interviewgenie.generation.llm import AnthropicClient

            client = AnthropicClient(api_key="k", endpoint=f"{server.url}/v1/messages")
            self.assertEqual("".join(client.stream("Question: hi")), "Claude says hi.")
        finally:
            server.stop()

    def test_anthropic_headers_are_sent(self):
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                # urllib title-cases header names, so compare case-insensitively.
                captured["headers"] = {k.lower(): v
                                       for k, v in self.headers.items()}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: [DONE]\n\n")

        server = _SSEServer(Handler)
        try:
            from interviewgenie.generation.llm import AnthropicClient

            list(AnthropicClient(api_key="sk-ant",
                                 endpoint=f"{server.url}/v1/messages").stream("hi"))
            self.assertEqual(captured["headers"].get("x-api-key"), "sk-ant")
            self.assertIn("anthropic-version", captured["headers"])
        finally:
            server.stop()

    def test_anthropic_error_frame_raises(self):
        from interviewgenie.errors import LLMUnavailableError
        from interviewgenie.generation.llm import AnthropicClient

        frames = [{"type": "error", "error": {"message": "overloaded"}}]
        server = _SSEServer(_sse_handler(frames, done=b""))
        try:
            client = AnthropicClient(api_key="k", endpoint=f"{server.url}/v1/messages")
            with self.assertRaises(LLMUnavailableError):
                list(client.stream("Question: hi"))
        finally:
            server.stop()


class GoogleDialectStreamTests(GenieTestCase):
    def test_candidate_parts_are_concatenated(self):
        frames = [
            {"candidates": [{"content": {"parts": [{"text": "Gemini "}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "reply."}]}}]},
        ]
        server = _SSEServer(_sse_handler(frames))
        try:
            from interviewgenie.generation.llm import GoogleClient

            client = GoogleClient(api_key="k",
                                  endpoint=f"{server.url}/v1beta/models",
                                  model="gemini-test")
            self.assertEqual("".join(client.stream("Question: hi")), "Gemini reply.")
        finally:
            server.stop()

    def test_key_is_passed_as_a_query_parameter(self):
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                captured["path"] = self.path
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: [DONE]\n\n")

        server = _SSEServer(Handler)
        try:
            from interviewgenie.generation.llm import GoogleClient

            client = GoogleClient(api_key="gk",
                                  endpoint=f"{server.url}/v1beta/models",
                                  model="gemini-2.0-flash")
            list(client.stream("Question: hi"))
            self.assertIn("key=gk", captured["path"])
            self.assertIn("streamGenerateContent", captured["path"])
            self.assertIn("alt=sse", captured["path"])
        finally:
            server.stop()


class ProviderSelectionTests(GenieTestCase):
    def test_provider_dispatch(self):
        from interviewgenie.generation.llm import (AnthropicClient, GoogleClient,
                                                   OpenAICompatibleClient,
                                                   client_for)
        cases = {
            "openai": OpenAICompatibleClient, "azure": OpenAICompatibleClient,
            "groq": OpenAICompatibleClient, "ollama": OpenAICompatibleClient,
            "deepseek": OpenAICompatibleClient, "openrouter": OpenAICompatibleClient,
            "anthropic": AnthropicClient, "claude": AnthropicClient,
            "google": GoogleClient, "gemini": GoogleClient,
        }
        for provider, cls in cases.items():
            self.assertIs(type(client_for(provider, {})), cls, msg=provider)

    def test_unknown_provider_raises_a_typed_error(self):
        from interviewgenie.errors import GenerationError
        from interviewgenie.generation.llm import client_for

        with self.assertRaises(GenerationError):
            client_for("not-a-provider", {})

    def test_no_credentials_falls_back_to_the_composer(self):
        from interviewgenie.generation.llm import DeterministicLLM, llm_from_config

        for section in ({}, {"provider": "openai"}, {"provider": "bogus"}):
            self.assertIsInstance(llm_from_config(section), DeterministicLLM,
                                  msg=str(section))

    def test_any_provider_with_a_key_is_used(self):
        import os

        from interviewgenie.generation.llm import (AnthropicClient,
                                                   available_providers,
                                                   llm_from_config)
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        try:
            found = available_providers({})
            self.assertIn("anthropic", [f["provider"] for f in found])
            # configured provider has no key, but anthropic does -> use it
            client = llm_from_config({"provider": "openai"})
            self.assertIsInstance(client, AnthropicClient)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_deterministic_fallback_still_streams(self):
        from interviewgenie.generation.llm import DeterministicLLM

        chunks = list(DeterministicLLM().stream(
            "Question: tell me about yourself\nEvidence: built a queue\n"
            "Draft the answer the candidate should give."))
        self.assertTrue(chunks)
        self.assertIn("tell me about yourself", "".join(chunks))

    def test_env_var_keys_are_discovered(self):
        import os

        from interviewgenie.generation.llm import OpenAICompatibleClient

        os.environ["OPENAI_API_KEY"] = "sk-env"
        try:
            self.assertTrue(OpenAICompatibleClient().available())
            self.assertEqual(OpenAICompatibleClient().api_key, "sk-env")
        finally:
            del os.environ["OPENAI_API_KEY"]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
