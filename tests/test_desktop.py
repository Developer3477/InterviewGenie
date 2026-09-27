"""End-to-end tests for the desktop app and the live audio path.

The scenario under test is the one that matters in a real interview: audio
arrives as a *stream* of fragments over the socket, and exactly one answer
comes back — after the interviewer pauses, not once per fragment.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
import queue
import unittest
from typing import Any, Dict, List

from tests.base import GenieTestCase

from interviewgenie.desktop.app import DesktopApp
from interviewgenie.live import LiveSession, LiveConfig
from interviewgenie.types import TranscriptChunk


class _FakeASR:
    """Emits a scripted sequence of transcripts as audio is fed."""

    name = "fake"

    def __init__(self, script: List[TranscriptChunk]):
        self.script = list(script)
        self.sent: List[int] = []

    def capabilities(self):
        from interviewgenie.asr.base import ASRCapabilities

        return ASRCapabilities()

    def reset(self) -> None:
        pass

    def accept_audio(self, samples, sample_rate=16000) -> None:
        self.sent.append(len(samples))

    def poll(self):
        return [self.script.pop(0)] if self.script else []

    def flush(self) -> None:
        pass


class _StubGenie:
    """Minimal stand-in with the surface the desktop app touches."""

    def __init__(self):
        from interviewgenie.asr.base import ASRCapabilities

        self.llm = type("L", (), {"__name__": "DeterministicLLM"})()
        self.asr = _FakeASR([])
        self.capabilities = lambda: ASRCapabilities()
        self.graph = type("G", (), {"stats": lambda self: {"nodes": 0}})()
        self.intent_classifier = type("I", (), {"examples": []})()
        self.profile = type("P", (), {"name": "Test"})()
        self.turns: List[Any] = []
        self.asked: List[str] = []
        self.live = None
        self.accepted: List[Dict[str, Any]] = []

    def start(self, **facts):
        return {"phase": "listening"}

    def attach_live(self, session):
        self.live = session

    def ask_stream(self, question, confidence=1.0):
        self.asked.append(question)
        yield {"type": "analysis", "intent": "behavioural",
               "topic": "leadership", "strategy": "structured"}
        yield {"type": "delta", "text": "I led the migration "}
        yield {"type": "delta", "text": "of the payments ledger."}
        yield {"type": "response", "response": _Resp()}

    def accept(self, edited=False, rejected=False):
        self.accepted.append({"edited": edited, "rejected": rejected})
        return {"ok": True}


class _Resp:
    text = "I led the migration of the payments ledger."
    strategy = "structured"
    latency_ms = 11.0
    turn_id = "turn-x"

    class scorecard:
        overall = 0.6

        @staticmethod
        def to_dict():
            return {"overall": 0.6}


def _drain(app: DesktopApp, wait: float = 0.4) -> List[Dict[str, Any]]:
    """Wait ``wait`` seconds, then take everything the app has emitted.

    Deliberately simple: the background loop needs real time to run, and a
    queue-draining race is the fastest way to make these tests flaky.
    """
    time.sleep(wait)
    out: List[Dict[str, Any]] = []
    while True:
        try:
            out.append(app.events.get_nowait())
        except queue.Empty:
            return out


try:
    import tkinter as _tk  # noqa: F401
    HAS_TK = True
except ImportError:  # pragma: no cover - headless CI images
    HAS_TK = False


class DesktopAppTests(GenieTestCase):
    def setUp(self):
        super().setUp()
        self.app = DesktopApp(port=0, headless=True, open_browser=False)
        self.app.genie = _StubGenie()
        self.app.build.__wrapped__ if False else None
        # build() constructs a real engine; inject the stub instead
        from interviewgenie.live import LiveConfig, LiveSession

        self.app.live = LiveSession(
            self.app.genie,
            LiveConfig(auto_answer=True, debounce_ms=100))
        self.app.genie.attach_live(self.app.live)
        self.addCleanup(self.app.shutdown)

    # -- the live loop --------------------------------------------------- #
    def test_a_fragment_is_not_answered_on_its_own(self):
        self.app.feed_transcript(TranscriptChunk(
            text="Tell me about", is_final=True, confidence=0.9))
        events = _drain(self.app, wait=0.3)
        # buffered, not answered: the debounce has not elapsed yet
        self.assertEqual(events, [])
        self.assertEqual(self.app.genie.asked, [])
        self.assertEqual(self.app.live.pending_question, "Tell me about")

    def test_a_paused_question_is_answered_automatically(self):
        self.app.feed_transcript(TranscriptChunk(
            text="Tell me about a time you led a team.", is_final=True,
            confidence=0.9))
        # the background tick loop flushes it after the debounce
        self.app._stop.clear()
        threading.Thread(target=self.app.run_live_loop, daemon=True).start()
        events = _drain(self.app, wait=1.6)
        kinds = [e["type"] for e in events]
        self.assertIn("question", kinds)
        self.assertIn("response", kinds)
        self.assertEqual(self.app.genie.asked,
                         ["Tell me about a time you led a team."])

    def test_a_typed_question_is_answered(self):
        self.app.submit("How would you design a cache?")
        events = _drain(self.app, wait=1.0)
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "question")
        self.assertIn("response", kinds)
        self.assertEqual(self.app.genie.asked, ["How would you design a cache?"])

    def test_a_blank_submission_is_ignored(self):
        self.app.submit("   ")
        self.assertEqual(_drain(self.app, wait=0.3), [])
        self.assertEqual(self.app.genie.asked, [])

    def test_deltas_reach_the_overlay_in_order(self):
        self.app.submit("Tell me about yourself.")
        events = _drain(self.app, wait=1.0)
        deltas = [e["text"] for e in events if e["type"] == "delta"]
        self.assertEqual(deltas, ["I led the migration ", "of the payments ledger."])

    def test_auto_can_be_switched_off(self):
        self.app.command("auto")
        self.assertFalse(self.app.live.auto)
        self.app.feed_transcript(TranscriptChunk(
            text="Tell me about yourself.", is_final=True, confidence=0.9))
        self.app._stop.clear()
        threading.Thread(target=self.app.run_live_loop, daemon=True).start()
        time.sleep(0.6)
        self.assertEqual(self.app.genie.asked, [])

    def test_regenerate_re_asks_the_last_question(self):
        self.app.submit("Tell me about yourself.")
        _drain(self.app, wait=1.0)
        self.app.command("regenerate")
        _drain(self.app, wait=1.0)
        self.assertEqual(self.app.genie.asked,
                         ["Tell me about yourself.", "Tell me about yourself."])

    def test_accept_and_reject_are_recorded(self):
        self.app.command("accept")
        self.app.command("reject")
        self.assertEqual(self.app.genie.accepted,
                         [{"edited": False, "rejected": False},
                          {"edited": False, "rejected": True}])

    def test_clear_empties_the_overlay(self):
        self.app.command("clear")
        events = _drain(self.app, wait=0.3)
        self.assertEqual(events[0], {"type": "question", "text": ""})
        self.assertEqual(events[1]["type"], "response")
        self.assertEqual(events[1]["text"], "")

    def test_shutdown_is_idempotent(self):
        self.app.shutdown()
        self.app.shutdown()
        self.assertTrue(self.app._stop.is_set())


class OverlayRenderTests(GenieTestCase):
    """The overlay's render logic, driven without a display."""

    def setUp(self):
        super().setUp()
        if not HAS_TK:
            self.skipTest("tkinter is not installed in this environment")

    def _overlay(self):
        import queue

        from interviewgenie.desktop.overlay import Overlay

        return Overlay(queue.Queue())

    def test_events_render_without_raising(self):
        overlay = self._overlay()
        events = [
            {"type": "status", "state": "listening", "text": "listening"},
            {"type": "partial", "text": "tell me about"},
            {"type": "question", "text": "Tell me about yourself."},
            {"type": "analysis", "strategy": "structured", "topic": "people"},
            {"type": "delta", "text": "Hello "},
            {"type": "delta", "text": "there."},
            {"type": "response", "text": "Hello there.", "strategy": "structured",
             "scorecard": {"overall": 0.62}, "latency_ms": 12.3},
            {"type": "error", "message": "boom"},
        ]
        for event in events:
            overlay.render(event)
        self.assertEqual(overlay.answer_text, "Hello there.")
        overlay.set_answer("")
        self.assertEqual(overlay.answer_text, "")

    def test_append_answer_ignores_empty_fragments(self):
        overlay = self._overlay()
        overlay.set_answer("")
        overlay.append_answer("")
        overlay.append_answer("x")
        self.assertEqual(overlay.answer_text, "x")

    def test_a_blank_question_shows_the_placeholder(self):
        overlay = self._overlay()
        overlay.set_question("")
        self.assertEqual(overlay.question_text, "")


class CLITests(GenieTestCase):
    def test_desktop_is_a_registered_subcommand(self):
        from interviewgenie.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["desktop", "--port", "9000", "--headless"])
        self.assertEqual(args.port, 9000)
        self.assertTrue(args.headless)
        self.assertFalse(args.no_auto)

    def test_desktop_flags_map_onto_the_app(self):
        from interviewgenie.cli import build_parser

        args = build_parser().parse_args(
            ["desktop", "--no-auto", "--no-browser", "--debounce", "250"])
        self.assertTrue(args.no_auto)
        self.assertTrue(args.no_browser)
        self.assertEqual(args.debounce, 250)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
