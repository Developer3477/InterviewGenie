"""Tests for the streaming, LLM-first turn pipeline.

The orchestrator exposes one streaming implementation that the blocking
``answer()`` wraps, so these tests pin down the contract the live overlay
depends on: an ``analysis`` frame first, then answer fragments, then the
finished ``response``.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, Iterator, List

from tests.base import GenieTestCase


class _ScriptedLLM:
    """A stand-in model that emits a fixed list of fragments."""

    name = "scripted"

    def __init__(self, fragments: List[str], *, fail: bool = False):
        self.fragments = fragments
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []

    def available(self) -> bool:
        return True

    def stream(self, prompt: str, system: str = "", max_tokens: int = 512
               ) -> Iterator[str]:
        self.calls.append({"prompt": prompt, "system": system,
                           "max_tokens": max_tokens})
        if self.fail:
            from interviewgenie.errors import LLMUnavailableError

            raise LLMUnavailableError("model exploded")
        for fragment in self.fragments:
            yield fragment

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
        return "".join(self.stream(prompt, system, max_tokens))


class TurnStreamTests(GenieTestCase):
    def _genie(self):
        from interviewgenie.factory import build_demo_genie

        genie = build_demo_genie()
        genie.start()
        return genie

    def test_frame_order_is_analysis_then_deltas_then_response(self):
        genie = self._genie()
        events = list(genie.ask_stream("How would you design a rate limiter?"))
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "analysis")
        self.assertEqual(kinds[-1], "response")
        self.assertTrue(all(k == "delta" for k in kinds[1:-1]))

    def test_analysis_frame_carries_everything_the_overlay_shows(self):
        genie = self._genie()
        analysis = list(genie.ask_stream("Tell me about a time you led a team."))[0]
        for key in ("question", "intent", "topic", "tone", "entities",
                    "evidence", "strategy", "latency_ms"):
            self.assertIn(key, analysis, msg=key)
        self.assertTrue(analysis["evidence"])
        self.assertIsInstance(analysis["entities"], list)

    def test_blocking_answer_returns_the_same_response(self):
        genie = self._genie()
        question = "Why do you want to work here?"
        streamed = list(genie.ask_stream(question))[-1]["response"]
        blocked = genie.ask(question)
        self.assertEqual(blocked.text, streamed.text)
        self.assertEqual(blocked.strategy, streamed.strategy)

    def test_deltas_concatenate_to_the_final_text(self):
        genie = self._genie()
        genie.llm = _ScriptedLLM(["I started in ", "backend work ", "and moved into ",
                                  "distributed systems."])
        events = list(genie.ask_stream("Tell me about yourself."))
        joined = "".join(e["text"] for e in events if e["type"] == "delta")
        self.assertEqual(joined, "I started in backend work and moved into "
                                 "distributed systems.")
        self.assertEqual(events[-1]["response"].text, joined)

    def test_llm_text_replaces_the_composer_draft(self):
        genie = self._genie()
        composer_text = genie.ask("Tell me about yourself.").text
        replacement = "I would lead with the payments migration and what I learned."
        genie.llm = _ScriptedLLM([replacement])
        response = genie.ask("Tell me about yourself.")
        self.assertEqual(response.text, replacement)
        self.assertNotEqual(response.text, composer_text)
        self.assertEqual(response.validation.get("backend"), "llm")

    def test_a_short_llm_reply_falls_back_to_the_composer(self):
        genie = self._genie()
        genie.llm = _ScriptedLLM(["Nope."])       # fewer than 6 words
        response = genie.ask("Tell me about yourself.")
        self.assertNotEqual(response.text, "Nope.")
        self.assertGreater(len(response.text.split()), 6)

    def test_a_failing_model_falls_back_to_the_composer(self):
        genie = self._genie()
        genie.llm = _ScriptedLLM([], fail=True)
        response = genie.ask("Tell me about yourself.")
        self.assertTrue(response.text)
        self.assertGreater(len(response.text.split()), 6)
        self.assertIsNone(response.validation.get("backend"))

    def test_composer_is_used_when_no_model_is_available(self):
        genie = self._genie()
        genie.llm = _ScriptedLLM(["ignored"])
        genie.llm.available = lambda: False
        response = genie.ask("Tell me about yourself.")
        self.assertIsNone(response.validation.get("backend"))
        self.assertTrue(response.text)

    def test_backend_composer_forces_the_offline_path(self):
        genie = self._genie()
        genie.llm = _ScriptedLLM(["ignored"])
        genie.config["generation"]["backend"] = "composer"
        response = genie.ask("Tell me about yourself.")
        self.assertEqual(genie.llm.calls, [])
        self.assertIsNone(response.validation.get("backend"))

    def test_backend_auto_is_the_default(self):
        from interviewgenie.config import DEFAULT_CONFIG

        self.assertEqual(DEFAULT_CONFIG["generation"]["backend"], "auto")

    def test_streaming_records_the_turn_in_memory(self):
        genie = self._genie()
        before = len(genie.turns)
        list(genie.ask_stream("Tell me about yourself."))
        self.assertEqual(len(genie.turns), before + 1)

    def test_streaming_updates_the_scorecard_and_metrics(self):
        genie = self._genie()
        before = len(genie.metrics.history) if hasattr(genie.metrics, "history") else 0
        response = list(genie.ask_stream("Tell me about yourself."))[-1]["response"]
        self.assertIsNotNone(response.scorecard)
        self.assertGreaterEqual(len(genie.metrics.history), before)

    def test_response_frame_includes_the_scorecard(self):
        genie = self._genie()
        response = list(genie.ask_stream("Tell me about yourself."))[-1]["response"]
        self.assertIsNotNone(response.scorecard.overall)
        self.assertGreaterEqual(response.latency_ms, 0.0)


class PromptContentTests(GenieTestCase):
    def _genie(self):
        from interviewgenie.factory import build_demo_genie

        genie = build_demo_genie()
        genie.start()
        return genie

    def _prompt_for(self, genie, question):
        genie.llm = _ScriptedLLM(["An answer that is long enough to keep."])
        genie.ask(question)
        self.assertTrue(genie.llm.calls)
        return genie.llm.calls[0]

    def test_prompt_contains_the_question(self):
        genie = self._genie()
        call = self._prompt_for(genie, "How do you handle conflicting priorities?")
        self.assertIn("How do you handle conflicting priorities?", call["prompt"])

    def test_prompt_contains_the_role_and_company(self):
        genie = self._genie()
        genie.update_profile(target_role="Staff Engineer",
                             target_company="Stripe")
        call = self._prompt_for(genie, "Tell me about yourself.")
        self.assertIn("Staff Engineer", call["prompt"])
        self.assertIn("Stripe", call["prompt"])

    def test_prompt_contains_the_job_description(self):
        genie = self._genie()
        genie.update_profile(job_description="Own payments reliability at scale.")
        call = self._prompt_for(genie, "Tell me about yourself.")
        self.assertIn("payments reliability", call["prompt"])

    def test_prompt_contains_retrieved_evidence(self):
        genie = self._genie()
        call = self._prompt_for(genie, "How would you design a cache?")
        self.assertIn("Evidence:", call["prompt"])

    def test_prompt_contains_the_candidate_skills(self):
        genie = self._genie()
        genie.update_profile(skills=["Rust", "Kafka"])
        call = self._prompt_for(genie, "Tell me about yourself.")
        self.assertIn("Rust", call["prompt"])

    def test_system_prompt_carries_register_and_tone(self):
        genie = self._genie()
        call = self._prompt_for(genie, "Tell me about yourself.")
        self.assertTrue(call["system"])
        self.assertIn("Response strategy", call["system"])

    def test_max_tokens_scale_with_the_word_budget(self):
        genie = self._genie()
        genie.config["generation"]["max_words"] = 50
        call = self._prompt_for(genie, "Tell me about yourself.")
        self.assertEqual(call["max_tokens"], 100)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
