"""Tests for the live loop: speech fragments in, answered questions out.

The whole point of ``LiveSession`` is that a recogniser emits a *stream* while
the orchestrator wants a *complete question*. These tests pin down the
boundary, using injected timestamps so nothing has to sleep.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, Iterator, List

from tests.base import GenieTestCase

from interviewgenie.live import (LiveConfig, LiveSession, looks_like_question,
                                 meaningful_word_count)
from interviewgenie.types import TranscriptChunk


def chunk(text: str, *, final: bool = True, confidence: float = 0.9) -> TranscriptChunk:
    return TranscriptChunk(text=text, is_final=final, confidence=confidence)


class _RecordingGenie:
    """Stands in for the orchestrator and records what it was asked."""

    def __init__(self, answer: str = "Here is a grounded answer."):
        self.answer = answer
        self.calls: List[Dict[str, Any]] = []

    def ask_stream(self, question: str, confidence: float = 1.0) -> Iterator[Dict[str, Any]]:
        self.calls.append({"question": question, "confidence": confidence})
        yield {"type": "analysis", "intent": "behavioural",
               "topic": "leadership", "strategy": "structured",
               "tone": "warm", "entities": ["Mentoring"], "evidence": []}
        yield {"type": "delta", "text": "Here is "}
        yield {"type": "delta", "text": "a grounded answer."}
        yield {"type": "response", "response": _Response(self.answer)}


class _Response:
    def __init__(self, text: str):
        self.text = text
        self.strategy = "structured"
        self.latency_ms = 12.5
        self.turn_id = "turn-1"

        class _Card:
            overall = 0.55

            def to_dict(self):
                return {"overall": 0.55}

        self.scorecard = _Card()


def session(**kwargs) -> LiveSession:
    genie = _RecordingGenie()
    live = LiveSession(genie, LiveConfig(**kwargs))
    return live, genie


class QuestionDetectionTests(GenieTestCase):
    def test_question_mark_is_decisive(self):
        self.assertTrue(looks_like_question("How would you debug Kafka?"))

    def test_question_word_without_a_mark(self):
        self.assertTrue(looks_like_question("tell me about a time you led a team"))

    def test_a_statement_is_not_a_question(self):
        self.assertFalse(looks_like_question("we use kafka at scale"))

    def test_a_short_opener_needs_a_following_word(self):
        # "however" must not match the "how" opener
        self.assertFalse(looks_like_question("however we proceeded"))

    def test_filler_does_not_count_towards_length(self):
        self.assertEqual(meaningful_word_count("um tell me uh about you know"), 3)
        self.assertEqual(meaningful_word_count("sort of kind of kafka"), 1)

    def test_empty_text_is_not_a_question(self):
        self.assertFalse(looks_like_question("   "))


class BufferingTests(GenieTestCase):
    def test_interim_results_are_shown_but_never_answered(self):
        live, genie = session()
        events = live.feed_transcript(chunk("tell me about", final=False), now=0.0)
        self.assertEqual([e["type"] for e in events], ["partial"])
        self.assertEqual(genie.calls, [])

    def test_a_question_is_answered_once_the_speaker_pauses(self):
        live, genie = session()
        live.feed_transcript(chunk("Tell me about a time you led a team."),
                             now=0.0)
        self.assertEqual(genie.calls, [], "not yet — no pause")
        events = live.tick(now=1.0)          # 1s later: past the 700ms debounce
        self.assertEqual(len(genie.calls), 1)
        self.assertEqual(genie.calls[0]["question"],
                         "Tell me about a time you led a team.")

    def test_fragments_are_joined_into_one_question(self):
        live, genie = session()
        live.feed_transcript(chunk("Tell me about"), now=0.0)
        live.feed_transcript(chunk("a time you showed leadership."), now=0.1)
        events = live.tick(now=1.0)
        self.assertEqual(len(genie.calls), 1)
        self.assertEqual(genie.calls[0]["question"],
                         "Tell me about a time you showed leadership.")
        self.assertIn("question", [e["type"] for e in events])

    def test_an_unfinished_question_keeps_accumulating(self):
        live, genie = session()
        live.feed_transcript(chunk("we use kafka at scale"), now=0.0)
        live.tick(now=1.0)
        live.tick(now=2.0)
        self.assertEqual(genie.calls, [], "a statement is not a question")
        self.assertEqual(live.pending_question, "we use kafka at scale")

    def test_a_statement_then_a_question_answers_only_the_question(self):
        live, genie = session()
        live.feed_transcript(chunk("we use kafka at scale"), now=0.0)
        live.tick(now=1.0)
        live.feed_transcript(chunk("how would you debug it?"), now=1.1)
        events = live.tick(now=2.0)
        self.assertEqual(len(genie.calls), 1)
        self.assertIn("how would you debug it?", genie.calls[0]["question"])

    def test_too_short_a_fragment_is_ignored(self):
        live, genie = session(min_question_words=4)
        live.feed_transcript(chunk("why?"), now=0.0)
        live.tick(now=5.0)
        self.assertEqual(genie.calls, [])

    def test_low_confidence_transcripts_are_dropped(self):
        live, genie = session(min_confidence=0.5)
        live.feed_transcript(chunk("tell me about yourself", confidence=0.2),
                             now=0.0)
        live.tick(now=5.0)
        self.assertEqual(genie.calls, [])

    def test_the_buffer_is_flushed_when_it_grows_unbounded(self):
        live, genie = session(max_buffer_words=8)
        live.feed_transcript(chunk("tell me about a very long story that "
                                   "never seems to end and keeps going"),
                             now=0.0)
        events = live.feed_transcript(chunk("and going"), now=0.1)
        self.assertTrue(genie.calls, "an unbounded buffer must not swallow speech")

    def test_reset_clears_the_buffer(self):
        live, genie = session()
        live.feed_transcript(chunk("we use kafka"), now=0.0)
        live.reset()
        live.tick(now=5.0)
        self.assertEqual(genie.calls, [])
        self.assertEqual(live.pending_question, "")


class AutoAnswerTests(GenieTestCase):
    def test_auto_answers_without_a_tick_when_the_question_is_complete(self):
        live, genie = session(debounce_ms=0)
        events = live.feed_transcript(chunk("Tell me about yourself."), now=0.0)
        self.assertEqual(len(genie.calls), 1)
        self.assertIn("question", [e["type"] for e in events])

    def test_auto_off_buffers_and_never_answers(self):
        live, genie = session()
        live.set_auto(False)
        live.feed_transcript(chunk("Tell me about yourself."), now=0.0)
        live.tick(now=99.0)
        self.assertEqual(genie.calls, [])
        self.assertEqual(live.pending_question, "Tell me about yourself.")

    def test_a_typed_question_is_answered_immediately(self):
        live, genie = session()
        events = list(live.submit("How would you design a cache?"))
        self.assertEqual(len(genie.calls), 1)
        self.assertEqual(events[0]["type"], "question")

    def test_a_blank_submission_does_nothing(self):
        live, genie = session()
        self.assertEqual(list(live.submit("   ")), [])
        self.assertEqual(genie.calls, [])

    def test_require_question_mark_rejects_statements(self):
        live, genie = session(require_question_mark=True, debounce_ms=0)
        live.feed_transcript(chunk("tell me about a time you led a team"),
                             now=0.0)
        self.assertEqual(genie.calls, [])

    def test_stats_reports_state(self):
        live, genie = session()
        list(live.submit("Tell me about yourself."))   # generators are lazy
        stats = live.stats
        self.assertEqual(stats["answered"], 1)
        self.assertTrue(stats["auto"])
        self.assertEqual(stats["recent"], ["Tell me about yourself."])


class EventShapeTests(GenieTestCase):
    def test_a_full_turn_yields_question_analysis_deltas_response(self):
        live, _ = session(debounce_ms=0)
        events = live.feed_transcript(chunk("Tell me about yourself."), now=0.0)
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "question")
        self.assertEqual(kinds[-1], "response")
        self.assertIn("analysis", kinds)
        self.assertIn("delta", kinds)

    def test_deltas_concatenate_to_the_response_text(self):
        live, _ = session(debounce_ms=0)
        events = live.feed_transcript(chunk("Tell me about yourself."), now=0.0)
        joined = "".join(e["text"] for e in events if e["type"] == "delta")
        final = events[-1]
        self.assertEqual(joined, "Here is a grounded answer.")
        self.assertEqual(final["text"], "Here is a grounded answer.")
        self.assertEqual(final["scorecard"]["overall"], 0.55)
        self.assertEqual(final["strategy"], "structured")

    def test_a_failing_turn_yields_an_error_not_an_exception(self):
        class _Boom:
            def ask_stream(self, question, confidence=1.0):
                raise RuntimeError("model exploded")
                yield  # pragma: no cover

        live = LiveSession(_Boom(), LiveConfig(debounce_ms=0))
        events = live.feed_transcript(chunk("Tell me about yourself."), now=0.0)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("exploded", events[-1]["message"])

    def test_listening_off_clears_the_buffer(self):
        live, genie = session()
        live.feed_transcript(chunk("we use kafka"), now=0.0)
        live.set_listening(False)
        self.assertEqual(live.pending_question, "")
        self.assertFalse(live.listening)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
