"""Tests for the supporting infrastructure: config, logging, events, errors."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from unittest import mock

from tests.base import GenieTestCase


class ConfigTests(GenieTestCase):
    def test_defaults_are_loaded(self):
        from interviewgenie.config import load_config

        config = load_config()
        self.assertEqual(config.get("asr.provider"), "local")
        self.assertEqual(config.get("system.name"), "InterviewGenie")
        self.assertEqual(config.get("generation.max_words"), 90)

    def test_dotted_get_and_sections(self):
        from interviewgenie.config import load_config

        config = load_config()
        self.assertEqual(config.get("knowledge.max_hops"), 2)
        self.assertEqual(config.section("asr")["provider"], "local")
        self.assertEqual(config.get("nonexistent.key", "fallback"), "fallback")

    def test_attribute_access(self):
        from interviewgenie.config import load_config

        config = load_config()
        self.assertEqual(config.asr["provider"], "local")

    def test_file_overrides(self):
        from interviewgenie.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"asr": {"provider": "mock", "sample_rate": 8000}}, fh)
            config = load_config(path)
            self.assertEqual(config.get("asr.provider"), "mock")
            self.assertEqual(config.get("asr.sample_rate"), 8000)
            self.assertEqual(config.get("system.name"), "InterviewGenie")

    def test_keyword_overrides_win(self):
        from interviewgenie.config import load_config

        config = load_config(asr={"provider": "mock"})
        self.assertEqual(config.get("asr.provider"), "mock")

    def test_environment_overrides(self):
        from interviewgenie.config import env_overrides

        os.environ["INTERVIEWGENIE_ASR__PROVIDER"] = "mock"
        try:
            overrides = env_overrides()
            self.assertEqual(overrides["asr"]["provider"], "mock")
        finally:
            del os.environ["INTERVIEWGENIE_ASR__PROVIDER"]

    def test_validation_rejects_bad_providers(self):
        from interviewgenie.config import Config
        from interviewgenie.errors import ErrorCode, InterviewGenieError

        config = Config()
        config.data["asr"]["provider"] = "nope"
        with self.assertRaises(InterviewGenieError) as ctx:
            config.validate()
        self.assertEqual(ctx.exception.code, ErrorCode.CONFIG_INVALID)

    def test_missing_config_file_is_a_hard_error(self):
        from interviewgenie.config import Config
        from interviewgenie.errors import InterviewGenieError

        with self.assertRaises(InterviewGenieError):
            Config.load("/definitely/not/here.json")

    def test_with_overrides_returns_a_new_config(self):
        from interviewgenie.config import load_config

        base = load_config()
        derived = base.with_overrides(asr={"provider": "mock"})
        self.assertEqual(base.get("asr.provider"), "local")
        self.assertEqual(derived.get("asr.provider"), "mock")


class LoggingTests(GenieTestCase):
    def test_context_logger_accepts_context(self):
        from interviewgenie.logging import ContextLogger, get_logger

        logger = get_logger("tests.context")
        self.assertIsInstance(logger, ContextLogger)
        # must not raise
        logger.info("hello", context={"key": "value"})
        logger.debug("debug", context={"key": "value"})

    def test_level_is_respected(self):
        import logging

        from interviewgenie.logging import configure_logging, get_logger

        # Restore the suite-wide level afterwards, otherwise every module that
        # runs later inherits DEBUG and floods the output.
        root = logging.getLogger("interviewgenie")
        previous = root.level
        try:
            configure_logging(level="ERROR", json_logs=False, force=True)
            logger = get_logger("tests.level")
            self.assertFalse(logger.isEnabledFor(logging.DEBUG))
            self.assertFalse(logger.isEnabledFor(logging.INFO))
            self.assertTrue(logger.isEnabledFor(logging.ERROR))
            configure_logging(level="DEBUG", json_logs=False, force=True)
            self.assertTrue(get_logger("tests.level").isEnabledFor(logging.DEBUG))
        finally:
            configure_logging(level=logging.getLevelName(previous),
                              json_logs=False, force=True)

    def test_explicit_level_is_not_overridden_by_the_config_file(self):
        """Regression: the factory re-applied ``system.log_level`` with
        ``force=True``, so a CLI ``--log-level`` flag had no effect at all."""
        import logging

        from interviewgenie import factory
        from interviewgenie.logging import configure_logging, get_logger, level_pinned

        root = logging.getLogger("interviewgenie")
        previous = root.level
        try:
            configure_logging(level="WARNING", json_logs=False, force=True)
            self.assertTrue(level_pinned())
            # The factory must keep an explicitly pinned level untouched.
            factory.build_default_genie()
            self.assertEqual(root.level, logging.WARNING)
            self.assertFalse(get_logger("tests.pin").isEnabledFor(logging.INFO))

            # ...but with nothing pinned it must apply the config file value.
            import interviewgenie.logging as logging_mod
            logging_mod._LEVEL_PINNED = False
            factory.build_default_genie()
            configured = factory.load_config().get("system.log_level", "INFO")
            self.assertEqual(logging.getLevelName(root.level), configured.upper())
        finally:
            configure_logging(level=logging.getLevelName(previous),
                              json_logs=False, force=True)
            logging_mod._LEVEL_PINNED = bool(os.environ.get("INTERVIEWGENIE_LOG_LEVEL"))

    def test_json_formatter_emits_context(self):
        from interviewgenie.logging import JsonFormatter

        import logging

        record = logging.LogRecord("interviewgenie.test", logging.INFO, "f", 1,
                                   "hello %s", ("world",), None)
        record.context = {"a": 1}
        payload = json.loads(JsonFormatter().format(record))
        self.assertEqual(payload["msg"], "hello world")
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["ctx"], {"a": 1})

    def test_pretty_formatter_includes_context(self):
        from interviewgenie.logging import PrettyFormatter

        import logging

        record = logging.LogRecord("interviewgenie.test", logging.WARNING, "f", 1,
                                   "careful", (), None)
        record.context = {"node": "x"}
        rendered = PrettyFormatter().format(record)
        self.assertIn("careful", rendered)
        self.assertIn("node=x", rendered)

    def test_log_context_manager(self):
        from interviewgenie.logging import get_logger, log_context

        logger = get_logger("tests.context_manager")
        with log_context(logger, turn_id="t1"):
            logger.info("inside")
        logger.info("outside")


class CliParserTests(GenieTestCase):
    """Guards for CLI argument handling that used to crash at runtime."""

    def test_listen_defaults_to_no_phrases(self):
        from interviewgenie.cli import build_parser

        args = build_parser().parse_args(["listen"])
        # None means "use the template bank"; cmd_listen must not iterate it.
        self.assertIsNone(args.phrases)

    def test_listen_accepts_explicit_phrases(self):
        from interviewgenie.cli import build_parser

        args = build_parser().parse_args(["listen", "--phrases", "a", "b"])
        self.assertEqual(args.phrases, ["a", "b"])

    def test_ask_and_demo_support_no_demo(self):
        from interviewgenie.cli import build_parser

        for command in ("ask", "demo"):
            args = build_parser().parse_args([command] + (["q"] if command == "ask" else []))
            self.assertTrue(args.demo_profile, msg=command)
            args = build_parser().parse_args(
                [command, "--no-demo"] + (["q"] if command == "ask" else []))
            self.assertFalse(args.demo_profile, msg=command)

    def test_log_level_defaults_to_none_so_the_config_can_win(self):
        from interviewgenie.cli import build_parser

        self.assertIsNone(build_parser().parse_args(["ask", "q"]).log_level)
        self.assertEqual(
            build_parser().parse_args(["--log-level", "DEBUG", "ask", "q"]).log_level,
            "DEBUG")

    def test_listen_falls_back_to_the_template_bank(self):
        """Regression: ``list(None)`` raised before the ``or`` fallback ran."""
        from interviewgenie.asr.local import LocalStreamingASR
        from interviewgenie.cli import cmd_listen

        args = argparse.Namespace(phrases=None, chunk_ms=320, seed=7,
                                  no_denoise=True)
        # One phrase keeps the run quick while still exercising the None branch.
        with mock.patch.object(LocalStreamingASR, "supported_phrases",
                               property(lambda self: ["tell me about yourself"])):
            self.assertEqual(cmd_listen(args), 0)


class EventTests(GenieTestCase):
    def test_publish_and_handle(self):
        from interviewgenie.events import EventBus

        bus = EventBus()
        seen = []
        bus.on("thing.happened", lambda event: seen.append(event.payload))
        bus.publish("thing.happened", {"value": 1})
        self.assertEqual(seen, [{"value": 1}])

    def test_wildcard_subscribers_see_everything(self):
        from interviewgenie.events import EventBus

        bus = EventBus()
        seen = []
        bus.on("*", lambda event: seen.append(event.type))
        bus.publish("a.b", {})
        bus.publish("c.d", {})
        self.assertEqual(seen, ["a.b", "c.d"])

    def test_handler_failures_are_isolated(self):
        from interviewgenie.events import EventBus

        bus = EventBus()

        def boom(_event):
            raise RuntimeError("bad handler")

        seen = []
        bus.on("x", boom)
        bus.on("x", lambda event: seen.append(1))
        bus.publish("x", {})
        self.assertEqual(seen, [1])
        self.assertEqual(bus.stats()["handler_failures"], 1)

    def test_history_is_recorded(self):
        from interviewgenie.events import EventBus

        bus = EventBus(history_size=4)
        for i in range(6):
            bus.publish("tick", {"i": i})
        history = bus.history
        self.assertEqual(len(history), 4)
        self.assertEqual(history[-1].payload["i"], 5)

    def test_off_unsubscribes(self):
        from interviewgenie.events import EventBus

        bus = EventBus()
        seen = []
        handler = lambda event: seen.append(1)  # noqa: E731
        bus.on("x", handler)
        bus.off("x", handler)
        bus.publish("x", {})
        self.assertEqual(seen, [])

    def test_once_fires_once(self):
        from interviewgenie.events import EventBus

        bus = EventBus()
        seen = []
        bus.once("x", lambda event: seen.append(1))
        bus.publish("x", {})
        bus.publish("x", {})
        self.assertEqual(seen, [1])

    def test_module_level_emit(self):
        from interviewgenie.events import DEFAULT_BUS, emit

        seen = []
        DEFAULT_BUS.on("smoke.test", lambda event: seen.append(event))
        emit("smoke.test", {"ok": True})
        self.assertTrue(seen)


class ErrorTests(GenieTestCase):
    def test_error_carries_code_and_context(self):
        from interviewgenie.errors import ErrorCode, InterviewGenieError

        error = InterviewGenieError("boom", code=ErrorCode.KG_NO_EVIDENCE,
                                    context={"query": "x"}, recoverable=True)
        self.assertEqual(error.code, ErrorCode.KG_NO_EVIDENCE)
        self.assertTrue(error.recoverable)
        payload = error.to_dict()
        self.assertEqual(payload["message"], "boom")
        self.assertEqual(payload["context"]["query"], "x")

    def test_error_hierarchy(self):
        from interviewgenie.errors import (ASRError, ContradictionError,
                                           DialogueError, EntityLinkError,
                                           GenerationError, InterviewGenieError,
                                           KnowledgeError, LearningError,
                                           LowConfidenceError, NLPError,
                                           UnsupportedClaimError)

        for cls in (ASRError, NLPError, KnowledgeError, GenerationError, DialogueError,
                    LearningError):
            self.assertTrue(issubclass(cls, InterviewGenieError))
        self.assertTrue(issubclass(LowConfidenceError, ASRError))
        self.assertTrue(issubclass(EntityLinkError, KnowledgeError))
        self.assertTrue(issubclass(ContradictionError, DialogueError))
        self.assertTrue(issubclass(UnsupportedClaimError, GenerationError))

    def test_recovery_plans_are_bounded(self):
        from interviewgenie.errors import ErrorCode, InterviewGenieError, RecoveryEngine

        engine = RecoveryEngine(max_attempts=2)
        error = InterviewGenieError("low confidence", code=ErrorCode.ASR_LOW_CONFIDENCE)
        for _ in range(4):
            plan = engine.plan_for(error)
            record = engine.execute(plan)
            self.assertIn(record.outcome, {"recovered", "escalated"})
        self.assertGreaterEqual(engine.stats()["total_errors"], 4)

    def test_recovery_error_rate(self):
        from interviewgenie.errors import ErrorCode, InterviewGenieError, RecoveryEngine

        engine = RecoveryEngine(max_attempts=1)
        error = InterviewGenieError("boom", code=ErrorCode.INTERNAL)
        engine.recover(error, outcome="recovered")
        engine.recover(error, outcome="failed")
        self.assertAlmostEqual(engine.error_rate, 0.5)

    def test_guard_converts_unexpected_exceptions(self):
        from interviewgenie.errors import ErrorCode, InterviewGenieError, guard

        @guard
        def explode():
            raise ValueError("nope")

        with self.assertRaises(InterviewGenieError) as ctx:
            explode()
        self.assertEqual(ctx.exception.code, ErrorCode.INTERNAL)
        self.assertIn("nope", str(ctx.exception))

    def test_guard_passes_typed_errors_through(self):
        from interviewgenie.errors import ASRError, guard

        @guard
        def boom():
            raise ASRError("already typed")

        with self.assertRaises(ASRError):
            boom()


class TypesTests(GenieTestCase):
    def test_scorecard_weights(self):
        from interviewgenie.types import Scorecard

        perfect = Scorecard(accuracy=1, relevance=1, engagement=1, personalization=1,
                            groundedness=1, fluency=1, error_rate=0)
        self.assertAlmostEqual(perfect.overall(), 1.0)
        zero = Scorecard()
        self.assertAlmostEqual(zero.overall(), 0.0)

    def test_scorecard_error_rate_penalises(self):
        from interviewgenie.types import Scorecard

        clean = Scorecard(accuracy=1, relevance=1, engagement=1, personalization=1,
                          groundedness=1, fluency=1, error_rate=0)
        broken = Scorecard(accuracy=1, relevance=1, engagement=1, personalization=1,
                           groundedness=1, fluency=1, error_rate=1)
        self.assertLess(broken.overall(), clean.overall())

    def test_turn_and_response_serialise(self):
        from interviewgenie.types import Response, Turn

        turn = Turn(speaker="interviewer", text="hello")
        response = Response(text="hi", turn_id=turn.id)
        turn.response = response
        payload = turn.to_dict()
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["response"]["text"], "hi")
        self.assertIn("scorecard", payload["response"])

    def test_emotion_scores_normalise(self):
        from interviewgenie.types import EmotionScores

        emotions = EmotionScores(joy=0.5, trust=0.25, anger=0.25)
        normalised = emotions.normalised()
        self.assertAlmostEqual(sum(normalised.values()), 1.0)
        self.assertIn("joy", normalised)

    def test_transcript_chunk_word_count(self):
        from interviewgenie.types import TranscriptChunk

        chunk = TranscriptChunk(text="one two three", is_final=True)
        self.assertEqual(chunk.word_count, 3)

    def test_helpers(self):
        from interviewgenie.types import chunked, merge_dicts

        self.assertEqual(list(chunked([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        self.assertEqual(merge_dicts({"a": 1}, {"b": 2}), {"a": 1, "b": 2})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
