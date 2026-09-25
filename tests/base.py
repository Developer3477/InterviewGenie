"""Shared test helpers.

Every test module derives from :class:`GenieTestCase`, which builds a single
:class:`~interviewgenie.factory.build_demo_genie` per class (not per test) so the
suite stays fast -- the knowledge-graph and index build dominate start-up cost.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewgenie.logging import configure_logging  # noqa: E402

configure_logging(level="CRITICAL", json_logs=False, force=True)

from interviewgenie.factory import build_demo_genie  # noqa: E402


class GenieTestCase(unittest.TestCase):
    """Base class exposing a shared, ready-to-run InterviewGenie."""

    @classmethod
    def setUpClass(cls) -> None:
        random.seed(20260925)
        cls.genie = shared_genie()
        cls.genie.start()
        cls.nlp = cls.genie.nlp
        cls.graph = cls.genie.graph
        cls.retriever = cls.genie.retriever
        cls.composer = cls.genie.composer
        cls.profile = cls.genie.profile

    @classmethod
    def tearDownClass(cls) -> None:
        cls.genie.stop()

    def ask(self, question: str):
        return self.genie.ask(question)


_SHARED = None


def shared_genie():
    """Build one InterviewGenie and reuse it across every test class."""
    global _SHARED
    if _SHARED is None:
        _SHARED = build_demo_genie()
        _SHARED.start()
    return _SHARED


def synth_audio(text: str, sample_rate: int = 16000):
    """Synthesise a phrase with the bundled formant synthesiser."""
    from interviewgenie.asr.synth import FormantSynthesizer

    return FormantSynthesizer(sample_rate=sample_rate).synthesize(text, sample_rate)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
