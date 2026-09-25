"""Deterministic speech recogniser used by tests, demos and benchmarks.

``MockASRProvider`` replays a scripted list of utterances whenever enough audio
has been fed, which makes the whole pipeline reproducible without a microphone.
It deliberately models real-world imperfections: an optional per-utterance
confidence penalty and a word-error rate, so downstream confidence gating and
error recovery can be exercised in tests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..logging import get_logger
from ..types import TranscriptChunk
from .base import ASRCapabilities, ASRProvider

LOG = get_logger("asr.mock")


@dataclass
class MockASRProvider(ASRProvider):
    """Replay a scripted transcript stream."""

    sample_rate: int = 16000
    script: List[str] = field(default_factory=list)
    word_error_rate: float = 0.0
    confidence: float = 0.92
    emit_partials: bool = True
    seed: int = 20260925
    name: str = "mock"

    def __post_init__(self) -> None:
        self._cursor = 0
        self._outbox: List[TranscriptChunk] = []
        self._samples_seen = 0
        self._emitted_partial = False
        self._rng = random.Random(self.seed)

    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            streaming=True, partial_results=self.emit_partials, diarisation=False,
            punctuation=False, word_timestamps=False, languages=("en-US",),
            noise_robust=True,
        )

    def reset(self) -> None:
        self._cursor = 0
        self._outbox = []
        self._samples_seen = 0
        self._emitted_partial = False

    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        self._samples_seen += len(samples)
        if self._cursor >= len(self.script):
            return
        utterance = self.script[self._cursor]
        needed = int(0.6 * self.sample_rate)          # 0.6 s of audio per utterance
        if self.emit_partials and not self._emitted_partial and self._samples_seen >= needed // 2:
            self._emitted_partial = True
            self._outbox.append(TranscriptChunk(
                text=utterance, is_final=False, confidence=self.confidence * 0.7,
                start=0.0, end=self._samples_seen / self.sample_rate,
                provider=self.name,
            ))
        if self._samples_seen >= needed:
            self._samples_seen = 0
            self._emitted_partial = False
            text = _corrupt(utterance, self.word_error_rate, self._rng)
            self._outbox.append(TranscriptChunk(
                text=text, is_final=True,
                confidence=round(self.confidence * (1.0 - self.word_error_rate), 4),
                end=needed / self.sample_rate, provider=self.name,
            ))
            self._cursor += 1

    def flush(self) -> None:
        if self._cursor < len(self.script):
            utterance = self.script[self._cursor]
            self._outbox.append(TranscriptChunk(
                text=_corrupt(utterance, self.word_error_rate, self._rng),
                is_final=True, confidence=self.confidence, provider=self.name,
            ))
            self._cursor += 1

    def poll(self) -> List[TranscriptChunk]:
        chunks, self._outbox = self._outbox, []
        return chunks


def _corrupt(text: str, wer: float, rng: random.Random) -> str:
    """Apply a simple word-level error model (deletion / substitution)."""
    if wer <= 0:
        return text
    words = text.split()
    out: List[str] = []
    substitutions = {"the": "a", "your": "you", "and": "an", "a": "the", "is": "are"}
    for word in words:
        if rng.random() < wer:
            if rng.random() < 0.5:
                continue                      # deletion
            word = substitutions.get(word.lower(), word + "uh")
        out.append(word)
    return " ".join(out)
