"""The speech-recognition provider contract.

Every recogniser -- local template matcher, Google / Azure / IBM cloud API,
Vosk, or the deterministic mock used in tests -- implements
:class:`ASRProvider`.  The orchestrator only ever talks to that interface, so
swapping the engine is a configuration change.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..errors import ASRError, LowConfidenceError, SilenceDetectedError
from ..logging import get_logger
from ..types import TranscriptChunk

LOG = get_logger("asr.base")


@dataclass
class ASRCapabilities:
    streaming: bool = True
    partial_results: bool = True
    diarisation: bool = False
    punctuation: bool = False
    word_timestamps: bool = True
    languages: Sequence[str] = ("en-US",)
    noise_robust: bool = False


class ASRProvider(ABC):
    """Interface every speech engine implements."""

    name: str = "abstract"

    @abstractmethod
    def capabilities(self) -> ASRCapabilities:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Drop all buffered state (called between interview turns)."""

    @abstractmethod
    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        """Feed a chunk of mono PCM."""

    @abstractmethod
    def poll(self) -> List[TranscriptChunk]:
        """Return any transcripts produced since the previous call."""

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> TranscriptChunk:
        """One-shot transcription of a complete utterance."""
        self.reset()
        self.accept_audio(list(audio), sample_rate)
        self.flush()
        chunks = self.poll()
        return chunks[-1] if chunks else TranscriptChunk(text="", is_final=True, provider=self.name)

    def flush(self) -> None:
        """Force endpointing of whatever is buffered."""

    def close(self) -> None:  # pragma: no cover - default no-op
        self.reset()

    # -- helpers ---------------------------------------------------------- #
    def _chunk(self, text: str, *, is_final: bool, confidence: float = 0.0,
               start: float = 0.0, end: float = 0.0, **extra: Any) -> TranscriptChunk:
        return TranscriptChunk(
            text=text, is_final=is_final, start=start, end=end,
            confidence=confidence, provider=self.name, **extra,
        )


class BufferingProvider(ASRProvider, ABC):
    """Base class for providers that accumulate PCM until an endpoint."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._buffer: List[float] = []
        self._outbox: List[TranscriptChunk] = []
        self._stream_time: float = 0.0

    def reset(self) -> None:
        self._buffer = []
        self._outbox = []

    def poll(self) -> List[TranscriptChunk]:
        chunks, self._outbox = self._outbox, []
        return chunks

    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        if sample_rate != self.sample_rate:
            from .dsp import resample

            samples = resample(samples, sample_rate, self.sample_rate)
        self._buffer.extend(float(s) for s in samples)
        self._stream_time += len(samples) / self.sample_rate
        self._process()

    @abstractmethod
    def _process(self) -> None:
        """Consume newly buffered audio (implementation specific)."""


def provider_from_config(config: Any) -> ASRProvider:
    """Instantiate the ASR provider described by a config section.

    ``config`` may be a :class:`~interviewgenie.config.Config`, a dict with an
    ``"asr"`` key, or the ``asr`` section itself.
    """
    if hasattr(config, "section"):
        section = config.section("asr")
    elif isinstance(config, dict) and isinstance(config.get("asr"), dict):
        section = config["asr"]
    else:
        section = config
    name = str(section.get("provider", "local")).lower()
    sample_rate = int(section.get("sample_rate", 16000))
    credentials = section.get("credentials", {}) or {}

    if name == "local":
        from .local import LocalStreamingASR

        return LocalStreamingASR(
            sample_rate=sample_rate,
            min_confidence=float(section.get("min_confidence", 0.45)),
            noise_suppression=bool(section.get("noise_suppression", True)),
            denoise_strength=float(section.get("denoise_strength", 0.75)),
            silence_timeout_ms=float(section.get("silence_timeout_ms", 900)),
            partial_interval_ms=float(section.get("partial_interval_ms", 240)),
            phrase_list=list(section.get("phrase_list", []) or []),
            vad_config=section.get("vad", {}) or {},
            model_path=str(section.get("model_path", "") or ""),
        )
    if name in {"google", "azure", "ibm"}:
        from .cloud import CloudASRProvider

        return CloudASRProvider(
            vendor=name, sample_rate=sample_rate,
            credentials=credentials,
            min_confidence=float(section.get("min_confidence", 0.45)),
        )
    if name == "mock":
        from .mock import MockASRProvider

        return MockASRProvider(sample_rate=sample_rate,
                               script=list(section.get("script", []) or []))
    raise ASRError(f"unknown ASR provider {name!r}")


__all__ = [
    "ASRProvider", "ASRCapabilities", "BufferingProvider", "provider_from_config",
    "TranscriptChunk", "ASRError", "LowConfidenceError", "SilenceDetectedError",
]
