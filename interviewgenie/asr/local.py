"""The offline / local streaming speech recogniser.

Pipeline per audio chunk::

    PCM -> noise suppression -> framing -> VAD -> endpointing
        -> MFCC -> DTW template decode -> SNR-aware confidence -> TranscriptChunk

Interim hypotheses come from a cheap prefix match while the candidate is still
speaking; the final hypothesis is produced when the VAD detects the end of the
utterance (or the silence timeout expires).

The acoustic model is a bundled template bank seeded by the formant synthesiser
(see :mod:`interviewgenie.asr.synth`).  It can be **re-calibrated at runtime** by
speaking each supported phrase once, which adapts the templates to the real
speaker and substantially improves accuracy -- see
:meth:`LocalStreamingASR.calibrate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import ASRError, LowConfidenceError
from ..logging import get_logger
from ..types import TranscriptChunk
from .base import ASRCapabilities, ASRProvider
from .decoder import PhraseDecoder
from .denoise import NoiseSuppressor, measure_snr
from .dsp import (Audio, MfccConfig, MfccExtractor, frame_signal, hamming,
                  power_spectrum, resample)
from .synth import FormantSynthesizer
from .templates import TemplateBank
from .vad import VadConfig, VoiceActivityDetector

LOG = get_logger("asr.local")

#: standard interview phrases the recogniser supports out of the box
DEFAULT_PHRASES: Tuple[str, ...] = (
    "tell me about yourself",
    "walk me through your resume",
    "tell me about a time you showed leadership",
    "what is your greatest weakness",
    "what is your greatest strength",
    "how would you design a url shortener",
    "what is the difference between tcp and udp",
    "why do you want to work here",
    "tell me about a difficult colleague",
    "do you have any questions for us",
    "could you repeat that please",
    "where do you see yourself in five years",
)

#: audio retained for feature extraction (seconds)
RETAIN_SECONDS = 12.0


@dataclass
class LocalStreamingASR(ASRProvider):
    """Dependency-free streaming recogniser with a template-matching decoder."""

    sample_rate: int = 16000
    min_confidence: float = 0.45
    noise_suppression: bool = True
    denoise_strength: float = 0.75
    silence_timeout_ms: float = 900.0
    partial_interval_ms: float = 240.0
    phrase_list: List[str] = field(default_factory=list)
    vad_config: Dict[str, Any] = field(default_factory=dict)
    model_path: str = ""
    name: str = "local"

    def __post_init__(self) -> None:
        self.extractor = MfccExtractor(MfccConfig(sample_rate=self.sample_rate))
        vad_kwargs = {k: v for k, v in (self.vad_config or {}).items()
                      if k in VadConfig.__dataclass_fields__}
        self.vad = VoiceActivityDetector(VadConfig(sample_rate=self.sample_rate, **vad_kwargs))
        self.suppressor = NoiseSuppressor(
            sample_rate=self.sample_rate, frame_len=512, hop_len=256,
            over_subtraction=max(0.5, 1.0 + self.denoise_strength),
            enabled=self.noise_suppression,
        )
        self.bank = TemplateBank(extractor=self.extractor)
        self.decoder = PhraseDecoder(bank=self.bank, extractor=self.extractor)
        self.synth = FormantSynthesizer(sample_rate=self.sample_rate)

        self._audio: List[float] = []            # absolute-indexed ring of samples
        self._audio_base: int = 0                # index of _audio[0] in stream time
        self._pending: List[float] = []          # samples not yet framed
        self._frame_cursor: int = 0              # next frame offset in _audio
        self._outbox: List[TranscriptChunk] = []
        self._speech_start: Optional[int] = None
        self._speech_end: Optional[int] = None
        self._frames_since_speech: int = 0
        self._last_partial: float = 0.0
        self._partial_text: str = ""
        self._stream_time: float = 0.0
        self.stats: Dict[str, Any] = {"chunks": 0, "final": 0, "partial": 0,
                                      "rejected": 0, "snr": 0.0}
        self._backend: Optional[str] = None
        self._seeded = False
        self._try_load_backend()

    # -- optional upgraded backend ---------------------------------------- #
    def _try_load_backend(self) -> None:
        """Upgrade to Vosk when a compatible model directory is configured."""
        if not self.model_path:
            return
        try:
            import vosk  # type: ignore

            self._vosk = vosk
            self._vosk_model = vosk.Model(self.model_path)
            self._vosk_recogniser = None
            self._backend = "vosk"
            LOG.info("vosk backend loaded", context={"path": self.model_path})
        except Exception as exc:  # noqa: BLE001 - optional dependency
            LOG.warning("vosk backend unavailable, using template decoder", context={"error": str(exc)})
            self._backend = None

    # -- phrase bank ------------------------------------------------------ #
    def ensure_seeded(self) -> None:
        """Seed the template bank on first use (synthesis is not free)."""
        if self._seeded:
            return
        self._seeded = True
        self.seed_default_phrases()

    def seed_default_phrases(self, phrases: Optional[Sequence[str]] = None) -> int:
        """Seed the template bank by synthesising the supported phrases."""
        self._seeded = True
        phrases = list(phrases or self.phrase_list or DEFAULT_PHRASES)
        added = 0
        for phrase in phrases:
            try:
                audio = self.synth.synthesize(phrase, self.sample_rate)
                self.bank.add(phrase, audio, source="generated", confidence=0.55)
                added += 1
            except Exception as exc:  # noqa: BLE001
                LOG.debug("could not synthesise %r: %s", phrase, exc)
        LOG.info("seeded template bank", context={"phrases": added})
        return added

    @property
    def supported_phrases(self) -> List[str]:
        self.ensure_seeded()
        return self.bank.phrases

    def calibrate(self, phrase: str, audio: Audio) -> Dict[str, Any]:
        """Re-enroll a phrase from real speech (speaker adaptation).

        The new template replaces the generated one for that phrase so the
        enrolled confidence is what the decoder actually uses.
        """
        self.ensure_seeded()
        self.bank.templates = [t for t in self.bank.templates
                               if t.phrase.lower() != phrase.lower()]
        self.bank._reindex()
        template = self.bank.add(phrase, audio, source="enrolled", confidence=0.92)
        return {"phrase": phrase, "frames": len(template.vectors), "source": template.source,
                "confidence": template.confidence}

    def calibrate_from_pcm(self, phrase: str, samples: Sequence[float]) -> Dict[str, Any]:
        return self.calibrate(phrase, Audio(list(samples), self.sample_rate, 1, phrase))

    # -- ASRProvider ------------------------------------------------------ #
    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            streaming=True, partial_results=True, diarisation=False,
            punctuation=False, word_timestamps=False, languages=("en-US",),
            noise_robust=self.noise_suppression,
        )

    def reset(self) -> None:
        self._audio = []
        self._audio_base = 0
        self._pending = []
        self._frame_cursor = 0
        self._outbox = []
        self._speech_start = None
        self._speech_end = None
        self._frames_since_speech = 0
        self._partial_text = ""
        self._stream_time = 0.0
        self.vad.reset()
        self.suppressor.reset()
        if self._backend == "vosk" and getattr(self, "_vosk_recogniser", None) is not None:
            self._vosk_recogniser = None

    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        self.ensure_seeded()
        if sample_rate != self.sample_rate:
            samples = resample(samples, sample_rate, self.sample_rate)
        block = [float(s) for s in samples]
        if not block:
            return
        self._stream_time += len(block) / self.sample_rate
        self._audio.extend(block)
        # drop audio older than the retention window
        limit = int(RETAIN_SECONDS * self.sample_rate)
        if len(self._audio) > limit:
            drop = len(self._audio) - limit
            self._audio = self._audio[drop:]
            self._audio_base += drop
        self._process()

    def flush(self) -> None:
        """Force endpointing of whatever speech is buffered."""
        if self._speech_start is not None:
            self._finalise()

    # -- frame loop ------------------------------------------------------- #
    def _process(self) -> None:
        frame_len = int(self.sample_rate * self.vad.config.frame_ms / 1000.0)
        hop_len = int(self.sample_rate * self.vad.config.hop_ms / 1000.0)
        window = hamming(frame_len)
        silence_frames = max(1, int(self.silence_timeout_ms / self.vad.config.hop_ms))
        min_speech_frames = max(1, int(self.vad.config.min_speech_ms / self.vad.config.hop_ms))

        while self._frame_cursor + frame_len <= len(self._audio):
            offset = self._frame_cursor
            frame = self._audio[offset : offset + frame_len]
            self._frame_cursor = offset + hop_len
            padded = list(frame) + [0.0] * (frame_len - len(frame))
            power = power_spectrum([s * w for s, w in zip(padded, window)], 512)
            info = {
                "start": offset,
                "end": offset + frame_len,
                "energy": sum(s * s for s in frame) / frame_len,
                "zcr": _zcr(frame),
                "flatness": _spectral_flatness(power),
            }
            if self.vad.is_speech(info):
                if self._speech_start is None:
                    self._speech_start = offset
                self._speech_end = offset + frame_len
                self._frames_since_speech = 0
            elif self._speech_start is not None:
                self._frames_since_speech += 1
                if self._frames_since_speech >= silence_frames:
                    if self._speech_end - self._speech_start >= min_speech_frames * hop_len:
                        self._finalise()
                    else:
                        self._speech_start = None
                        self._speech_end = None
                    continue
            self._maybe_partial()

    def _maybe_partial(self) -> None:
        if self._speech_start is None or self._speech_end is None:
            return
        now = self._speech_end / self.sample_rate
        if now - self._last_partial < self.partial_interval_ms / 1000.0:
            return
        self._last_partial = now
        features = self._features(self._speech_start, self._speech_end)
        if not features:
            return
        result = self.decoder.partial_decode(features)
        if result is None or not result.phrase or result.phrase == self._partial_text:
            return
        self._partial_text = result.phrase
        self.stats["partial"] += 1
        self._outbox.append(TranscriptChunk(
            text=result.phrase, is_final=False,
            start=self._speech_start / self.sample_rate,
            end=self._speech_end / self.sample_rate,
            confidence=result.confidence, provider=self.name,
        ))

    def _finalise(self) -> None:
        start = self._speech_start or 0
        end = self._speech_end or start
        self._speech_start = None
        self._speech_end = None
        self._partial_text = ""

        features = self._features(start, end)
        self.stats["final"] += 1
        if not features:
            self._outbox.append(TranscriptChunk(
                text="", is_final=True, start=start / self.sample_rate,
                end=end / self.sample_rate, confidence=0.0, provider=self.name))
            return

        segment = Audio(self._slice(start, end), self.sample_rate, 1)
        snr = measure_snr(segment)
        self.stats["snr"] = round(snr, 2)

        result = self.decoder.decode(features)
        if result is None:
            return
        confidence = min(1.0, result.confidence * _snr_factor(snr))
        chunk = TranscriptChunk(
            text=result.phrase, is_final=True,
            start=start / self.sample_rate, end=end / self.sample_rate,
            confidence=round(confidence, 4), provider=self.name,
        )
        if confidence < self.min_confidence:
            self.stats["rejected"] += 1
        self._outbox.append(chunk)

    # -- helpers ---------------------------------------------------------- #
    def _slice(self, start: int, end: int) -> List[float]:
        lo = max(0, start - self._audio_base)
        hi = max(0, end - self._audio_base)
        return self._audio[lo:hi]

    def _features(self, start: int, end: int) -> List[List[float]]:
        samples = self._slice(start, end)
        if len(samples) < int(self.sample_rate * 0.12):
            return []
        return self.extractor.extract(Audio(samples, self.sample_rate, 1))

    def poll(self) -> List[TranscriptChunk]:
        chunks, self._outbox = self._outbox, []
        self.stats["chunks"] += len(chunks)
        return chunks

    def describe(self) -> Dict[str, Any]:
        self.ensure_seeded()
        return {
            "provider": self.name,
            "backend": self._backend or "template-matching",
            "sample_rate": self.sample_rate,
            "phrases": len(self.bank.phrases),
            "noise_suppression": self.noise_suppression,
            "min_confidence": self.min_confidence,
            "stats": self.stats,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _zcr(frame: Sequence[float]) -> float:
    if len(frame) < 2:
        return 0.0
    crossings = sum(1 for i in range(1, len(frame)) if (frame[i - 1] >= 0) != (frame[i] >= 0))
    return crossings / (len(frame) - 1)


def _spectral_flatness(power: Sequence[float]) -> float:
    if not power:
        return 0.0
    eps = 1e-12
    geo = math.exp(sum(math.log(max(p, eps)) for p in power) / len(power))
    ari = sum(power) / len(power)
    return geo / ari if ari > 0 else 0.0


def _snr_factor(snr_db: float) -> float:
    """Map an estimated SNR onto a confidence multiplier."""
    if snr_db >= 25:
        return 1.0
    if snr_db <= 0:
        return 0.55
    return 0.55 + 0.45 * (snr_db / 25.0)
