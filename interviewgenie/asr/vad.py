"""Voice activity detection and endpointing.

Three complementary cues are combined per frame:

* **short-time energy** relative to an adaptive noise floor,
* **zero crossing rate** (speech sits in a middle band; noise is either very
  low -- low frequency hum -- or very high -- fricatives),
* **spectral flatness** (tonal speech is much less flat than broadband noise).

The detector applies hysteresis (separate attack/release thresholds) and a
hangover so that short pauses inside a sentence do not split an utterance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..logging import get_logger
from .dsp import (Audio, frame_energy, frame_signal, hamming, power_spectrum,
                  spectral_centroid, spectral_flatness, zero_crossing_rate)

LOG = get_logger("asr.vad")


@dataclass
class VadConfig:
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    energy_threshold: float = 0.006       # absolute energy floor
    adaptive_factor: float = 2.6          # multiple of the tracked noise floor
    zcr_min: float = 0.02
    zcr_max: float = 0.35
    flatness_max: float = 0.55
    hangover_ms: float = 260.0
    min_speech_ms: float = 180.0
    min_silence_ms: float = 320.0
    calibration_frames: int = 12          # leading frames assumed to be noise
    attack: float = 0.55                  # downward adaptation speed
    release: float = 0.0012               # slow upward drift per frame


@dataclass
class VadSegment:
    """A contiguous run of speech."""

    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    energy: float = 0.0
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration": round(self.duration, 4),
            "energy": round(self.energy, 5),
            "confidence": round(self.confidence, 4),
        }


class VoiceActivityDetector:
    """Hysteretic, adaptive voice activity detector."""

    def __init__(self, config: Optional[VadConfig] = None) -> None:
        self.config = config or VadConfig()
        self._noise_floor = 0.0
        self._frame_index = 0
        self._initialised = False

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._noise_floor = 0.0
        self._frame_index = 0
        self._initialised = False

    def _adapt_noise_floor(self, energy: float) -> float:
        """Minimum-statistics noise floor: fast down, very slow up.

        Adapting in both directions with an asymmetric time constant is what
        keeps the tracker from climbing into the speech itself (the classic
        failure mode of symmetric smoothing).
        """
        cfg = self.config
        if not self._initialised or self._frame_index < cfg.calibration_frames:
            # leading frames are treated as noise-only for calibration
            self._noise_floor = energy if not self._initialised else (
                0.5 * self._noise_floor + 0.5 * energy
            )
            self._initialised = True
            self._frame_index += 1
            return self._noise_floor
        if energy < self._noise_floor:
            self._noise_floor = cfg.attack * energy + (1 - cfg.attack) * self._noise_floor
        else:
            self._noise_floor *= (1.0 + cfg.release)
        self._frame_index += 1
        return self._noise_floor

    def frame_scores(self, audio: Audio | Sequence[float],
                     sample_rate: Optional[int] = None) -> List[dict]:
        """Return per-frame cue values (energy, zcr, flatness, centroid)."""
        samples, rate = _resolve(audio, sample_rate, self.config.sample_rate)
        frame_len = int(rate * self.config.frame_ms / 1000.0)
        hop_len = int(rate * self.config.hop_ms / 1000.0)
        frames = frame_signal(samples, frame_len, hop_len)
        window = hamming(frame_len)
        out: List[dict] = []
        for frame in frames:
            padded = list(frame.samples) + [0.0] * (frame_len - len(frame.samples))
            power = power_spectrum([s * w for s, w in zip(padded, window)], 512)
            out.append({
                "start": frame.start,
                "end": frame.end,
                "energy": frame_energy(frame.samples),
                "zcr": zero_crossing_rate(frame.samples),
                "flatness": spectral_flatness(power),
                "centroid": spectral_centroid(power, rate),
            })
        return out

    # ------------------------------------------------------------------ #
    def detect(self, audio: Audio | Sequence[float],
               sample_rate: Optional[int] = None) -> List[VadSegment]:
        """Segment ``audio`` into speech runs."""
        self.reset()
        frames = self.frame_scores(audio, sample_rate)
        if not frames:
            return []

        speech_flags = [False] * len(frames)
        for i, frame in enumerate(frames):
            floor = self._adapt_noise_floor(frame["energy"])
            energy_ok = (
                frame["energy"] > self.config.energy_threshold
                and frame["energy"] > floor * self.config.adaptive_factor
            )
            zcr_ok = self.config.zcr_min <= frame["zcr"] <= self.config.zcr_max
            flat_ok = frame["flatness"] < self.config.flatness_max
            # speech needs an energy cue plus at least one spectral cue
            speech_flags[i] = bool(energy_ok and (zcr_ok or flat_ok))

        return self._segments_from_flags(frames, speech_flags)

    def _segments_from_flags(self, frames: Sequence[dict],
                             flags: Sequence[bool]) -> List[VadSegment]:
        cfg = self.config
        hangover = max(1, int(cfg.hangover_ms / cfg.hop_ms))
        min_speech = max(1, int(cfg.min_speech_ms / cfg.hop_ms))
        min_silence = max(1, int(cfg.min_silence_ms / cfg.hop_ms))

        segments: List[VadSegment] = []
        i = 0
        n = len(frames)
        while i < n:
            if not flags[i]:
                i += 1
                continue
            start = i
            silence_run = 0
            end = i
            while i < n:
                if flags[i]:
                    end = i
                    silence_run = 0
                else:
                    silence_run += 1
                    if silence_run >= min_silence:
                        break
                i += 1
            # keep a short hangover of context
            stop = min(n - 1, end + hangover)
            length = stop - start + 1
            if length >= min_speech:
                energies = [frames[k]["energy"] for k in range(start, stop + 1)]
                peak = max(energies) if energies else 0.0
                mean = sum(energies) / len(energies) if energies else 0.0
                segments.append(VadSegment(
                    start_sample=frames[start]["start"],
                    end_sample=frames[stop]["end"],
                    start_time=frames[start]["start"] / self.config.sample_rate,
                    end_time=frames[stop]["end"] / self.config.sample_rate,
                    energy=mean,
                    confidence=min(1.0, mean / peak) if peak > 0 else 0.0,
                ))
        return segments

    # ------------------------------------------------------------------ #
    def is_speech(self, frame: dict) -> bool:
        """Online single-frame decision (used by the streaming recogniser)."""
        floor = self._adapt_noise_floor(frame["energy"])
        energy_ok = (
            frame["energy"] > self.config.energy_threshold
            and frame["energy"] > floor * self.config.adaptive_factor
        )
        zcr_ok = self.config.zcr_min <= frame["zcr"] <= self.config.zcr_max
        flat_ok = frame["flatness"] < self.config.flatness_max
        return bool(energy_ok and (zcr_ok or flat_ok))


def _resolve(audio, sample_rate, default_rate) -> Tuple[List[float], int]:
    if isinstance(audio, Audio):
        return list(audio.samples), audio.sample_rate
    return list(audio), sample_rate or default_rate


#: module level singleton
_DEFAULT_VAD = VoiceActivityDetector()


def detect_speech(audio: Audio) -> List[VadSegment]:
    return _DEFAULT_VAD.detect(audio)
