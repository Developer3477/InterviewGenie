"""Noise suppression and SNR estimation.

The front-end uses a two-stage strategy that is standard in robust ASR:

1. **Noise estimation** -- a minimum-statistics tracker keeps a running lower
   envelope of the smoothed power spectrum, which converges on the noise floor
   even during continuous speech.
2. **Spectral subtraction with a spectral floor** followed by an optional
   **Wiener gain** stage, which removes the residual "musical noise" that plain
   subtraction leaves behind.

The module also provides synthetic noise generators, which the test-suite uses
to demonstrate that recognition degrades far less with suppression enabled.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..logging import get_logger
from .dsp import Audio, MfccExtractor, frame_signal, hamming, power_spectrum

LOG = get_logger("asr.denoise")

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Synthetic noise (used by tests and by the noise-robustness benchmark)
# --------------------------------------------------------------------------- #
def add_noise(audio: Audio, snr_db: float, kind: str = "white",
              seed: Optional[int] = None) -> Audio:
    """Return a copy of ``audio`` with additive noise at the requested SNR."""
    if not audio.samples:
        return Audio([], audio.sample_rate, audio.channels, audio.label)
    rng = random.Random(seed)
    signal_power = sum(s * s for s in audio.samples) / len(audio.samples)
    if signal_power <= 0:
        signal_power = EPS
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    if kind == "white":
        noise = [rng.gauss(0.0, 1.0) for _ in audio.samples]
    elif kind == "pink":
        white = [rng.gauss(0.0, 1.0) for _ in audio.samples]
        noise = []
        b0 = b1 = b2 = 0.0
        for w in white:
            b0 = 0.99765 * b0 + w * 0.0990460
            b1 = 0.96300 * b1 + w * 0.2965164
            b2 = 0.57000 * b2 + w * 1.0526913
            noise.append(b0 + b1 + b2 + w * 0.1848)
    elif kind == "babble":
        # modulated noise approximating competing speech
        noise = []
        for i, _ in enumerate(audio.samples):
            t = i / audio.sample_rate
            envelope = 0.55 + 0.45 * math.sin(2 * math.pi * 3.7 * t) * math.sin(2 * math.pi * 11.3 * t)
            noise.append(rng.gauss(0.0, 1.0) * envelope)
    elif kind == "hum":
        noise = [
            math.sin(2 * math.pi * 100 * i / audio.sample_rate)
            + 0.4 * math.sin(2 * math.pi * 60 * i / audio.sample_rate)
            for i in range(len(audio.samples))
        ]
    else:
        raise ValueError(f"unknown noise kind {kind!r}")

    scale = math.sqrt(noise_power / max(EPS, sum(n * n for n in noise) / len(noise)))
    mixed = [max(-1.0, min(1.0, s + n * scale)) for s, n in zip(audio.samples, noise)]
    return Audio(mixed, audio.sample_rate, audio.channels, f"{audio.label}+{kind}@{snr_db}dB")


def measure_snr(audio: Audio) -> float:
    """Estimate the SNR of ``audio`` from its upper/lower spectral envelope."""
    if not audio.samples:
        return 0.0
    cfg_len = 512
    hop = 256
    window = hamming(cfg_len)
    frames = frame_signal(audio.samples, cfg_len, hop)
    if not frames:
        return 0.0
    powers: List[List[float]] = []
    for frame in frames:
        padded = list(frame.samples[:cfg_len]) + [0.0] * max(0, cfg_len - len(frame.samples))
        powers.append(power_spectrum([s * w for s, w in zip(padded, window)], cfg_len))
    # noise floor = 10th percentile per bin across frames
    n_bins = len(powers[0])
    floor: List[float] = []
    for b in range(n_bins):
        column = sorted(p[b] for p in powers)
        floor.append(column[max(0, int(0.10 * len(column)))])
    signal_power = sum(sum(p) for p in powers) / len(powers)
    noise_power = sum(floor)
    if noise_power <= EPS:
        return 60.0
    return round(10.0 * math.log10(max(EPS, signal_power) / max(EPS, noise_power)), 2)


# --------------------------------------------------------------------------- #
# Noise estimation
# --------------------------------------------------------------------------- #
@dataclass
class MinimumStatisticsNoiseEstimator:
    """Minimum-statistics noise tracker over a sliding window of frames."""

    window_frames: int = 24          # ~0.6 s at 10 ms hop
    smoothing: float = 0.85
    bias: float = 1.45               # compensation for the minimum being optimistic
    _buffer: List[List[float]] = field(default_factory=list, repr=False)
    _noise: Optional[List[float]] = field(default=None, repr=False)

    def update(self, power: Sequence[float]) -> List[float]:
        """Feed one power spectrum; return the current noise estimate."""
        if self._noise is None or len(self._noise) != len(power):
            self._noise = list(power)
            self._buffer = [list(power)]
            return list(self._noise)
        self._buffer.append(list(power))
        if len(self._buffer) > self.window_frames:
            self._buffer.pop(0)
        n_bins = len(power)
        smoothed = [
            self.smoothing * self._noise[b] + (1 - self.smoothing) * power[b]
            for b in range(n_bins)
        ]
        minimum = [
            min(frame[b] for frame in self._buffer) for b in range(n_bins)
        ]
        self._noise = [min(smoothed[b], minimum[b] * self.bias) for b in range(n_bins)]
        return list(self._noise)

    def reset(self) -> None:
        self._buffer = []
        self._noise = None

    @property
    def noise_spectrum(self) -> Optional[List[float]]:
        return list(self._noise) if self._noise is not None else None


# --------------------------------------------------------------------------- #
# Suppression filters
# --------------------------------------------------------------------------- #
def spectral_subtraction(power: Sequence[float], noise: Sequence[float],
                         over_subtraction: float = 1.0, floor: float = 0.05) -> List[float]:
    """Magnitude spectral subtraction with a spectral floor."""
    out: List[float] = []
    for p, n in zip(power, noise):
        clean = p - over_subtraction * n
        out.append(max(clean, floor * p))
    return out


def wiener_gain(power: Sequence[float], noise: Sequence[float],
                alpha: float = 0.98) -> List[float]:
    """Decision-directed style Wiener gain per bin."""
    gains: List[float] = []
    for p, n in zip(power, noise):
        snr_post = max(0.0, p - n) / max(EPS, p)
        gains.append(max(floor_gain, snr_post ** alpha) if p > 0 else 0.0)
    return gains


floor_gain = 0.10


@dataclass
class NoiseSuppressor:
    """Stateful frame-by-frame noise suppressor."""

    sample_rate: int = 16000
    frame_len: int = 512
    hop_len: int = 256
    over_subtraction: float = 1.6
    use_wiener: bool = True
    enabled: bool = True
    estimator: MinimumStatisticsNoiseEstimator = field(
        default_factory=MinimumStatisticsNoiseEstimator, repr=False)

    def __post_init__(self) -> None:
        self._window = hamming(self.frame_len)

    # -- public ----------------------------------------------------------- #
    def process(self, samples: Sequence[float]) -> List[float]:
        """Return the noise-suppressed version of ``samples``."""
        if not self.enabled or not samples:
            return list(samples)
        frames = frame_signal(samples, self.frame_len, self.hop_len)
        if not frames:
            return list(samples)

        out = [0.0] * len(samples)
        norm = [0.0] * len(samples)
        from .dsp import fft, ifft

        for frame in frames:
            padded = list(frame.samples[: self.frame_len])
            padded += [0.0] * (self.frame_len - len(padded))
            windowed = [s * w for s, w in zip(padded, self._window)]
            power = power_spectrum(windowed, self.frame_len)
            noise = self.estimator.update(power)
            clean_power = spectral_subtraction(power, noise, self.over_subtraction)
            if self.use_wiener:
                gains = wiener_gain(power, noise)
                clean_power = [max(g * g * p, c) for g, p, c in zip(gains, power, clean_power)]

            # rebuild the frame with the estimated clean magnitude spectrum
            re, im = fft(windowed, None)
            half = self.frame_len // 2 + 1
            magnitudes = [math.sqrt(max(p, 0.0)) for p in power]
            phases = [math.atan2(im[i], re[i]) for i in range(half)]
            clean_mag = [math.sqrt(max(p, 0.0)) for p in clean_power]
            new_re = [0.0] * self.frame_len
            new_im = [0.0] * self.frame_len
            for i in range(half):
                new_re[i] = clean_mag[i] * math.cos(phases[i])
                new_im[i] = clean_mag[i] * math.sin(phases[i])
                if 0 < i < self.frame_len // 2:
                    new_re[self.frame_len - i] = new_re[i]
                    new_im[self.frame_len - i] = -new_im[i]
            time_re, time_im = ifft(new_re, new_im)
            for k, v in enumerate(time_re):
                idx = frame.start + k
                if idx < len(out):
                    out[idx] += v * self._window[k]
                    norm[idx] += self._window[k] ** 2
        return [out[i] / norm[i] if norm[i] > 1e-6 else samples[i] for i in range(len(out))]

    def process_audio(self, audio: Audio) -> Audio:
        return Audio(self.process(audio.samples), audio.sample_rate, audio.channels,
                     audio.label + "+denoised")

    def reset(self) -> None:
        self.estimator.reset()


# --------------------------------------------------------------------------- #
# Convenience: end-to-end feature extraction with suppression
# --------------------------------------------------------------------------- #
def robust_features(audio: Audio, extractor: Optional[MfccExtractor] = None,
                    suppressor: Optional[NoiseSuppressor] = None) -> List[List[float]]:
    """Extract MFCCs after optional noise suppression."""
    extractor = extractor or MfccExtractor()
    if suppressor is not None and suppressor.enabled:
        audio = suppressor.process_audio(audio)
    return extractor.extract(audio)
