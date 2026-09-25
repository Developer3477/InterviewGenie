"""Digital signal processing primitives for the speech front-end.

Everything here is pure Python (``math`` + ``array``) so the recogniser runs on
a bare interpreter.  The pipeline follows the classic MFCC recipe:

    PCM -> pre-emphasis -> framing -> Hamming window -> FFT
        -> power spectrum -> mel filterbank -> log -> DCT -> MFCC
        -> delta + delta-delta -> cepstral mean/variance normalisation
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from ..logging import get_logger

LOG = get_logger("asr.dsp")


# --------------------------------------------------------------------------- #
# FFT
# --------------------------------------------------------------------------- #
def fft(real: Sequence[float], imag: Optional[Sequence[float]] = None) -> Tuple[List[float], List[float]]:
    """In-place iterative radix-2 Cooley-Tukey FFT (length must be a power of 2)."""
    n = len(real)
    if n == 0:
        return [], []
    if n & (n - 1):
        raise ValueError(f"FFT length must be a power of two, got {n}")
    re = [float(v) for v in real]
    im = [float(v) for v in (imag if imag is not None else [0.0] * n)]

    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]

    length = 2
    while length <= n:
        ang = -2.0 * math.pi / length
        wr = math.cos(ang)
        wi = math.sin(ang)
        half = length // 2
        for i in range(0, n, length):
            cr, ci = 1.0, 0.0
            for k in range(half):
                a = i + k
                b = a + half
                tr = cr * re[b] - ci * im[b]
                ti = cr * im[b] + ci * re[b]
                re[b] = re[a] - tr
                im[b] = im[a] - ti
                re[a] += tr
                im[a] += ti
                cr, ci = cr * wr - ci * wi, cr * wi + ci * wr
        length <<= 1
    return re, im


def ifft(real: Sequence[float], imag: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Inverse FFT (used by the noise-suppression filters)."""
    n = len(real)
    if n == 0:
        return [], []
    re = [v / n for v in real]
    im = [-v / n for v in imag]
    out_re, out_im = fft(re, im)
    return out_re, [-v for v in out_im]


def power_spectrum(frame: Sequence[float], n_fft: int = 512) -> List[float]:
    """One-sided power spectrum of ``frame`` (zero padded to ``n_fft``)."""
    buf = [0.0] * n_fft
    for i, v in enumerate(frame[:n_fft]):
        buf[i] = v
    re, im = fft(buf)
    half = n_fft // 2 + 1
    return [(re[i] * re[i] + im[i] * im[i]) / n_fft for i in range(half)]


def magnitude_spectrum(frame: Sequence[float], n_fft: int = 512) -> List[float]:
    buf = [0.0] * n_fft
    for i, v in enumerate(frame[:n_fft]):
        buf[i] = v
    re, im = fft(buf)
    half = n_fft // 2 + 1
    return [math.sqrt(re[i] * re[i] + im[i] * im[i]) for i in range(half)]


# --------------------------------------------------------------------------- #
# Filters and windows
# --------------------------------------------------------------------------- #
def hamming(length: int) -> List[float]:
    """Symmetric Hamming window."""
    if length <= 1:
        return [1.0] * max(0, length)
    return [0.54 - 0.46 * math.cos(2.0 * math.pi * i / (length - 1)) for i in range(length)]


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_filters: int = 26, n_fft: int = 512, sample_rate: int = 16000,
                   low_freq: float = 20.0, high_freq: Optional[float] = None) -> List[List[float]]:
    """Triangular mel-spaced filterbank over the one-sided spectrum."""
    if high_freq is None:
        high_freq = sample_rate / 2.0
    n_bins = n_fft // 2 + 1
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    points = [mel_to_hz(low_mel + (high_mel - low_mel) * i / (n_filters + 1)) for i in range(n_filters + 2)]
    bin_idx = [int(math.floor(p / (sample_rate / 2.0) * (n_bins - 1))) for p in points]
    bin_idx = [max(0, min(n_bins - 1, b)) for b in bin_idx]

    filters: List[List[float]] = []
    for m in range(1, n_filters + 1):
        left, centre, right = bin_idx[m - 1], bin_idx[m], bin_idx[m + 1]
        weights = [0.0] * n_bins
        if centre == left:
            centre = min(left + 1, n_bins - 1)
        if right == centre:
            right = min(centre + 1, n_bins - 1)
        for k in range(left, centre):
            if centre > left:
                weights[k] = (k - left) / (centre - left)
        for k in range(centre, right):
            if right > centre:
                weights[k] = (right - k) / (right - centre)
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        filters.append(weights)
    return filters


def dct_matrix(n_out: int, n_in: int) -> List[List[float]]:
    """Orthonormal DCT-II basis of shape ``(n_out, n_in)``."""
    out: List[List[float]] = []
    scale = math.sqrt(2.0 / n_in)
    for k in range(n_out):
        row = [scale * math.cos(math.pi * k * (2 * i + 1) / (2 * n_in)) for i in range(n_in)]
        if k == 0:
            row = [v * math.sqrt(0.5) for v in row]
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Audio containers
# --------------------------------------------------------------------------- #
@dataclass
class Audio:
    """A block of mono floating point PCM in ``[-1, 1]``."""

    samples: List[float]
    sample_rate: int = 16000
    channels: int = 1
    label: str = ""

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate) if self.sample_rate else 0.0

    def rms(self) -> float:
        if not self.samples:
            return 0.0
        return math.sqrt(sum(s * s for s in self.samples) / len(self.samples))

    def peak(self) -> float:
        return max((abs(s) for s in self.samples), default=0.0)

    def slice(self, start: int, end: int) -> "Audio":
        return Audio(self.samples[start:end], self.sample_rate, self.channels, self.label)

    def to_dict(self) -> dict:
        return {"samples": len(self.samples), "sample_rate": self.sample_rate,
                "duration": round(self.duration, 4), "rms": round(self.rms(), 5),
                "label": self.label}


def read_wav(path: str) -> Audio:
    """Read a 16-bit PCM WAV file into an :class:`Audio` block."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    if width == 2:
        ints = struct.unpack(f"<{len(raw) // 2}h", raw)
        samples = [v / 32768.0 for v in ints]
    elif width == 1:
        samples = [(v - 128) / 128.0 for v in raw]
    elif width == 4:
        ints = struct.unpack(f"<{len(raw) // 4}i", raw)
        samples = [v / 2147483648.0 for v in ints]
    else:  # pragma: no cover - unusual formats
        raise ValueError(f"unsupported sample width {width}")
    if channels > 1:
        samples = [
            sum(samples[i : i + channels]) / channels
            for i in range(0, len(samples) - channels + 1, channels)
        ]
    return Audio([max(-1.0, min(1.0, s)) for s in samples], sample_rate, channels, path)


def resample(samples: Sequence[float], src_rate: int, dst_rate: int) -> List[float]:
    """Linear interpolation resampler (adequate for 8k/16k/44.1k sources)."""
    if src_rate == dst_rate or not samples:
        return list(samples)
    ratio = dst_rate / src_rate
    n_out = int(len(samples) * ratio)
    out: List[float] = []
    for i in range(n_out):
        pos = i / ratio
        left = int(math.floor(pos))
        right = min(left + 1, len(samples) - 1)
        frac = pos - left
        out.append(samples[left] * (1 - frac) + samples[right] * frac)
    return out


def pcm16_to_floats(data: bytes) -> List[float]:
    """Decode little-endian signed 16-bit PCM bytes."""
    n = len(data) // 2
    return list(struct.unpack(f"<{n}h", data[: n * 2])) if n else []


def int16_from_floats(samples: Iterable[float]) -> bytes:
    """Encode floats as little-endian signed 16-bit PCM (with clipping)."""
    out = bytearray()
    for s in samples:
        v = int(max(-32768, min(32767, round(s * 32767.0))))
        out += struct.pack("<h", v)
    return bytes(out)


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """One analysis frame with its time bounds."""

    samples: List[float]
    start: int
    end: int
    index: int

    @property
    def time(self) -> float:
        return self.start


def frame_signal(samples: Sequence[float], frame_len: int, hop_len: int) -> List[Frame]:
    """Split ``samples`` into overlapping frames."""
    frames: List[Frame] = []
    if len(samples) < frame_len:
        if samples:
            padded = list(samples) + [0.0] * (frame_len - len(samples))
            frames.append(Frame(padded, 0, len(samples), 0))
        return frames
    index = 0
    start = 0
    while start + frame_len <= len(samples):
        frames.append(Frame(list(samples[start : start + frame_len]), start, start + frame_len, index))
        start += hop_len
        index += 1
    # tail frame
    if start < len(samples) and frames:
        tail = list(samples[start:]) + [0.0] * (start + frame_len - len(samples))
        frames.append(Frame(tail, start, len(samples), index))
    return frames


# --------------------------------------------------------------------------- #
# Frame level descriptors
# --------------------------------------------------------------------------- #
def frame_energy(frame: Sequence[float]) -> float:
    return sum(v * v for v in frame) / max(1, len(frame))


def zero_crossing_rate(frame: Sequence[float]) -> float:
    if len(frame) < 2:
        return 0.0
    crossings = sum(1 for i in range(1, len(frame)) if (frame[i - 1] >= 0) != (frame[i] >= 0))
    return crossings / (len(frame) - 1)


def spectral_flatness(power: Sequence[float]) -> float:
    """Geometric / arithmetic mean ratio; ~0 for tonal, ~1 for noise-like."""
    if not power:
        return 0.0
    eps = 1e-12
    log_sum = sum(math.log(max(p, eps)) for p in power)
    geo = math.exp(log_sum / len(power))
    ari = sum(power) / len(power)
    if ari <= 0:
        return 0.0
    return geo / ari


def spectral_centroid(power: Sequence[float], sample_rate: int = 16000) -> float:
    total = sum(power)
    if total <= 0:
        return 0.0
    n_bins = len(power)
    weighted = sum(i * p for i, p in enumerate(power))
    return (weighted / total) * (sample_rate / 2.0) / max(1, n_bins - 1)


# --------------------------------------------------------------------------- #
# MFCC extractor
# --------------------------------------------------------------------------- #
@dataclass
class MfccConfig:
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_mfcc: int = 13
    n_filters: int = 26
    n_fft: int = 512
    preemphasis: float = 0.97
    low_freq: float = 20.0
    high_freq: Optional[float] = None
    use_deltas: bool = True
    cmvn: bool = True
    lifter: int = 22

    @property
    def frame_len(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))

    @property
    def hop_len(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))


class MfccExtractor:
    """Extract (delta-augmented, normalised) MFCC features from audio."""

    def __init__(self, config: Optional[MfccConfig] = None) -> None:
        self.config = config or MfccConfig()
        self._window = hamming(self.config.frame_len)
        self._filters = mel_filterbank(
            self.config.n_filters, self.config.n_fft,
            self.config.sample_rate, self.config.low_freq, self.config.high_freq,
        )
        self._dct = dct_matrix(self.config.n_mfcc, self.config.n_filters)

    # -- core ------------------------------------------------------------- #
    def extract(self, audio: Audio | Sequence[float], sample_rate: Optional[int] = None) -> List[List[float]]:
        """Return a list of feature vectors, one per frame."""
        samples, rate = self._resolve(audio, sample_rate)
        cfg = self.config
        if rate != cfg.sample_rate:
            samples = resample(samples, rate, cfg.sample_rate)
        if not samples:
            return []

        frames = frame_signal(samples, cfg.frame_len, cfg.hop_len)
        if not frames:
            return []

        mfccs: List[List[float]] = []
        previous_sample = samples[0] if samples else 0.0
        for frame in frames:
            emphasised = self._preemphasise(frame.samples, previous_sample)
            previous_sample = frame.samples[-1]
            power = power_spectrum([s * w for s, w in zip(emphasised, self._window)], cfg.n_fft)
            energies = [sum(w * p for w, p in zip(filt, power)) for filt in self._filters]
            log_energies = [math.log(max(e, 1e-10)) for e in energies]
            coeffs = [sum(c * e for c, e in zip(row, log_energies)) for row in self._dct]
            if cfg.lifter > 0:
                coeffs = [
                    c * (1.0 + (cfg.lifter / 2.0) * math.sin(math.pi * k / cfg.lifter))
                    for k, c in enumerate(coeffs)
                ]
            mfccs.append(coeffs)

        if cfg.use_deltas and len(mfccs) > 1:
            delta = _delta(mfccs, 2)
            ddelta = _delta(delta, 2)
            mfccs = [m + d + dd for m, d, dd in zip(mfccs, delta, ddelta)]

        if cfg.cmvn and mfccs:
            mfccs = _cmvn(mfccs)
        return mfccs

    def _resolve(self, audio: Audio | Sequence[float], sample_rate: Optional[int]) -> Tuple[List[float], int]:
        if isinstance(audio, Audio):
            return list(audio.samples), audio.sample_rate
        return list(audio), sample_rate or self.config.sample_rate

    @staticmethod
    def _preemphasise(frame: Sequence[float], previous_sample: float, coef: float = 0.97) -> List[float]:
        out: List[float] = []
        prev = previous_sample
        for v in frame:
            out.append(v - coef * prev)
            prev = v
        return out

    @property
    def feature_dim(self) -> int:
        return self.config.n_mfcc * (3 if self.config.use_deltas else 1)


def _delta(features: Sequence[Sequence[float]], width: int = 2) -> List[List[float]]:
    """Regression based first-order derivative over a sliding window."""
    n = len(features)
    dim = len(features[0]) if n else 0
    if n == 0:
        return []
    denom = 2.0 * sum(i * i for i in range(1, width + 1))
    out: List[List[float]] = []
    for t in range(n):
        acc = [0.0] * dim
        for k in range(1, width + 1):
            prev_idx = max(0, t - k)
            next_idx = min(n - 1, t + k)
            for d in range(dim):
                acc[d] += k * (features[next_idx][d] - features[prev_idx][d])
        out.append([a / denom for a in acc])
    return out


def _cmvn(features: Sequence[Sequence[float]]) -> List[List[float]]:
    """Cepstral mean and variance normalisation over the utterance."""
    n = len(features)
    dim = len(features[0])
    means = [sum(f[d] for f in features) / n for d in range(dim)]
    var = [sum((f[d] - means[d]) ** 2 for f in features) / n for d in range(dim)]
    std = [math.sqrt(v) if v > 1e-10 else 1.0 for v in var]
    return [[(f[d] - means[d]) / std[d] for d in range(dim)] for f in features]
