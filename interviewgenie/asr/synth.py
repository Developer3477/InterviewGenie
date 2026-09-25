"""A compact formant (source-filter) speech syntheser.

Two jobs:

1. **Template bootstrapping** -- the offline recogniser needs acoustic templates
   before anybody has enrolled anything, so the phrase bank is seeded by
   synthesising each supported phrase from its phoneme sequence.
2. **Spoken prompts** -- the cockpit can read a question back to the operator.

The synthesiser is a classic cascade of three 2nd-order resonators driven by a
glottal pulse train (voiced) or filtered noise (unvoiced).  It is deliberately
small -- it produces intelligible, robotic speech, which is all the template
matcher needs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..logging import get_logger
from .dsp import Audio

LOG = get_logger("asr.synth")


# --------------------------------------------------------------------------- #
# Phoneme table: (F1, F2, F3, B1, B2, B3, voiced, duration_scale)
# --------------------------------------------------------------------------- #
_VOWELS: Dict[str, Tuple[float, float, float, float, float, float, bool, float]] = {
    "iy": (270, 2290, 3010, 60, 90, 120, True, 1.0),
    "ih": (400, 1990, 2550, 70, 100, 130, True, 0.9),
    "ey": (530, 1840, 2480, 70, 100, 130, True, 1.1),
    "eh": (530, 1790, 2490, 70, 100, 130, True, 0.9),
    "ae": (660, 1720, 2410, 80, 110, 140, True, 1.0),
    "aa": (730, 1090, 2440, 80, 110, 140, True, 1.0),
    "ao": (570, 840, 2410, 80, 110, 140, True, 1.1),
    "ow": (490, 910, 2350, 80, 110, 140, True, 1.1),
    "uh": (640, 1190, 2390, 80, 110, 140, True, 0.8),
    "uw": (300, 870, 2240, 60, 90, 120, True, 1.0),
    "ah": (640, 1250, 2600, 80, 110, 140, True, 0.85),
    "er": (490, 1350, 1690, 70, 100, 130, True, 1.0),
    "ax": (500, 1500, 2500, 90, 120, 150, True, 0.5),
}
_CONSONANTS: Dict[str, Tuple[float, float, float, float, float, float, bool, float]] = {
    "p": (400, 1100, 2200, 90, 120, 160, False, 0.7),
    "b": (400, 1100, 2200, 90, 120, 160, True, 0.7),
    "t": (400, 1600, 2600, 100, 140, 180, False, 0.65),
    "d": (400, 1600, 2600, 100, 140, 180, True, 0.65),
    "k": (400, 1900, 2700, 110, 150, 190, False, 0.7),
    "g": (400, 1900, 2700, 110, 150, 190, True, 0.7),
    "f": (400, 1100, 2200, 120, 160, 200, False, 0.8),
    "v": (400, 1100, 2200, 120, 160, 200, True, 0.8),
    "th": (400, 1400, 2400, 130, 170, 210, False, 0.8),
    "dh": (400, 1400, 2400, 130, 170, 210, True, 0.8),
    "s": (400, 4000, 5000, 140, 200, 260, False, 0.9),
    "z": (400, 4000, 5000, 140, 200, 260, True, 0.9),
    "sh": (400, 2000, 2800, 140, 200, 260, False, 0.9),
    "zh": (400, 2000, 2800, 140, 200, 260, True, 0.9),
    "h": (500, 1500, 2500, 150, 200, 260, False, 0.7),
    "m": (280, 900, 2200, 80, 110, 140, True, 0.8),
    "n": (280, 1700, 2600, 80, 110, 140, True, 0.8),
    "ng": (280, 2300, 2750, 80, 110, 140, True, 0.8),
    "l": (380, 1300, 2800, 70, 100, 130, True, 0.9),
    "r": (310, 1060, 1380, 70, 100, 130, True, 0.9),
    "w": (290, 610, 2150, 60, 90, 120, True, 0.8),
    "y": (270, 2200, 3000, 60, 90, 120, True, 0.8),
    "ch": (400, 2000, 2800, 140, 200, 260, False, 0.8),
    "jh": (400, 2000, 2800, 140, 200, 260, True, 0.8),
}
PHONEMES: Dict[str, Tuple[float, float, float, float, float, float, bool, float]] = {**_VOWELS, **_CONSONANTS}

# --------------------------------------------------------------------------- #
# Grapheme -> phoneme (a compact rule set; good enough for interview vocabulary)
# --------------------------------------------------------------------------- #
_G2P_RULES: List[Tuple[str, str]] = [
    (r"tion", "sh ax n"), (r"sion", "zh ax n"), (r"cious", "sh ax s"),
    (r"tious", "sh ax s"), (r"ough", "ah f"), (r"augh", "ae f"),
    (r"ight", "ay t"), (r"ould", "uh d"), (r"ould", "uh d"),
    (r"ing", "ih ng"), (r"ies", "iy z"), (r"ied", "iy d"),
    (r"ee", "iy"), (r"ea", "iy"), (r"oo", "uw"), (r"ou", "aw"),
    (r"ow", "aw"), (r"ai", "ey"), (r"ay", "ey"), (r"ei", "iy"),
    (r"ey", "iy"), (r"au", "ao"), (r"aw", "ao"), (r"oi", "oy"),
    (r"oy", "oy"), (r"oa", "ow"), (r"ue", "uw"), (r"ui", "uw"),
    (r"er", "er"), (r"ir", "er"), (r"ur", "er"), (r"or", "er"),
    (r"ar", "aa r"), (r"al", "ah l"), (r"ph", "f"), (r"gh", "g"),
    (r"ck", "k"), (r"ch", "ch"), (r"sh", "sh"), (r"th", "th"),
    (r"wh", "w"), (r"ng", "ng"), (r"qu", "k w"), (r"kn", "n"),
    (r"wr", "r"), (r"gn", "n"), (r"ps", "s"), (r"mn", "m"),
    (r"ce", "s"), (r"ci", "s ih"), (r"cy", "s iy"), (r"ge", "jh"),
    (r"gi", "jh ih"), (r"gy", "jh iy"), (r"y", "iy"), (r"x", "k s"),
]
_G2P_SINGLE: Dict[str, str] = {
    "a": "ae", "b": "b", "c": "k", "d": "d", "e": "eh", "f": "f", "g": "g",
    "h": "h", "i": "ih", "j": "jh", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "aa", "p": "p", "q": "k", "r": "r", "s": "s", "t": "t", "u": "ah",
    "v": "v", "w": "w", "x": "k s", "y": "iy", "z": "z",
}
#: phonemes produced by the rules that are not in the table
_EXTRA = {"ay": "ey", "aw": "ao", "oy": "ao ih", "ae": "ae", "ah": "ah", "ax": "ax"}


def g2p(word: str) -> List[str]:
    """Rule-based grapheme-to-phoneme conversion."""
    low = re.sub(r"[^a-z]", "", word.lower())
    if not low:
        return []
    out: List[str] = []
    i = 0
    while i < len(low):
        matched = False
        for pattern, replacement in _G2P_RULES:
            if low.startswith(pattern, i):
                out.extend(replacement.split())
                i += len(pattern)
                matched = True
                break
        if matched:
            continue
        phoneme = _G2P_SINGLE.get(low[i], "ah")
        out.extend(phoneme.split())
        i += 1
    resolved: List[str] = []
    for phoneme in out:
        if phoneme in PHONEMES:
            resolved.append(phoneme)
        elif phoneme in _EXTRA:
            resolved.extend(p for p in _EXTRA[phoneme].split() if p in PHONEMES)
        else:
            resolved.append("ax")
    return resolved or ["ax"]


def _stable_hash(*parts: Any) -> int:
    """A deterministic stand-in for ``hash()`` over strings.

    Python randomises ``str`` hashing per process (``PYTHONHASHSEED``), so the
    previous ``hash((i, phoneme))`` made every synthesised waveform — and hence
    every noise-robustness number — differ between runs. This FNV-1a style mix
    keeps the noise-like quality of the unvoiced excitation source while being
    fully reproducible.
    """
    h = 2166136261
    for part in parts:
        for ch in str(part):
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        h = ((h ^ 0x1F) * 16777619) & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
# Resonator + synthesiser
# --------------------------------------------------------------------------- #
@dataclass
class _Resonator:
    a1: float = 0.0
    a2: float = 0.0
    b0: float = 0.0
    y1: float = 0.0
    y2: float = 0.0

    def set(self, frequency: float, bandwidth: float, sample_rate: int) -> None:
        r = math.exp(-math.pi * bandwidth / sample_rate)
        theta = 2.0 * math.pi * frequency / sample_rate
        self.a2 = -r * r
        self.a1 = 2.0 * r * math.cos(theta)
        self.b0 = (1.0 - r) * math.sqrt(1.0 - 2.0 * r * math.cos(2 * theta) + r * r)

    def step(self, x: float) -> float:
        y = self.b0 * x + self.a1 * self.y1 + self.a2 * self.y2
        self.y2 = self.y1
        self.y1 = y
        return y


@dataclass
class FormantSynthesizer:
    """Cascade formant synthesiser with co-articulation smoothing."""

    sample_rate: int = 16000
    base_f0: float = 120.0
    words_per_minute: float = 145.0
    jitter: float = 0.0

    def synthesize(self, text: str, sample_rate: Optional[int] = None) -> Audio:
        rate = sample_rate or self.sample_rate
        words = re.findall(r"[A-Za-z']+", text)
        if not words:
            return Audio([], rate, 1, text)
        chunks: List[float] = []
        for word in words:
            phonemes = g2p(word)
            chunks.extend(self._synthesise_phonemes(phonemes, rate))
            chunks.extend([0.0] * int(rate * 0.05))
        peak = max((abs(v) for v in chunks), default=1.0) or 1.0
        gain = 0.72 / peak
        return Audio([v * gain for v in chunks], rate, 1, text)

    def _synthesise_phonemes(self, phonemes: Sequence[str], rate: int) -> List[float]:
        if not phonemes:
            return []
        resonators = [_Resonator() for _ in range(3)]
        out: List[float] = []
        phase = 0.0
        # seconds per phoneme: shorter for consonants, proportional to WPM
        base = 60.0 / self.words_per_minute
        for idx, phoneme in enumerate(phonemes):
            spec = PHONEMES.get(phoneme)
            if spec is None:
                continue
            f1, f2, f3, b1, b2, b3, voiced, dur_scale = spec
            duration = base * 0.22 * dur_scale
            n = max(1, int(duration * rate))
            # co-articulation: glide the previous formants toward this one
            prev = PHONEMES.get(phonemes[idx - 1], spec)
            for i, (res, freq, bw, prev_freq, prev_bw) in enumerate(zip(
                resonators, (f1, f2, f3), (b1, b2, b3),
                (prev[0], prev[1], prev[2]), (prev[3], prev[4], prev[5]),
            )):
                res.set(freq, bw, rate)
            start = len(out)
            for i in range(n):
                t = i / n
                if voiced:
                    f0 = self.base_f0 * (1.0 + 0.02 * math.sin(2 * math.pi * 3 * t))
                    phase += f0 / rate
                    if phase >= 1.0:
                        phase -= 1.0
                    source = math.exp(-30.0 * phase) * (1.0 - 0.6 * phase)
                else:
                    source = (_stable_hash(i, phoneme) % 2000 - 1000) / 1000.0
                sample = 0.0
                for res, freq in zip(resonators, (f1, f2, f3)):
                    sample += res.step(source) * (1.0 if freq == f1 else 0.45)
                # amplitude envelope with 8 ms ramps
                ramp = min(1.0, t / 0.25, (1.0 - t) / 0.25)
                out.append(max(-1.0, min(1.0, sample * ramp * 0.5)))
            # silence between phonemes
            out.extend([0.0] * int(rate * 0.012))
        return out


#: module level singleton
_SYNTH = FormantSynthesizer()


def synthesize(text: str, sample_rate: int = 16000) -> Audio:
    return _SYNTH.synthesize(text, sample_rate)
