"""Dynamic time warping decoder for the offline speech recogniser.

Given a sequence of MFCC frames, the decoder finds the template whose warping
path has the lowest normalised cost.  Confidence is derived from the margin
between the best and second-best candidate, so the caller can ask for
clarification when the recogniser is unsure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..logging import get_logger
from .dsp import Audio, MfccExtractor
from .templates import PhraseTemplate, TemplateBank

LOG = get_logger("asr.decoder")


def dtw_cost(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]],
             band: Optional[int] = None) -> Tuple[float, List[Tuple[int, int]]]:
    """Dynamic time warping cost and optimal path between two frame sequences.

    ``a`` is the query and ``b`` the template.  ``band`` limits the Sakoe-Chiba
    band width, which both speeds up decoding and prevents pathological
    alignments when the sequences have very different lengths.
    """
    if not a or not b:
        return float("inf"), []
    n, m = len(a), len(b)
    if band is None:
        band = max(n, m)
    band = max(abs(n - m), min(band, max(n, m)))

    inf = float("inf")
    # rolling rows keep memory at O(m)
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    back: List[List[Tuple[int, int]]] = [[(0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        lo = max(1, i - band)
        hi = min(m, i + band)
        row_back = back[i]
        for j in range(lo, hi + 1):
            dist = _frame_distance(a[i - 1], b[j - 1])
            best = prev[j]          # insertion (advance in a)
            step = (i - 1, j)
            if cur[j - 1] < best:   # deletion (advance in b)
                best = cur[j - 1]
                step = (i, j - 1)
            if prev[j - 1] < best:  # match
                best = prev[j - 1]
                step = (i - 1, j - 1)
            cur[j] = dist + best
            row_back[j] = step
        if all(v == inf for v in cur):
            return inf, []
        prev = cur

    # backtrack
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i, j))
        pi, pj = back[i][j]
        if pi == i and pj == j:
            break
        i, j = pi, pj
    path.reverse()
    return prev[m] / max(1, n + m), path


def _frame_distance(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        d = x - y
        total += d * d
    return math.sqrt(total)


def normalised_cost(cost: float, n_frames: int) -> float:
    """Convert a raw DTW cost into a ``[0, 1]`` dissimilarity score."""
    if not math.isfinite(cost):
        return 1.0
    scale = 12.0
    return 1.0 - math.exp(-cost / scale)


@dataclass
class DecodeResult:
    """One hypothesis produced by the decoder."""

    phrase: str
    confidence: float
    cost: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    frames: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "phrase": self.phrase,
            "confidence": round(self.confidence, 4),
            "cost": round(self.cost, 4),
            "frames": self.frames,
            "alternatives": [[p, round(c, 4)] for p, c in self.alternatives],
        }


@dataclass
class PhraseDecoder:
    """Template-matching decoder over a :class:`TemplateBank`."""

    bank: TemplateBank = field(default_factory=TemplateBank)
    extractor: MfccExtractor = field(default_factory=MfccExtractor)
    band_ratio: float = 0.35
    margin_weight: float = 0.6

    # -- decoding --------------------------------------------------------- #
    def decode(self, features: Sequence[Sequence[float]],
               top_k: int = 3) -> Optional[DecodeResult]:
        """Match ``features`` against every template in the bank."""
        if not features or not self.bank.templates:
            return None
        scored: List[Tuple[float, PhraseTemplate]] = []
        for template in self.bank.templates:
            if not template.vectors:
                continue
            band = max(8, int(self.band_ratio * max(len(features), len(template.vectors))))
            cost, _ = dtw_cost(features, template.vectors, band=band)
            if math.isfinite(cost):
                scored.append((cost, template))
        if not scored:
            return None
        scored.sort(key=lambda ct: ct[0])
        best_cost, best_template = scored[0]
        best_dissimilarity = normalised_cost(best_cost, len(features))
        # margin against the runner-up sharpens confidence
        margin = 0.0
        if len(scored) > 1:
            second = normalised_cost(scored[1][0], len(features))
            margin = max(0.0, second - best_dissimilarity)
        confidence = (1.0 - best_dissimilarity) * best_template.confidence
        confidence = min(1.0, confidence * (1.0 + self.margin_weight * margin))
        alternatives = [
            (t.phrase, round(max(0.0, (1.0 - normalised_cost(c, len(features))) * t.confidence), 4))
            for c, t in scored[1:top_k]
        ]
        return DecodeResult(
            phrase=best_template.phrase,
            confidence=round(confidence, 4),
            cost=round(best_cost, 4),
            alternatives=alternatives,
            frames=len(features),
        )

    def decode_audio(self, audio: Audio) -> Optional[DecodeResult]:
        return self.decode(self.extractor.extract(audio))

    # -- partial / streaming --------------------------------------------- #
    def partial_decode(self, features: Sequence[Sequence[float]]) -> Optional[DecodeResult]:
        """Cheap prefix match used to emit interim hypotheses."""
        if not features:
            return None
        # only compare against the tail of each template
        results: List[Tuple[float, PhraseTemplate]] = []
        for template in self.bank.templates:
            if not template.vectors:
                continue
            tail = template.vectors[-max(1, len(features)):]
            band = max(4, int(self.band_ratio * max(len(features), len(tail))))
            cost, _ = dtw_cost(features, tail, band=band)
            if math.isfinite(cost):
                results.append((cost, template))
        if not results:
            return None
        results.sort(key=lambda ct: ct[0])
        cost, template = results[0]
        return DecodeResult(
            phrase=template.phrase,
            confidence=round(max(0.0, 1.0 - normalised_cost(cost, len(features))) * 0.75, 4),
            cost=round(cost, 4),
            frames=len(features),
        )

    # -- calibration ------------------------------------------------------ #
    def calibrate(self, phrases: Sequence[str], audio_by_phrase: Dict[str, Audio]) -> Dict[str, Any]:
        """Enroll a set of phrases; returns per-phrase outcomes."""
        outcomes: Dict[str, Any] = {"enrolled": [], "skipped": []}
        for phrase in phrases:
            audio = audio_by_phrase.get(phrase)
            if audio is None:
                outcomes["skipped"].append(phrase)
                continue
            self.bank.add(phrase, audio, source="enrolled", confidence=0.9)
            outcomes["enrolled"].append(phrase)
        return outcomes
