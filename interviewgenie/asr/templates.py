"""Acoustic template bank for the offline phrase decoder.

The local recogniser is an isolated-phrase / template matcher: each supported
utterance has a stored MFCC template, either

* **enrolled** -- the operator (or candidate) speaks the phrase once and the
  extracted features become the template, or
* **generated** -- the template is derived from the phrase's phoneme sequence
  using the bundled formant model (see :mod:`interviewgenie.asr.synth`).

Templates persist as JSON so a calibrated recogniser survives restarts.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..logging import get_logger
from .dsp import Audio, MfccExtractor

LOG = get_logger("asr.templates")


@dataclass
class PhraseTemplate:
    """One acoustic template."""

    phrase: str
    vectors: List[List[float]]
    source: str = "enrolled"          # enrolled | generated | averaged
    samples: int = 1
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase,
            "vectors": self.vectors,
            "source": self.source,
            "samples": self.samples,
            "confidence": round(self.confidence, 4),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhraseTemplate":
        return cls(
            phrase=data["phrase"],
            vectors=[list(map(float, v)) for v in data["vectors"]],
            source=data.get("source", "enrolled"),
            samples=int(data.get("samples", 1)),
            confidence=float(data.get("confidence", 0.8)),
            created_at=float(data.get("created_at", time.time())),
        )


def average_templates(templates: Sequence[PhraseTemplate]) -> Optional[PhraseTemplate]:
    """Average a set of same-phrase templates by resampling to a common length."""
    if not templates:
        return None
    if len(templates) == 1:
        return templates[0]
    lengths = [len(t.vectors) for t in templates]
    target = max(1, int(sum(lengths) / len(lengths)))
    dim = len(templates[0].vectors[0]) if templates[0].vectors else 0
    if dim == 0:
        return None
    resampled: List[List[List[float]]] = []
    for template in templates:
        resampled.append(_resample(template.vectors, target))
    averaged = [
        [sum(frame[d] for frame in batch) / len(batch) for d in range(dim)]
        for batch in zip(*resampled)
    ]
    return PhraseTemplate(
        phrase=templates[0].phrase,
        vectors=averaged,
        source="averaged",
        samples=len(templates),
        confidence=min(1.0, sum(t.confidence for t in templates) / len(templates) + 0.05),
    )


def _resample(vectors: Sequence[Sequence[float]], target: int) -> List[List[float]]:
    if not vectors:
        return [[0.0]] * target
    if len(vectors) == target:
        return [list(v) for v in vectors]
    out: List[List[float]] = []
    for i in range(target):
        pos = i * (len(vectors) - 1) / max(1, target - 1)
        left = int(math.floor(pos))
        right = min(left + 1, len(vectors) - 1)
        frac = pos - left
        out.append([a * (1 - frac) + b * frac for a, b in zip(vectors[left], vectors[right])])
    return out


@dataclass
class TemplateBank:
    """A searchable collection of phrase templates."""

    extractor: MfccExtractor = field(default_factory=MfccExtractor)
    templates: List[PhraseTemplate] = field(default_factory=list)
    _by_phrase: Dict[str, List[PhraseTemplate]] = field(default_factory=dict, repr=False)

    # -- construction ----------------------------------------------------- #
    def __post_init__(self) -> None:
        self._reindex()

    def _reindex(self) -> None:
        self._by_phrase = {}
        for template in self.templates:
            self._by_phrase.setdefault(template.phrase.lower(), []).append(template)

    def add(self, phrase: str, audio: Audio, *, source: str = "enrolled",
            confidence: float = 0.8) -> PhraseTemplate:
        """Enroll ``phrase`` from ``audio``."""
        vectors = self.extractor.extract(audio)
        if not vectors:
            raise ValueError("no features could be extracted from the audio")
        template = PhraseTemplate(phrase=phrase, vectors=vectors, source=source,
                                  confidence=confidence)
        self.templates.append(template)
        self._by_phrase.setdefault(phrase.lower(), []).append(template)
        return template

    def add_vectors(self, phrase: str, vectors: Sequence[Sequence[float]],
                    *, source: str = "generated", confidence: float = 0.6) -> PhraseTemplate:
        template = PhraseTemplate(phrase=phrase, vectors=[list(v) for v in vectors],
                                  source=source, confidence=confidence)
        self.templates.append(template)
        self._by_phrase.setdefault(phrase.lower(), []).append(template)
        return template

    def consolidate(self) -> int:
        """Collapse multiple templates per phrase into a single averaged one."""
        merged: List[PhraseTemplate] = []
        for phrase, group in self._by_phrase.items():
            avg = average_templates(group)
            if avg is not None:
                merged.append(avg)
        self.templates = merged
        self._reindex()
        return len(self.templates)

    # -- access ----------------------------------------------------------- #
    @property
    def phrases(self) -> List[str]:
        return sorted(self._by_phrase)

    def template_for(self, phrase: str) -> Optional[PhraseTemplate]:
        group = self._by_phrase.get(phrase.lower())
        return group[0] if group else None

    def __len__(self) -> int:
        return len(self.templates)

    def __contains__(self, phrase: object) -> bool:
        return isinstance(phrase, str) and phrase.lower() in self._by_phrase

    # -- persistence ------------------------------------------------------ #
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "feature_dim": self.extractor.feature_dim,
            "templates": [t.to_dict() for t in self.templates],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        LOG.info("template bank saved", context={"path": path, "templates": len(self.templates)})
        return path

    @classmethod
    def load(cls, path: str, extractor: Optional[MfccExtractor] = None) -> "TemplateBank":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        bank = cls(extractor=extractor or MfccExtractor())
        bank.templates = [PhraseTemplate.from_dict(t) for t in payload.get("templates", [])]
        bank._reindex()
        LOG.info("template bank loaded", context={"path": path, "templates": len(bank.templates)})
        return bank

    def stats(self) -> Dict[str, Any]:
        return {
            "templates": len(self.templates),
            "phrases": len(self._by_phrase),
            "feature_dim": self.extractor.feature_dim,
            "by_source": {
                source: sum(1 for t in self.templates if t.source == source)
                for source in ("enrolled", "generated", "averaged")
            },
        }
