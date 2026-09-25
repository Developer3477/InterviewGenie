"""Emotional intelligence: reading the interviewer, choosing the response tone.

:class:`EmotionalIntelligence` wraps the sentiment/emotion analysers with the
stateful behaviour an interview actually needs:

* **tone tracking** -- smoothed emotion trajectory across the interview,
* **rapport modelling** -- is the interviewer warming up, cooling off, probing
  hard, or testing for culture fit,
* **empathy policy** -- map the detected emotion onto a response register and a
  set of concrete linguistic moves (acknowledge, validate, de-escalate, match
  energy), and
* **engagement scoring** -- how engaging and empathetic the system's own output
  is, which feeds the evaluation metrics.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from ..logging import get_logger
from ..nlp.sentiment import SentimentAnalyzer
from ..types import EmotionScores, Sentiment

LOG = get_logger("context.emotion")


#: emotion -> (register, opening move, closing move)
EMPATHY_PLAYBOOK: Dict[str, Dict[str, str]] = {
    "joy": {
        "register": "warm",
        "opening": "That is great to hear.",
        "closing": "I would love to keep that energy going.",
    },
    "trust": {
        "register": "steady",
        "opening": "Happy to walk you through it.",
        "closing": "Let me know if you want more detail.",
    },
    "fear": {
        "register": "reassuring",
        "opening": "That is a fair thing to press on.",
        "closing": "Happy to go slower on any part of that.",
    },
    "surprise": {
        "register": "engaged",
        "opening": "That is an interesting angle.",
        "closing": "Did you want me to dig into that part?",
    },
    "sadness": {
        "register": "supportive",
        "opening": "I appreciate you asking.",
        "closing": "That one still matters to me.",
    },
    "disgust": {
        "register": "grounded",
        "opening": "I understand the concern.",
        "closing": "I would rather be straight about the trade-offs.",
    },
    "anger": {
        "register": "calm",
        "opening": "That is a legitimate frustration.",
        "closing": "Happy to take that offline with more detail.",
    },
    "anticipation": {
        "register": "encouraging",
        "opening": "Good question to be thinking about.",
        "closing": "That is exactly the direction I want to grow in.",
    },
    "neutral": {
        "register": "professional",
        "opening": "",
        "closing": "",
    },
}

#: interviewer behaviour -> what it usually means
RAPPORT_SIGNALS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "warming": (("great", "love", "impressive", "excellent", "interesting", "nice"),
                "the interviewer is engaged and receptive"),
    "cooling": (("hmm", "okay", "i see", "moving on", "let's move", "anyway"),
                "the interviewer may be losing interest -- tighten the answer"),
    "probing": (("why", "how exactly", "specifically", "what about", "but", "actually"),
                "the interviewer is stress-testing the claim"),
    "rushing": (("quickly", "time", "short on", "wrap", "briefly", "last question"),
                "time pressure -- lead with the headline"),
    "supportive": (("no pressure", "take your time", "don't worry", "that's fine", "good"),
                   "psychological safety is high -- be candid"),
}

#: linguistic markers that make an answer feel empathetic and engaged
EMPATHY_MARKERS: Tuple[str, ...] = (
    "appreciate", "understand", "fair", "that makes sense", "i can see",
    "good question", "interesting", "i would love", "happy to", "glad",
    "thank you", "that is a fair", "i hear you", "let me",
)
ENGAGEMENT_MARKERS: Tuple[str, ...] = (
    "for example", "for instance", "specifically", "in practice", "one project",
    "we shipped", "i built", "i led", "the result", "as a result", "which meant",
    "so that", "because", "the trade-off", "i learned",
)


@dataclass
class EmotionalIntelligence:
    """Stateful emotional intelligence module."""

    smoothing: float = 0.35
    empathy_level: float = 0.7
    analyser: SentimentAnalyzer = field(default_factory=SentimentAnalyzer)
    emotion: EmotionScores = field(default_factory=EmotionScores)
    trajectory: Deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=40))
    rapport: str = "neutral"

    def __post_init__(self) -> None:
        self.emotion = EmotionScores()
        self.trajectory = deque(maxlen=40)
        self._cache: Dict[str, Dict[str, Any]] = {}

    # -- perception ------------------------------------------------------- #
    def perceive(self, text: str) -> Dict[str, Any]:
        """Analyse one interviewer utterance and update the emotion state."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        result = self._perceive(text)
        self._cache[text] = result
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        return result

    def _perceive(self, text: str) -> Dict[str, Any]:
        instant = self.analyser.emotions(text)
        sentiment = self.analyser.analyze(text)
        alpha = self.smoothing
        for field_name in EmotionScores().__dataclass_fields__:  # type: ignore[attr-defined]
            new = getattr(instant, field_name)
            old = getattr(self.emotion, field_name)
            setattr(self.emotion, field_name, round((1 - alpha) * old + alpha * new, 4))
        self.rapport = self.detect_rapport(text)
        self.trajectory.append({**self.emotion.to_dict(), "polarity": sentiment.polarity})
        dominant = self.emotion.dominant()
        return {
            "instant": instant.to_dict(),
            "smoothed": self.emotion.to_dict(),
            "dominant": dominant if self.emotion.intensity() >= 0.12 else "neutral",
            "intensity": round(self.emotion.intensity(), 4),
            "sentiment": sentiment.to_dict(),
            "rapport": self.rapport,
        }

    def detect_rapport(self, text: str) -> str:
        low = text.lower()
        scores: Dict[str, int] = {}
        for label, (cues, _meaning) in RAPPORT_SIGNALS.items():
            hits = sum(1 for cue in cues if cue in low)
            if hits:
                scores[label] = hits
        if not scores:
            return "neutral"
        return max(scores, key=scores.get)

    def rapport_meaning(self) -> str:
        for label, (_cues, meaning) in RAPPORT_SIGNALS.items():
            if label == self.rapport:
                return meaning
        return "steady, neutral engagement"

    # -- expression ------------------------------------------------------- #
    def respond_with(self, text: str) -> Dict[str, str]:
        """Return the register and linguistic moves for responding to ``text``."""
        perception = self.perceive(text)
        dominant = perception["dominant"]
        playbook = EMPATHY_PLAYBOOK.get(dominant, EMPATHY_PLAYBOOK["neutral"])
        moves: List[str] = []
        if dominant in {"fear", "sadness"}:
            moves.append("acknowledge the concern before answering")
        if dominant == "anger":
            moves.append("de-escalate: agree with the frustration, then be concrete")
        if dominant == "surprise":
            moves.append("show genuine curiosity")
        if dominant == "joy":
            moves.append("match the energy briefly, then return to substance")
        if perception["rapport"] == "probing":
            moves.append("lead with the evidence, not the narrative")
        if perception["rapport"] == "cooling":
            moves.append("shorten and land the headline in the first sentence")
        if perception["rapport"] == "rushing":
            moves.append("answer in one sentence, then offer to expand")
        if perception["rapport"] == "supportive":
            moves.append("be candid about what went wrong")
        return {
            "register": playbook["register"],
            "opening": playbook["opening"] if self.empathy_level > 0.4 else "",
            "closing": playbook["closing"] if self.empathy_level > 0.6 else "",
            "dominant_emotion": dominant,
            "rapport": perception["rapport"],
            "moves": "; ".join(moves),
        }

    # -- self evaluation -------------------------------------------------- #
    def engagement_score(self, text: str) -> float:
        """How empathetic and engaging ``text`` reads (0..1)."""
        low = text.lower()
        empathy = sum(1 for marker in EMPATHY_MARKERS if marker in low)
        engagement = sum(1 for marker in ENGAGEMENT_MARKERS if marker in low)
        words = max(1, len(text.split()))
        score = 0.0
        score += min(0.5, 0.18 * empathy)
        score += min(0.5, 0.12 * engagement)
        # questions back to the interviewer show engagement
        score += 0.15 if "?" in text else 0.0
        # reward concreteness: numbers and specifics
        score += 0.1 if any(ch.isdigit() for ch in text) else 0.0
        # penalise extreme verbosity
        if words > 130:
            score -= 0.1
        return round(max(0.0, min(1.0, score)), 4)

    def empathy_score(self, text: str, interviewer_text: str = "") -> float:
        """How well ``text`` matches the emotional needs of ``interviewer_text``."""
        base = self.engagement_score(text)
        if not interviewer_text:
            return base
        guidance = self.respond_with(interviewer_text)
        register = guidance["register"]
        bonuses = {
            "professional": ("good question", "happy to", "let me", "here is",
                             "the evidence", "in practice"),
            "warm": ("great", "love", "glad", "happy"),
            "steady": ("happy to", "let me", "here is"),
            "reassuring": ("fair", "understand", "makes sense", "appreciate"),
            "engaged": ("interesting", "angle", "curious"),
            "supportive": ("appreciate", "matters", "thank"),
            "grounded": ("understand", "straight", "trade-off"),
            "calm": ("legitimate", "understand", "concrete"),
            "encouraging": ("direction", "grow", "love"),
        }
        markers = bonuses.get(register, ())
        if any(marker in text.lower() for marker in markers):
            base = min(1.0, base + 0.2)
        return round(base, 4)

    def reset(self) -> None:
        self.emotion = EmotionScores()
        self.trajectory.clear()
        self.rapport = "neutral"
        self._cache.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion": self.emotion.to_dict(),
            "dominant": self.emotion.dominant(),
            "intensity": round(self.emotion.intensity(), 4),
            "rapport": self.rapport,
            "rapport_meaning": self.rapport_meaning(),
        }
