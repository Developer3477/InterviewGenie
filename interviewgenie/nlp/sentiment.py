"""Sentiment and emotion analysis (the "emotional intelligence" front-end).

The sentiment model is a VADER-style rule engine: a hand-tuned valence lexicon
combined with negation scope, degree modifiers, punctuation emphasis and
capitalisation weighting.  The emotion model maps the same lexical evidence onto
Plutchik's eight primary emotions, which the response generator uses to pick an
empathetic register.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..types import EmotionScores, Sentiment
from .textnorm import tokenize

# --------------------------------------------------------------------------- #
# Valence lexicon  (-4 .. +4)
# --------------------------------------------------------------------------- #
VALENCE: Dict[str, float] = {
    # strongly positive
    "excellent": 3.0, "outstanding": 3.2, "exceptional": 3.0, "brilliant": 2.8,
    "fantastic": 2.9, "amazing": 2.8, "awesome": 2.7, "superb": 2.9,
    "impressive": 2.4, "remarkable": 2.3, "terrific": 2.6, "wonderful": 2.7,
    "delighted": 2.6, "thrilled": 2.7, "love": 2.6, "loved": 2.5, "loves": 2.5,
    "perfect": 2.5, "ideal": 1.9, "best": 2.4, "great": 2.3, "good": 1.7,
    "strong": 1.6, "solid": 1.5, "proud": 2.1, "excited": 2.2, "exciting": 2.2,
    "enjoy": 1.9, "enjoyed": 1.9, "passionate": 2.2, "motivated": 1.7,
    "confident": 1.8, "success": 2.1, "successful": 2.2, "achieve": 1.8,
    "achieved": 1.8, "improve": 1.4, "improved": 1.5, "win": 1.9, "won": 1.9,
    "helpful": 1.6, "supportive": 1.7, "collaborative": 1.5, "reliable": 1.7,
    "efficient": 1.5, "effective": 1.6, "clear": 1.0, "smooth": 1.4,
    "seamless": 1.8, "robust": 1.5, "scalable": 1.2, "innovative": 1.8,
    "creative": 1.6, "insightful": 1.9, "thoughtful": 1.5, "honest": 1.6,
    "grateful": 2.0, "thanks": 1.5, "thank": 1.4, "appreciate": 1.8,
    "welcome": 1.6, "glad": 1.8, "happy": 2.1, "pleased": 1.8, "satisfied": 1.7,
    "comfortable": 1.3, "easy": 1.2, "faster": 1.2, "fast": 1.0, "clean": 1.2,
    "recommend": 1.5, "kudos": 2.0, "nice": 1.6, "cool": 1.3, "fun": 1.7,
    # mildly positive
    "okay": 0.5, "ok": 0.5, "fine": 0.6, "decent": 0.9, "reasonable": 0.9,
    "adequate": 0.4, "fair": 0.5, "interesting": 1.1, "curious": 0.9,
    "willing": 0.8, "ready": 0.9, "able": 0.7, "capable": 1.2, "promising": 1.4,
    "positive": 1.6, "optimistic": 1.6, "hopeful": 1.5, "encouraging": 1.7,
    # neutral / mildly negative
    "busy": -0.4, "tired": -0.9, "late": -0.8, "slow": -0.9, "difficult": -1.2,
    "hard": -0.8, "complex": -0.5, "unclear": -1.1, "confusing": -1.3,
    "confused": -1.2, "risky": -1.1, "concern": -1.2, "concerned": -1.3,
    "worried": -1.5, "worry": -1.4, "unsure": -1.0, "uncertain": -1.0,
    "doubt": -1.2, "issue": -1.0, "problem": -1.3, "bug": -1.0, "fail": -2.0,
    "failed": -2.1, "failure": -2.2, "failing": -2.0, "error": -1.4,
    "broke": -1.6, "broken": -1.7, "crash": -1.8, "down": -1.2, "outage": -2.0,
    "delay": -1.3, "delayed": -1.4, "blocked": -1.5, "stuck": -1.4,
    "regression": -1.6, "leak": -1.4, "vulnerable": -1.5, "insecure": -1.5,
    "frustrated": -1.9, "frustrating": -1.9, "annoyed": -1.7, "annoying": -1.8,
    "angry": -2.3, "upset": -1.9, "disappointed": -2.0, "disappointing": -2.0,
    "sad": -1.9, "unhappy": -1.9, "miserable": -2.4, "depressed": -2.5,
    "anxious": -1.8, "anxiety": -1.8, "nervous": -1.4, "stress": -1.7,
    "stressed": -1.8, "overwhelmed": -1.9, "pressure": -1.1, "burnout": -2.3,
    "afraid": -1.9, "scared": -1.9, "terrified": -2.5, "fear": -1.8,
    "hate": -2.6, "hated": -2.5, "dislike": -1.6, "awful": -2.6,
    "terrible": -2.5, "horrible": -2.5, "dreadful": -2.5, "worst": -2.7,
    "bad": -2.0, "poor": -1.9, "weak": -1.4, "wrong": -1.5, "mistake": -1.4,
    "missed": -1.2, "miss": -1.1, "lose": -1.6, "lost": -1.5, "loss": -1.7,
    "reject": -1.9, "rejected": -2.0, "blame": -1.6, "conflict": -1.3,
    "argue": -1.3, "argument": -1.4, "disagree": -1.0, "tense": -1.5,
    "awkward": -1.1, "boring": -1.5, "waste": -1.8, "wasted": -1.8,
    "useless": -2.0, "pointless": -1.8, "impossible": -1.6, "never": -0.8,
    "quit": -1.3, "fired": -2.3, "layoff": -2.2, "toxic": -2.4, "painful": -1.9,
    "suffer": -2.0, "struggle": -1.5, "struggling": -1.6, "harder": -0.9,
    # intensifiers are handled separately but appear here for completeness
}

#: words that scale the following valence
INTENSIFIERS: Dict[str, float] = {
    "very": 1.35, "really": 1.3, "extremely": 1.5, "incredibly": 1.5,
    "absolutely": 1.5, "completely": 1.4, "totally": 1.35, "utterly": 1.45,
    "highly": 1.3, "deeply": 1.3, "truly": 1.25, "genuinely": 1.25,
    "seriously": 1.25, "so": 1.2, "such": 1.15, "particularly": 1.2,
    "especially": 1.2, "exceptionally": 1.45, "remarkably": 1.35,
    "super": 1.35, "quite": 0.85, "rather": 0.9, "somewhat": 0.8,
    "slightly": 0.7, "a bit": 0.75, "a little": 0.75, "kind of": 0.8,
    "sort of": 0.8, "fairly": 0.85, "pretty": 0.9, "mostly": 0.95,
}

#: negators flip and damp the valence of the following words within a window
NEGATORS: frozenset = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "cannot", "cant", "can't", "wont", "won't", "dont", "don't", "doesnt",
    "doesn't", "didnt", "didn't", "isnt", "isn't", "arent", "aren't",
    "wasnt", "wasn't", "werent", "weren't", "hasnt", "hasn't", "havent",
    "haven't", "hadnt", "hadn't", "wouldnt", "wouldn't", "couldnt", "couldn't",
    "shouldnt", "shouldn't", "without", "lack", "lacks", "lacking", "absent",
    "hardly", "barely", "scarcely", "rarely", "seldom",
})

#: emotion lexicon: word -> {emotion: weight}
EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
    "love": {"joy": 0.9}, "loved": {"joy": 0.85}, "happy": {"joy": 0.9},
    "delighted": {"joy": 0.95}, "thrilled": {"joy": 0.95, "surprise": 0.4},
    "excited": {"joy": 0.8, "anticipation": 0.7}, "great": {"joy": 0.7},
    "wonderful": {"joy": 0.85}, "proud": {"joy": 0.8, "trust": 0.4},
    "enjoy": {"joy": 0.8}, "glad": {"joy": 0.7}, "pleased": {"joy": 0.7},
    "grateful": {"joy": 0.6, "trust": 0.5}, "optimistic": {"joy": 0.6, "anticipation": 0.6},
    "confident": {"trust": 0.7, "joy": 0.3}, "trust": {"trust": 0.9},
    "reliable": {"trust": 0.8}, "supportive": {"trust": 0.8}, "collaborate": {"trust": 0.6},
    "respect": {"trust": 0.8}, "honest": {"trust": 0.75}, "safe": {"trust": 0.6},
    "welcome": {"trust": 0.6, "joy": 0.4}, "appreciate": {"trust": 0.6, "joy": 0.5},
    "afraid": {"fear": 0.9}, "scared": {"fear": 0.9}, "terrified": {"fear": 1.0},
    "nervous": {"fear": 0.7}, "anxious": {"fear": 0.8}, "anxiety": {"fear": 0.8},
    "worried": {"fear": 0.8}, "worry": {"fear": 0.75}, "concerned": {"fear": 0.7},
    "risky": {"fear": 0.6}, "unsure": {"fear": 0.5}, "doubt": {"fear": 0.55},
    "threat": {"fear": 0.9}, "panic": {"fear": 1.0}, "overwhelmed": {"fear": 0.85, "sadness": 0.4},
    "surprised": {"surprise": 0.9}, "surprising": {"surprise": 0.8},
    "unexpected": {"surprise": 0.7}, "shocked": {"surprise": 0.95},
    "wow": {"surprise": 0.8, "joy": 0.4}, "sudden": {"surprise": 0.5},
    "interesting": {"surprise": 0.4, "anticipation": 0.3}, "curious": {"surprise": 0.4, "anticipation": 0.5},
    "sad": {"sadness": 0.9}, "unhappy": {"sadness": 0.85}, "disappointed": {"sadness": 0.85},
    "depressed": {"sadness": 1.0}, "miserable": {"sadness": 0.95},
    "lonely": {"sadness": 0.8}, "hopeless": {"sadness": 0.9},
    "regret": {"sadness": 0.7}, "miss": {"sadness": 0.5}, "lost": {"sadness": 0.5},
    "failed": {"sadness": 0.7}, "failure": {"sadness": 0.7}, "rejected": {"sadness": 0.85},
    "burnout": {"sadness": 0.8, "fear": 0.3},
    "angry": {"anger": 0.9}, "annoyed": {"anger": 0.7}, "furious": {"anger": 1.0},
    "frustrated": {"anger": 0.8, "sadness": 0.3}, "irritated": {"anger": 0.75},
    "mad": {"anger": 0.85}, "hate": {"anger": 0.9, "disgust": 0.6},
    "blame": {"anger": 0.6}, "unfair": {"anger": 0.7, "disgust": 0.4},
    "toxic": {"anger": 0.7, "disgust": 0.8}, "argue": {"anger": 0.6},
    "disgusting": {"disgust": 0.95}, "gross": {"disgust": 0.8},
    "awful": {"disgust": 0.8, "anger": 0.4}, "terrible": {"disgust": 0.7, "anger": 0.4},
    "unacceptable": {"disgust": 0.7, "anger": 0.5}, "waste": {"disgust": 0.5},
    "excited": {"anticipation": 0.7}, "eager": {"anticipation": 0.8},
    "looking forward": {"anticipation": 0.9}, "hope": {"anticipation": 0.7},
    "plan": {"anticipation": 0.4}, "goal": {"anticipation": 0.5},
    "ready": {"anticipation": 0.5}, "upcoming": {"anticipation": 0.6},
    "soon": {"anticipation": 0.4}, "next": {"anticipation": 0.3},
}

#: emotion -> empathetic response tone
EMPATHY_TONE: Dict[str, str] = {
    "joy": "warm",
    "trust": "steady",
    "fear": "reassuring",
    "surprise": "engaged",
    "sadness": "supportive",
    "disgust": "grounded",
    "anger": "calm",
    "anticipation": "encouraging",
    "neutral": "neutral",
}


@dataclass
class SentimentAnalyzer:
    """Rule-based valence/emotion analyser."""

    lexicon: Dict[str, float] = field(default_factory=lambda: dict(VALENCE))
    negation_window: int = 3
    emotion_smoothing: float = 0.35

    # ------------------------------------------------------------------ #
    def analyze(self, text: str) -> Sentiment:
        """Compute polarity, magnitude and subjectivity for ``text``."""
        if not text or not text.strip():
            return Sentiment()
        tokens = [t.lower() for t in tokenize(text, keep_punct=False)]
        if not tokens:
            return Sentiment()

        total = 0.0
        hits = 0
        negations = 0
        for i, tok in enumerate(tokens):
            valence = self.lexicon.get(tok)
            if valence is None:
                continue
            hits += 1
            # intensifier / dampener immediately before the word
            scale = 1.0
            for j in range(max(0, i - 2), i):
                if tokens[j] in INTENSIFIERS:
                    scale *= INTENSIFIERS[tokens[j]]
            # negation in a preceding window
            negated = any(tokens[j] in NEGATORS for j in range(max(0, i - self.negation_window), i))
            if negated:
                negations += 1
                valence = -valence * 0.74
            total += valence * scale

        # punctuation / capitalisation emphasis
        exclamations = text.count("!")
        if exclamations:
            total *= 1.0 + min(0.30, 0.08 * exclamations)
        caps_words = sum(1 for t in tokenize(text, keep_punct=False) if t.isupper() and len(t) > 2)
        if caps_words:
            total *= 1.0 + min(0.25, 0.06 * caps_words)

        # normalise with a saturating function (VADER style)
        norm = total / math.sqrt(total * total + 15.0) if total else 0.0
        magnitude = min(1.0, abs(total) / 6.0)
        subjectivity = min(1.0, hits / max(4, len(tokens) * 0.35)) if hits else 0.0
        if norm > 0.15:
            label = "positive"
        elif norm < -0.15:
            label = "negative"
        else:
            label = "neutral"
        return Sentiment(
            polarity=round(max(-1.0, min(1.0, norm)), 4),
            magnitude=round(magnitude, 4),
            subjectivity=round(subjectivity, 4),
            label=label,
        )

    # ------------------------------------------------------------------ #
    def emotions(self, text: str) -> EmotionScores:
        """Activate the eight Plutchik emotions for ``text``."""
        low = text.lower()
        tokens = [t.lower() for t in tokenize(text, keep_punct=False)]
        scores: Dict[str, float] = {k: 0.0 for k in EmotionScores().__dataclass_fields__}  # type: ignore[attr-defined]

        for phrase, mapping in EMOTION_LEXICON.items():
            weight = 0.0
            if " " in phrase:
                weight = 1.0 if phrase in low else 0.0
            else:
                weight = 1.0 if phrase in tokens else 0.0
            if weight <= 0:
                continue
            negated = any(t in NEGATORS for t in tokens[max(0, tokens.index(phrase) - 3):tokens.index(phrase)]) if " " not in phrase else False
            for emotion, value in mapping.items():
                scores[emotion] += weight * value * (0.35 if negated else 1.0)

        # polarity from the sentiment analyser feeds joy/sadness
        sent = self.analyze(text)
        if sent.polarity > 0.2:
            scores["joy"] += sent.polarity * 0.8
        elif sent.polarity < -0.2:
            scores["sadness"] += abs(sent.polarity) * 0.6
            scores["anger"] += abs(sent.polarity) * 0.3
        # uncertainty cues feed fear
        for cue in ("not sure", "unsure", "unclear", "maybe", "perhaps", "i guess"):
            if cue in low:
                scores["fear"] += 0.25
        # urgency cues feed anticipation
        for cue in ("asap", "urgent", "deadline", "soon", "quickly", "immediately"):
            if cue in low:
                scores["anticipation"] += 0.3

        # saturate into [0, 1]
        return EmotionScores(**{k: round(min(1.0, v), 4) for k, v in scores.items()})

    # ------------------------------------------------------------------ #
    def empathy_tone(self, text: str) -> Tuple[str, float]:
        """Return ``(tone, intensity)`` the responder should adopt."""
        emotions = self.emotions(text)
        intensity = emotions.intensity()
        dominant = emotions.dominant() if intensity >= 0.15 else "neutral"
        tone = EMPATHY_TONE.get(dominant, "steady")
        return tone, round(intensity, 4)

    def emotional_volatility(self, texts: Sequence[str]) -> float:
        """Variance of polarity across a sequence of turns (0 = stable)."""
        if len(texts) < 2:
            return 0.0
        pols = [self.analyze(t).polarity for t in texts]
        mean = sum(pols) / len(pols)
        var = sum((p - mean) ** 2 for p in pols) / len(pols)
        return round(math.sqrt(var), 4)


#: module level singleton used by the pipeline
_ANALYZER = SentimentAnalyzer()


def analyze_sentiment(text: str) -> Sentiment:
    return _ANALYZER.analyze(text)


def analyze_emotion(text: str) -> EmotionScores:
    return _ANALYZER.emotions(text)
