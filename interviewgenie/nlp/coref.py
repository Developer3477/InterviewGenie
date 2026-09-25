"""Light-weight anaphora / coreference resolution.

Interviewer questions routinely refer back to earlier turns ("*it* was slow,
how would you fix **that**?").  A full neural coreference model is out of scope
for a dependency-free deployment, so this module implements a classical
Hobbs-style scorer:

* candidate antecedents are entity mentions and noun phrases from the current
  and previous turns,
* candidates are filtered by gender / number agreement with the pronoun,
* survivors are ranked by recency, grammatical role and sentence distance.

It resolves the cases that actually occur in interviews while degrading
gracefully (returning ``None``) rather than guessing wildly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..logging import get_logger
from ..types import Entity
from .textnorm import Tokenizer, tokenize

LOG = get_logger("nlp.coref")

#: pronoun -> (number, gender)
PRONOUN_INFO: Dict[str, Tuple[str, Optional[str]]] = {
    "he": ("sg", "m"), "him": ("sg", "m"), "his": ("sg", "m"), "himself": ("sg", "m"),
    "she": ("sg", "f"), "her": ("sg", "f"), "hers": ("sg", "f"), "herself": ("sg", "f"),
    "it": ("sg", None), "its": ("sg", None), "itself": ("sg", None),
    "they": ("pl", None), "them": ("pl", None), "their": ("pl", None),
    "theirs": ("pl", None), "themselves": ("pl", None),
    "we": ("pl", None), "us": ("pl", None), "our": ("pl", None),
    "you": ("sg/pl", None), "i": ("sg", None), "me": ("sg", None), "my": ("sg", None),
}

#: words whose presence marks a referent as feminine / masculine
FEMININE_CUES = frozenset({"she", "her", "hers", "woman", "girl", "mother", "sister",
                           "daughter", "mrs", "ms", "miss", "queen", "actress"})
MASCULINE_CUES = frozenset({"he", "him", "his", "man", "boy", "father", "brother",
                            "son", "mr", "sir", "king", "actor"})
PLURAL_CUES = frozenset({"they", "them", "their", "we", "us", "our", "these", "those"})
PLURAL_SUFFIXES = ("s", "es", "ies")

_NP_PATTERN = re.compile(
    r"\b(?:the\s+|a\s+|an\s+|this\s+|that\s+|these\s+|those\s+|my\s+|our\s+|their\s+)?"
    r"([A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,3})",
)


@dataclass
class Mention:
    """A candidate antecedent."""

    text: str
    kind: str                      # entity | noun_phrase | person
    number: str = "sg"
    gender: Optional[str] = None
    turn_index: int = 0
    position: int = 0              # token position inside the turn
    role: str = "object"           # subject | object
    score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {"text": self.text, "kind": self.kind, "number": self.number,
                "gender": self.gender, "turn_index": self.turn_index, "score": round(self.score, 3)}


@dataclass
class CoreferenceResolver:
    """Rank-and-filter coreference resolution."""

    max_back_turns: int = 4
    gender_window: int = 6

    def resolve(self, text: str, history: Sequence[str]) -> List[Tuple[str, str]]:
        """Return ``[(pronoun, antecedent)]`` pairs found in ``text``."""
        pronouns = self._pronouns(text)
        if not pronouns:
            return []
        candidates = self._candidates(history)
        results: List[Tuple[str, str]] = []
        used: set[str] = set()
        for pronoun in pronouns:
            best = self._best_antecedent(pronoun, candidates, text)
            if best is not None and best.text not in used:
                results.append((pronoun, best.text))
                used.add(best.text)
        return results

    # ------------------------------------------------------------------ #
    def _pronouns(self, text: str) -> List[str]:
        return [t.lower() for t in tokenize(text, keep_punct=False) if t.lower() in PRONOUN_INFO]

    def _candidates(self, history: Sequence[str]) -> List[Mention]:
        mentions: List[Mention] = []
        for turn_index, turn in enumerate(history[-self.max_back_turns:]):
            relative_index = len(history) - len(history[-self.max_back_turns:]) + turn_index
            tokens = tokenize(turn, keep_punct=False)
            # entities and capitalised runs
            for match in _NP_PATTERN.finditer(turn):
                span = match.group(1).strip()
                if not span or span.lower() in {"i", "you", "we", "they"}:
                    continue
                words = span.split()
                if not words or not words[0][:1].isalpha():
                    continue
                lowered = span.lower()
                if lowered in PRONOUN_INFO:
                    continue
                gender = "f" if lowered in FEMININE_CUES else ("m" if lowered in MASCULINE_CUES else None)
                number = "pl" if (len(words) > 1 and words[-1].endswith(PLURAL_SUFFIXES)) else "sg"
                if lowered in PLURAL_CUES or any(w.lower() in PLURAL_CUES for w in words):
                    number = "pl"
                mentions.append(Mention(
                    text=span, kind="noun_phrase", number=number, gender=gender,
                    turn_index=relative_index, position=turn.lower().find(lowered),
                    role="subject" if match.start() <= len(turn) * 0.3 else "object",
                ))
            # scan for gender cues in the whole turn
            for cue in FEMININE_CUES & set(t.lower() for t in tokens):
                for mention in mentions:
                    if mention.turn_index == relative_index and mention.gender is None:
                        mention.gender = "f"
            for cue in MASCULINE_CUES & set(t.lower() for t in tokens):
                for mention in mentions:
                    if mention.turn_index == relative_index and mention.gender is None:
                        mention.gender = "m"
            for cue in PLURAL_CUES & set(t.lower() for t in tokens):
                for mention in mentions:
                    if mention.turn_index == relative_index and mention.number == "sg":
                        mention.number = "pl"
        return mentions

    def _best_antecedent(self, pronoun: str, candidates: Sequence[Mention],
                         current_text: str) -> Optional[Mention]:
        number, gender = PRONOUN_INFO.get(pronoun, ("sg", None))
        compatible: List[Mention] = []
        for cand in candidates:
            if cand.number == "sg/pl":
                continue
            if number != "sg/pl":
                if cand.number != number:
                    continue
            if gender is not None and cand.gender is not None and cand.gender != gender:
                continue
            compatible.append(cand)
        if not compatible:
            return None

        total_turns = max((c.turn_index for c in candidates), default=0) + 1
        for cand in compatible:
            score = 0.0
            # recency dominates
            score += 2.2 * (1.0 - (total_turns - 1 - cand.turn_index) / max(1, total_turns))
            # grammatical role: subjects are preferred antecedents
            score += 0.8 if cand.role == "subject" else 0.35
            # shorter mentions are more salient
            score += max(0.0, 0.6 - 0.05 * len(cand.text.split()))
            # proximity to the pronoun inside the turn
            score += 0.4 * (1.0 - min(1.0, cand.position / max(1, len(current_text))))
            # capitalised entities are more salient
            if cand.text[:1].isupper():
                score += 0.3
            cand.score = score
        compatible.sort(key=lambda c: c.score, reverse=True)
        return compatible[0]


#: module level singleton
_RESOLVER = CoreferenceResolver()


def resolve(text: str, history: Sequence[str]) -> List[Tuple[str, str]]:
    return _RESOLVER.resolve(text, history)
