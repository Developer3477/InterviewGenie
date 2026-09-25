"""Response templates and the surface realiser.

Generation is a three-stage NLG pipeline:

1. **text planning** -- :class:`ResponsePlanner` picks the content plan
   (which evidence, which story, which angle) for the question's intent,
2. **sentence planning** -- each plan step is turned into a proposition,
3. **surface realisation** -- :class:`SurfaceRealiser` renders the propositions
   into fluent text in the requested register.

Keeping these stages separate is what makes the output steerable: the emotional
intelligence module changes the register, the personalisation module changes the
parameters, and the validator can reject a plan before any text exists.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..logging import get_logger
from ..types import Analysis, Evidence, Intent, Response

LOG = get_logger("generation.templates")


# --------------------------------------------------------------------------- #
# Openers / closers per register and emotion
# --------------------------------------------------------------------------- #
OPENERS: Dict[str, List[str]] = {
    "warm": ["Great question.", "I love this one.", "That is a great thing to dig into."],
    "steady": ["Sure, let me walk you through it.", "Happy to.", "Let me take that directly."],
    "reassuring": ["That is a fair thing to ask.", "I appreciate you pressing on that.",
                   "Good question, and I want to be precise about it."],
    "engaged": ["Interesting angle.", "That is a good one to think about.",
                "I have not been asked that before, so let me think it through."],
    "supportive": ["I appreciate you asking.", "Thanks for raising it.",
                   "That one still matters to me."],
    "grounded": ["I understand the concern.", "Fair challenge.", "Let me be straight about it."],
    "calm": ["That is a legitimate frustration.", "I hear you.", "Let me take that head on."],
    "encouraging": ["Good question to be thinking about.", "I am glad you asked.",
                    "That is the direction I want to grow in."],
    "professional": ["Good question.", "Let me answer that directly.",
                     "Happy to walk you through it."],
}

CLOSERS: Dict[str, List[str]] = {
    "warm": ["I would love to keep that momentum going.", "Happy to go deeper on any part."],
    "steady": ["Happy to expand on any part of that.", "Let me know what else would help."],
    "reassuring": ["Happy to slow down on any part of that.",
                   "Tell me if you want me to unpack a piece."],
    "engaged": ["Did you want me to dig into that part?", "Curious what your take is."],
    "supportive": ["That experience still shapes how I work.", "I am glad to talk it through."],
    "grounded": ["I would rather be straight about the trade-offs.",
                 "Happy to share the numbers behind that."],
    "calm": ["Happy to take the detail offline.", "I would rather show than tell here."],
    "encouraging": ["That is exactly where I want to grow next.",
                    "Happy to talk about how I am building towards it."],
    "professional": ["Happy to go deeper on any part of that.",
                     "Let me know if you would like more detail."],
}

HEDGES: Dict[str, List[str]] = {
    "low": ["", "", ""],
    "medium": ["I would say", "In my experience", "My read is", "From what I have seen"],
    "high": ["I believe", "It seems to me that", "I could be wrong, but",
             "Roughly speaking"],
}

TRANSITIONS: List[str] = [
    "Concretely,", "In practice,", "The way I approach it is:",
    "What I did was", "So the plan was", "The key decision was",
]


# --------------------------------------------------------------------------- #
# Content plans per intent
# --------------------------------------------------------------------------- #
@dataclass
class PlanStep:
    """One unit of content to be realised as a sentence."""

    kind: str                       # headline | evidence | story | reasoning | tradeoff | close
    payload: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload, "optional": self.optional}


INTENT_PLANS: Dict[str, List[str]] = {
    "self_introduction": ["headline", "evidence", "evidence", "reasoning", "close"],
    "past_experience": ["headline", "story", "reasoning", "tradeoff", "close"],
    "self_assessment": ["headline", "evidence", "reasoning", "close"],
    "people": ["headline", "story", "reasoning", "tradeoff", "close"],
    "hypothetical": ["headline", "reasoning", "tradeoff", "close"],
    "knowledge": ["headline", "evidence", "reasoning", "tradeoff", "close"],
    "design": ["headline", "reasoning", "evidence", "tradeoff", "close"],
    "coding": ["headline", "reasoning", "close"],
    "troubleshooting": ["headline", "reasoning", "evidence", "close"],
    "background": ["headline", "evidence", "reasoning", "close"],
    "motivation": ["headline", "reasoning", "evidence", "close"],
    "goals": ["headline", "reasoning", "close"],
    "culture": ["headline", "reasoning", "evidence", "close"],
    "measurement": ["headline", "reasoning", "evidence", "close"],
    "logistics": ["headline", "close"],
    "meta": ["headline", "close"],
}


@dataclass
class ResponsePlanner:
    """Turn evidence + state into an ordered content plan."""

    rng: random.Random = field(default_factory=lambda: random.Random(11))

    def plan(self, intent_name: str, strategy: str, evidence: Sequence[Evidence],
             profile: Any = None, stories: Optional[Sequence[Dict[str, str]]] = None,
             memory: Optional[Sequence[str]] = None) -> List[PlanStep]:
        steps: List[PlanStep] = []
        kinds = INTENT_PLANS.get(intent_name, ["headline", "reasoning", "close"])

        if strategy == "clarify":
            return [PlanStep("clarify"), PlanStep("close")]
        if strategy == "question_back":
            kinds = kinds + ["question_back"]

        evidence_list = list(evidence)
        story_list = list(stories or [])
        memory_list = list(memory or [])

        for index, kind in enumerate(kinds):
            step = PlanStep(kind=kind)
            if kind == "evidence":
                if evidence_list:
                    step.payload = {"evidence": evidence_list.pop(0)}
                else:
                    step.optional = True
            elif kind == "story":
                if story_list:
                    step.payload = {"story": story_list.pop(0)}
                elif evidence_list:
                    step.payload = {"evidence": evidence_list.pop(0)}
                    step.optional = True
                else:
                    step.optional = True
            elif kind == "reasoning":
                step.payload = {"memory": memory_list[:1]}
            steps.append(step)
        return [s for s in steps if not s.optional or s.kind in {"headline", "close"}]


# --------------------------------------------------------------------------- #
# Surface realisation
# --------------------------------------------------------------------------- #
@dataclass
class SurfaceRealiser:
    """Render a content plan into text in the requested register."""

    rng: random.Random = field(default_factory=lambda: random.Random(5))

    def realise(self, steps: Sequence[PlanStep], *, register: str = "professional",
                hedging: str = "medium", max_words: int = 90,
                opener: str = "", closer: str = "",
                lead_with_headline: bool = False,
                numbers: bool = True) -> Tuple[str, List[str]]:
        """Return ``(text, plan_labels)``."""
        sentences: List[str] = []
        labels: List[str] = []

        if opener:
            sentences.append(opener)

        for step in steps:
            sentence = self._realise_step(step, register=register, hedging=hedging,
                                          numbers=numbers)
            if sentence:
                sentences.append(sentence)
                labels.append(step.kind)

        if closer:
            sentences.append(closer)
            labels.append("closer")

        text = " ".join(s for s in sentences if s)
        text = _tidy(text)
        if max_words and len(text.split()) > max_words:
            text = _truncate_to_words(text, max_words)
        return text, labels

    # -- per-step --------------------------------------------------------- #
    def _realise_step(self, step: PlanStep, *, register: str, hedging: str,
                      numbers: bool) -> str:
        kind = step.kind
        payload = step.payload

        if kind == "headline":
            return str(payload.get("text", "")).strip()
        if kind == "clarify":
            return str(payload.get("text", "")).strip()
        if kind == "question_back":
            return str(payload.get("text", "")).strip()
        if kind == "evidence":
            evidence = payload.get("evidence")
            if evidence is None:
                return ""
            return self._realise_evidence(evidence, hedging=hedging, numbers=numbers)
        if kind == "story":
            story = payload.get("story")
            if story is None:
                return ""
            return self._realise_story(story)
        if kind == "reasoning":
            memory = payload.get("memory") or []
            return str(memory[0]).strip() if memory else ""
        if kind == "tradeoff":
            return str(payload.get("text", "")).strip()
        if kind == "close":
            return str(payload.get("text", "")).strip()
        return ""

    def _realise_evidence(self, evidence: Evidence, *, hedging: str,
                          numbers: bool) -> str:
        text = evidence.text.strip()
        if not text:
            return ""
        # kg paths read better when described rather than dumped
        if evidence.kind == "kg_path" and "->" in text:
            parts = [p.strip() for p in text.split("->")]
            if len(parts) >= 2:
                head = parts[0]
                rest = ", ".join(parts[1:-1]) if len(parts) > 2 else ""
                tail = parts[-1]
                if rest:
                    return f"{head} relates to {rest} and ultimately to {tail}."
                return f"{head} connects directly to {tail}."
        if evidence.kind == "fact":
            return f"To ground that: {text}."
        hedge = self._pick(HEDGES.get(hedging, HEDGES["medium"]))
        return f"{hedge} {text}".replace("  ", " ").strip()

    def _realise_story(self, story: Dict[str, str]) -> str:
        title = story.get("title", "").strip()
        situation = story.get("situation", "").strip()
        action = story.get("action", "").strip()
        result = story.get("result", "").strip()
        if not (title or situation):
            return ""
        parts: List[str] = []
        if title:
            parts.append(f"A good example is {title}.")
        if situation:
            parts.append(f"{situation.rstrip('.')}.")
        if action:
            parts.append(f"What I did: {action.rstrip('.')}.")
        if result:
            parts.append(f"The outcome: {result.rstrip('.')}.")
        return " ".join(parts)

    def _pick(self, options: Sequence[str]) -> str:
        options = [o for o in options if o]
        return self.rng.choice(options) if options else ""


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.!?])\1+", r"\1", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    # cut at the last sentence boundary that fits
    truncated = " ".join(words[:max_words])
    boundary = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if boundary > len(truncated) * 0.5:
        return truncated[: boundary + 1]
    return truncated.rstrip(",;: ") + "."


def count_words(text: str) -> int:
    return len(text.split())


def reading_seconds(text: str, wpm: int = 150) -> float:
    return round(count_words(text) / wpm * 60.0, 2)
