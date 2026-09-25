"""Continuous learning: feedback intake, online model updates and metric tracking.

Three loops run during and after an interview:

* **feedback loop** -- explicit signals from the candidate ("too long", "that
  worked") and implicit signals (did they accept, edit or reject the suggestion?)
  are normalised into a reward and fed to the style model and the dialogue bandit,
* **online loop** -- the intent classifier and the knowledge-graph edge weights
  are updated incrementally with ``partial_fit``-style updates, so the system
  improves on this interviewer's vocabulary without retraining from scratch, and
* **metrics loop** -- :class:`LearningMetrics` keeps the running evaluation
  numbers (response accuracy, relevance, engagement, personalisation, error rate)
  and can report whether the system is actually improving.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ..errors import LearningError
from ..logging import get_logger
from ..nlp.intent import IntentClassifier
from ..nlp.textnorm import similarity
from ..types import Analysis, Response, Scorecard, Turn

LOG = get_logger("learning.online")

#: positive feedback phrases
POSITIVE = ("perfect", "great", "excellent", "that worked", "love it", "use that",
            "exactly", "yes", "good", "nice", "keep it", "more like this")
#: negative feedback phrases -> what to change
NEGATIVE = {
    "too long": "shorten", "too verbose": "shorten", "shorter": "shorten",
    "too short": "expand", "more detail": "expand", "longer": "expand",
    "too formal": "relax", "too casual": "formalise", "too vague": "concretise",
    "too technical": "simplify", "not technical enough": "deepen",
    "too much detail": "simplify", "more numbers": "quantify",
    "less hedging": "be direct", "more enthusiasm": "energise",
    "calmer": "calm down", "more about the team": "people focus",
    "less about me": "less personal", "not what i asked": "refocus",
    "wrong": "refocus", "no": "refocus",
}


@dataclass
class FeedbackSignal:
    """One normalised piece of feedback."""

    raw: str
    reward: float
    action: str = ""
    source: str = "explicit"          # explicit | implicit | automatic
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"raw": self.raw, "reward": round(self.reward, 4),
                "action": self.action, "source": self.source,
                "created_at": self.created_at}


def normalise_feedback(text: str) -> Optional[FeedbackSignal]:
    """Turn free-text feedback into a reward and an action."""
    low = text.lower().strip()
    if not low:
        return None
    if any(phrase in low for phrase in POSITIVE):
        return FeedbackSignal(text, 1.0, "reinforce", "explicit")
    for phrase, action in NEGATIVE.items():
        if phrase in low:
            severity = 0.75 if any(w in low for w in ("much", "way", "really", "very")) else 0.5
            return FeedbackSignal(text, -severity, action, "explicit")
    # a near-duplicate of an earlier question is a signal to refocus
    return FeedbackSignal(text, 0.0, "observe", "explicit")


@dataclass
class OnlineLearner:
    """Incrementally updates the intent classifier, KG weights and style model."""

    intent_classifier: Optional[IntentClassifier] = None
    retriever: Any = None
    profile: Any = None
    dialogue: Any = None
    feedback_history: Deque[FeedbackSignal] = field(
        default_factory=lambda: deque(maxlen=200))
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    kg_updates: int = 0
    model_updates: int = 0

    # -- explicit feedback ------------------------------------------------ #
    def learn(self, feedback: str, turn: Optional[Turn] = None) -> Dict[str, Any]:
        signal = normalise_feedback(feedback)
        if signal is None:
            raise LearningError("empty feedback")
        self.feedback_history.append(signal)
        applied: Dict[str, Any] = {"reward": signal.reward, "action": signal.action}

        if self.profile is not None and signal.action:
            applied["style_changes"] = self.profile.apply_feedback(feedback)

        if turn is not None and self.dialogue is not None:
            self.dialogue.update_policy(turn.strategy or "direct", max(0.0, signal.reward))
            applied["policy_updated"] = True

        if turn is not None and turn.analysis is not None and signal.reward < 0:
            # a negative signal is a supervised correction for the classifier
            self._record_correction(turn, feedback)

        if turn is not None and turn.response is not None and self.retriever is not None:
            nodes = [e.node_id for e in turn.analysis.entities if e.node_id] \
                if turn.analysis else []
            if nodes:
                self.retriever.record_usage(nodes, reward=0.05 * signal.reward)
                self.kg_updates += 1
                applied["kg_reinforced"] = len(nodes)

        LOG.info("feedback applied", context={"reward": signal.reward,
                                              "action": signal.action})
        return applied

    # -- implicit feedback ------------------------------------------------- #
    def learn_from_outcome(self, turn: Turn, accepted: bool, edited: bool = False,
                           rejected: bool = False) -> Dict[str, Any]:
        """Learn from what the candidate actually did with the suggestion."""
        if rejected:
            reward = -0.6
        elif edited:
            reward = -0.2
        elif accepted:
            reward = 0.8
        else:
            reward = 0.0
        signal = FeedbackSignal(turn.text, reward,
                                "accept" if accepted else ("edit" if edited else "reject"),
                                "implicit")
        self.feedback_history.append(signal)

        if self.dialogue is not None:
            self.dialogue.update_policy(turn.strategy or "direct", max(0.0, reward))
        if self.profile is not None and turn.response is not None:
            self.profile.observe_answer(turn.response.text, reward=max(0.0, reward))
        if self.retriever is not None and turn.analysis is not None:
            nodes = [e.node_id for e in turn.analysis.entities if e.node_id]
            if nodes and reward != 0:
                self.retriever.record_usage(nodes, reward=0.04 * reward)
                self.kg_updates += 1
        return {"reward": reward, "accepted": accepted, "edited": edited,
                "rejected": rejected}

    # -- supervised corrections -------------------------------------------- #
    def _record_correction(self, turn: Turn, feedback: str) -> None:
        analysis = turn.analysis
        if analysis is None or analysis.intent is None:
            return
        corrected = self._infer_intent_from_feedback(feedback)
        if not corrected:
            return
        self.corrections.append({
            "text": turn.text, "from": analysis.intent.name, "to": corrected,
            "feedback": feedback, "created_at": time.time(),
        })
        if self.intent_classifier is not None:
            try:
                self.intent_classifier.learn(analysis.text, corrected,
                                             analysis.intent.topic)
                self.model_updates += 1
            except Exception as exc:  # noqa: BLE001
                LOG.warning("could not update the intent model: %s", exc)

    @staticmethod
    def _infer_intent_from_feedback(feedback: str) -> Optional[str]:
        low = feedback.lower()
        if "behavioural" in low or "star" in low or "tell me about a time" in low:
            return "past_experience"
        if "design" in low or "architecture" in low:
            return "design"
        if "code" in low or "algorithm" in low or "complexity" in low:
            return "coding"
        if "salary" in low or "logistics" in low or "notice" in low:
            return "logistics"
        if "culture" in low:
            return "culture"
        if "why" in low and ("company" in low or "here" in low):
            return "motivation"
        return None

    # -- periodic consolidation -------------------------------------------- #
    def consolidate(self) -> Dict[str, Any]:
        """Apply pending corrections and rebuild derived state."""
        applied = {"corrections": len(self.corrections), "model_updates": self.model_updates,
                   "kg_updates": self.kg_updates}
        if self.intent_classifier is not None and self.corrections:
            try:
                self.intent_classifier.fit_from_dataset(use_fallback=False)
                applied["model_refitted"] = True
            except Exception as exc:  # noqa: BLE001
                LOG.warning("model refit failed: %s", exc)
        if self.retriever is not None:
            self.retriever.rebuild_indexes(force=True)
            applied["indexes_rebuilt"] = True
        return applied

    def stats(self) -> Dict[str, Any]:
        rewards = [f.reward for f in self.feedback_history]
        return {
            "signals": len(rewards),
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "corrections": len(self.corrections),
            "model_updates": self.model_updates,
            "kg_updates": self.kg_updates,
            "bandit": self.dialogue.bandit_stats() if self.dialogue else {},
        }


@dataclass
class LearningMetrics:
    """Running evaluation metrics with trend detection."""

    window: int = 50
    history: Deque[Scorecard] = field(default_factory=lambda: deque(maxlen=50))
    baseline: Optional[Scorecard] = None

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.window)

    def record(self, scorecard: Scorecard) -> None:
        self.history.append(scorecard)
        if self.baseline is None:
            self.baseline = scorecard

    # -- aggregates -------------------------------------------------------- #
    def means(self) -> Dict[str, float]:
        if not self.history:
            return {}
        keys = ["accuracy", "relevance", "engagement", "personalization",
                "groundedness", "fluency", "error_rate"]
        out: Dict[str, float] = {}
        for key in keys:
            values = [getattr(s, key) for s in self.history]
            out[key] = round(sum(values) / len(values), 4)
        out["overall"] = round(sum(s.overall() for s in self.history) / len(self.history), 4)
        return out

    def trend(self, key: str = "overall") -> Dict[str, Any]:
        """Compare the recent half of the window against the older half."""
        if len(self.history) < 4:
            return {"direction": "insufficient data", "samples": len(self.history)}
        values = ([s.overall() for s in self.history] if key == "overall"
                  else [getattr(s, key) for s in self.history])
        mid = len(values) // 2
        older = sum(values[:mid]) / mid
        recent = sum(values[mid:]) / (len(values) - mid)
        delta = recent - older
        direction = "improving" if delta > 0.01 else (
            "declining" if delta < -0.01 else "flat")
        return {"direction": direction, "delta": round(delta, 4),
                "older": round(older, 4), "recent": round(recent, 4),
                "samples": len(values)}

    def error_rate(self) -> float:
        if not self.history:
            return 0.0
        return round(sum(s.error_rate for s in self.history) / len(self.history), 4)

    def is_improving(self) -> bool:
        return self.trend()["direction"] == "improving"

    def to_dict(self) -> Dict[str, Any]:
        return {"samples": len(self.history), "means": self.means(),
                "trend": self.trend(), "error_rate": self.error_rate()}
