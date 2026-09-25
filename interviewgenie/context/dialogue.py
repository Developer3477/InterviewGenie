"""Multi-turn dialogue management: state, topics, memory and policy.

The dialogue manager owns the interview's evolving state:

* a **turn stack** of recent utterances with their analyses,
* a **slot store** filled from entities and answers (``role``, ``company``,
  ``skills``, ``last_project``, ...),
* a **topic stack** with exponential decay so the system knows which subject is
  current and when the interviewer has moved on,
* **coreference chains** so "it", "that project" and "the same approach" resolve
  to the right antecedent,
* a lightweight **bandit policy** that learns which response strategies work for
  this particular interviewer.

The policy is deliberately simple and inspectable: it picks a *strategy*
(direct answer, structured STAR, clarify, hedge, ask a question back) from the
current state, and the composer realises that strategy into text.
"""

from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..errors import ContradictionError, DialogueError
from ..logging import get_logger
from ..nlp.coref import CoreferenceResolver, Mention
from ..nlp.textnorm import tokenize
from ..types import Analysis, EmotionScores, Response, Sentiment, Turn

LOG = get_logger("context.dialogue")

STRATEGIES: Tuple[str, ...] = (
    "direct",            # answer plainly and concisely
    "structured",        # STAR / numbered structure
    "example_first",     # lead with a concrete example
    "clarify",           # ask the interviewer to disambiguate
    "bridge",            # answer, then steer back to a strength
    "question_back",     # end with a question for the interviewer
    "hedge",             # answer but flag uncertainty honestly
)

#: intent -> default strategy preferences (ordered)
STRATEGY_PRIORS: Dict[str, Tuple[str, ...]] = {
    "self_introduction": ("structured", "direct"),
    "past_experience": ("structured", "example_first"),
    "self_assessment": ("structured", "direct"),
    "people": ("structured", "example_first"),
    "hypothetical": ("structured", "direct"),
    "knowledge": ("direct", "structured"),
    "design": ("structured", "direct"),
    "coding": ("direct", "structured"),
    "troubleshooting": ("structured", "direct"),
    "background": ("structured", "direct"),
    "motivation": ("direct", "bridge"),
    "goals": ("direct", "structured"),
    "culture": ("direct", "structured"),
    "measurement": ("structured", "direct"),
    "logistics": ("direct",),
    "meta": ("direct",),
}


@dataclass
class DialogueState:
    """Everything the system remembers about the conversation so far."""

    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=60))
    slots: Dict[str, Any] = field(default_factory=dict)
    topics: List[Tuple[str, float]] = field(default_factory=list)
    emotion: EmotionScores = field(default_factory=EmotionScores)
    sentiment_history: List[float] = field(default_factory=list)
    current_strategy: str = "direct"
    interviewer_style: Dict[str, float] = field(default_factory=dict)
    open_questions: List[str] = field(default_factory=list)
    turn_index: int = 0

    # -- derived ---------------------------------------------------------- #
    @property
    def last_question(self) -> Optional[Turn]:
        for turn in reversed(self.turns):
            if turn.dialogue_act == "question":
                return turn
        return None

    @property
    def last_answer(self) -> Optional[Turn]:
        for turn in reversed(self.turns):
            if turn.dialogue_act == "answer":
                return turn
        return None

    def recent_texts(self, n: int = 6) -> List[str]:
        return [t.text for t in list(self.turns)[-n:]]

    def topic(self) -> Optional[str]:
        return self.topics[0][0] if self.topics else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "turns": len(self.turns),
            "slots": self.slots,
            "topics": [[t, round(s, 3)] for t, s in self.topics[:6]],
            "emotion": self.emotion.to_dict(),
            "sentiment_trend": [round(s, 3) for s in self.sentiment_history[-12:]],
            "current_strategy": self.current_strategy,
            "interviewer_style": {k: round(v, 3) for k, v in self.interviewer_style.items()},
            "open_questions": self.open_questions,
        }


@dataclass
class DialogueManager:
    """Owns dialogue state and decides the response strategy for each turn."""

    max_turns: int = 40
    topic_decay: float = 0.85
    emotion_smoothing: float = 0.35
    empathy_level: float = 0.7
    coref: CoreferenceResolver = field(default_factory=CoreferenceResolver)
    rng: random.Random = field(default_factory=lambda: random.Random(20260925))
    #: strategy -> (pulls, total reward) for the contextual bandit
    _bandit: Dict[str, List[float]] = field(default_factory=dict, repr=False)
    state: DialogueState = field(default_factory=DialogueState)

    def __post_init__(self) -> None:
        self.state = DialogueState()
        for strategy in STRATEGIES:
            self._bandit[strategy] = [0.0, 0.0]

    # -- ingestion -------------------------------------------------------- #
    def observe(self, turn: Turn) -> Dict[str, Any]:
        """Fold a new interviewer turn into the dialogue state."""
        self.state.turns.append(turn)
        self.state.turn_index += 1
        events: Dict[str, Any] = {}

        analysis = turn.analysis
        if analysis is None:
            return events

        # topic stack with decay
        self._update_topics(analysis)
        events["topic"] = self.state.topic()

        # slots
        self._update_slots(turn, analysis)
        events["slots"] = dict(self.state.slots)

        # emotion: exponential smoothing towards the newest reading
        if analysis.emotion is not None:
            self._update_emotion(analysis.emotion)
        if analysis.sentiment is not None:
            self.state.sentiment_history.append(analysis.sentiment.polarity)
            if len(self.state.sentiment_history) > 60:
                self.state.sentiment_history = self.state.sentiment_history[-60:]
        events["emotion"] = self.state.emotion.dominant()
        events["emotion_intensity"] = round(self.state.emotion.intensity(), 3)

        # interviewer style
        self._update_interviewer_style(analysis)

        # coreference against the running history
        if analysis.tokens:
            history = [t.text for t in list(self.state.turns)[:-1]]
            try:
                self.state.slots["coreferences"] = self.coref.resolve(turn.text, history)
            except Exception as exc:  # noqa: BLE001
                LOG.debug("coreference failed: %s", exc)

        # contradiction check against earlier answers
        contradictions = self._detect_contradictions(turn)
        if contradictions:
            events["contradictions"] = contradictions

        # A change of subject is normal in an interview, so it is recorded as
        # an event rather than raised as an error; only a genuine contradiction
        # is exceptional.
        previous = self.state.topics[1][0] if len(self.state.topics) > 1 else None
        current = self.state.topic()
        if previous and current and previous != current:
            events["topic_shift"] = {"from": previous, "to": current}
        return events

    def record_answer(self, turn: Turn, response: Response) -> None:
        """Attach a generated response to its question turn."""
        turn.response = response
        turn.dialogue_act = "answer"
        self.state.turns.append(turn)
        self.state.open_questions = [
            q for q in self.state.open_questions if q != turn.text
        ]

    def _update_topics(self, analysis: Analysis) -> None:
        decayed = [(topic, score * self.topic_decay) for topic, score in self.state.topics]
        boosted: Dict[str, float] = {t: s for t, s in decayed if s > 0.02}
        for term, score in analysis.keywords[:6]:
            boosted[term] = boosted.get(term, 0.0) + score * (1.0 - self.topic_decay)
        if analysis.intent is not None and analysis.intent.topic:
            key = analysis.intent.topic
            boosted[key] = boosted.get(key, 0.0) + 0.35
        ranked = sorted(boosted.items(), key=lambda kv: kv[1], reverse=True)
        self.state.topics = ranked[:12]

    def _update_slots(self, turn: Turn, analysis: Analysis) -> None:
        slots = self.state.slots
        if analysis.intent is not None:
            slots["last_intent"] = analysis.intent.name
            slots["last_topic"] = analysis.intent.topic
        for entity in analysis.entities:
            label = (entity.label or "").upper()
            if label in {"ORG", "COMPANY"}:
                slots.setdefault("companies_mentioned", [])
                if entity.text not in slots["companies_mentioned"]:
                    slots["companies_mentioned"].append(entity.text)
            elif label in {"SKILL", "TECH"}:
                slots.setdefault("skills_mentioned", [])
                if entity.text not in slots["skills_mentioned"]:
                    slots["skills_mentioned"].append(entity.text)
            elif label in {"PERSON"}:
                slots.setdefault("people_mentioned", [])
                if entity.text not in slots["people_mentioned"]:
                    slots["people_mentioned"].append(entity.text)
        if turn.dialogue_act == "question":
            self.state.open_questions.append(turn.text)
            self.state.open_questions = self.state.open_questions[-5:]

    def _update_emotion(self, emotion: EmotionScores) -> None:
        alpha = self.emotion_smoothing
        current = self.state.emotion
        for field_name in EmotionScores().__dataclass_fields__:  # type: ignore[attr-defined]
            new_value = getattr(emotion, field_name)
            old_value = getattr(current, field_name)
            setattr(current, field_name, round((1 - alpha) * old_value + alpha * new_value, 4))

    def _update_interviewer_style(self, analysis: Analysis) -> None:
        style = self.state.interviewer_style
        text = analysis.text.lower()
        words = max(1, len(analysis.tokens))
        style["verbosity"] = 0.9 * style.get("verbosity", 0.5) + 0.1 * min(1.0, words / 40.0)
        style["formality"] = 0.9 * style.get("formality", 0.5) + 0.1 * _formality(text)
        if analysis.sentiment is not None:
            style["warmth"] = 0.9 * style.get("warmth", 0.5) + 0.1 * (
                0.5 + analysis.sentiment.polarity / 2)
        if analysis.intent is not None:
            style["directness"] = 0.9 * style.get("directness", 0.5) + 0.1 * (
                0.8 if analysis.intent.name in {"coding", "knowledge", "logistics"} else 0.4)

    def _detect_contradictions(self, turn: Turn) -> List[Dict[str, Any]]:
        from ..knowledge.commonsense import CommonsenseEngine

        engine = CommonsenseEngine()
        previous = [t.text for t in list(self.state.turns)[-4:-1] if t.dialogue_act == "answer"]
        if not previous:
            return []
        return engine.contradictions(previous + [turn.text])

    # -- policy ----------------------------------------------------------- #
    def choose_strategy(self, analysis: Analysis) -> str:
        """Pick a response strategy using priors plus a contextual bandit."""
        intent = analysis.intent.name if analysis.intent else "knowledge"
        priors = STRATEGY_PRIORS.get(intent, ("direct",))

        # hard constraints from the current state
        if analysis.intent is not None and analysis.intent.requires_clarification:
            return "clarify"
        if intent == "meta" and analysis.text.lower().startswith(("could you repeat", "sorry")):
            return "clarify"
        if self.state.emotion.intensity() > 0.55 and self.state.emotion.dominant() in {
                "fear", "sadness", "anger"} and self.empathy_level > 0.5:
            if "structured" not in priors:
                priors = priors + ("structured",)

        candidates = list(priors) or ["direct"]
        # epsilon-greedy over the prior-ranked candidates
        if self.rng.random() < 0.12 and len(candidates) > 1:
            choice = self.rng.choice(candidates)
        else:
            choice = max(candidates, key=lambda s: self._expected_reward(s, intent))
        self.state.current_strategy = choice
        return choice

    def _expected_reward(self, strategy: str, intent: str) -> float:
        pulls, reward = self._bandit[strategy]
        if pulls < 2:
            return 0.5
        mean = reward / pulls
        # optimism bonus shrinks as we learn
        return mean + 0.25 / math.sqrt(pulls)

    def update_policy(self, strategy: str, reward: float) -> None:
        entry = self._bandit.setdefault(strategy, [0.0, 0.0])
        entry[0] += 1
        entry[1] += max(0.0, min(1.0, reward))

    def bandit_stats(self) -> Dict[str, Dict[str, float]]:
        return {
            strategy: {"pulls": pulls, "mean_reward": round(reward / pulls, 4) if pulls else 0.0}
            for strategy, (pulls, reward) in sorted(self._bandit.items())
        }

    # -- queries ---------------------------------------------------------- #
    def resolve_reference(self, text: str) -> List[Tuple[str, str]]:
        history = [t.text for t in list(self.state.turns)]
        return self.coref.resolve(text, history)

    def summary(self) -> Dict[str, Any]:
        return {
            "turns": self.state.turn_index,
            "strategy": self.state.current_strategy,
            "topic": self.state.topic(),
            "slots": {k: v for k, v in self.state.slots.items() if k != "coreferences"},
            "emotion": self.state.emotion.dominant(),
        }

    def reset(self) -> None:
        self.state = DialogueState()


def _formality(text: str) -> float:
    formal = {"would", "could", "shall", "regarding", "furthermore", "however",
              "therefore", "additionally", "sincerely"}
    informal = {"gonna", "wanna", "yeah", "cool", "stuff", "kinda", "hey", "ok"}
    tokens = {t.lower() for t in tokenize(text, keep_punct=False)}
    score = 0.5
    score += 0.1 * len(tokens & formal)
    score -= 0.15 * len(tokens & informal)
    return max(0.0, min(1.0, score))
