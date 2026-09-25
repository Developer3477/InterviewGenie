"""Core value objects shared across the whole pipeline.

Everything the system exchanges between stages is an immutable-ish dataclass
with a ``to_dict`` serialiser, which keeps the REST/WebSocket API and the
persistence layer trivial and makes the pipeline easy to test.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# Enumerations (plain strings so they serialise cleanly to JSON)
# --------------------------------------------------------------------------- #
class Speaker:
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    SYSTEM = "system"


class Phase:
    IDLE = "idle"
    PREPARING = "preparing"
    LISTENING = "listening"
    ANALYSING = "analysing"
    RETRIEVING = "retrieving"
    DIALOGUE = "dialogue"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    LEARNING = "learning"
    RECOVERING = "recovering"
    CLOSING = "closing"
    REPORTING = "reporting"
    PRE_INTERVIEW = "pre_interview"
    INTERVIEW = "interview"
    RESPONSE = "response"
    POST_RESPONSE = "post_response"
    RECOVERY = "recovery"


class Severity:
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# --------------------------------------------------------------------------- #
# Speech layer
# --------------------------------------------------------------------------- #
@dataclass
class Token:
    """A single recognised word plus timing information."""

    text: str
    start: float = 0.0
    end: float = 0.0
    confidence: float = 1.0
    speaker: str = Speaker.INTERVIEWER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptChunk:
    """One emission from the streaming speech recogniser."""

    text: str
    is_final: bool
    start: float = 0.0
    end: float = 0.0
    confidence: float = 0.0
    provider: str = "local"
    tokens: List[Token] = field(default_factory=list)
    chunk_id: str = field(default_factory=lambda: _new_id("chunk"))
    created_at: float = field(default_factory=_now)

    @property
    def word_count(self) -> int:
        return len([t for t in self.text.split() if t])

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "word_count": self.word_count,
        }


@dataclass
class AudioFrame:
    """A chunk of mono 16 kHz PCM audio."""

    samples: Sequence[float]
    sample_rate: int = 16000
    timestamp: float = field(default_factory=_now)
    channel: str = "interviewer"

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)


# --------------------------------------------------------------------------- #
# NLP layer
# --------------------------------------------------------------------------- #
@dataclass
class PosToken:
    text: str
    lemma: str
    tag: str
    index: int
    is_stop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Entity:
    """A named entity linked into the knowledge graph when possible."""

    text: str
    label: str                      # PERSON, ORG, SKILL, CONCEPT, DATE, ...
    start: int = 0
    end: int = 0
    confidence: float = 0.5
    node_id: Optional[str] = None   # KG node once linked
    canonical: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmotionScores:
    """Plutchik-style emotion activation in ``[0, 1]``."""

    joy: float = 0.0
    trust: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    sadness: float = 0.0
    disgust: float = 0.0
    anger: float = 0.0
    anticipation: float = 0.0

    def dominant(self) -> str:
        values = self.to_dict()
        return max(values, key=lambda k: values[k])

    def intensity(self) -> float:
        return max(self.to_dict().values())

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "EmotionScores":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: float(v) for k, v in data.items() if k in known})

    def normalised(self) -> Dict[str, float]:
        """Return the scores as a distribution summing to 1.0."""
        total = sum(vars(self).values())
        if total <= 0:
            return {k: 1.0 / len(vars(self)) for k in vars(self)}
        return {k: v / total for k, v in vars(self).items()}

    #: British spelling kept for callers that prefer it
    normalized = normalised

@dataclass
class Sentiment:
    polarity: float = 0.0            # -1 .. +1
    magnitude: float = 0.0           # 0 .. 1
    subjectivity: float = 0.0        # 0 .. 1
    label: str = "neutral"           # negative | neutral | positive

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Intent:
    """Classification of what the interviewer is asking for."""

    name: str                                   # e.g. ``behavioral``
    wh_type: Optional[str] = None               # what/why/how/when/where/who
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    is_question: bool = True
    requires_clarification: bool = False
    topic: str = "meta"
    topic_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Analysis:
    """Everything the NLP pipeline knows about one utterance."""

    text: str
    normalized: str = ""
    tokens: List[PosToken] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    intent: Optional[Intent] = None
    sentiment: Optional[Sentiment] = None
    emotion: Optional[EmotionScores] = None
    embedding: List[float] = field(default_factory=list)
    topics: List[Tuple[str, float]] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0

    @property
    def lemmas(self) -> List[str]:
        return [t.lemma for t in self.tokens]

    @property
    def content_words(self) -> List[str]:
        return [t.lemma for t in self.tokens if not t.is_stop and len(t.lemma) > 2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "tokens": [t.to_dict() for t in self.tokens],
            "sentences": self.sentences,
            "entities": [e.to_dict() for e in self.entities],
            "keywords": [[k, round(v, 4)] for k, v in self.keywords],
            "intent": self.intent.to_dict() if self.intent else None,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "emotion": self.emotion.to_dict() if self.emotion else None,
            "topics": [[k, round(v, 4)] for k, v in self.topics],
            "questions": self.questions,
            "language": self.language,
            "duration": round(self.duration, 4),
        }


# --------------------------------------------------------------------------- #
# Dialogue layer
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    id: str = field(default_factory=lambda: _new_id("turn"))
    speaker: str = Speaker.INTERVIEWER
    text: str = ""
    analysis: Optional[Analysis] = None
    audio_start: float = 0.0
    audio_end: float = 0.0
    created_at: float = field(default_factory=_now)
    dialogue_act: str = "question"          # question | statement | answer | greeting
    response: Optional["Response"] = None
    phase: str = Phase.INTERVIEW
    confidence: float = 0.0
    start: float = 0.0
    end: float = 0.0
    strategy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "speaker": self.speaker,
            "text": self.text,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "dialogue_act": self.dialogue_act,
            "response": self.response.to_dict() if self.response else None,
            "phase": self.phase,
            "created_at": self.created_at,
        }


@dataclass
class Evidence:
    """A piece of knowledge-graph support backing a generated claim."""

    kind: str                      # kg_path | fact | commonsense | memory | profile
    text: str
    source: str = ""
    score: float = 0.0
    nodes: List[str] = field(default_factory=list)
    relation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Response:
    id: str = field(default_factory=lambda: _new_id("resp"))
    text: str = ""
    turn_id: Optional[str] = None
    plan: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    tone: str = "neutral"
    register: str = "professional"
    confidence: float = 0.0
    validation: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)
    latency_ms: float = 0.0
    strategy: str = ""
    scorecard: Optional["Scorecard"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "turn_id": self.turn_id,
            "plan": self.plan,
            "evidence": [e.to_dict() for e in self.evidence],
            "tone": self.tone,
            "register": self.register,
            "confidence": round(self.confidence, 4),
            "validation": self.validation,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "latency_ms": round(self.latency_ms, 2),
            "strategy": self.strategy,
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
        }


# --------------------------------------------------------------------------- #
# Evaluation / learning
# --------------------------------------------------------------------------- #
@dataclass
class Scorecard:
    """Evaluation metrics for a single response (or a whole session)."""

    accuracy: float = 0.0
    relevance: float = 0.0
    engagement: float = 0.0
    personalization: float = 0.0
    error_rate: float = 0.0
    groundedness: float = 0.0
    fluency: float = 0.0
    empathy: float = 0.0
    detail: Dict[str, float] = field(default_factory=dict)

    def overall(self) -> float:
        weights = {
            "accuracy": 0.25,
            "relevance": 0.25,
            "engagement": 0.15,
            "personalization": 0.10,
            "groundedness": 0.15,
            "fluency": 0.10,
        }
        total = sum(getattr(self, k) * w for k, w in weights.items())
        total -= 0.10 * self.error_rate
        return round(max(0.0, min(1.0, total)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "overall": self.overall()}


@dataclass
class PipelineEvent:
    """A structured event emitted by any stage; consumed by UI, logs, metrics."""

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    phase: str = Phase.INTERVIEW
    severity: str = Severity.INFO
    ts: float = field(default_factory=_now)
    event_id: str = field(default_factory=lambda: _new_id("evt"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "phase": self.phase,
            "severity": self.severity,
            "ts": self.ts,
            "payload": self.payload,
        }


def merge_dicts(*sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow-merge a sequence of dictionaries, skipping ``None``."""
    out: Dict[str, Any] = {}
    for src in sources:
        if src:
            out.update(src)
    return out


def chunked(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Yield lists of at most ``size`` items from ``items``."""
    bucket: List[Any] = []
    for item in items:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
