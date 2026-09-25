"""Three-tier conversational memory.

* :class:`WorkingMemory` -- the last few turns, always in context.
* :class:`EpisodicMemory` -- turn-level episodes with salience scores, retrieved
  by recency and relevance (this is what lets the system refer back to "the
  project you mentioned earlier").
* :class:`SemanticMemory` -- durable facts learned about the candidate and the
  interview (skills claimed, stories used, preferences expressed).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ..logging import get_logger
from ..nlp.embeddings import TfidfIndex, sparse_cosine
from ..types import Analysis, Response, Turn

LOG = get_logger("context.memory")


@dataclass
class Episode:
    """One memorable interaction."""

    turn_id: str
    question: str
    answer: str
    intent: str = ""
    topic: str = ""
    salience: float = 0.5
    sentiment: float = 0.0
    reward: float = 0.5
    created_at: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "question": self.question,
            "answer": self.answer,
            "intent": self.intent,
            "topic": self.topic,
            "salience": round(self.salience, 3),
            "reward": round(self.reward, 3),
            "tags": self.tags,
        }


@dataclass
class WorkingMemory:
    """A short FIFO of the most recent turns."""

    capacity: int = 8
    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=8))

    def __post_init__(self) -> None:
        self.turns = deque(maxlen=self.capacity)

    def push(self, turn: Turn) -> None:
        self.turns.append(turn)

    def recent(self, n: int = 4) -> List[Turn]:
        return list(self.turns)[-n:]

    def texts(self, n: int = 6) -> List[str]:
        return [t.text for t in self.recent(n)]

    def clear(self) -> None:
        self.turns.clear()


@dataclass
class EpisodicMemory:
    """Salience-weighted store of past interactions."""

    capacity: int = 200
    episodes: Deque[Episode] = field(default_factory=deque)
    _index: Optional[TfidfIndex] = field(default=None, repr=False)
    _dirty: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        self.episodes = deque(maxlen=self.capacity)

    def add(self, episode: Episode) -> Episode:
        self.episodes.append(episode)
        self._dirty = True
        return episode

    def add_from_turn(self, turn: Turn, salience: float = 0.5) -> Optional[Episode]:
        if turn.response is None:
            return None
        analysis = turn.analysis
        return self.add(Episode(
            turn_id=turn.id, question=turn.text, answer=turn.response.text,
            intent=analysis.intent.name if analysis and analysis.intent else "",
            topic=analysis.intent.topic if analysis and analysis.intent else "",
            salience=salience,
            sentiment=analysis.sentiment.polarity if analysis and analysis.sentiment else 0.0,
            reward=turn.response.confidence,
            created_at=turn.created_at,
            tags=[e.text for e in (analysis.entities if analysis else [])][:5],
        ))

    def _ensure_index(self) -> None:
        if not self._dirty and self._index is not None:
            return
        documents = [f"{e.question} {e.answer}" for e in self.episodes]
        self._index = TfidfIndex(use_stem=True, dim=32)
        self._index.fit(documents)
        self._dirty = False

    def recall(self, query: str, k: int = 3) -> List[Episode]:
        """Retrieve episodes relevant to ``query``, weighted by salience."""
        if not self.episodes:
            return []
        self._ensure_index()
        assert self._index is not None
        query_vec = self._index.transform(query)
        scored: List[Tuple[float, Episode]] = []
        for episode in self.episodes:
            doc_vec = self._index.transform(f"{episode.question} {episode.answer}")
            similarity = sparse_cosine(query_vec, doc_vec)
            scored.append((similarity * episode.salience, episode))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [episode for score, episode in scored[:k] if score > 0.02]

    def best_episodes(self, k: int = 5) -> List[Episode]:
        return sorted(self.episodes, key=lambda e: e.reward, reverse=True)[:k]

    def stats(self) -> Dict[str, Any]:
        return {
            "episodes": len(self.episodes),
            "mean_reward": round(sum(e.reward for e in self.episodes) / len(self.episodes), 4)
            if self.episodes else 0.0,
        }


@dataclass
class SemanticMemory:
    """Durable facts about the candidate and the interview."""

    facts: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.facts[key] = value
        self.counts[key] = self.counts.get(key, 0) + 1

    def get(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def add_to_set(self, key: str, value: Any, limit: int = 20) -> None:
        current = self.facts.setdefault(key, [])
        if value not in current:
            current.append(value)
            del current[:-limit]

    def merge(self, other: Dict[str, Any]) -> None:
        for key, value in other.items():
            self.set(key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": self.facts, "counts": self.counts}


@dataclass
class ConversationMemory:
    """Facade over the three memory tiers."""

    working: WorkingMemory = field(default_factory=WorkingMemory)
    episodic: EpisodicMemory = field(default_factory=EpisodicMemory)
    semantic: SemanticMemory = field(default_factory=SemanticMemory)

    def observe(self, turn: Turn, salience: float = 0.5) -> None:
        self.working.push(turn)
        if turn.response is not None:
            self.episodic.add_from_turn(turn, salience)

    def recall(self, query: str, k: int = 3) -> List[Episode]:
        return self.episodic.recall(query, k)

    def context_for(self, query: str, k: int = 3) -> List[str]:
        """Human-readable memory snippets to feed the composer."""
        snippets: List[str] = []
        for episode in self.recall(query, k):
            snippets.append(f"Earlier you answered '{episode.question[:70]}' with: "
                            f"{episode.answer[:140]}")
        for key, value in list(self.semantic.facts.items())[:4]:
            if isinstance(value, (str, int, float, bool)):
                snippets.append(f"{key}: {value}")
        return snippets

    def stats(self) -> Dict[str, Any]:
        return {
            "working": len(self.working.turns),
            "episodic": self.episodic.stats(),
            "semantic": len(self.semantic.facts),
        }

    def clear(self) -> None:
        self.working.clear()
        self.episodic.episodes.clear()
        self.episodic._dirty = True
        self.semantic = SemanticMemory()
