"""The live-interview loop: accumulate speech, detect questions, answer.

This is the piece that makes the product usable during a real interview. The
orchestrator answers one *complete* question; a speech recogniser emits a
*stream* of fragments, one per phrase, and answering each fragment as it
arrives would produce a stream of half-answers. ``LiveSession`` sits between
the two:

  * final transcripts accumulate into a buffer
  * the buffer is answered when it holds a complete question *and* the
    interviewer has paused long enough that the question is finished
  * interim transcripts are surfaced for display but never answered

Time is passed in explicitly (``now`` in seconds) rather than read from the
clock, so the debounce logic is deterministic and testable without sleeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .logging import get_logger
from .types import TranscriptChunk

LOG = get_logger("live")

# Openers that mark an utterance as a question even without a question mark.
# Ordered longest-first so "tell me about" wins over "tell".
_QUESTION_OPENERS: Sequence[str] = (
    "tell me about", "talk me through", "walk me through", "walk me through how",
    "describe a time", "describe how", "describe what", "give me an example",
    "can you tell me", "could you tell me", "would you tell me",
    "can you walk", "could you walk", "can you describe", "could you describe",
    "how would you", "how do you", "how did you", "how have you",
    "what would you", "what do you", "what did you", "what have you",
    "why would you", "why do you", "why did you",
    "when would you", "when do you", "when did you",
    "where would you", "who would you",
    "which would you", "have you ever", "do you have", "did you ever",
    "are you able", "is there a time", "let's say", "imagine that",
    "suppose that", "share an example", "explain how", "explain what",
    "tell me", "talk me", "walk me", "describe", "explain", "share",
    "what", "why", "how", "when", "where", "who", "which",
)

# Filler that trails a question and should not count towards its length.
_FILLER = {"um", "uh", "erm", "like", "you know", "i mean", "sort of",
           "kind of", "basically", "actually", "right", "okay", "so"}


@dataclass
class LiveConfig:
    """Tuning for the live loop."""

    auto_answer: bool = True
    debounce_ms: int = 700          # pause that marks a question as finished
    min_question_words: int = 3     # ignore fragments shorter than this
    min_confidence: float = 0.45    # ASR confidence floor
    max_buffer_words: int = 90      # flush before the buffer grows unbounded
    require_question_mark: bool = False


@dataclass
class _Buffer:
    """The in-progress question."""

    text: str = ""
    confidence: float = 0.0
    updated_at: float = 0.0
    words: int = 0

    def append(self, fragment: str, confidence: float, now: float) -> None:
        fragment = " ".join(fragment.split())
        if not fragment:
            return
        self.text = (self.text + " " + fragment).strip()
        self.confidence = max(self.confidence, confidence)
        self.updated_at = now
        self.words = len(self.text.split())


def looks_like_question(text: str) -> bool:
    """True when ``text`` reads as a complete question.

    A question mark is decisive. Without one, the utterance must *open* with a
    question word or a "tell me about"-style invitation, because a bare
    statement like "we use Kafka at scale" is context, not a question.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return False
    if "?" in cleaned:
        return True
    low = cleaned.lower()
    for opener in _QUESTION_OPENERS:
        if low.startswith(opener):
            # "how" alone would match "however"; require a following word
            if len(opener) <= 5 and len(low) > len(opener):
                if not low[len(opener)].isspace():
                    continue
            return True
    return False


def meaningful_word_count(text: str) -> int:
    """Word count with filler removed, so "um, tell me about, uh" is not a question.

    Multi-word filler ("you know", "sort of") is stripped as a phrase first;
    splitting on whitespace alone would never match it, and the count would be
    inflated by exactly the words that carry no meaning.
    """
    low = " ".join((text or "").lower().split())
    for phrase in _FILLER:
        if " " in phrase:
            low = low.replace(phrase, " ")
    return len([w for w in low.split() if w and w not in _FILLER])


class LiveSession:
    """Turns a stream of speech fragments into answered questions."""

    def __init__(self, genie: Any, config: Optional[LiveConfig] = None):
        self.genie = genie
        self.config = config or LiveConfig()
        self._buffer = _Buffer()
        self._asked: List[str] = []
        self._answered = 0
        self._auto = self.config.auto_answer
        self._listening = False
        self._last_error = ""

    # -- state ----------------------------------------------------------- #
    @property
    def auto(self) -> bool:
        return self._auto

    def set_auto(self, on: bool) -> None:
        self._auto = bool(on)
        LOG.info("auto-answer %s", "on" if self._auto else "off")

    @property
    def listening(self) -> bool:
        return self._listening

    def set_listening(self, on: bool) -> None:
        self._listening = bool(on)
        if not on:
            self._buffer = _Buffer()

    @property
    def pending_question(self) -> str:
        return self._buffer.text

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "answered": self._answered,
            "auto": self._auto,
            "listening": self._listening,
            "pending": self._buffer.text,
            "recent": list(self._asked[-5:]),
            "error": self._last_error,
        }

    def reset(self) -> None:
        self._buffer = _Buffer()
        self._asked = []

    # -- speech in ------------------------------------------------------- #
    def feed_transcript(self, chunk: TranscriptChunk,
                        now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Consume one recogniser emission and return any events to show.

        Interim results are returned as a ``partial`` event so the overlay can
        show what is being said, but they are never answered.
        """
        now = time.time() if now is None else now
        text = " ".join((chunk.text or "").split())
        if not text:
            return []

        if not chunk.is_final:
            return [{"type": "partial", "text": text,
                     "confidence": chunk.confidence}]

        if chunk.confidence < self.config.min_confidence:
            LOG.debug("ignoring low-confidence transcript: %.2f", chunk.confidence)
            return []

        self._buffer.append(text, chunk.confidence, now)

        # An unbounded buffer means the interviewer never paused: flush it so
        # the overlay stays useful rather than silently swallowing speech.
        if len(self._buffer.text.split()) > self.config.max_buffer_words:
            return self._flush(now)

        if self._auto and self._buffer_is_complete(now):
            return self._flush(now)
        return []

    def tick(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Called on a timer: answer once the interviewer has paused."""
        now = time.time() if now is None else now
        if not self._auto or not self._buffer.text:
            return []
        if self._buffer_is_complete(now):
            return self._flush(now)
        return []

    def _buffer_is_complete(self, now: float) -> bool:
        buffer = self._buffer
        if not buffer.text:
            return False
        if meaningful_word_count(buffer.text) < self.config.min_question_words:
            return False
        paused_ms = (now - buffer.updated_at) * 1000.0
        if paused_ms < self.config.debounce_ms:
            return False
        if self.config.require_question_mark and "?" not in buffer.text:
            return False
        return looks_like_question(buffer.text)

    def _flush(self, now: float) -> List[Dict[str, Any]]:
        question = self._buffer.text
        self._buffer = _Buffer()
        if not question:
            return []
        self._asked.append(question)
        return list(self.answer(question, confidence=0.9, now=now))

    # -- answering ------------------------------------------------------- #
    def answer(self, question: str, confidence: float = 1.0,
               now: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        """Answer one question, yielding display events as it goes."""
        started = time.perf_counter()
        question = " ".join(question.split())
        if not question:
            return

        yield {"type": "question", "text": question, "confidence": confidence}

        try:
            for event in self.genie.ask_stream(question, confidence=confidence):
                kind = event.get("type")
                if kind == "analysis":
                    yield {
                        "type": "analysis",
                        "question": question,
                        "intent": event.get("intent"),
                        "topic": event.get("topic"),
                        "strategy": event.get("strategy"),
                        "tone": event.get("tone"),
                        "entities": event.get("entities") or [],
                        "evidence": event.get("evidence") or [],
                    }
                elif kind == "delta":
                    yield {"type": "delta", "text": event.get("text", "")}
                elif kind == "response":
                    response = event.get("response", event)
                    self._answered += 1
                    yield {
                        "type": "response",
                        "question": question,
                        "text": response.text,
                        "strategy": response.strategy,
                        "scorecard": response.scorecard.to_dict()
                        if hasattr(response.scorecard, "to_dict") else {},
                        "latency_ms": response.latency_ms,
                        "turn_id": response.turn_id,
                        "wall_ms": (time.perf_counter() - started) * 1000.0,
                    }
        except Exception as exc:  # noqa: BLE001 - a dead turn must not kill the loop
            self._last_error = str(exc)
            LOG.warning("live turn failed: %s", exc)
            yield {"type": "error", "message": str(exc), "question": question}

    # -- manual text ----------------------------------------------------- #
    def submit(self, text: str) -> Iterator[Dict[str, Any]]:
        """Answer a question typed or pasted by the candidate."""
        text = " ".join((text or "").split())
        if not text:
            return
        self._asked.append(text)
        yield from self.answer(text, confidence=1.0)


__all__ = ["LiveSession", "LiveConfig", "looks_like_question",
           "meaningful_word_count"]
