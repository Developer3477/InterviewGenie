"""The InterviewGenie orchestrator: the whole system in one object.

This is the top-level coordinator.  It owns one instance of every subsystem and
implements the interview lifecycle described in the product requirements:

**Pre-interview**
    load configuration and the knowledge graph, initialise NLP and ASR, set up
    the context and emotional-intelligence modules, and restore the personalisation
    profile.

**Interview**
    audio in -> ASR -> NLP analysis -> knowledge retrieval -> dialogue state ->
    response generation -> validation -> response out, all inside a guarded,
    event-emitting turn loop.

**Post-response**
    evaluate the answer, update the knowledge graph and the context, and feed the
   continuous-learning loop.

**Error handling and recovery**
    every stage is wrapped by :class:`~interviewgenie.errors.RecoveryEngine`; a
    failure produces a :class:`~interviewgenie.errors.RecoveryPlan` which is
    executed (retry, degrade, ask for clarification, or abort) and recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .asr.base import ASRProvider
from .context.dialogue import DialogueManager, STRATEGIES
from .context.emotion import EmotionalIntelligence
from .context.memory import ConversationMemory
from .errors import (ErrorCode, InterviewGenieError, RecoveryEngine, RecoveryPlan,
                     RecoveryRecord)
from .events import DEFAULT_BUS, EventBus, emit
from .generation.composer import ResponseComposer
from .generation.llm import LLMClient, ResponseValidator, llm_from_config
from .knowledge.commonsense import CommonsenseEngine
from .knowledge.graph import PropertyGraph, graph_from_config
from .knowledge.retrieval import KnowledgeRetriever
from .learning.online import LearningMetrics, OnlineLearner
from .nlp.intent import IntentClassifier
from .nlp.pipeline import NLPPipeline
from .personalization.profile import IntervieweeProfile, StyleAdapter
from .types import Analysis, Phase, Response, Scorecard, Speaker, TranscriptChunk, Turn
from .logging import get_logger

LOG = get_logger("orchestrator")


@dataclass
class InterviewGenie:
    """The live interview copilot."""

    config: Any = None
    bus: EventBus = field(default_factory=EventBus)

    # -- construction ----------------------------------------------------- #
    def __post_init__(self) -> None:
        if self.config is None:
            from .config import load_config

            self.config = load_config()
        self.bus = self.bus or DEFAULT_BUS

        self.phase: str = Phase.IDLE
        self.recovery = RecoveryEngine()
        self.memory = ConversationMemory()
        self.emotional = EmotionalIntelligence(
            empathy_level=float(self.config.get("context.empathy_level", 0.7)))
        self.dialogue = DialogueManager(empathy_level=self.emotional.empathy_level)
        self.commonsense = CommonsenseEngine()
        self.validator = ResponseValidator(
            max_words=int(self.config.get("generation.max_words", 110)),
            min_words=int(self.config.get("generation.min_words", 8)))

        # knowledge
        self.graph = graph_from_config(self.config)
        self.retriever = KnowledgeRetriever(
            graph=self.graph,
            max_hops=int(self.config.get("knowledge.max_hops", 2)),
            max_evidence=int(self.config.get("knowledge.max_evidence", 8)),
        )

        # nlp
        from .nlp.tagger import PosTagger

        self.nlp = NLPPipeline(
            tagger=PosTagger(backend=str(self.config.get("nlp.tagger_backend", "builtin"))),
            enable_coref=bool(self.config.get("nlp.use_coref", True)),
        )
        self.intent_classifier = IntentClassifier(
            alpha=float(self.config.get("nlp.intent_alpha", 0.22)))
        self.intent_classifier.fit_from_dataset()
        self.nlp.attach_linker(self.retriever.linker)
        self.nlp.attach_index(self.retriever.index)

        # personalisation
        self.profile = IntervieweeProfile()
        profile_path = self.config.get("personalization.profile_path", "")
        if profile_path:
            try:
                from pathlib import Path

                if Path(profile_path).exists():
                    self.profile = IntervieweeProfile.load(profile_path)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("could not load profile: %s", exc)
        self.style = StyleAdapter(profile=self.profile)

        # generation
        self.composer = ResponseComposer(
            profile=self.profile, style=self.style, emotional=self.emotional,
            commonsense=self.commonsense,
            max_words=int(self.config.get("generation.max_words", 90)))
        self.llm = llm_from_config(self.config.section("generation").get("llm", {}) or {})

        # speech
        self.asr = self._build_asr()

        # learning
        self.learner = OnlineLearner(
            intent_classifier=self.intent_classifier, retriever=self.retriever,
            profile=self.profile, dialogue=self.dialogue)
        self.metrics = LearningMetrics()
        self.turns: List[Turn] = []
        self.started_at: Optional[float] = None
        self._response_count = 0

    def _build_asr(self) -> ASRProvider:
        from .asr.base import provider_from_config

        try:
            return provider_from_config(self.config)
        except Exception as exc:  # noqa: BLE001 - ASR must never block startup
            LOG.warning("falling back to the mock recogniser: %s", exc)
            from .asr.mock import MockASRProvider

            return MockASRProvider()

    # -- lifecycle -------------------------------------------------------- #
    def start(self, **profile_facts: Any) -> Dict[str, Any]:
        """Pre-interview setup."""
        self.phase = Phase.PREPARING
        self.started_at = time.time()
        if profile_facts:
            self.profile.update_facts(**profile_facts)
        self.asr.reset()
        self.nlp.reset()
        summary = {
            "phase": self.phase,
            "graph": self.graph.stats(),
            "asr": self.asr.capabilities().__dict__,
            "nlp": {"intents": len(self.intent_classifier.examples)},
            "profile": self.profile.name,
        }
        self.phase = Phase.LISTENING
        self.bus.publish("interview.started", summary)
        LOG.info("interview session started", context=summary)
        return summary

    def stop(self) -> Dict[str, Any]:
        """Post-interview teardown and reporting."""
        self.phase = Phase.CLOSING
        report = self.report()
        self.bus.publish("interview.stopped", report)
        self.phase = Phase.IDLE
        return report

    # -- the turn loop ---------------------------------------------------- #
    def ingest(self, samples: Sequence[float], sample_rate: int = 16000) -> List[TranscriptChunk]:
        """Feed raw audio into the recogniser and return new transcripts."""
        try:
            self.asr.accept_audio(samples, sample_rate)
            return self.asr.poll()
        except InterviewGenieError as exc:
            plan = self.recovery.plan_for(exc)
            record = self.recovery.execute(plan)
            self.bus.publish("asr.error", record.to_dict())
            return []

    def flush(self) -> List[TranscriptChunk]:
        try:
            self.asr.flush()
            return self.asr.poll()
        except InterviewGenieError as exc:  # pragma: no cover - defensive
            plan = self.recovery.plan_for(exc)
            self.recovery.execute(plan)
            return []

    def handle_transcript(self, chunk: TranscriptChunk) -> Optional[Response]:
        """Full pipeline for one transcript chunk."""
        if not chunk.is_final or not chunk.text.strip():
            return None
        if chunk.confidence < float(self.config.get("asr.min_confidence", 0.45)):
            return self._recover_low_confidence(chunk)

        turn = Turn(speaker=Speaker.INTERVIEWER, text=chunk.text,
                    confidence=chunk.confidence, start=chunk.start, end=chunk.end)
        return self.answer(turn)

    def ask(self, question: str, confidence: float = 1.0) -> Response:
        """Text-in path (no ASR) -- used by the CLI, tests and the web cockpit."""
        turn = Turn(speaker=Speaker.INTERVIEWER, text=question, confidence=confidence)
        return self.answer(turn)

    # -- core ------------------------------------------------------------- #
    def answer(self, turn: Turn) -> Response:
        """Analyse, retrieve, plan, generate, validate and score one question."""
        started = time.perf_counter()
        self.phase = Phase.ANALYSING

        # 1. analysis ----------------------------------------------------- #
        try:
            analysis = self.nlp.analyze(turn.text)
        except InterviewGenieError as exc:
            return self._recover(exc, turn, started)
        turn.analysis = analysis
        turn.dialogue_act = "question"

        # 2. retrieval ---------------------------------------------------- #
        self.phase = Phase.RETRIEVING
        try:
            retrieval = self.retriever.retrieve(analysis)
        except InterviewGenieError as exc:
            return self._recover(exc, turn, started)
        self.bus.publish("knowledge.retrieved", {
            "entities": [str(n.get("name", n.id)) for n in retrieval.entities],
            "evidence": len(retrieval.evidence),
        })

        # 3. dialogue state ------------------------------------------------ #
        self.phase = Phase.DIALOGUE
        try:
            events = self.dialogue.observe(turn)
            self.bus.publish("dialogue.updated", events)
        except InterviewGenieError as exc:
            # a contradiction or topic shift is recoverable: keep going
            self.recovery.execute(self.recovery.plan_for(exc))
            events = {}

        strategy = self.dialogue.choose_strategy(analysis)
        turn.strategy = strategy

        # 4. generation --------------------------------------------------- #
        self.phase = Phase.GENERATING
        response = self._generate(turn, analysis, retrieval, strategy)
        if response is None:
            return self._recover(InterviewGenieError("generation produced nothing",
                                                     code=ErrorCode.GEN_NO_PLAN),
                                 turn, started)

        # 5. validation and repair ---------------------------------------- #
        validation = self.validator.validate(response, history=self._answer_history())
        if not validation["ok"]:
            response = self.validator.repair(response, validation["issues"])
            validation = self.validator.validate(response, history=self._answer_history())
        response.validation = {**response.validation, "validator": validation}

        # 6. scoring ------------------------------------------------------ #
        self.phase = Phase.EVALUATING
        from .evaluation.metrics import get_evaluator

        scorecard = get_evaluator().score(analysis, response)
        response.scorecard = scorecard
        self.metrics.record(scorecard)

        # 7. memory + learning -------------------------------------------- #
        # the response must be attached before the memory observes the turn,
        # otherwise there is nothing worth remembering yet
        self.dialogue.record_answer(turn, response)
        self.memory.observe(turn)
        self.turns.append(turn)
        self._response_count += 1
        self.phase = Phase.LISTENING

        response.latency_ms = (time.perf_counter() - started) * 1000.0
        response.turn_id = turn.id
        self.bus.publish("response.generated", {
            "turn_id": turn.id, "text": response.text[:120],
            "strategy": strategy, "confidence": response.confidence,
            "scorecard": scorecard.to_dict(),
        })
        return response

    def _generate(self, turn: Turn, analysis: Analysis, retrieval: Any,
                  strategy: str) -> Optional[Response]:
        """Compose (or delegate to the LLM) and post-process the answer."""
        memory = self.memory.context_for(turn.text, k=2)
        try:
            response = self.composer.compose(
                analysis, retrieval, strategy, question_text=turn.text,
                memory=memory)
        except InterviewGenieError:
            raise
        except Exception as exc:  # noqa: BLE001 - composer must never crash a turn
            LOG.warning("composer failed, retrying with defaults: %s", exc)
            response = None

        if response is None:
            return None

        # optional LLM upgrade, with the grounded composer as fallback
        use_llm = (str(self.config.get("generation.backend", "composer")).lower() == "llm"
                   and getattr(self.llm, "available", lambda: False)())
        if use_llm:
            try:
                prompt = self._build_llm_prompt(turn, analysis, retrieval, strategy)
                text = self.llm.complete(
                    prompt,
                    system=self._llm_system_prompt(strategy),
                    max_tokens=int(self.config.get("generation.max_words", 90)) * 2)
                if text and len(text.split()) >= 6:
                    response.text = text
                    response.validation = {**response.validation, "backend": "llm"}
            except InterviewGenieError as exc:
                LOG.warning("LLM unavailable, using the composer: %s", exc)
        response.strategy = strategy
        return response

    def _llm_system_prompt(self, strategy: str) -> str:
        from .generation.llm import SYSTEM_PROMPT

        return SYSTEM_PROMPT.format(
            max_words=int(self.config.get("generation.max_words", 90)),
            register=self.style.register(),
            tone=self.emotional.emotion.dominant(),
        ) + f"\nResponse strategy: {strategy}."

    def _build_llm_prompt(self, turn: Turn, analysis: Analysis,
                          retrieval: Any, strategy: str) -> str:
        lines = [
            f"Question: {turn.text}",
            f"Intent: {analysis.intent.name} / topic: {analysis.intent.topic}",
            f"Register: {self.style.register()}",
            f"Emotion detected: {self.emotional.emotion.dominant()}",
        ]
        if retrieval.entities:
            names = ", ".join(str(n.get("name", n.id)) for n in retrieval.entities[:5])
            lines.append(f"Relevant knowledge entities: {names}")
        for evidence in retrieval.evidence[:4]:
            lines.append(f"Evidence: {evidence.text}")
        if self.profile.strengths:
            lines.append(f"Candidate strengths: {', '.join(self.profile.strengths[:4])}")
        if self.profile.skills:
            lines.append(f"Candidate skills: {', '.join(self.profile.skills[:6])}")
        story = self.composer._story_sentence(analysis)
        if story:
            lines.append(f"Candidate story: {story}")
        lines.append(f"Strategy: {strategy}")
        lines.append("Draft the answer the candidate should give.")
        return "\n".join(lines)

    # -- feedback / learning ---------------------------------------------- #
    def feedback(self, text: str) -> Dict[str, Any]:
        """Explicit feedback from the candidate."""
        turn = self.turns[-1] if self.turns else None
        result = self.learner.learn(text, turn)
        self.bus.publish("learning.feedback", result)
        return result

    def accept(self, edited: bool = False, rejected: bool = False) -> Dict[str, Any]:
        """Implicit feedback: what the candidate did with the last suggestion."""
        if not self.turns:
            return {"reward": 0.0}
        return self.learner.learn_from_outcome(self.turns[-1], accepted=not (edited or rejected),
                                               edited=edited, rejected=rejected)

    def consolidate(self) -> Dict[str, Any]:
        return self.learner.consolidate()

    # -- recovery --------------------------------------------------------- #
    def _recover(self, exc: InterviewGenieError, turn: Turn,
                 started: float) -> Response:
        plan = self.recovery.plan_for(exc)
        record = self.recovery.execute(plan)
        self.bus.publish("error.recovered", record.to_dict())
        fallback = Response(
            text=(f"I want to make sure I answer the right question -- could you say a "
                  f"little more about what you are after?"),
            turn_id=turn.id, plan=["clarify"], confidence=0.2,
            tone="neutral", register="professional",
        )
        fallback.validation = {"recovered": True, "error": str(exc),
                               "plan": plan.to_dict() if hasattr(plan, "to_dict") else str(plan)}
        fallback.metrics = {"relevance": 0.1, "accuracy": 0.1, "groundedness": 0.0,
                            "engagement": 0.1, "personalization": 0.0, "fluency": 0.8,
                            "empathy": 0.4, "concision": 1.0}
        from .evaluation.metrics import get_evaluator

        analysis = turn.analysis or self.nlp.analyze(turn.text)
        fallback.scorecard = get_evaluator().score(analysis, fallback)
        self.metrics.record(fallback.scorecard)
        fallback.latency_ms = (time.perf_counter() - started) * 1000.0
        return fallback

    def _recover_low_confidence(self, chunk: TranscriptChunk) -> Response:
        """Ask for clarification when the ASR is not confident enough."""
        self.bus.publish("asr.low_confidence", {"text": chunk.text,
                                              "confidence": chunk.confidence})
        response = Response(
            text=("Sorry, I did not quite catch that -- could you repeat the question?"),
            plan=["clarify"], confidence=chunk.confidence,
            tone="neutral", register="professional",
        )
        response.validation = {"low_confidence": True, "asr_confidence": chunk.confidence,
                               "heard": chunk.text}
        response.metrics = {"relevance": 0.1, "accuracy": 0.1, "groundedness": 0.0,
                            "engagement": 0.2, "personalization": 0.0, "fluency": 0.8,
                            "empathy": 0.6, "concision": 1.0}
        return response

    # -- reporting -------------------------------------------------------- #
    def _answer_history(self) -> List[str]:
        return [t.response.text for t in self.turns if t.response is not None][-6:]

    def report(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "duration_s": round(time.time() - self.started_at, 2) if self.started_at else 0.0,
            "turns": len(self.turns),
            "responses": self._response_count,
            "metrics": self.metrics.to_dict(),
            "dialogue": self.dialogue.summary(),
            "memory": self.memory.stats(),
            "learning": self.learner.stats(),
            "recovery": self.recovery.stats(),
            "knowledge": self.retriever.stats(),
            "emotion": self.emotional.to_dict(),
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "asr": self.asr.describe() if hasattr(self.asr, "describe") else
            {"provider": getattr(self.asr, "name", "unknown")},
            "graph": self.graph.stats(),
            "profile": self.profile.to_dict(),
            "strategies": list(STRATEGIES),
            "llm": getattr(self.llm, "name", "none"),
            "phrases": getattr(self.asr, "supported_phrases", None),
        }

    # -- profile persistence ---------------------------------------------- #
    def save_profile(self, path: str) -> str:
        return self.profile.save(path)
