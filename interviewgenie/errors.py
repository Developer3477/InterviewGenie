"""Exception hierarchy and the error-handling / recovery machinery.

Every failure inside InterviewGenie is expressed as an :class:`InterviewGenieError`
carrying a machine readable ``code``, a ``severity`` and a ``recoverable`` flag.
The :class:`RecoveryEngine` turns those failures into concrete recovery actions
(re-ask, rephrase, degrade gracefully, fall back to another provider, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging import get_logger

LOG = get_logger("errors")


class ErrorCode(str, Enum):
    # speech
    ASR_LOW_CONFIDENCE = "asr.low_confidence"
    ASR_SILENCE = "asr.silence"
    ASR_PROVIDER_UNAVAILABLE = "asr.provider_unavailable"
    ASR_TIMEOUT = "asr.timeout"
    # nlp
    NLP_PARSE_FAILURE = "nlp.parse_failure"
    NLP_UNKNOWN_INTENT = "nlp.unknown_intent"
    NLP_AMBIGUOUS_QUESTION = "nlp.ambiguous_question"
    NLP_LOW_QUALITY_TRANSCRIPT = "nlp.low_quality_transcript"
    # knowledge
    KG_NO_EVIDENCE = "kg.no_evidence"
    KG_ENTITY_UNLINKED = "kg.entity_unlinked"
    KG_QUERY_TIMEOUT = "kg.query_timeout"
    # generation
    GEN_NO_PLAN = "generation.no_plan"
    GEN_UNSUPPORTED_CLAIM = "generation.unsupported_claim"
    GEN_TOO_LONG = "generation.too_long"
    LLM_UNAVAILABLE = "llm.unavailable"
    LLM_RATE_LIMITED = "llm.rate_limited"
    # dialogue
    DLG_TOPIC_SHIFT = "dialogue.topic_shift"
    DLG_CONTRADICTION = "dialogue.contradiction"
    DLG_CONTEXT_LOST = "dialogue.context_lost"
    # infra
    CONFIG_INVALID = "config.invalid"
    STORAGE_FAILURE = "storage.failure"
    INTERNAL = "internal"


class InterviewGenieError(Exception):
    """Base class for every recoverable failure raised inside the system."""

    code = ErrorCode.INTERNAL
    severity = "error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[ErrorCode] = None,
        severity: Optional[str] = None,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if severity is not None:
            self.severity = severity
        self.recoverable = recoverable
        self.context: Dict[str, Any] = context or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value if isinstance(self.code, ErrorCode) else str(self.code),
            "message": self.message,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "context": self.context,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.code} {self.message!r}>"


# --- speech --------------------------------------------------------------- #
class ASRError(InterviewGenieError):
    code = ErrorCode.ASR_PROVIDER_UNAVAILABLE
    severity = "warning"


class LowConfidenceError(ASRError):
    code = ErrorCode.ASR_LOW_CONFIDENCE


class SilenceDetectedError(ASRError):
    code = ErrorCode.ASR_SILENCE


class ASRTimeoutError(ASRError):
    code = ErrorCode.ASR_TIMEOUT


# --- nlp ------------------------------------------------------------------ #
class NLPError(InterviewGenieError):
    code = ErrorCode.NLP_PARSE_FAILURE
    severity = "warning"


class AmbiguousQuestionError(NLPError):
    code = ErrorCode.NLP_AMBIGUOUS_QUESTION


class UnknownIntentError(NLPError):
    code = ErrorCode.NLP_UNKNOWN_INTENT


# --- knowledge ------------------------------------------------------------ #
class KnowledgeError(InterviewGenieError):
    code = ErrorCode.KG_NO_EVIDENCE
    severity = "warning"


class EntityLinkError(KnowledgeError):
    code = ErrorCode.KG_ENTITY_UNLINKED


# --- generation ----------------------------------------------------------- #
class GenerationError(InterviewGenieError):
    code = ErrorCode.GEN_NO_PLAN
    severity = "error"


class UnsupportedClaimError(GenerationError):
    """A generated claim is not supported by the retrieved evidence."""
    code = ErrorCode.GEN_UNSUPPORTED_CLAIM


# --- dialogue ------------------------------------------------------------- #
class LLMUnavailableError(GenerationError):
    """The configured language-model backend could not be reached."""


class LearningError(InterviewGenieError):
    """The continuous-learning loop could not apply an update."""


class EvaluationError(InterviewGenieError):
    """Evaluation could not be completed."""


class DialogueError(InterviewGenieError):
    code = ErrorCode.DLG_TOPIC_SHIFT
    severity = "warning"


class ContradictionError(DialogueError):
    code = ErrorCode.DLG_CONTRADICTION


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
class RecoveryAction(str, Enum):
    RETRY = "retry"
    REPHRASE = "rephrase"
    CLARIFY = "clarify"                # ask the interviewer to repeat / disambiguate
    FALLBACK_PROVIDER = "fallback_provider"
    DEGRADE = "degrade"                # answer with what we do know
    ESCALATE = "escalate"              # hand control to the candidate
    IGNORE = "ignore"                  # treat as noise, keep listening
    RESET_CONTEXT = "reset_context"


@dataclass
class RecoveryPlan:
    action: RecoveryAction
    reason: str
    attempts: int = 0
    max_attempts: int = 2
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "payload": self.payload,
        }


@dataclass
class RecoveryRecord:
    error: Dict[str, Any]
    plan: Dict[str, Any]
    outcome: str = "pending"     # pending | recovered | failed | escalated
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"error": self.error, "plan": self.plan, "outcome": self.outcome, "detail": self.detail}


class RecoveryEngine:
    """Maps errors to recovery plans and executes them with bounded retries."""

    #: error code -> preferred recovery action
    _TABLE: Dict[str, RecoveryAction] = {
        ErrorCode.ASR_LOW_CONFIDENCE.value: RecoveryAction.REPHRASE,
        ErrorCode.ASR_SILENCE.value: RecoveryAction.IGNORE,
        ErrorCode.ASR_PROVIDER_UNAVAILABLE.value: RecoveryAction.FALLBACK_PROVIDER,
        ErrorCode.ASR_TIMEOUT.value: RecoveryAction.FALLBACK_PROVIDER,
        ErrorCode.NLP_PARSE_FAILURE.value: RecoveryAction.CLARIFY,
        ErrorCode.NLP_UNKNOWN_INTENT.value: RecoveryAction.DEGRADE,
        ErrorCode.NLP_AMBIGUOUS_QUESTION.value: RecoveryAction.CLARIFY,
        ErrorCode.NLP_LOW_QUALITY_TRANSCRIPT.value: RecoveryAction.REPHRASE,
        ErrorCode.KG_NO_EVIDENCE.value: RecoveryAction.DEGRADE,
        ErrorCode.KG_ENTITY_UNLINKED.value: RecoveryAction.DEGRADE,
        ErrorCode.KG_QUERY_TIMEOUT.value: RecoveryAction.DEGRADE,
        ErrorCode.GEN_NO_PLAN.value: RecoveryAction.ESCALATE,
        ErrorCode.GEN_UNSUPPORTED_CLAIM.value: RecoveryAction.REPHRASE,
        ErrorCode.GEN_TOO_LONG.value: RecoveryAction.REPHRASE,
        ErrorCode.LLM_UNAVAILABLE.value: RecoveryAction.FALLBACK_PROVIDER,
        ErrorCode.LLM_RATE_LIMITED.value: RecoveryAction.RETRY,
        ErrorCode.DLG_TOPIC_SHIFT.value: RecoveryAction.RESET_CONTEXT,
        ErrorCode.DLG_CONTRADICTION.value: RecoveryAction.REPHRASE,
        ErrorCode.DLG_CONTEXT_LOST.value: RecoveryAction.RESET_CONTEXT,
        ErrorCode.CONFIG_INVALID.value: RecoveryAction.ESCALATE,
        ErrorCode.STORAGE_FAILURE.value: RecoveryAction.RETRY,
        ErrorCode.INTERNAL.value: RecoveryAction.ESCALATE,
    }

    def __init__(self, max_attempts: int = 2) -> None:
        self.max_attempts = max_attempts
        self.history: List[RecoveryRecord] = []
        self._counts: Dict[str, int] = {}

    def plan_for(self, error: InterviewGenieError) -> RecoveryPlan:
        code = error.code.value if isinstance(error.code, ErrorCode) else str(error.code)
        action = self._TABLE.get(code, RecoveryAction.ESCALATE)
        attempts = self._counts.get(code, 0)
        return RecoveryPlan(
            action=action,
            reason=error.message,
            attempts=attempts,
            max_attempts=self.max_attempts if error.recoverable else 0,
            payload=dict(error.context),
        )

    def record(self, error: InterviewGenieError, plan: RecoveryPlan, outcome: str, detail: str = "") -> RecoveryRecord:
        code = error.code.value if isinstance(error.code, ErrorCode) else str(error.code)
        self._counts[code] = self._counts.get(code, 0) + 1
        rec = RecoveryRecord(error=error.to_dict(), plan=plan.to_dict(), outcome=outcome, detail=detail)
        self.history.append(rec)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        LOG.warning(
            "recovery %s -> %s (%s)",
            code,
            plan.action.value,
            outcome,
            context={"detail": detail},
        )
        return rec

    def execute(self, plan: RecoveryPlan, *, detail: str = "recovered",
                outcome: Optional[str] = None) -> RecoveryRecord:
        """Record a plan as executed.

        The orchestrator performs the actual recovery work (falling back to the
        local recogniser, asking for clarification, ...); this method is the
        single place that books the attempt so retries stay bounded.
        """
        error = InterviewGenieError(plan.reason, code=ErrorCode.INTERNAL,
                                    recoverable=True, context=plan.payload)
        plan.attempts += 1
        state = outcome or ("recovered" if not plan.exhausted else "escalated")
        return self.record(error, plan, state, detail)

    def recover(self, error: InterviewGenieError, *, detail: str = "recovered",
                outcome: Optional[str] = None) -> Tuple[RecoveryPlan, RecoveryRecord]:
        """Plan for ``error`` and immediately record the plan as executed."""
        plan = self.plan_for(error)
        return plan, self.execute(plan, detail=detail, outcome=outcome)

    @property
    def error_rate(self) -> float:
        """Fraction of handled errors that were *not* recovered."""
        if not self.history:
            return 0.0
        failed = sum(1 for r in self.history if r.outcome in {"failed", "escalated"})
        return failed / len(self.history)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_errors": len(self.history),
            "error_rate": round(self.error_rate, 4),
            "by_code": dict(self._counts),
            "by_outcome": {
                outcome: sum(1 for r in self.history if r.outcome == outcome)
                for outcome in ("recovered", "failed", "escalated", "pending")
            },
        }


def guard(fn: Callable[..., Any], *, fallback: Any = None, code: ErrorCode = ErrorCode.INTERNAL) -> Callable[..., Any]:
    """Decorator that converts arbitrary exceptions into typed errors."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except InterviewGenieError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InterviewGenieError(
                f"{fn.__name__} failed: {exc}",
                code=code,
                context={"function": fn.__name__},
                cause=exc,
            ) from exc

    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    wrapper.__doc__ = fn.__doc__
    return wrapper
