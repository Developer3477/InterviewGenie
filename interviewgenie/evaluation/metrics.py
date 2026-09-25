"""Evaluation: the five required metrics plus the benchmark harness.

The five metrics the system is judged on are

* **Response Accuracy** -- is the answer factually consistent with the knowledge
  graph and free of commonsense violations?
* **Response Relevance** -- does it answer the question that was asked?
* **Engagement** -- is it concrete, specific and interesting?
* **Personalization** -- does it use this candidate's own material and style?
* **Error Rate** -- how often does it fail, ask for clarification, or produce an
  answer that has to be repaired?

:class:`Evaluator` scores a generated response against a question.  A
:class:`Benchmark` runs the same scoring over a labelled dataset so the numbers
are reproducible and comparable between runs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..errors import EvaluationError
from ..logging import get_logger
from ..nlp.textnorm import tokenize
from ..types import Analysis, Response, Scorecard

LOG = get_logger("evaluation.metrics")


# --------------------------------------------------------------------------- #
# Reference answer similarity
# --------------------------------------------------------------------------- #
def token_f1(answer: str, reference: str) -> float:
    """Token-level F1 between the answer and a reference answer."""
    a = [w.lower() for w in tokenize(answer, keep_punct=False)]
    b = [w.lower() for w in tokenize(reference, keep_punct=False)]
    if not a or not b:
        return 0.0
    from collections import Counter

    common = Counter(a) & Counter(b)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def rouge_l(answer: str, reference: str) -> float:
    """Longest-common-subsequence based ROUGE-L F1."""
    a = [w.lower() for w in tokenize(answer, keep_punct=False)]
    b = [w.lower() for w in tokenize(reference, keep_punct=False)]
    if not a or not b:
        return 0.0
    # DP over the LCS length
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    lcs = previous[len(b)]
    if lcs == 0:
        return 0.0
    precision = lcs / len(a)
    recall = lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(answer: str, reference: str, index: Any = None) -> float:
    """TF-IDF/LSA cosine between the answer and a reference, if an index exists."""
    if index is None:
        from ..nlp.embeddings import TfidfIndex

        index = TfidfIndex(use_stem=True, dim=32)
        index.fit([answer, reference])
    try:
        va = index.transform_dense(answer)
        vb = index.transform_dense(reference)
    except Exception:  # noqa: BLE001
        return token_f1(answer, reference)
    if not va or not vb:
        return token_f1(answer, reference)
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(y * y for y in vb) ** 0.5
    if na == 0 or nb == 0:
        return token_f1(answer, reference)
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
@dataclass
class Evaluator:
    """Scores a single generated response."""

    def score(self, analysis: Analysis, response: Response,
              reference: Optional[str] = None,
              grounded_entities: Optional[Sequence[str]] = None) -> Scorecard:
        metrics = response.metrics or {}
        scorecard = Scorecard(
            accuracy=self._accuracy(analysis, response, reference, grounded_entities),
            relevance=metrics.get("relevance", self._relevance(response, analysis)),
            engagement=metrics.get("engagement", 0.0),
            personalization=metrics.get("personalization", 0.0),
            groundedness=metrics.get("groundedness", 0.0),
            fluency=metrics.get("fluency", 0.0),
            empathy=metrics.get("empathy", 0.0),
            error_rate=self._error_rate(analysis, response),
        )
        scorecard.detail = {k: round(v, 4) for k, v in metrics.items()}
        return scorecard

    # -- components -------------------------------------------------------- #
    def _accuracy(self, analysis: Analysis, response: Response,
                  reference: Optional[str],
                  grounded_entities: Optional[Sequence[str]]) -> float:
        if reference:
            return round(0.5 * token_f1(response.text, reference)
                         + 0.3 * rouge_l(response.text, reference)
                         + 0.2 * semantic_similarity(response.text, reference), 4)
        # no reference: use plausibility plus KG support
        plausibility = float(response.validation.get("plausibility", 0.5))
        grounded = float(response.metrics.get("groundedness", 0.0))
        return round(0.6 * plausibility + 0.4 * grounded, 4)

    def _relevance(self, response: Response, analysis: Analysis) -> float:
        question_terms = {w.lower() for w in analysis.text.split() if len(w) > 4}
        answer_terms = {w.lower() for w in response.text.split() if len(w) > 4}
        if not question_terms:
            return 0.5
        return round(len(question_terms & answer_terms) / len(question_terms), 4)

    @staticmethod
    def _error_rate(analysis: Analysis, response: Response) -> float:
        errors = 0.0
        if not response.text.strip():
            errors += 1.0
        violations = response.validation.get("violations", []) or []
        errors += 0.25 * min(4, len(violations))
        if response.validation.get("repaired"):
            errors += 0.5
        if response.confidence < 0.35:
            errors += 0.5
        if analysis.intent is not None and analysis.intent.requires_clarification:
            errors += 0.25
        return round(min(1.0, errors), 4)


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkCase:
    """One labelled evaluation case."""

    question: str
    reference_answer: str = ""
    intent: str = ""
    topic: str = ""
    key_terms: List[str] = field(default_factory=list)
    must_mention: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkCase":
        return cls(
            question=data["question"],
            reference_answer=data.get("reference_answer", ""),
            intent=data.get("intent", ""),
            topic=data.get("topic", ""),
            key_terms=list(data.get("key_terms", [])),
            must_mention=list(data.get("must_mention", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "reference_answer": self.reference_answer,
                "intent": self.intent, "topic": self.topic,
                "key_terms": self.key_terms, "must_mention": self.must_mention}


@dataclass
class BenchmarkResult:
    """Aggregate benchmark numbers."""

    cases: int = 0
    means: Dict[str, float] = field(default_factory=dict)
    by_intent: Dict[str, Dict[str, float]] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"cases": self.cases, "means": self.means,
                "by_intent": self.by_intent, "failures": self.failures[:20],
                "elapsed_ms": round(self.elapsed_ms, 2)}

    def summary(self) -> str:
        lines = [f"cases: {self.cases}"]
        for key, value in self.means.items():
            lines.append(f"  {key:16s} {value:.4f}")
        if self.failures:
            lines.append(f"  worst cases: {len(self.failures)}")
        return "\n".join(lines)


@dataclass
class Benchmark:
    """Runs a labelled case set through the full pipeline and scores it."""

    evaluator: Evaluator = field(default_factory=Evaluator)
    cases: List[BenchmarkCase] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "Benchmark":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cases = [BenchmarkCase.from_dict(c) for c in payload.get("cases", [])]
        return cls(cases=cases)

    def run(self, pipeline: Any, composer: Any, retriever: Any,
            limit: Optional[int] = None) -> BenchmarkResult:
        """Evaluate every case.

        ``pipeline`` is an :class:`~interviewgenie.nlp.pipeline.NLPPipeline`,
        ``composer`` a :class:`~interviewgenie.generation.composer.ResponseComposer`
        and ``retriever`` a
        :class:`~interviewgenie.knowledge.retrieval.KnowledgeRetriever`.
        """
        started = time.perf_counter()
        cases = self.cases[:limit] if limit else self.cases
        scorecards: List[Scorecard] = []
        by_intent: Dict[str, List[Scorecard]] = {}
        failures: List[Dict[str, Any]] = []

        for case in cases:
            try:
                analysis = pipeline.analyze(case.question)
                retrieval = retriever.retrieve(analysis)
                response = composer.compose(analysis, retrieval, "structured",
                                            question_text=case.question)
                scorecard = self.evaluator.score(analysis, response,
                                                 case.reference_answer or None)
                scorecards.append(scorecard)
                by_intent.setdefault(case.intent or "unknown", []).append(scorecard)
                if scorecard.overall() < 0.45:
                    failures.append({
                        "question": case.question,
                        "overall": round(scorecard.overall(), 4),
                        "answer": response.text[:160],
                    })
            except Exception as exc:  # noqa: BLE001 - benchmark must continue
                LOG.warning("benchmark case failed: %s", exc)
                failures.append({"question": case.question, "error": str(exc)})

        result = BenchmarkResult(
            cases=len(scorecards),
            means=self._means(scorecards),
            by_intent={intent: self._means(cards)
                       for intent, cards in sorted(by_intent.items())},
            failures=failures,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return result

    @staticmethod
    def _means(scorecards: Sequence[Scorecard]) -> Dict[str, float]:
        if not scorecards:
            return {}
        keys = ["accuracy", "relevance", "engagement", "personalization",
                "groundedness", "fluency", "error_rate"]
        out = {k: round(sum(getattr(s, k) for s in scorecards) / len(scorecards), 4)
               for k in keys}
        out["overall"] = round(sum(s.overall() for s in scorecards) / len(scorecards), 4)
        return out


#: module level singleton
_EVALUATOR: Optional[Evaluator] = None


def get_evaluator() -> Evaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = Evaluator()
    return _EVALUATOR
