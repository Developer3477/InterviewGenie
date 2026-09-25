"""Common sense and world knowledge.

The knowledge graph encodes *specific* facts.  This module encodes the
*general* knowledge an interviewer expects a competent candidate to have --
the rules that make an answer "reasonable" rather than merely plausible:

* **conceptual rules** (``if X is a distributed system then X must handle
  partial failure``),
* **category ontologies** (a *database* is a *system*; a *language* is a
  *tool*), which let the system reason about things it has never seen,
* **sanity checks** that flag internally inconsistent or physically impossible
  claims before they reach the candidate, and
* **default expectations** ("candidates should quantify impact", "design
  answers should mention trade-offs").

The plausibility score produced here feeds the response validator: a claim that
violates a rule costs confidence and triggers the error-recovery path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..logging import get_logger
from ..nlp.textnorm import tokenize
from ..types import Analysis

LOG = get_logger("knowledge.commonsense")


# --------------------------------------------------------------------------- #
# Category ontology
# --------------------------------------------------------------------------- #
#: child -> parents (multiple inheritance)
ONTOLOGY: Dict[str, Tuple[str, ...]] = {
    "language": ("tool",),
    "framework": ("tool",),
    "database": ("system",),
    "cloud": ("service",),
    "infrastructure": ("system",),
    "data": ("tool",),
    "ml": ("tool",),
    "practice": ("method",),
    "concept": ("idea",),
    "competency": ("trait",),
    "metric": ("measure",),
    "ic": ("role",),
    "management": ("role",),
    "partner": ("role",),
    "bigtech": ("company",),
    "scaleup": ("company",),
    "system": ("artifact",),
    "service": ("artifact",),
    "tool": ("artifact",),
    "method": ("artifact",),
    "artifact": ("thing",),
    "idea": ("thing",),
    "trait": ("thing",),
    "measure": ("thing",),
    "role": ("thing",),
    "company": ("thing",),
}

#: things every member of a category necessarily has
CATEGORY_PROPERTIES: Dict[str, Tuple[str, ...]] = {
    "system": ("a failure mode", "an owner", "a scaling limit"),
    "database": ("a consistency model", "a query interface", "a durability guarantee"),
    "distributed system": ("partial failure", "replication", "a consensus or coordination story"),
    "service": ("an interface", "a latency profile", "a dependency graph"),
    "language": ("a type system", "a concurrency model", "an ecosystem"),
    "tool": ("a learning curve", "an operational cost"),
    "method": ("a cost", "a benefit"),
    "company": ("a business model", "customers", "competitors"),
    "role": ("responsibilities", "success criteria"),
    "measure": ("a direction", "a definition"),
    "trait": ("an observable behaviour"),
}


# --------------------------------------------------------------------------- #
# Conceptual rules
# --------------------------------------------------------------------------- #
@dataclass
class CommonsenseRule:
    """A single piece of world knowledge expressed as a trigger and an implication."""

    name: str
    triggers: Tuple[str, ...]              # any of these terms activates the rule
    implication: str                       # what a competent answer should mention
    requires: Tuple[str, ...] = ()         # terms that should co-occur
    penalty: float = 0.25                  # confidence cost when violated
    domain: str = "general"

    def applies(self, text: str) -> bool:
        return any(trigger in text for trigger in self.triggers)


RULES: Tuple[CommonsenseRule, ...] = (
    # --- distributed systems --------------------------------------------- #
    CommonsenseRule("partial failure", ("distributed", "microservice", "replicat", "shard",
                                       "cluster", "multi region", "multi-region"),
                    "any distributed design must say what happens when a node fails",
                    ("fail", "failure", "down", "unavailable", "degrade", "retry", "timeout"),
                    domain="architecture"),
    CommonsenseRule("consistency choice", ("eventual consistency", "strong consistency",
                                          "cap theorem", "partition"),
                    "a consistency trade-off must be justified for the use case",
                    ("consistency", "trade-off", "tradeoff", "latency", "availability"),
                    domain="architecture"),
    CommonsenseRule("capacity maths", ("scale", "million", "billion", "qps", "requests per second",
                                      "throughput"),
                    "large-scale answers should include a rough capacity estimate",
                    ("qps", "rps", "per second", "storage", "bandwidth", "estimate", "roughly",
                     "approximately", "order of"),
                    domain="architecture"),
    CommonsenseRule("bottleneck naming", ("slow", "latency", "performance", "optimis", "optimiz"),
                    "performance answers must identify the actual bottleneck",
                    ("bottleneck", "profile", "measure", "index", "cache", "query plan",
                     "n+1", "hot path", "cpu", "memory", "io"),
                    domain="performance"),
    # --- data ------------------------------------------------------------- #
    CommonsenseRule("data modelling keys", ("database", "schema", "data model", "table",
                                           "postgres", "mysql", "sql"),
                    "a data model answer must mention keys and access patterns",
                    ("primary key", "foreign key", "index", "access pattern", "cardinality",
                     "normalis", "normaliz", "join"),
                    domain="data"),
    CommonsenseRule("migration safety", ("migrat", "rewrite", "refactor", "cutover"),
                    "migrations need a rollback plan and a compatibility strategy",
                    ("rollback", "backward", "dual write", "backfill", "shadow", "feature flag",
                     "compatib"),
                    domain="delivery"),
    # --- process ---------------------------------------------------------- #
    CommonsenseRule("quantified impact", ("project", "built", "led", "improved", "shipped"),
                    "accomplishment claims should be quantified",
                    ("%", "percent", "x faster", "reduced", "increased", "saved", "hours",
                     "days", "ms", "from", "to"),
                    domain="behavioural"),
    CommonsenseRule("learning from failure", ("i failed", "my mistake", "we failed",
                                             "i made a mistake", "went wrong", "incident",
                                             "postmortem", "my fault", "i got it wrong"),
                    "failure stories must show what changed afterwards",
                    ("learned", "changed", "now", "since then", "process", "guardrail",
                     "monitor", "test"),
                    domain="behavioural"),
    CommonsenseRule("conflict resolution", ("disagree", "conflict", "push back", "pushed back",
                                           "difficult"),
                    "conflict answers should show empathy plus a path to a decision",
                    ("listen", "understand", "data", "evidence", "align", "agree", "escalat",
                     "compromise", "decided"),
                    domain="behavioural"),
    CommonsenseRule("trade-off articulation", ("design", "architecture", "choose", "versus",
                                              "vs", "trade-off", "tradeoff"),
                    "design answers should name the trade-off being made",
                    ("trade-off", "tradeoff", "instead of", "rather than", "cost", "downside",
                     "because", "however"),
                    penalty=0.18, domain="design"),
    CommonsenseRule("testing mention", ("ship", "deploy", "release", "production", "code"),
                    "delivery answers should mention verification",
                    ("test", "ci", "canary", "staging", "review", "monitor", "rollout"),
                    domain="delivery"),
    CommonsenseRule("ownership signal", ("team", "we", "our"),
                    "team stories should make the candidate's own contribution explicit",
                    ("i ", "my ", "i led", "i built", "i owned", "personally", "i decided"),
                    domain="behavioural"),
    CommonsenseRule("security awareness", ("auth", "password", "api key", "secret",
                                          "pii", "encrypt", "permission", "oauth", "jwt"),
                    "security answers must mention threat modelling or least privilege",
                    ("least privilege", "threat", "encrypt", "rotate", "audit", "validate",
                     "sanitis", "sanitiz", "hash"),
                    domain="security"),
    CommonsenseRule("measurement habit", ("success", "impact", "goal", "metric", "kpi"),
                    "success claims should say how success would be measured",
                    ("metric", "measure", "kpi", "baseline", "target", "track"),
                    domain="measurement"),
)

#: expectations that apply to every answer regardless of topic
UNIVERSAL_EXPECTATIONS: Tuple[CommonsenseRule, ...] = (
    CommonsenseRule("specificity", ("*",), "answers should contain a concrete example or number",
                    ("for example", "for instance", "specifically", "in one project",
                     "last year", "we ", "i ", "%", "ms", "seconds"),
                    penalty=0.15),
    CommonsenseRule("concision", ("*",), "answers should stay under two minutes of speech",
                    (), penalty=0.05),
)


# --------------------------------------------------------------------------- #
# Plausibility scoring
# --------------------------------------------------------------------------- #
@dataclass
class PlausibilityReport:
    score: float = 1.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    satisfied: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "violations": self.violations,
            "satisfied": self.satisfied,
            "categories": self.categories,
        }


@dataclass
class CommonsenseEngine:
    """Rule-based plausibility and world-knowledge reasoning."""

    rules: Sequence[CommonsenseRule] = RULES
    ontology: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: dict(ONTOLOGY))
    category_properties: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(CATEGORY_PROPERTIES))

    # -- ontology --------------------------------------------------------- #
    def ancestors(self, category: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [category]
        while stack:
            current = stack.pop()
            for parent in self.ontology.get(current, ()):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    def is_a(self, child: str, parent: str) -> bool:
        return child == parent or parent in self.ancestors(child)

    def expected_properties(self, category: str) -> List[str]:
        out: List[str] = []
        for cat in [category, *sorted(self.ancestors(category))]:
            out.extend(self.category_properties.get(cat, ()))
        return out

    # -- checking --------------------------------------------------------- #
    def check(self, answer_text: str, question_text: str = "",
              categories: Optional[Sequence[str]] = None) -> PlausibilityReport:
        """Score how commonsensical ``answer_text`` is for ``question_text``."""
        combined = f"{question_text} {answer_text}".lower()
        report = PlausibilityReport()
        report.categories = list(categories or [])

        for rule in list(self.rules) + list(UNIVERSAL_EXPECTATIONS):
            if not rule.applies(combined):
                continue
            if not rule.requires:
                # pure length/style expectation handled by the validator
                continue
            if any(term in combined for term in rule.requires):
                report.satisfied.append(rule.name)
                report.score = min(1.0, report.score + 0.02)
            else:
                report.violations.append({
                    "rule": rule.name,
                    "domain": rule.domain,
                    "expectation": rule.implication,
                    "penalty": rule.penalty,
                })
                report.score = max(0.0, report.score - rule.penalty)

        # implied properties from the categories of any entity mentioned
        for category in report.categories:
            if not category:
                continue
            expected = self.expected_properties(category)
            missing = [prop for prop in expected
                       if prop.split() and prop.split()[0] not in combined]
            if missing and len(missing) < len(expected):
                report.score = max(0.0, report.score - 0.03 * min(3, len(missing)))
        report.score = round(report.score, 4)
        return report

    # -- inference -------------------------------------------------------- #
    def implications(self, text: str) -> List[str]:
        """Return what must be true given ``text`` (forward chaining)."""
        low = text.lower()
        out: List[str] = []
        if any(k in low for k in ("distributed", "microservice", "replicat", "shard")):
            out.append("the system must tolerate partial failure and network partitions")
        if any(k in low for k in ("database", "sql", "postgres", "mysql")):
            out.append("the data model needs keys, indexes and an explicit consistency level")
        if any(k in low for k in ("cache", "redis", "caching")):
            out.append("cache invalidation and staleness must be addressed")
        if any(k in low for k in ("queue", "kafka", "async", "asynchronous")):
            out.append("the consumer must handle duplicates, ordering and backpressure")
        if any(k in low for k in ("api", "endpoint", "rest", "graphql")):
            out.append("the interface needs versioning, auth and error semantics")
        if any(k in low for k in ("deploy", "release", "ship", "production")):
            out.append("the change needs a rollout and rollback plan")
        if any(k in low for k in ("team", "we", "colleague", "stakeholder")):
            out.append("the candidate's personal contribution should be identifiable")
        return out

    def contradictions(self, statements: Sequence[str]) -> List[Dict[str, Any]]:
        """Detect statements that cannot both be true."""
        found: List[Dict[str, Any]] = []
        lowered = [s.lower() for s in statements]
        negation_markers = ("never", "no ", "not ", "avoid", "without", "cannot", "can't")
        for i, first in enumerate(lowered):
            for j in range(i + 1, len(lowered)):
                second = lowered[j]
                shared = _shared_content(first, second)
                if not shared:
                    continue
                first_neg = any(m in first for m in negation_markers)
                second_neg = any(m in second for m in negation_markers)
                if first_neg != second_neg:
                    found.append({
                        "statements": [statements[i], statements[j]],
                        "shared_terms": shared,
                        "issue": "one statement affirms what the other denies",
                    })
        return found


def _shared_content(a: str, b: str, min_len: int = 5) -> List[str]:
    words_a = {w for w in tokenize(a, keep_punct=False) if len(w) >= min_len}
    words_b = {w for w in tokenize(b, keep_punct=False) if len(w) >= min_len}
    stop = {"about", "would", "could", "should", "there", "their", "which", "because"}
    return sorted((words_a & words_b) - stop)[:4]


#: module level singleton
_ENGINE: Optional[CommonsenseEngine] = None


def get_engine() -> CommonsenseEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CommonsenseEngine()
    return _ENGINE


def check_commonsense(answer: str, question: str = "",
                      categories: Optional[Sequence[str]] = None) -> PlausibilityReport:
    return get_engine().check(answer, question, categories)
