"""Named entity recognition and entity linking into the knowledge graph.

Three complementary extractors run over every utterance:

1. **gazetteer matching** against the knowledge-graph alias index (this is what
   links ``"k8s"`` to the *Kubernetes* node),
2. **capitalisation / shape patterns** for unseen proper nouns,
3. **regular-expression patterns** for dates, durations, versions, money and
   percentages.

The result is a list of :class:`~interviewgenie.types.Entity` objects carrying
character offsets and, where linking succeeded, the target KG node id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Pattern, Sequence, Set, Tuple

from ..logging import get_logger
from ..types import Entity
from .textnorm import Tokenizer, is_acronym, normalize

LOG = get_logger("nlp.entities")

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
ORG_SUFFIXES = (
    "inc", "llc", "ltd", "corp", "corporation", "company", "co", "gmbh", "plc",
    "group", "labs", "technologies", "systems", "studios", "university",
    "institute", "college", "school", "foundation", "association",
)

_DATE_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?\b", re.I),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"),
    re.compile(r"\b(19|20)\d{2}\b"),
    re.compile(r"\b(?:q[1-4]|h[12])\s*(?:of\s*)?(?:19|20)\d{2}\b", re.I),
    re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b", re.I),
)

_DURATION_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:years?|yrs?|months?|mos?|weeks?|days?|hours?|hrs?)\b", re.I)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|x|k|m|bn|b|ms|s|qps|rps|rpm)?\b", re.I)
_VERSION_PATTERN = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b")
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_URL_PATTERN = re.compile(r"\bhttps?://\S+\b")

#: multi-word technology terms that lowercase matching would otherwise miss
TECH_TERMS: Set[str] = {
    "machine learning", "deep learning", "neural network", "neural networks",
    "natural language processing", "computer vision", "reinforcement learning",
    "data science", "data engineering", "data pipeline", "data pipelines",
    "big data", "distributed systems", "distributed system", "system design",
    "operating system", "operating systems", "version control", "code review",
    "continuous integration", "continuous delivery", "continuous deployment",
    "unit test", "unit tests", "integration test", "integration tests",
    "load balancing", "rate limiting", "service mesh", "message queue",
    "message queues", "event driven", "event-driven", "domain driven design",
    "test driven development", "object oriented", "object-oriented",
    "functional programming", "design pattern", "design patterns",
    "big o", "time complexity", "space complexity", "data structure",
    "data structures", "relational database", "relational databases",
    "no sql", "nosql", "primary key", "foreign key", "database index",
    "query plan", "query optimization", "connection pool", "connection pooling",
    "cache invalidation", "write ahead log", "consensus algorithm",
    "distributed transaction", "eventual consistency", "strong consistency",
    "service level objective", "service level agreement", "error budget",
    "graceful degradation", "circuit breaker", "back pressure", "blue green",
    "canary release", "feature flag", "a b testing", "technical debt",
    "pair programming", "agile", "scrum", "kanban", "retrospective",
    "post mortem", "postmortem", "root cause analysis", "on call", "on-call",
    "incident response", "capacity planning", "cost optimization",
}

#: single tokens that are technical skills even when lower case
TECH_TOKENS: Set[str] = {
    "python", "java", "javascript", "typescript", "golang", "rust", "ruby",
    "scala", "kotlin", "swift", "matlab", "julia", "haskell", "elixir",
    "sql", "nosql", "graphql", "rest", "grpc", "http", "https", "tcp", "udp",
    "json", "yaml", "xml", "html", "css", "sass", "jsx", "tsx",
    "react", "angular", "vue", "svelte", "django", "flask", "fastapi",
    "spring", "rails", "express", "node", "deno", "bun",
    "tensorflow", "pytorch", "keras", "scikit", "pandas", "numpy", "scipy",
    "spark", "hadoop", "kafka", "rabbitmq", "celery", "airflow", "dbt",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "puppet", "chef",
    "jenkins", "gitlab", "github", "circleci", "travis",
    "aws", "gcp", "azure", "ec2", "s3", "rds", "lambda", "dynamodb",
    "postgres", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "cassandra", "elasticsearch", "opensearch", "neo4j", "clickhouse",
    "linux", "unix", "bash", "zsh", "git", "svn", "mercurial",
    "api", "apis", "sdk", "cli", "ide", "ci", "cd", "cpu", "gpu", "ram",
    "ssd", "cdn", "dns", "vpc", "iam", "sso", "jwt", "oauth", "ldap",
    "ml", "ai", "nlp", "llm", "rag", "etl", "oltp", "olap",
    "microservices", "monolith", "serverless", "devops", "sre", "mlops",
    "agile", "scrum", "kanban", "jira", "confluence", "slack",
}

_GAZETTEER_HOOK: Optional[Callable[[str], List[Tuple[str, str, float]]]] = None


def register_gazetteer(hook: Callable[[str], List[Tuple[str, str, float]]]) -> None:
    """Install a knowledge-graph lookup hook.

    The hook receives a lower-cased span and returns ``(canonical, label,
    confidence)`` triples.  It is called by :class:`EntityExtractor` to link
    mentions to graph nodes.
    """
    global _GAZETTEER_HOOK
    _GAZETTEER_HOOK = hook


@dataclass
class EntityExtractor:
    """Rule + gazetteer NER with KG linking."""

    gazetteer: Optional[Callable[[str], List[Tuple[str, str, float]]]] = None
    max_span: int = 6
    _cache: Dict[str, List[Entity]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.gazetteer is None:
            self.gazetteer = _GAZETTEER_HOOK

    # ------------------------------------------------------------------ #
    def extract(self, text: str) -> List[Entity]:
        """Return non-overlapping entities found in ``text``."""
        if not text or not text.strip():
            return []
        cached = self._cache.get(text)
        if cached is not None:
            return list(cached)
        spans: List[Entity] = []
        spans.extend(self._gazetteer_spans(text))
        spans.extend(self._tech_spans(text))
        spans.extend(self._pattern_spans(text))
        spans.extend(self._capitalized_spans(text))
        merged = _merge_spans(spans)
        self._cache[text] = merged
        if len(self._cache) > 2048:
            self._cache.clear()
        return list(merged)

    # -- strategies ------------------------------------------------------- #
    def _gazetteer_spans(self, text: str) -> List[Entity]:
        if self.gazetteer is None:
            return []
        low = text.lower()
        out: List[Entity] = []
        tokenizer = Tokenizer()
        words = [w for w in tokenizer.tokenize(text) if w.isalnum() or "-" in w or "." in w]
        # try progressively longer n-grams (up to max_span) for multi-word names
        for n in range(self.max_span, 0, -1):
            for i in range(len(words) - n + 1):
                span = " ".join(words[i : i + n])
                if len(span) < 2:
                    continue
                try:
                    hits = self.gazetteer(span.lower())
                except Exception:  # noqa: BLE001 - gazetteer must never break NLP
                    return out
                for canonical, label, conf in hits:
                    start = _find_span(text, span)
                    if start < 0:
                        continue
                    out.append(Entity(
                        text=span, label=label, start=start, end=start + len(span),
                        confidence=conf, canonical=canonical,
                    ))
                    break
        return out

    def _tech_spans(self, text: str) -> List[Entity]:
        low = text.lower()
        out: List[Entity] = []
        for phrase in TECH_TERMS:
            start = low.find(phrase)
            while start >= 0:
                if _is_word_boundary(low, start, start + len(phrase)):
                    out.append(Entity(
                        text=text[start : start + len(phrase)],
                        label="TECH", start=start, end=start + len(phrase),
                        confidence=0.72,
                    ))
                start = low.find(phrase, start + 1)
        for token, s, e in Tokenizer().spans(text):
            if token.lower() in TECH_TOKENS:
                out.append(Entity(text=token, label="TECH", start=s, end=e, confidence=0.78))
        return out

    def _pattern_spans(self, text: str) -> List[Entity]:
        out: List[Entity] = []
        for pattern, label in (
            (_EMAIL_PATTERN, "EMAIL"), (_URL_PATTERN, "URL"),
            (_DATE_PATTERNS[0], "DATE"), (_DATE_PATTERNS[1], "DATE"),
            (_DATE_PATTERNS[2], "DATE"), (_DATE_PATTERNS[3], "DATE"),
            (_DATE_PATTERNS[4], "DATE"),
            (_DURATION_PATTERN, "DURATION"), (_VERSION_PATTERN, "VERSION"),
        ):
            for match in pattern.finditer(text):
                out.append(Entity(
                    text=match.group(0), label=label,
                    start=match.start(), end=match.end(), confidence=0.85,
                ))
        return out

    def _capitalized_spans(self, text: str) -> List[Entity]:
        """Detect capitalised proper-noun sequences not already covered."""
        out: List[Entity] = []
        pattern = re.compile(r"\b([A-Z][a-zA-Z0-9&.+/-]*(?:\s+(?:of|the|and|for)?\s*[A-Z][a-zA-Z0-9&.+/-]*)*)\b")
        for match in pattern.finditer(text):
            span = match.group(1).strip()
            words = span.split()
            if not words:
                continue
            # skip sentence-initial single common words
            if len(words) == 1 and match.start() <= 1:
                continue
            if span.lower() in TECH_TOKENS or span.lower() in TECH_TERMS:
                continue
            label = "ORG" if words[-1].lower().strip(".,") in ORG_SUFFIXES else "PROPER"
            if is_acronym(span):
                label = "ACRONYM"
            out.append(Entity(
                text=span, label=label, start=match.start(), end=match.end(),
                confidence=0.62 if label == "PROPER" else 0.8,
            ))
        return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_word_boundary(text: str, start: int, end: int) -> bool:
    before_ok = start == 0 or not (text[start - 1].isalnum() or text[start - 1] in "-_")
    after_ok = end >= len(text) or not (text[end].isalnum() or text[end] in "-_")
    return before_ok and after_ok


def _find_span(text: str, span: str) -> int:
    """Locate ``span`` inside ``text`` case-insensitively, preferring boundaries."""
    low = text.lower()
    needle = span.lower()
    idx = low.find(needle)
    while idx >= 0:
        if _is_word_boundary(low, idx, idx + len(needle)):
            return idx
        idx = low.find(needle, idx + 1)
    return -1


def _merge_spans(spans: Sequence[Entity]) -> List[Entity]:
    """Resolve overlaps: keep the highest-confidence, longest span."""
    ordered = sorted(spans, key=lambda e: (e.start, -(e.end - e.start), -e.confidence))
    kept: List[Entity] = []
    for ent in ordered:
        if ent.end <= ent.start:
            continue
        overlap = False
        for existing in kept:
            if ent.start < existing.end and existing.start < ent.end:
                # overlapping: prefer higher confidence then longer span
                if (ent.confidence, ent.end - ent.start) > (existing.confidence, existing.end - existing.start):
                    kept.remove(existing)
                    kept.append(ent)
                    kept.sort(key=lambda e: e.start)
                overlap = True
                break
        if not overlap:
            kept.append(ent)
    kept.sort(key=lambda e: e.start)
    return kept


# --------------------------------------------------------------------------- #
# Entity linking
# --------------------------------------------------------------------------- #
@dataclass
class EntityLinker:
    """Link extracted entities to knowledge-graph nodes."""

    alias_index: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # alias -> (node_id, label)
    fuzzy_threshold: float = 0.86

    def index_aliases(self, aliases: Dict[str, Tuple[str, str]]) -> None:
        self.alias_index = {k.lower(): v for k, v in aliases.items()}

    def link(self, entities: Sequence[Entity]) -> List[Entity]:
        out: List[Entity] = []
        for ent in entities:
            key = ent.text.lower().strip()
            hit = self.alias_index.get(key)
            if hit is None and len(key) > 3:
                hit = self._fuzzy(key)
            if hit is not None:
                ent.node_id, label = hit
                if ent.label in {"PROPER", "TECH", "ACRONYM", "ORG"} or ent.confidence < 0.7:
                    ent.label = label
                ent.canonical = ent.text
                ent.confidence = max(ent.confidence, 0.9)
            out.append(ent)
        return out

    def _fuzzy(self, key: str) -> Optional[Tuple[str, str]]:
        from .textnorm import similarity

        best: Optional[Tuple[str, str]] = None
        best_score = self.fuzzy_threshold
        for alias, hit in self.alias_index.items():
            if abs(len(alias) - len(key)) > 3:
                continue
            score = similarity(key, alias)
            if score > best_score:
                best_score = score
                best = hit
        return best


#: module level singleton
_DEFAULT_EXTRACTOR = EntityExtractor()


def extract_entities(text: str) -> List[Entity]:
    return _DEFAULT_EXTRACTOR.extract(text)
