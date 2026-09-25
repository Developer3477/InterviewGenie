"""Knowledge retrieval: turn an analysed question into ranked evidence.

The retriever is the bridge between language and the graph:

1. **entity linking** -- map mentions in the utterance onto KG nodes using the
   alias index (with fuzzy fallback),
2. **concept matching** -- keywords and topics are matched against node names
   and descriptions using the TF-IDF / LSA index,
3. **neighbourhood expansion** -- weighted BFS around the matched nodes collects
   typed paths,
4. **path ranking** -- paths are scored by edge weight, hop count, and overlap
   with the question's key terms,
5. **corpus retrieval** -- a second index over node descriptions and answer
   snippets supplies prose evidence for the composer.

The result is a list of :class:`~interviewgenie.types.Evidence` objects, each
traceable back to the graph element that produced it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..errors import EntityLinkError, KnowledgeError
from ..logging import get_logger
from ..nlp.embeddings import TfidfIndex, sparse_cosine
from ..nlp.entities import EntityLinker
from ..types import Analysis, Evidence
from .graph import Edge, Node, Path, PropertyGraph

LOG = get_logger("knowledge.retrieval")


#: canonical concept clusters used when the question names no known entity.
#: These are the nodes an expert would reach for given the *kind* of question.
INTENT_CONCEPTS: Dict[str, Tuple[str, ...]] = {
    "design": ("System design", "Load balancing", "Caching", "Sharding",
               "Rate limiting", "Latency", "Availability", "Microservices",
               "Replication", "Queue"),
    "troubleshooting": ("Observability", "Incident response", "Latency",
                        "Error rate", "Postmortem", "Performance tuning",
                        "Race condition", "Circuit breaker"),
    "knowledge": ("Distributed systems", "Data structures", "Algorithms",
                  "Index", "Transaction isolation", "Concurrency"),
    "coding": ("Data structures", "Algorithms", "Big O notation", "Hash table",
               "Testing"),
    "measurement": ("Service level objective", "Error budget", "Availability",
                    "p99 latency", "A/B test"),
    "past_experience": ("Ownership", "Problem solving", "Communication"),
    "people": ("Collaboration", "Conflict resolution", "Mentoring",
               "Influence without authority", "Psychological safety"),
    "self_assessment": ("Craft", "Communication", "Ownership"),
    "background": ("Testing", "CI/CD", "Code review", "Observability"),
    "motivation": ("Ownership", "Craft", "Communication"),
    "goals": ("Ownership", "Mentorship", "Decision making"),
    "culture": ("Psychological safety", "Communication", "Collaboration"),
    "hypothetical": ("Decision making", "Problem solving", "Technical debt"),
    "self_introduction": ("Ownership", "Communication", "Problem solving"),
    "logistics": (),
    "meta": (),
}


@dataclass
class RetrievalResult:
    """Everything the generator needs to ground a response."""

    entities: List[Node] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    paths: List[Path] = field(default_factory=list)
    unmatched_terms: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    question_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [{"id": n.id, "name": n.get("name", n.id), "label": n.label()}
                         for n in self.entities],
            "evidence": [e.to_dict() for e in self.evidence],
            "paths": [p.to_dict() for p in self.paths],
            "unmatched_terms": self.unmatched_terms,
            "queries": self.queries,
        }


@dataclass
class KnowledgeRetriever:
    """Rank evidence from the knowledge graph for a single question."""

    graph: PropertyGraph
    max_hops: int = 2
    max_evidence: int = 8
    min_edge_weight: float = 0.15
    linker: EntityLinker = field(default_factory=EntityLinker)
    index: Optional[TfidfIndex] = None
    _alias_index: Dict[str, Tuple[str, str]] = field(default_factory=dict, repr=False)
    _corpus: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _dirty: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        self._alias_index = {}
        self._corpus = []
        if self.linker is None:
            self.linker = EntityLinker()
        self.rebuild_indexes()

    # -- indexes ---------------------------------------------------------- #
    def rebuild_indexes(self, force: bool = False) -> None:
        """Recompute the alias and text indexes from the graph.

        Index construction is relatively expensive, so it is skipped unless the
        graph has actually changed since the last build.
        """
        if not self._dirty and not force:
            return
        self._dirty = False
        self._alias_index = {}
        documents: List[str] = []
        corpus: List[Dict[str, Any]] = []
        for alias, node_id in self.graph.aliases().items():
            node = self.graph.get(node_id)
            if node is not None:
                self._alias_index[alias] = (node_id, node.label() or "ENTITY")
        for node in self.graph.nodes.values():
            name = str(node.get("name", node.id))
            description = str(node.get("description", ""))
            category = str(node.get("category", node.label() or ""))
            doc = f"{name} | {category} | {description}"
            documents.append(doc)
            corpus.append({"node": node, "document": doc})
        self._corpus = corpus
        self.index = TfidfIndex(use_stem=True, dim=48)
        self.index.fit(documents)
        if self.linker is not None:
            self.linker.index_aliases(self._alias_index)
        LOG.debug("retriever indexes rebuilt", context={
            "aliases": len(self._alias_index), "documents": len(documents)})

    # -- linking ---------------------------------------------------------- #
    def link_entities(self, analysis: Analysis) -> List[Node]:
        """Resolve the entities and keywords of ``analysis`` to graph nodes."""
        found: Dict[str, Node] = {}
        for entity in analysis.entities:
            hit = self._alias_index.get(entity.text.lower().strip())
            if hit is None:
                hit = self._fuzzy_alias(entity.text.lower().strip())
            if hit is not None:
                node = self.graph.get(hit[0])
                if node is not None:
                    found[node.id] = node
                    entity.node_id = node.id
                    entity.canonical = str(node.get("name", node.id))
                    entity.label = node.label() or entity.label

        # keywords and content words as a second pass
        for term, _score in analysis.keywords:
            hit = self._alias_index.get(term)
            if hit is None and len(term) > 4:
                hit = self._fuzzy_alias(term)
            if hit is not None:
                node = self.graph.get(hit[0])
                if node is not None:
                    found.setdefault(node.id, node)

        # Text-search fallback for multi-word concepts.  It is deliberately
        # strict: a loosely-similar LSA neighbour or a single shared word is not
        # enough to claim the question is "about" an entity.
        if not found and self.index is not None and self.index.fitted:
            low = analysis.text.lower()
            for idx, score in self.index.bm25(analysis.text, k=8):
                if score < 1.5:
                    continue
                node = self._corpus[idx]["node"]
                name = str(node.get("name", node.id)).lower()
                if not name:
                    continue
                if name in low or _term_overlap(name, analysis.text) >= 0.75:
                    found.setdefault(node.id, node)
        return list(found.values())

    def _fuzzy_alias(self, term: str) -> Optional[Tuple[str, str]]:
        from ..nlp.textnorm import similarity

        if len(term) < 4:
            return None
        best: Optional[Tuple[str, str]] = None
        best_score = 0.84
        for alias, hit in self._alias_index.items():
            if abs(len(alias) - len(term)) > 3:
                continue
            score = similarity(term, alias)
            if score > best_score:
                best_score = score
                best = hit
        return best

    # -- retrieval -------------------------------------------------------- #
    def retrieve(self, analysis: Analysis, intent_name: Optional[str] = None) -> RetrievalResult:
        """Collect ranked evidence for an analysed question."""
        result = RetrievalResult(question_text=analysis.text)
        entities = self.link_entities(analysis)
        result.entities = entities

        seeds = [n.id for n in entities]
        intent_name = (analysis.intent.name if analysis.intent else "") or ""
        if not seeds:
            # no entity was named, so reason from the concepts that define this
            # kind of question; they are surfaced as *inferred* entities.
            inferred: List[Node] = []
            for concept in INTENT_CONCEPTS.get(intent_name, ()):
                node = self.graph.find_by_name(concept)
                if node is not None:
                    seeds.append(node.id)
                    inferred.append(node)
            if inferred:
                entities = inferred
                result.entities = inferred
        if not seeds:
            # fall back to the best text match so the answer is never empty
            if self.index is not None and self.index.fitted:
                for idx, score in self.index.bm25(analysis.text, k=3):
                    if score >= 1.0:
                        seeds.append(self._corpus[idx]["node"].id)
                if not seeds:
                    for idx, score in self.index.search(analysis.text, k=3):
                        if score > 0.35:
                            seeds.append(self._corpus[idx]["node"].id)

        paths = self.graph.expand(seeds, max_hops=self.max_hops,
                                  min_weight=self.min_edge_weight, limit=48)
        ranked = self._rank_paths(paths, analysis)
        result.paths = ranked

        for path in ranked[: self.max_evidence]:
            result.evidence.append(Evidence(
                kind="kg_path",
                text=self._describe_path(path),
                source=path.nodes[0].get("name", path.nodes[0].id),
                score=round(path.weight, 4),
                nodes=[n.id for n in path.nodes],
                relation=path.edges[0].type if path.edges else None,
            ))

        # entities are ranked so that the most question-relevant ones are first
        result.entities = _rank_entities(result.entities, analysis.text)

        # prose evidence from the corpus index
        if self.index is not None and self.index.fitted:
            for idx, score in self.index.search(analysis.text, k=4):
                if score < 0.10:
                    continue
                node = self._corpus[idx]["node"]
                description = str(node.get("description", ""))
                if not description:
                    continue
                result.evidence.append(Evidence(
                    kind="fact",
                    text=f"{node.get('name', node.id)}: {description}",
                    source=str(node.get("name", node.id)),
                    score=round(score, 4),
                    nodes=[node.id],
                ))

        # de-duplicate while preserving rank order
        seen: Set[str] = set()
        deduped: List[Evidence] = []
        for ev in result.evidence:
            if ev.text in seen:
                continue
            seen.add(ev.text)
            deduped.append(ev)
        result.evidence = deduped[: self.max_evidence]

        linked_terms = {str(n.get("name", n.id)).lower() for n in entities}
        result.unmatched_terms = [
            term for term, _ in analysis.keywords
            if term not in linked_terms and len(term) > 3
        ][:6]
        result.queries = self._suggest_queries(analysis, entities)
        return result

    # -- ranking ---------------------------------------------------------- #
    def _rank_paths(self, paths: Sequence[Path], analysis: Analysis) -> List[Path]:
        question_terms = set(t.lower() for t, _ in analysis.keywords)
        for path in paths:
            score = path.weight
            # reward short paths
            score *= 1.0 / (1.0 + 0.25 * path.length)
            # reward overlap with the question's key terms
            path_text = path.describe().lower()
            overlap = sum(1 for term in question_terms if term in path_text)
            score *= 1.0 + 0.35 * overlap
            # prefer paths that end on a well-described node
            end_node = path.nodes[-1]
            if end_node.get("description"):
                score *= 1.1
            path._score = score  # type: ignore[attr-defined]
        return sorted(paths, key=lambda p: getattr(p, "_score", 0.0), reverse=True)

    @staticmethod
    def _describe_path(path: Path) -> str:
        if not path.nodes:
            return ""
        parts = [str(path.nodes[0].get("name", path.nodes[0].id))]
        for edge, node in zip(path.edges, path.nodes[1:]):
            parts.append(f"{edge.type} {node.get('name', node.id)}")
        return " -> ".join(parts)

    def _suggest_queries(self, analysis: Analysis, entities: Sequence[Node]) -> List[str]:
        out: List[str] = []
        for node in entities[:4]:
            name = str(node.get("name", node.id))
            out.append(f"MATCH (n {{name: '{name}'}})-[r]->(m) RETURN n, r, m")
        return out

    # -- learning --------------------------------------------------------- #
    def record_usage(self, node_ids: Sequence[str], reward: float = 0.05) -> None:
        """Reinforce the edges between nodes that supported a good answer."""
        self._dirty = True
        for node_id in node_ids:
            for edge, _node in self.graph.neighbors(node_id, "both"):
                if reward >= 0:
                    edge.weight = min(1.0, edge.weight * (1.0 + reward))
                else:
                    edge.weight = max(0.05, edge.weight * (1.0 + reward))

    def stats(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.stats(),
            "aliases": len(self._alias_index),
            "index": self.index.stats() if self.index else None,
        }


def _term_overlap(name: str, text: str) -> float:
    """Fraction of the words of ``name`` that also occur in ``text``."""
    from ..nlp.textnorm import tokenize

    name_words = {w.lower() for w in tokenize(name, keep_punct=False) if len(w) > 2}
    if not name_words:
        return 0.0
    text_words = {w.lower() for w in tokenize(text, keep_punct=False)}
    return len(name_words & text_words) / len(name_words)


#: module level singleton
_RETRIEVER: Optional[KnowledgeRetriever] = None


def get_retriever(graph: PropertyGraph) -> KnowledgeRetriever:
    global _RETRIEVER
    if _RETRIEVER is None or _RETRIEVER.graph is not graph:
        _RETRIEVER = KnowledgeRetriever(graph=graph)
    return _RETRIEVER


def _rank_entities(entities: Sequence[Node], question: str) -> List[Node]:
    """Order entities so the best "subject" candidates come first."""
    low = question.lower()
    label_rank = {"SKILL": 0, "CONCEPT": 1, "COMPANY": 2, "ROLE": 3,
                  "COMPETENCY": 4, "METRIC": 5}
    scored: List[Tuple[int, int, Node]] = []
    for index, node in enumerate(entities):
        name = str(node.get("name", node.id))
        in_question = 0 if name.lower() in low else 1
        scored.append((in_question, label_rank.get(str(node.label()), 6) + index * 0.01, node))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [node for _a, _b, node in scored]
