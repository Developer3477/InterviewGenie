"""An in-memory labelled property graph.

The graph is the system's long-term memory of the world: entities (skills,
companies, roles, concepts, metrics, ...) and typed, weighted relationships
between them.  It supports

* node/edge CRUD with secondary indices on label, property and alias,
* breadth-first and weighted (Dijkstra) traversal,
* neighbourhood expansion used by the retriever,
* a small declarative pattern-matching query language,
* JSON / triple serialisation, and
* pluggable remote backends (GraphDB, Amazon Neptune) via
  :class:`RemoteGraphBackend`.
"""

from __future__ import annotations

import heapq
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..errors import InterviewGenieError, KnowledgeError, ErrorCode
from ..logging import get_logger

LOG = get_logger("knowledge.graph")


# --------------------------------------------------------------------------- #
# Elements
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    id: str
    labels: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)

    def label(self) -> Optional[str]:
        """Primary label, preferring a specific one over the generic ``ENTITY``."""
        if not self.labels:
            return None
        specific = sorted(l for l in self.labels if l != "ENTITY")
        return specific[0] if specific else "ENTITY"

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "labels": sorted(self.labels), "properties": self.properties}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(id=data["id"], labels=set(data.get("labels", [])),
                   properties=dict(data.get("properties", {})))


@dataclass
class Edge:
    source: str
    target: str
    type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str, str]:
        return (self.source, self.type, self.target)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "type": self.type,
                "weight": self.weight, "properties": self.properties}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(source=data["source"], target=data["target"], type=data["type"],
                   weight=float(data.get("weight", 1.0)),
                   properties=dict(data.get("properties", {})))


def _as_traversed(edge: Edge, from_node: str) -> Edge:
    """Return ``edge`` oriented so ``source == from_node``.

    Neighbour lookups follow edges in both directions; re-orienting them keeps
    every :class:`Path` readable left-to-right regardless of the stored
    direction.
    """
    if edge.source == from_node:
        return edge
    return Edge(source=from_node, target=edge.source, type=edge.type,
                weight=edge.weight, properties=edge.properties)


@dataclass
class Path:
    """A walk through the graph, used as evidence for generated answers."""

    nodes: List[Node]
    edges: List[Edge]

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def weight(self) -> float:
        if not self.edges:
            return 0.0
        return math.exp(sum(math.log(max(e.weight, 1e-6)) for e in self.edges) / len(self.edges))

    def describe(self) -> str:
        if not self.nodes:
            return ""
        parts = [self.nodes[0].get("name", self.nodes[0].id)]
        for edge, node in zip(self.edges, self.nodes[1:]):
            parts.append(f"--{edge.type}--> {node.get('name', node.id)}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": [n.id for n in self.nodes],
                "edges": [e.type for e in self.edges],
                "weight": round(self.weight, 4),
                "description": self.describe()}


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
class PropertyGraph:
    """A labelled, weighted, directed multi-graph with query support."""

    def __init__(self, name: str = "interviewgenie") -> None:
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._out: Dict[str, List[Edge]] = defaultdict(list)
        self._in: Dict[str, List[Edge]] = defaultdict(list)
        self._by_label: Dict[str, Set[str]] = defaultdict(set)
        self._aliases: Dict[str, str] = {}
        self._edge_keys: Set[Tuple[str, str, str]] = set()

    # -- mutation --------------------------------------------------------- #
    def add_node(self, node_id: str, labels: Optional[Iterable[str]] = None,
                 **properties: Any) -> Node:
        node = self.nodes.get(node_id)
        if node is None:
            node = Node(id=node_id, labels=set(labels or ()), properties=dict(properties))
            self.nodes[node_id] = node
        else:
            if labels:
                node.labels.update(labels)
            node.properties.update(properties)
        for label in node.labels:
            self._by_label[label].add(node_id)
        name = node.properties.get("name")
        if name:
            self._aliases.setdefault(str(name).lower(), node_id)
        return node

    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0,
                 **properties: Any) -> Optional[Edge]:
        if source not in self.nodes or target not in self.nodes:
            missing = [n for n in (source, target) if n not in self.nodes]
            raise KnowledgeError(
                f"cannot add edge {edge_type}: unknown node(s) {missing}",
                code=ErrorCode.KG_ENTITY_UNLINKED)
        key = (source, edge_type, target)
        if key in self._edge_keys:
            existing = self._find_edge(key)
            if existing is not None:
                existing.weight = max(existing.weight, weight)
                existing.properties.update(properties)
                return existing
        edge = Edge(source=source, target=target, type=edge_type, weight=float(weight),
                    properties=dict(properties))
        self.edges.append(edge)
        self._edge_keys.add(key)
        self._out[source].append(edge)
        self._in[target].append(edge)
        return edge

    def register_alias(self, alias: str, node_id: str) -> None:
        """Map an alternative spelling (``k8s``) onto a node id."""
        if node_id in self.nodes:
            self._aliases[alias.lower()] = node_id

    def update_node(self, node_id: str, **properties: Any) -> Node:
        if node_id not in self.nodes:
            raise KnowledgeError(f"unknown node {node_id}", code=ErrorCode.KG_ENTITY_UNLINKED)
        self.nodes[node_id].properties.update(properties)
        return self.nodes[node_id]

    def reinforce(self, source: str, edge_type: str, target: str, amount: float = 0.05) -> float:
        """Strengthen an edge (used by continuous learning)."""
        edge = self._find_edge((source, edge_type, target))
        if edge is None:
            return 0.0
        edge.weight = min(1.0, edge.weight + amount)
        return edge.weight

    def weaken(self, source: str, edge_type: str, target: str, amount: float = 0.05) -> float:
        edge = self._find_edge((source, edge_type, target))
        if edge is None:
            return 0.0
        edge.weight = max(0.0, edge.weight - amount)
        return edge.weight

    def _find_edge(self, key: Tuple[str, str, str]) -> Optional[Edge]:
        for edge in self._out[key[0]]:
            if edge.type == key[1] and edge.target == key[2]:
                return edge
        return None

    # -- lookup ----------------------------------------------------------- #
    def get(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def find_by_alias(self, alias: str) -> Optional[Node]:
        node_id = self._aliases.get(alias.lower().strip())
        return self.nodes.get(node_id) if node_id else None

    def find_by_name(self, name: str) -> Optional[Node]:
        return self.find_by_alias(name)

    def by_label(self, label: str) -> List[Node]:
        return [self.nodes[n] for n in sorted(self._by_label.get(label, ()))]

    def aliases(self) -> Dict[str, str]:
        return dict(self._aliases)

    # -- traversal -------------------------------------------------------- #
    def neighbors(self, node_id: str, direction: str = "both",
                  edge_types: Optional[Sequence[str]] = None) -> List[Tuple[Edge, Node]]:
        """Return ``(edge, node)`` pairs adjacent to ``node_id``."""
        out: List[Tuple[Edge, Node]] = []
        buckets = []
        if direction in ("out", "both"):
            buckets.append(self._out.get(node_id, []))
        if direction in ("in", "both"):
            buckets.append(self._in.get(node_id, []))
        for bucket in buckets:
            for edge in bucket:
                if edge_types and edge.type not in edge_types:
                    continue
                other = self.nodes.get(edge.target if edge.source == node_id else edge.source)
                if other is not None:
                    out.append((edge, other))
        return out

    def bfs(self, start: str, max_hops: int = 2,
            edge_types: Optional[Sequence[str]] = None,
            min_weight: float = 0.0) -> List[Path]:
        """Breadth-first paths from ``start`` up to ``max_hops``."""
        if start not in self.nodes:
            return []
        results: List[Path] = []
        queue: deque = deque([(start, [self.nodes[start]], [])])
        seen: Set[Tuple[str, int]] = {(start, 0)}
        while queue:
            node_id, nodes, edges = queue.popleft()
            depth = len(edges)
            if depth >= max_hops:
                if nodes:
                    results.append(Path(nodes, edges))
                continue
            for edge, node in self.neighbors(node_id, "both", edge_types):
                if edge.weight < min_weight:
                    continue
                signature = (node.id, depth + 1)
                if signature in seen:
                    continue
                seen.add(signature)
                queue.append((node.id, nodes + [node],
                              edges + [_as_traversed(edge, node_id)]))
            if edges:
                results.append(Path(nodes, edges))
        return results

    def shortest_path(self, source: str, target: str,
                      max_hops: int = 4) -> Optional[Path]:
        """Uniform-cost search minimising hop count then edge weight."""
        if source not in self.nodes or target not in self.nodes:
            return None
        frontier: List[Tuple[int, float, str, List[Node], List[Edge]]] = [
            (0, 0.0, source, [self.nodes[source]], [])]
        best: Dict[str, float] = {source: 0.0}
        while frontier:
            hops, neg_weight, node_id, nodes, edges = heapq.heappop(frontier)
            if node_id == target:
                return Path(nodes, edges)
            if hops >= max_hops:
                continue
            for edge, node in self.neighbors(node_id, "both"):
                cost = -math.log(max(edge.weight, 1e-6))
                key = node.id
                if cost < best.get(key, float("inf")):
                    best[key] = cost
                    heapq.heappush(frontier, (hops + 1, neg_weight - cost, key,
                                              nodes + [node],
                                              edges + [_as_traversed(edge, node_id)]))
        return None

    def expand(self, seeds: Sequence[str], max_hops: int = 2,
               min_weight: float = 0.15, limit: int = 64) -> List[Path]:
        """Collect evidence paths around a set of seed nodes."""
        paths: List[Path] = []
        for seed in seeds:
            if seed not in self.nodes:
                continue
            paths.extend(self.bfs(seed, max_hops=max_hops, min_weight=min_weight))
        paths.sort(key=lambda p: (p.length, -p.weight))
        return paths[:limit]

    # -- querying --------------------------------------------------------- #
    def query(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match a declarative pattern.

        ``pattern`` looks like::

            {"start": {"label": "SKILL", "where": {"category": "language"}},
             "rel": ["requires", "related_to"],
             "hops": 2,
             "limit": 10}

        Returns a list of bindings: ``{"start": Node, "end": Node, "path": Path}``.
        """
        start_spec = pattern.get("start", {})
        relations = pattern.get("rel")
        if isinstance(relations, str):
            relations = [relations]
        hops = int(pattern.get("hops", 1))
        limit = int(pattern.get("limit", 20))
        min_weight = float(pattern.get("min_weight", 0.0))

        seeds = self._match_nodes(start_spec)
        results: List[Dict[str, Any]] = []
        for seed in seeds:
            for path in self.bfs(seed.id, max_hops=hops, edge_types=relations,
                                 min_weight=min_weight):
                if not path.edges:
                    continue
                results.append({
                    "start": path.nodes[0],
                    "end": path.nodes[-1],
                    "path": path,
                    "depth": path.length,
                })
                if len(results) >= limit:
                    return results
        return results

    def _match_nodes(self, spec: Dict[str, Any]) -> List[Node]:
        label = spec.get("label")
        where = spec.get("where", {})
        ids = spec.get("ids")
        if ids:
            candidates = [self.nodes[i] for i in ids if i in self.nodes]
        elif label:
            candidates = self.by_label(label)
        else:
            candidates = list(self.nodes.values())
        out: List[Node] = []
        for node in candidates:
            if all(node.properties.get(k) == v for k, v in where.items()):
                out.append(node)
        return out

    def search(self, text: str, limit: int = 10) -> List[Node]:
        """Fuzzy node search over names and aliases."""
        low = text.lower().strip()
        scored: List[Tuple[float, Node]] = []
        for alias, node_id in self._aliases.items():
            node = self.nodes[node_id]
            if not low:
                continue
            if alias == low:
                scored.append((1.0, node))
            elif low in alias:
                scored.append((0.75 + 0.2 * len(low) / max(1, len(alias)), node))
            elif alias in low and len(alias) > 3:
                scored.append((0.5, node))
        seen: Set[str] = set()
        out: List[Node] = []
        for score, node in sorted(scored, key=lambda kv: kv[0], reverse=True):
            if node.id in seen:
                continue
            seen.add(node.id)
            out.append(node)
            if len(out) >= limit:
                break
        return out

    # -- serialisation ---------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "aliases": self._aliases,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropertyGraph":
        graph = cls(name=data.get("name", "interviewgenie"))
        for node in data.get("nodes", []):
            graph.add_node(node["id"], node.get("labels", []), **node.get("properties", {}))
        for edge in data.get("edges", []):
            graph.add_edge(edge["source"], edge["target"], edge["type"],
                           edge.get("weight", 1.0), **edge.get("properties", {}))
        for alias, node_id in (data.get("aliases") or {}).items():
            graph.register_alias(alias, node_id)
        return graph

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False)
        LOG.info("graph saved", context={"path": path, "nodes": len(self.nodes),
                                         "edges": len(self.edges)})
        return path

    @classmethod
    def load(cls, path: str) -> "PropertyGraph":
        with open(path, "r", encoding="utf-8") as fh:
            graph = cls.from_dict(json.load(fh))
        LOG.info("graph loaded", context={"path": path, "nodes": len(graph.nodes),
                                          "edges": len(graph.edges)})
        return graph

    def to_triples(self) -> List[Tuple[str, str, str]]:
        """Export as ``(subject, predicate, object)`` triples."""
        out: List[Tuple[str, str, str]] = []
        for node in self.nodes.values():
            out.append((node.id, "rdf:type", node.label() or "Thing"))
            out.append((node.id, "rdfs:label", str(node.get("name", node.id))))
        for edge in self.edges:
            out.append((edge.source, edge.type, edge.target))
        return out

    # -- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        by_label = {label: len(ids) for label, ids in sorted(self._by_label.items())}
        by_type: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            by_type[edge.type] += 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "labels": by_label,
            "edge_types": dict(sorted(by_type.items())),
            "aliases": len(self._aliases),
        }


# --------------------------------------------------------------------------- #
# Remote backends (GraphDB / Neptune)
# --------------------------------------------------------------------------- #
class RemoteGraphBackend:
    """Thin REST adapter for GraphDB / Amazon Neptune style endpoints."""

    def __init__(self, endpoint: str, graph: str = "interviewgenie",
                 repository: str = "interviewgenie", timeout: float = 15.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.graph = graph
        self.repository = repository
        self.timeout = timeout

    def sparql(self, query: str) -> Dict[str, Any]:
        """Execute a SPARQL SELECT against the remote repository."""
        import urllib.parse
        import urllib.request

        url = (f"{self.endpoint}/repositories/{self.repository}"
               f"?query={urllib.parse.quote(query)}")
        request = urllib.request.Request(
            url, headers={"Accept": "application/sparql-results+json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            return json.loads(response.read())

    def insert_triples(self, triples: Sequence[Tuple[str, str, str]]) -> int:
        import urllib.request

        body = "\n".join(f"<{s}> <{p}> <{o}> ." for s, p, o in triples)
        request = urllib.request.Request(
            f"{self.endpoint}/repositories/{self.repository}/statements",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/x-turtle"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            return response.status

    def health(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.endpoint}/rest/repositories",
                                        timeout=self.timeout) as response:  # noqa: S310
                return response.status == 200
        except Exception:  # noqa: BLE001
            return False


def graph_from_config(config: Any) -> PropertyGraph:
    """Build the configured knowledge graph (memory, file or remote)."""
    section = config.section("knowledge") if hasattr(config, "section") else config
    backend = str(section.get("backend", "memory")).lower()
    seed_file = section.get("seed_file", "")
    path = section.get("persist_path", "")

    graph: Optional[PropertyGraph] = None
    if backend == "memory" and seed_file and os.path.exists(seed_file):
        try:
            graph = PropertyGraph.load(seed_file)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            LOG.warning("could not load seed graph (%s); building a fresh one", exc)
            graph = None
    elif backend in {"graphdb", "neptune"} and section.get("endpoint"):
        remote = RemoteGraphBackend(section["endpoint"], section.get("graph", "interviewgenie"))
        if remote.health():
            LOG.info("remote knowledge backend reachable", context={"endpoint": section["endpoint"]})
        else:
            LOG.warning("remote knowledge backend unreachable; using the local graph")

    if graph is None:
        from .seed import build_seed_graph

        graph = build_seed_graph()
    if path and backend == "memory":
        try:
            graph.save(path)
        except OSError as exc:  # pragma: no cover - disk issues
            LOG.warning("could not persist graph: %s", exc)
    return graph
