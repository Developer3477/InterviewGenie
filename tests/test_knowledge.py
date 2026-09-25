"""Tests for the knowledge graph, retrieval and common-sense layers."""

from __future__ import annotations

import unittest

from tests.base import GenieTestCase


class GraphTests(GenieTestCase):
    def test_seed_graph_is_populated(self):
        stats = self.graph.stats()
        self.assertGreater(stats["nodes"], 100)
        self.assertGreater(stats["edges"], 100)
        self.assertGreater(stats["aliases"], 100)

    def test_node_and_edge_crud(self):
        from interviewgenie.knowledge.graph import PropertyGraph

        graph = PropertyGraph("test")
        graph.add_node("a", {"THING"}, name="Alpha", description="first")
        graph.add_node("b", {"THING"}, name="Beta")
        edge = graph.add_edge("a", "b", "related_to", 0.5)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.weight, 0.5)
        self.assertEqual(len(graph.neighbors("a", "out")), 1)
        self.assertEqual(len(graph.neighbors("b", "in")), 1)

    def test_adding_an_edge_to_a_missing_node_raises(self):
        from interviewgenie.knowledge.graph import PropertyGraph
        from interviewgenie.errors import KnowledgeError

        graph = PropertyGraph("test")
        graph.add_node("a", {"THING"})
        with self.assertRaises(KnowledgeError):
            graph.add_edge("a", "missing", "related_to")

    def test_duplicate_edges_are_merged(self):
        from interviewgenie.knowledge.graph import PropertyGraph

        graph = PropertyGraph("test")
        graph.add_node("a", {"THING"})
        graph.add_node("b", {"THING"})
        graph.add_edge("a", "b", "related_to", 0.3)
        graph.add_edge("a", "b", "related_to", 0.8)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].weight, 0.8)

    def test_alias_lookup(self):
        node = self.graph.find_by_alias("k8s")
        self.assertIsNotNone(node)
        self.assertEqual(node.get("name"), "Kubernetes")

    def test_shortest_path(self):
        path = self.graph.shortest_path("skill:Redis", "skill:System design")
        self.assertIsNotNone(path)
        self.assertEqual(path.nodes[0].id, "skill:Redis")
        self.assertEqual(path.nodes[-1].id, "skill:System design")
        # every edge must be oriented along the walk, not against it
        for edge, node in zip(path.edges, path.nodes[1:]):
            self.assertEqual(edge.source, path.nodes[path.edges.index(edge)].id)
            self.assertEqual(edge.target, node.id)
        self.assertGreater(path.length, 0)

    def test_shortest_path_between_unconnected_nodes(self):
        self.assertIsNone(self.graph.shortest_path("skill:Redis", "metric:Error budget",
                                                   max_hops=2))

    def test_bfs_expansion_is_bounded(self):
        paths = self.graph.bfs("skill:Python", max_hops=2)
        self.assertTrue(paths)
        self.assertTrue(all(p.length <= 2 for p in paths))

    def test_pattern_query(self):
        results = self.graph.query({"start": {"label": "ROLE"},
                                    "rel": ["requires"], "hops": 1, "limit": 5})
        self.assertTrue(results)
        for binding in results:
            self.assertIn("start", binding)
            self.assertIn("end", binding)
            self.assertIn("path", binding)

    def test_search_ranks_exact_matches_first(self):
        results = self.graph.search("postgres")
        self.assertTrue(results)
        self.assertEqual(results[0].get("name"), "PostgreSQL")

    def test_serialisation_round_trip(self):
        from interviewgenie.knowledge.graph import PropertyGraph

        restored = PropertyGraph.from_dict(self.graph.to_dict())
        self.assertEqual(len(restored.nodes), len(self.graph.nodes))
        self.assertEqual(len(restored.edges), len(self.graph.edges))
        self.assertEqual(restored.aliases(), self.graph.aliases())

    def test_persistence_round_trip(self):
        import os
        import tempfile

        from interviewgenie.knowledge.graph import PropertyGraph

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.json")
            self.graph.save(path)
            loaded = PropertyGraph.load(path)
            self.assertEqual(len(loaded.nodes), len(self.graph.nodes))

    def test_triple_export(self):
        triples = self.graph.to_triples()
        self.assertTrue(triples)
        self.assertTrue(any(p == "rdf:type" for _s, p, _o in triples))
        self.assertTrue(any(p == "related_to" for _s, p, _o in triples))

    def test_reinforce_and_weaken(self):
        source, edge_type, target = ("concept:Caching", "related_to", "skill:Redis")
        edge = self.graph._find_edge((source, edge_type, target))
        self.assertIsNotNone(edge)
        before = edge.weight
        self.assertGreater(self.graph.reinforce(source, edge_type, target, 0.1), before)
        after = self.graph.reinforce(source, edge_type, target, 0.1)
        self.assertLess(self.graph.weaken(source, edge_type, target, 0.1), after)
        # an unknown edge is a no-op rather than an error
        self.assertEqual(self.graph.reinforce("a", "b", "c", 0.1), 0.0)


class RetrievalTests(GenieTestCase):
    def test_entities_are_linked_from_the_question(self):
        analysis = self.nlp.analyze("How would you use Kubernetes with Redis?")
        result = self.retriever.retrieve(analysis)
        names = {str(n.get("name")) for n in result.entities}
        self.assertIn("Kubernetes", names)
        self.assertIn("Redis", names)

    def test_evidence_is_returned_and_scored(self):
        analysis = self.nlp.analyze("How would you design a rate limiter with Redis?")
        result = self.retriever.retrieve(analysis)
        self.assertTrue(result.evidence)
        for evidence in result.evidence:
            self.assertTrue(evidence.text)
            self.assertGreaterEqual(evidence.score, 0.0)
            self.assertIn(evidence.kind, {"kg_path", "fact", "commonsense", "memory",
                                          "profile"})

    def test_unmatched_questions_still_produce_evidence(self):
        analysis = self.nlp.analyze("How would you architect a news feed for a million users?")
        result = self.retriever.retrieve(analysis)
        self.assertTrue(result.evidence)

    def test_queries_are_suggested(self):
        analysis = self.nlp.analyze("Tell me about Redis")
        result = self.retriever.retrieve(analysis)
        self.assertTrue(result.queries)
        self.assertTrue(all("MATCH" in q for q in result.queries))

    def test_record_usage_reinforces_edges(self):
        edge = self.graph._find_edge(("concept:Rate limiting", "related_to", "skill:Redis"))
        self.assertIsNotNone(edge)
        before = edge.weight
        self.retriever.record_usage(["skill:Redis"], reward=0.2)
        self.assertGreater(edge.weight, before)

    def test_index_is_cached_until_the_graph_changes(self):
        before = self.retriever.index
        self.retriever.rebuild_indexes()
        self.assertIs(self.retriever.index, before)
        self.retriever.record_usage(["skill:Redis"], reward=0.05)
        self.retriever.rebuild_indexes(force=True)
        self.assertIsNotNone(self.retriever.index)

    def test_retriever_stats(self):
        stats = self.retriever.stats()
        self.assertIn("graph", stats)
        self.assertIn("aliases", stats)


class CommonsenseTests(GenieTestCase):
    def test_ontology_transitivity(self):
        from interviewgenie.knowledge.commonsense import CommonsenseEngine

        engine = CommonsenseEngine()
        self.assertTrue(engine.is_a("database", "thing"))
        self.assertTrue(engine.is_a("database", "system"))
        self.assertFalse(engine.is_a("database", "trait"))

    def test_expected_properties_are_inherited(self):
        from interviewgenie.knowledge.commonsense import CommonsenseEngine

        engine = CommonsenseEngine()
        props = engine.expected_properties("database")
        self.assertIn("a failure mode", props)          # from "system"
        self.assertIn("a consistency model", props)     # from "database"

    def test_violation_is_detected(self):
        from interviewgenie.knowledge.commonsense import CommonsenseEngine

        engine = CommonsenseEngine()
        naive = engine.check("I would use a microservice because it is modern.",
                             "How would you design this?", [])
        careful = engine.check(
            "I would use a microservice and explicitly handle partial failure with "
            "retries and a timeout.", "How would you design this?", [])
        naive_rules = {v["rule"] for v in naive.violations}
        careful_rules = {v["rule"] for v in careful.violations}
        self.assertIn("partial failure", naive_rules)
        self.assertNotIn("partial failure", careful_rules)
        self.assertGreater(careful.score, naive.score)

    def test_implications_are_derived(self):
        from interviewgenie.knowledge.commonsense import CommonsenseEngine

        engine = CommonsenseEngine()
        implications = engine.implications("we will add a cache in front of the database")
        self.assertTrue(any("invalidation" in i for i in implications))

    def test_contradiction_detection(self):
        from interviewgenie.knowledge.commonsense import CommonsenseEngine

        engine = CommonsenseEngine()
        found = engine.contradictions([
            "We always deploy on Fridays",
            "We never deploy on Fridays",
        ])
        self.assertTrue(found)
        self.assertEqual(found[0]["issue"],
                         "one statement affirms what the other denies")

    def test_no_contradiction_for_consistent_statements(self):
        from interviewgenie.knowledge.commonsense import CommonsenseEngine

        engine = CommonsenseEngine()
        self.assertEqual(engine.contradictions([
            "We deploy on Fridays", "We deploy after the tests pass"]), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
