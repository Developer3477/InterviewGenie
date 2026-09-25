"""Tests for the natural-language-processing layer."""

from __future__ import annotations

import json
import math
import random
import unittest

from interviewgenie.types import Entity
from tests.base import GenieTestCase


class TextNormTests(GenieTestCase):
    def test_normalisation_is_idempotent(self):
        from interviewgenie.nlp.textnorm import normalize

        raw = "  I  can't   do it\u00a0— it's  too hard!!  "
        once = normalize(raw)
        twice = normalize(once)
        self.assertEqual(once, twice)
        self.assertIn("can not", once)
        self.assertNotIn("  ", once)

    def test_contraction_expansion(self):
        from interviewgenie.nlp.textnorm import expand_contraction

        self.assertIn("I am", expand_contraction("I'm ready"))
        self.assertIn("we have", expand_contraction("we've done it"))
        self.assertEqual(expand_contraction("nothing to expand"), "nothing to expand")

    def test_sentence_splitting_respects_abbreviations(self):
        from interviewgenie.nlp.textnorm import split_sentences

        text = ("Dr. Smith worked at Acme Inc. for 5 years. "
                "Then she moved to a startup. It was hard.")
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].startswith("Dr."))

    def test_tokenizer_spans_line_up_with_text(self):
        from interviewgenie.nlp.textnorm import Tokenizer

        tokenizer = Tokenizer()
        text = "I worked at Stripe, building APIs."
        tokens = tokenizer.tokenize(text)
        spans = tokenizer.spans(text)
        self.assertEqual(len(tokens), len(spans))
        for token, start, end in spans:
            self.assertEqual(text[start:end], token)
        self.assertEqual([t for t, _s, _e in spans], tokens)

    def test_ngrams_and_similarity(self):
        from interviewgenie.nlp.textnorm import ngrams, jaccard, edit_distance

        self.assertEqual(ngrams(["a", "b", "c"], 2), [("a", "b"), ("b", "c")])
        self.assertAlmostEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0)
        self.assertEqual(edit_distance("kitten", "sitting"), 3)


class StemmerTests(GenieTestCase):
    def test_porter_stemming(self):
        from interviewgenie.nlp.stemmer import PorterStemmer

        stemmer = PorterStemmer()
        cases = {
            "caresses": "caress", "ponies": "poni", "ties": "ti",
            "feed": "feed", "agreed": "agre", "motoring": "motor",
            "sing": "sing", "conflated": "conflat",
        }
        for word, expected in cases.items():
            self.assertEqual(stemmer.stem(word), expected, msg=word)

    def test_lemmatiser_handles_irregulars(self):
        from interviewgenie.nlp.stemmer import lemmatize

        self.assertEqual(lemmatize("went"), "go")
        self.assertEqual(lemmatize("better"), "good")
        self.assertEqual(lemmatize("children"), "child")
        self.assertEqual(lemmatize("running"), "run")

    def test_stemming_is_stable(self):
        from interviewgenie.nlp.stemmer import stem

        self.assertEqual(stem("debugging"), stem("debug"))


class TaggerTests(GenieTestCase):
    def test_pronouns_and_auxiliaries(self):
        from interviewgenie.nlp.tagger import PosTagger

        tagger = PosTagger()
        tagged = tagger.tag_text("I am sure you can do it")
        lookup = {t.text: t.tag for t in tagged}
        self.assertEqual(lookup["I"], "PRP")
        self.assertEqual(lookup["am"], "VBP")
        self.assertEqual(lookup["can"], "MD")
        self.assertEqual(lookup["you"], "PRP")

    def test_punctuation_is_never_a_noun(self):
        from interviewgenie.nlp.tagger import PosTagger

        tagger = PosTagger()
        for token in tagger.tag_text("Really? Yes, absolutely!"):
            if token.text in {"?", ",", "!", "."}:
                self.assertEqual(token.tag, "PUNCT")

    def test_determiners_and_prepositions(self):
        from interviewgenie.nlp.tagger import PosTagger

        tagger = PosTagger()
        lookup = {t.text: t.tag for t in tagger.tag_text("the cat sat on the mat")}
        self.assertEqual(lookup["the"], "DT")
        self.assertEqual(lookup["on"], "IN")
        self.assertEqual(lookup["cat"], "NN")

    def test_tag_accuracy_is_reported(self):
        from interviewgenie.nlp.tagger import tag_accuracy

        gold = [("I", "PRP"), ("am", "VBP"), ("here", "RB")]
        self.assertAlmostEqual(tag_accuracy(gold, gold), 1.0)

    def test_every_token_has_a_lemma(self):
        analysis = self.nlp.analyze("We shipped the feature behind a flag.")
        self.assertTrue(analysis.tokens)
        for token in analysis.tokens:
            self.assertTrue(token.lemma)


class IntentTests(GenieTestCase):
    def test_classifier_trains_and_reports_accuracy(self):
        from interviewgenie.nlp.intent import IntentClassifier

        scores = IntentClassifier().evaluate()
        self.assertGreater(scores["intent_accuracy"], 0.5)
        self.assertGreater(scores["topic_accuracy"], 0.4)
        self.assertGreater(scores["support"], 10)

    def test_rules_classify_the_canonical_intents(self):
        cases = {
            "Tell me about yourself.": "self_introduction",
            "How would you design a URL shortener?": "design",
            "What is the difference between TCP and UDP?": "knowledge",
            "Tell me about a time you showed leadership": "people",
            "Why do you want to work here?": "motivation",
            "Could you repeat that please?": "meta",
            "What are your salary expectations?": "logistics",
            "This service is down, how would you debug it?": "troubleshooting",
        }
        for text, expected in cases.items():
            analysis = self.nlp.analyze(text)
            self.assertEqual(analysis.intent.name, expected, msg=text)

    def test_clarification_flag(self):
        for text in ("Sorry, could you repeat that?",
                     "I did not quite follow, say that again?",
                     "Could you clarify what you mean?"):
            self.assertTrue(self.nlp.analyze(text).intent.requires_clarification, msg=text)
        self.assertFalse(
            self.nlp.analyze("How would you design a cache?").intent.requires_clarification)

    def test_wh_type_detection(self):
        analysis = self.nlp.analyze("How would you scale this?")
        self.assertEqual(analysis.intent.wh_type, "how")

    def test_online_learning_updates_the_model(self):
        from interviewgenie.nlp.intent import IntentClassifier

        classifier = IntentClassifier()
        before = classifier.classify("Tell me about a time you mentored someone")[0]
        classifier.learn("Tell me about a time you mentored someone", "past_experience")
        after = classifier.classify("Tell me about a time you mentored someone")[0]
        self.assertIn(before, classifier.classifier.classes)
        self.assertIn(after, classifier.classifier.classes)
        self.assertEqual(len(classifier.history), 1)

    def test_fallback_samples_are_not_appended_to_a_loaded_dataset(self):
        """Regression: the embedded fallback used to be appended on top of the
        real dataset, duplicating rows and mislabelling every fallback topic."""
        from interviewgenie.nlp.intent import DEFAULT_DATASET, IntentClassifier

        with open(DEFAULT_DATASET, "r", encoding="utf-8") as fh:
            on_disk = len(json.load(fh)["examples"])

        classifier = IntentClassifier()
        self.assertEqual(len(classifier.examples), on_disk)
        self.assertEqual(len(classifier.examples), 290)

    def test_fallback_samples_carry_their_own_topics(self):
        """Regression: every fallback sample was labelled topic='meta'."""
        from interviewgenie.nlp.intent import _FALLBACK_SAMPLES

        by_text = {t: topic for t, _i, topic in _FALLBACK_SAMPLES}
        self.assertEqual(by_text["explain the cap theorem"], "technical_concept")
        self.assertEqual(by_text["how would you design a url shortener"], "architecture")
        self.assertEqual(by_text["tell me about yourself"], "self")
        self.assertEqual(by_text["why do you want to work here"], "company")
        self.assertEqual(by_text["could you repeat that please"], "meta")
        self.assertNotEqual(by_text["what are your salary expectations"], "meta")

    def test_holdout_refits_the_models_on_the_training_split_only(self):
        """Regression: the old holdout assigned ``.examples`` without re-fitting,
        so the models stayed trained on the rows being scored (leakage)."""
        from interviewgenie.nlp.intent import IntentClassifier

        classifier = IntentClassifier()
        twin = classifier._clone_for_training()
        # A freshly cloned twin must not have seen any example at all.
        self.assertEqual(twin.examples, [])
        self.assertEqual(len(twin.classifier.classes), 0)

        rows = classifier.evaluate_holdout(seeds=(1, 2))
        self.assertEqual(rows["seeds"], 2)
        self.assertEqual(len(rows["per_seed"]), 2)
        # The deployed path must beat the model alone, but cannot be perfect.
        self.assertGreater(rows["intent_accuracy"], rows["model_only_intent_accuracy"])
        self.assertLess(rows["intent_accuracy"], 1.0)
        self.assertGreater(rows["topic_accuracy"], 0.4)
        self.assertLess(rows["topic_accuracy"], 1.0)

    def test_holdout_is_deterministic(self):
        from interviewgenie.nlp.intent import IntentClassifier

        first = IntentClassifier().evaluate_holdout(seeds=(3,))
        second = IntentClassifier().evaluate_holdout(seeds=(3,))
        self.assertEqual(first, second)


class SentimentTests(GenieTestCase):
    def test_polarity_direction(self):
        from interviewgenie.nlp.sentiment import SentimentAnalyzer

        analyser = SentimentAnalyzer()
        positive = analyser.analyze("I loved that project, it was excellent and rewarding")
        negative = analyser.analyze("That was a terrible, frustrating experience")
        self.assertGreater(positive.polarity, negative.polarity)
        self.assertGreater(positive.polarity, 0.0)
        self.assertLess(negative.polarity, 0.0)

    def test_negation_flips_valence(self):
        from interviewgenie.nlp.sentiment import SentimentAnalyzer

        analyser = SentimentAnalyzer()
        plain = analyser.analyze("this is good")
        negated = analyser.analyze("this is not good")
        self.assertGreater(plain.polarity, negated.polarity)

    def test_emotion_scores_sum_to_one(self):
        from interviewgenie.nlp.sentiment import SentimentAnalyzer

        emotions = SentimentAnalyzer().emotions(
            "I am really excited about this opportunity and I trust the team")
        normalised = emotions.normalised()
        self.assertAlmostEqual(sum(normalised.values()), 1.0, places=4)
        self.assertGreater(normalised["joy"], 0.0)
        self.assertGreater(normalised["trust"], 0.0)

    def test_empathy_tone_mapping(self):
        from interviewgenie.nlp.sentiment import SentimentAnalyzer

        tone, intensity = SentimentAnalyzer().empathy_tone(
            "I am so frustrated, nothing works and I am worried about the deadline")
        self.assertIn(tone, {"calm", "supportive", "reassuring", "grounded",
                             "encouraging", "steady", "warm", "engaged",
                             "professional"})
        self.assertGreaterEqual(intensity, 0.0)


class EmbeddingTests(GenieTestCase):
    def test_tfidf_search_ranks_relevant_documents_first(self):
        from interviewgenie.nlp.embeddings import TfidfIndex

        index = TfidfIndex(use_stem=True, dim=8)
        index.fit(["redis is an in memory cache",
                   "postgres is a relational database",
                   "kubernetes schedules containers"])
        hits = index.search("cache", k=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], 0)

    def test_cosine_of_identical_vectors_is_one(self):
        from interviewgenie.nlp.embeddings import cosine

        self.assertAlmostEqual(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, places=6)
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_keyword_extraction(self):
        from interviewgenie.nlp.embeddings import extract_keywords

        text = ("Distributed systems need consensus. Consensus is hard. "
                "Consensus is the reason Raft exists.")
        from interviewgenie.nlp.stemmer import stem

        keywords = [word for word, _score in extract_keywords(text, top_k=3)]
        self.assertIn(stem("consensus"), keywords)


class EntityTests(GenieTestCase):
    def test_tech_entities_are_detected(self):
        from interviewgenie.nlp.entities import EntityExtractor

        extractor = EntityExtractor()
        found = {e.text.lower() for e in extractor.extract("We use Python and Kubernetes daily")}
        self.assertIn("python", found)
        self.assertIn("kubernetes", found)

    def test_pattern_entities(self):
        from interviewgenie.nlp.entities import EntityExtractor

        extractor = EntityExtractor()
        found = {e.text for e in extractor.extract("Email me at dev@example.com")}
        self.assertTrue(any("example.com" in f for f in found))

    def test_entity_linker_resolves_aliases(self):
        from interviewgenie.nlp.entities import EntityLinker

        linker = EntityLinker()
        linker.index_aliases({"k8s": ("skill:Kubernetes", "SKILL"),
                              "postgres": ("skill:PostgreSQL", "SKILL")})
        entity = Entity(text="k8s", label="TECH")
        linked = linker.link([entity])
        self.assertEqual(linked[0].node_id, "skill:Kubernetes")
        self.assertEqual(linked[0].label, "SKILL")

    def test_gazetteer_hook(self):
        from interviewgenie.nlp.entities import register_gazetteer, EntityExtractor

        aliases = {"interviewgenie": ("skill:InterviewGenie", "SKILL")}

        register_gazetteer(lambda span: [
            (canonical, label, 0.95)
            for alias, (canonical, label) in aliases.items() if alias == span
        ])
        extractor = EntityExtractor()
        found = {e.text for e in extractor.extract("InterviewGenie is the tool")}
        self.assertIn("InterviewGenie", found)


class CorefTests(GenieTestCase):
    def test_pronoun_resolution(self):
        from interviewgenie.nlp.coref import CoreferenceResolver

        resolver = CoreferenceResolver()
        history = ["Priya joined the platform team.",
                   "She owned the delivery pipeline."]
        resolved = resolver.resolve("She shipped it", history)
        self.assertTrue(resolved)

    def test_resolution_is_deterministic(self):
        from interviewgenie.nlp.coref import CoreferenceResolver

        resolver = CoreferenceResolver()
        history = ["The team shipped the service.", "It was hard."]
        first = resolver.resolve("It was hard", history)
        second = resolver.resolve("It was hard", history)
        self.assertEqual(first, second)


class PipelineTests(GenieTestCase):
    def test_analysis_is_complete(self):
        analysis = self.nlp.analyze(
            "How would you design a rate limiter using Redis for our public API?")
        self.assertTrue(analysis.normalized)
        self.assertTrue(analysis.tokens)
        self.assertTrue(analysis.keywords)
        self.assertIsNotNone(analysis.intent)
        self.assertIsNotNone(analysis.sentiment)
        self.assertIsNotNone(analysis.emotion)
        self.assertEqual(analysis.intent.name, "design")

    def test_entities_are_linked_to_the_graph(self):
        analysis = self.nlp.analyze("How would you use Kubernetes with Redis?")
        linked = [e for e in analysis.entities if e.node_id]
        self.assertTrue(linked, "expected at least one entity linked to the KG")

    def test_history_is_capped(self):
        for i in range(120):
            self.nlp.analyze(f"question number {i}", update_history=True)
        self.assertLessEqual(len(self.nlp.history), 100)

    def test_reset_clears_history(self):
        self.nlp.analyze("something to remember")
        self.nlp.reset()
        self.assertEqual(self.nlp.history, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
