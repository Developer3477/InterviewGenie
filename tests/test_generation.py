"""Tests for generation, personalisation, learning and evaluation."""

from __future__ import annotations

import unittest

from tests.base import GenieTestCase


class ComposerTests(GenieTestCase):
    def test_answer_is_produced_for_every_intent(self):
        questions = {
            "self_introduction": "Tell me about yourself.",
            "design": "How would you design a URL shortener?",
            "knowledge": "What is the difference between TCP and UDP?",
            "people": "Tell me about a time you showed leadership.",
            "self_assessment": "What is your greatest weakness?",
            "motivation": "Why do you want to work here?",
            "troubleshooting": "This service is down, how would you debug it?",
            "meta": "Could you repeat that please?",
        }
        for intent, question in questions.items():
            response = self.ask(question)
            self.assertTrue(response.text.strip(), msg=question)
            self.assertGreater(len(response.text.split()), 5, msg=question)
            self.assertEqual(response.validation.get("plausibility", 0) >= 0, True)

    def test_answers_stay_within_the_word_budget(self):
        response = self.ask("How would you design a news feed for a hundred million users?")
        self.assertLessEqual(len(response.text.split()), 110)

    def test_answers_reference_the_question_entities(self):
        response = self.ask("How would you design a rate limiter using Redis?")
        self.assertIn("Redis", response.text)

    def test_stories_are_selected_by_topic(self):
        leadership = self.ask("Tell me about a time you showed leadership.")
        reliability = self.ask("Tell me about a time you debugged a production issue.")
        self.assertIn("schema migration", leadership.text)
        self.assertIn("payments retry", reliability.text)

    def test_scorecard_is_populated(self):
        response = self.ask("How would you design a rate limiter?")
        card = response.scorecard
        self.assertGreaterEqual(card.accuracy, 0.0)
        self.assertGreaterEqual(card.relevance, 0.0)
        self.assertGreaterEqual(card.engagement, 0.0)
        self.assertGreaterEqual(card.personalization, 0.0)
        self.assertLessEqual(card.error_rate, 1.0)
        self.assertGreaterEqual(card.overall(), 0.0)
        self.assertLessEqual(card.overall(), 1.0)

    def test_personalisation_uses_the_profile(self):
        response = self.ask("Tell me about yourself.")
        self.assertIn("Senior Software Engineer", response.text)
        self.assertIn("Python", response.text)

    def test_register_follows_the_style_vector(self):
        self.profile.style["formality"] = 0.1
        self.assertIn(self.genie.style.register(),
                      {"casual", "conversational", "professional", "formal"})
        response = self.ask("What is your greatest strength?")
        self.assertTrue(response.register)

    def test_empathy_moves_with_the_interviewer_tone(self):
        supportive = self.ask("This is a really tough problem and I am worried we cannot solve it.")
        self.assertIn(supportive.register, {"reassuring", "supportive", "calm", "steady",
                                            "professional", "engaged"})


class ValidatorTests(GenieTestCase):
    def test_empty_answer_is_blocked(self):
        from interviewgenie.generation.llm import ResponseValidator
        from interviewgenie.types import Response

        validator = ResponseValidator()
        report = validator.validate(Response(text=""))
        self.assertFalse(report["ok"])
        self.assertTrue(any(i["code"] == "empty" for i in report["issues"]))

    def test_overlong_answer_is_flagged_and_repaired(self):
        from interviewgenie.generation.llm import ResponseValidator
        from interviewgenie.types import Response

        validator = ResponseValidator(max_words=20)
        response = Response(text=" ".join(["word"] * 80))
        report = validator.validate(response)
        self.assertTrue(any(i["code"] == "too_long" for i in report["issues"]))
        repaired = validator.repair(response, report["issues"])
        self.assertLessEqual(len(repaired.text.split()), 21)

    def test_duplicate_answers_are_flagged(self):
        from interviewgenie.generation.llm import ResponseValidator
        from interviewgenie.types import Response

        validator = ResponseValidator()
        response = Response(text="I would use a cache in front of the database.")
        report = validator.validate(
            response, history=["I would use a cache in front of the database."])
        self.assertTrue(any(i["code"] == "duplicate_answer" for i in report["issues"]))

    def test_repetitive_answer_is_flagged_and_repaired(self):
        from interviewgenie.generation.llm import ResponseValidator
        from interviewgenie.types import Response

        validator = ResponseValidator(max_repetition=0.2)
        response = Response(text="Cache the cache. Cache the cache. Cache the cache.")
        report = validator.validate(response)
        self.assertTrue(any(i["code"] == "repetitive" for i in report["issues"]))
        repaired = validator.repair(response, report["issues"])
        self.assertLessEqual(len(repaired.text.split()), len(response.text.split()))

    def test_llm_falls_back_when_unconfigured(self):
        from interviewgenie.generation.llm import DeterministicLLM, llm_from_config

        client = llm_from_config({})
        self.assertIsInstance(client, DeterministicLLM)
        self.assertFalse(client.available())

    def test_openai_client_requires_a_key(self):
        from interviewgenie.errors import LLMUnavailableError
        from interviewgenie.generation.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="")
        self.assertFalse(client.available())
        with self.assertRaises(LLMUnavailableError):
            client.complete("hello")


class PersonalisationTests(GenieTestCase):
    def test_explicit_feedback_moves_the_style_vector(self):
        before = dict(self.profile.style)
        changes = self.profile.apply_feedback("that was way too long")
        self.assertTrue(changes)
        self.assertLess(self.profile.style["verbosity"], before["verbosity"])

    def test_unknown_feedback_is_a_no_op(self):
        before = dict(self.profile.style)
        self.assertEqual(self.profile.apply_feedback("hmm interesting"), {})
        self.assertEqual(self.profile.style, before)

    def test_style_is_clamped(self):
        for _ in range(20):
            self.profile.apply_feedback("much too long")
        self.assertGreaterEqual(self.profile.style["verbosity"], 0.0)
        self.assertLessEqual(self.profile.style["verbosity"], 1.0)

    def test_story_retrieval_matches_tags(self):
        stories = self.profile.stories_for("collaboration")
        self.assertTrue(stories)
        self.assertEqual(stories[0]["title"], "the schema migration standoff")

    def test_personalisation_score_rewards_profile_content(self):
        from interviewgenie.personalization.profile import IntervieweeProfile

        profile = IntervieweeProfile(name="Priya", target_role="Staff Engineer",
                                     skills=["Rust", "Kafka"])
        with_profile = profile.personalisation_score(
            "As a Staff Engineer working in Rust and Kafka I owned delivery.")
        without = profile.personalisation_score("I like computers.")
        self.assertGreater(with_profile, without)

    def test_profile_round_trip(self):
        import os
        import tempfile

        from interviewgenie.personalization.profile import IntervieweeProfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "profile.json")
            self.profile.save(path)
            loaded = IntervieweeProfile.load(path)
            self.assertEqual(loaded.name, self.profile.name)
            self.assertEqual(loaded.skills, self.profile.skills)
            self.assertEqual(loaded.style, self.profile.style)

    def test_style_adapter_parameters(self):
        from interviewgenie.personalization.profile import StyleAdapter

        adapter = StyleAdapter(profile=self.profile)
        params = adapter.parameters()
        self.assertIn("max_words", params)
        self.assertIn("register", params)
        self.assertGreater(params["max_words"], 0)
        self.assertIn(adapter.register(), {"formal", "casual", "professional",
                                           "conversational"})


class LearningTests(GenieTestCase):
    def test_feedback_is_normalised(self):
        from interviewgenie.learning.online import normalise_feedback

        positive = normalise_feedback("that was perfect")
        self.assertIsNotNone(positive)
        self.assertGreater(positive.reward, 0)

        negative = normalise_feedback("way too long")
        self.assertIsNotNone(negative)
        self.assertLess(negative.reward, 0)
        self.assertEqual(negative.action, "shorten")

        self.assertIsNone(normalise_feedback(""))

    def test_orchestrator_feedback_updates_the_profile(self):
        before = dict(self.profile.style)
        result = self.genie.feedback("that was much too technical")
        self.assertEqual(result["action"], "simplify")
        self.assertNotEqual(self.profile.style["technical_depth"],
                            before["technical_depth"])

    def test_implicit_feedback_reward_scale(self):
        self.ask("Tell me about yourself.")
        accepted = self.genie.accept()
        rejected = self.genie.accept(rejected=True)
        edited = self.genie.accept(edited=True)
        self.assertGreater(accepted["reward"], 0)
        self.assertLess(rejected["reward"], 0)
        self.assertLess(edited["reward"], 0)

    def test_supervised_correction_updates_the_classifier(self):
        from interviewgenie.types import Turn

        self.genie.ask("Tell me about a time you mentored someone.")
        turn = Turn(speaker="interviewer", text="Tell me about a time you mentored someone.")
        turn.analysis = self.nlp.analyze(turn.text)
        before = self.genie.intent_classifier.classify(turn.text)[0]
        self.genie.learner._record_correction(turn, "that is a behavioural question")
        after = self.genie.intent_classifier.classify(turn.text)[0]
        self.assertIn(after, self.genie.intent_classifier.classifier.classes)
        self.assertIn(before, self.genie.intent_classifier.classifier.classes)
        self.assertEqual(len(self.genie.learner.corrections), 1)

    def test_metrics_track_trend(self):
        from interviewgenie.learning.online import LearningMetrics
        from interviewgenie.types import Scorecard

        metrics = LearningMetrics(window=8)
        for value in (0.2, 0.25, 0.3, 0.35, 0.6, 0.65, 0.7, 0.75):
            metrics.record(Scorecard(accuracy=value, relevance=value,
                                     engagement=value, personalization=value,
                                     groundedness=value, fluency=value, error_rate=0.0))
        trend = metrics.trend()
        self.assertEqual(trend["direction"], "improving")
        self.assertGreater(trend["delta"], 0.0)
        self.assertTrue(metrics.is_improving())

    def test_consolidate_rebuilds_indexes(self):
        result = self.genie.consolidate()
        self.assertTrue(result.get("indexes_rebuilt"))


class EvaluationTests(GenieTestCase):
    def test_token_f1_and_rouge(self):
        from interviewgenie.evaluation.metrics import rouge_l, token_f1

        self.assertAlmostEqual(token_f1("a b c", "a b c"), 1.0)
        self.assertLess(token_f1("a b c", "x y z"), 0.001)
        self.assertGreater(rouge_l("the cat sat on the mat",
                                   "the cat sat on a mat"), 0.7)

    def test_semantic_similarity_of_identical_texts(self):
        from interviewgenie.evaluation.metrics import semantic_similarity

        self.assertAlmostEqual(semantic_similarity("redis is a cache",
                                                   "redis is a cache"), 1.0, places=3)

    def test_evaluator_scores_a_response(self):
        from interviewgenie.evaluation.metrics import get_evaluator

        response = self.ask("How would you design a rate limiter?")
        analysis = self.nlp.analyze("How would you design a rate limiter?")
        card = get_evaluator().score(analysis, response)
        self.assertGreaterEqual(card.overall(), 0.0)
        self.assertLessEqual(card.overall(), 1.0)

    def test_benchmark_runs_end_to_end(self):
        from interviewgenie.evaluation.dataset import default_cases
        from interviewgenie.evaluation.metrics import Benchmark

        benchmark = Benchmark(cases=default_cases()[:5])
        result = benchmark.run(self.nlp, self.composer, self.retriever)
        self.assertEqual(result.cases, 5)
        self.assertIn("accuracy", result.means)
        self.assertIn("overall", result.means)
        self.assertGreater(result.elapsed_ms, 0.0)

    def test_dataset_stats(self):
        from interviewgenie.evaluation.dataset import stats

        info = stats()
        self.assertGreater(info["cases"], 20)
        self.assertIn("design", info["by_intent"])

    def test_dataset_loading_falls_back(self):
        from interviewgenie.evaluation.dataset import default_cases, load_cases

        self.assertEqual(len(load_cases("/nonexistent/path.json")), len(default_cases()))


class DialogueTests(GenieTestCase):
    def test_topic_stack_is_maintained(self):
        self.genie.ask("How would you design a rate limiter?")
        self.genie.ask("What is the difference between TCP and UDP?")
        state = self.genie.dialogue.state
        self.assertTrue(state.topics)
        self.assertGreater(state.turn_index, 0)

    def test_slots_are_filled_from_entities(self):
        self.genie.ask("How would you use Kubernetes with Redis at Stripe?")
        slots = self.genie.dialogue.state.slots
        self.assertIn("Kubernetes", slots.get("skills_mentioned", []))
        self.assertIn("Stripe", slots.get("companies_mentioned", []))

    def test_strategy_is_chosen_from_the_intent(self):
        self.genie.ask("Tell me about yourself.")
        self.assertEqual(self.genie.dialogue.state.current_strategy, "structured")

    def test_bandit_learns(self):
        self.genie.dialogue.update_policy("direct", 1.0)
        stats = self.genie.dialogue.bandit_stats()
        self.assertEqual(stats["direct"]["pulls"], 1)
        self.assertAlmostEqual(stats["direct"]["mean_reward"], 1.0)

    def test_memory_recalls_earlier_turns(self):
        self.genie.ask("How would you design a rate limiter?")
        recalled = self.genie.memory.recall("rate limiter", k=2)
        self.assertTrue(recalled)
        self.assertIn("rate limiter", recalled[0].question.lower())

    def test_memory_context_for(self):
        self.genie.ask("Tell me about yourself.")
        snippets = self.genie.memory.context_for("yourself", k=1)
        self.assertTrue(snippets)

    def test_coreference_resolution(self):
        self.genie.ask("How would you design a URL shortener?")
        self.genie.ask("How would you scale it?")
        resolved = self.genie.dialogue.resolve_reference("How would you scale it?")
        self.assertIsInstance(resolved, list)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
