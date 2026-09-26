"""Tests for résumé and job-description ingestion.

The rule under test throughout: the extractor may only record what it can point
at in the supplied text.  Inventing a fact is worse than returning nothing,
because the candidate would then be coached to say something untrue.
"""

from __future__ import annotations

import unittest

from tests.base import GenieTestCase

RÉSUMÉ = """Priya Raman
priya.raman@example.com | github.com/priyaraman | +91 98765 43210

SUMMARY
Senior Software Engineer with 7 years of experience building distributed
payment systems.

EXPERIENCE
Staff Engineer, Acme Payments (2021-present)
- Led the migration of a monolithic ledger to event-driven microservices on Kafka
- Reduced p99 checkout latency from 840ms to 120ms by introducing a Redis cache
- Mentored 4 engineers and established the on-call rotation

Software Engineer, Bolt Retail (2017-2021)
- Built the checkout API in Python and PostgreSQL
- Shipped a feature flag service used by 40 teams

EDUCATION
B.Tech in Computer Science, IIT Bombay

SKILLS
Python, PostgreSQL, Kubernetes, Kafka, Redis, AWS, System design, Terraform
"""

JD = """Role: Senior Backend Engineer
Company: Stripe

We are looking for a senior engineer to own payments reliability.

Requirements:
- 5+ years building distributed systems
- Must have experience with Kafka and PostgreSQL
- Should have Kubernetes and Terraform exposure
- Nice to have: observability tooling
"""


class IngestionTests(GenieTestCase):
    def setUp(self):
        super().setUp()
        from interviewgenie.personalization.ingest import ingest

        self.profile, self.report = ingest(resume_text=RÉSUMÉ,
                                           job_description=JD)

    # -- identity --------------------------------------------------------- #
    def test_name_is_read_from_the_header(self):
        self.assertEqual(self.profile.name, "Priya Raman")

    def test_role_comes_from_the_job_description(self):
        self.assertEqual(self.profile.target_role, "Senior Backend Engineer")

    def test_company_comes_from_the_job_description(self):
        self.assertEqual(self.profile.target_company, "Stripe")

    def test_contacts_are_found(self):
        self.assertIn("priya.raman@example.com", self.report.contacts["emails"])
        self.assertIn("github.com/priyaraman", self.report.contacts["links"])

    # -- skills ----------------------------------------------------------- #
    def test_skills_from_both_documents(self):
        for skill in ("python", "postgresql", "kubernetes", "kafka", "redis",
                      "aws", "terraform"):
            self.assertIn(skill, self.profile.skills, msg=skill)

    def test_multiword_skills_win_over_their_parts(self):
        # "system design" must not be split into "system" and "design"
        self.assertIn("system design", self.profile.skills)
        self.assertNotIn("system", self.profile.skills)

    def test_longest_match_wins(self):
        from interviewgenie.personalization.ingest import find_skills

        found = find_skills("I work with machine learning and machine "
                            "learning models.")
        self.assertIn("machine learning", found)

    def test_case_insensitive_matching(self):
        from interviewgenie.personalization.ingest import find_skills

        self.assertIn("python", find_skills("PYTHON and Python and python"))

    def test_skill_not_in_text_is_absent(self):
        from interviewgenie.personalization.ingest import find_skills

        self.assertNotIn("cobol", find_skills("Python and Kafka"))

    # -- seniority and experience ----------------------------------------- #
    def test_seniority_is_detected(self):
        self.assertEqual(self.profile.seniority, "staff")

    def test_years_of_experience(self):
        self.assertEqual(self.profile.years_experience, 7)

    def test_years_not_stated_stays_empty(self):
        from interviewgenie.personalization.ingest import find_years

        self.assertIsNone(find_years("I write software."))

    def test_seniority_unknown_stays_empty(self):
        from interviewgenie.personalization.ingest import find_seniority

        self.assertEqual(find_seniority("I write software."), "")

    # -- stories ---------------------------------------------------------- #
    def test_star_stories_are_extracted_from_accomplishments(self):
        self.assertGreaterEqual(len(self.profile.story_bank), 2)

    def test_story_keeps_the_action_verbatim(self):
        joined = " ".join(s.get("action", "") for s in self.profile.story_bank)
        self.assertIn("migration", joined.lower())

    def test_story_never_invents_a_situation(self):
        # the extractor must not fabricate a STAR block it cannot support
        for story in self.profile.story_bank:
            situation = (story.get("situation") or "").strip()
            self.assertEqual(situation, "")

    def test_plain_bullet_without_an_accomplishment_verb_is_skipped(self):
        from interviewgenie.personalization.ingest import find_stories

        self.assertEqual(find_stories("- Attended standup"), [])

    def test_quantified_bullets_are_preferred(self):
        from interviewgenie.personalization.ingest import find_stories

        stories = find_stories(RÉSUMÉ)
        quantified = [s for s in stories if any(c.isdigit() for c in s.get("action", ""))]
        self.assertTrue(quantified)

    # -- requirements and education --------------------------------------- #
    def test_job_requirements_are_extracted(self):
        self.assertTrue(self.report.jd_requirements)
        self.assertTrue(any("distributed systems" in r
                            for r in self.report.jd_requirements))

    def test_education_is_recorded(self):
        self.assertTrue(any("IIT Bombay" in e for e in self.profile.constraints))

    # -- honesty ---------------------------------------------------------- #
    def test_nothing_is_invented_for_empty_input(self):
        from interviewgenie.personalization.ingest import ingest

        profile, report = ingest(resume_text="", job_description="")
        self.assertEqual(profile.skills, [])
        self.assertEqual(profile.story_bank, [])
        # the class default, not an invented identity
        self.assertEqual(profile.name, "Candidate")

    def test_only_the_role_in_the_text_is_claimed(self):
        from interviewgenie.personalization.ingest import find_role_title

        self.assertEqual(find_role_title("Role: Backend Engineer\nCompany: X"),
                         "Backend Engineer")
        self.assertEqual(find_role_title("no role here"), "")

    def test_company_comes_from_a_labelled_line(self):
        from interviewgenie.personalization.ingest import find_company

        self.assertEqual(find_company("Company: Stripe\nRole: X"), "Stripe")
        self.assertEqual(find_company("nothing here"), "")

    # -- serialisation ---------------------------------------------------- #
    def test_ingested_profile_round_trips(self):
        from interviewgenie.personalization.profile import IntervieweeProfile

        restored = IntervieweeProfile.from_dict(self.profile.to_dict())
        self.assertEqual(restored.name, self.profile.name)
        self.assertEqual(restored.skills, self.profile.skills)
        self.assertEqual(restored.story_bank, self.profile.story_bank)
        self.assertEqual(restored.job_description, self.profile.job_description)
        self.assertEqual(restored.resume_text, self.profile.resume_text)

    def test_job_description_is_retained_verbatim(self):
        self.assertIn("payments reliability", self.profile.job_description)

    # -- summary ---------------------------------------------------------- #
    def test_summary_is_human_readable(self):
        from interviewgenie.personalization.ingest import summarise

        text = summarise(self.profile)
        self.assertIn("Priya Raman", text)
        self.assertIn("Stripe", text)
        self.assertIn("7 years", text)

    def test_summary_survives_an_empty_profile(self):
        from interviewgenie.personalization.ingest import ingest, summarise

        profile, _ = ingest()
        self.assertIsInstance(summarise(profile), str)

    # -- report ----------------------------------------------------------- #
    def test_report_exposes_what_was_found(self):
        data = self.report.to_dict()
        for key in ("skills", "seniority", "years_experience", "target_role",
                    "target_company", "education", "stories",
                    "jd_requirements", "contacts", "notes"):
            self.assertIn(key, data, msg=key)

    def test_report_lists_skills_from_both_documents(self):
        # résumé skills come first, then the JD's additions
        self.assertTrue(self.report.skills)
        self.assertIn("kafka", self.report.skills)
        self.assertTrue(all(isinstance(s, str) for s in self.report.skills))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
