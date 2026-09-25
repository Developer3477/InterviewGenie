"""The labelled evaluation dataset.

Each case carries the question, a reference answer, the expected intent/topic and
the terms a good answer must mention.  The reference answers are deliberately
short "ideal" answers so that the token-F1 / ROUGE-L / semantic-similarity
metrics have something meaningful to compare against.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..errors import EvaluationError
from ..logging import get_logger
from .metrics import BenchmarkCase

LOG = get_logger("evaluation.dataset")

CASES: List[Dict[str, Any]] = [
    # ---------------- knowledge ----------------------------------------- #
    {
        "question": "What is the difference between TCP and UDP?",
        "intent": "knowledge", "topic": "technical_concept",
        "reference_answer": (
            "TCP is reliable, ordered and connection oriented, so it retransmits "
            "lost packets and guarantees delivery, while UDP is unreliable and "
            "connectionless with lower overhead, which makes it better for "
            "latency-sensitive traffic like video and voice."),
        "key_terms": ["reliable", "ordered", "connection", "overhead", "retransmit"],
        "must_mention": ["reliable", "overhead"],
    },
    {
        "question": "Explain the CAP theorem.",
        "intent": "knowledge", "topic": "architecture",
        "reference_answer": (
            "The CAP theorem says that during a network partition a distributed "
            "system must choose between consistency and availability; you cannot "
            "have all three at once. Most systems choose availability and settle "
            "for eventual consistency."),
        "key_terms": ["partition", "consistency", "availability", "choose"],
        "must_mention": ["partition", "consistency"],
    },
    {
        "question": "What is the difference between a mutex and a semaphore?",
        "intent": "knowledge", "topic": "technical_concept",
        "reference_answer": (
            "A mutex is a mutual exclusion lock owned by one thread, so only the "
            "owner can release it, while a semaphore is a counter that bounds how "
            "many threads can hold a resource at once."),
        "key_terms": ["lock", "owner", "counter", "threads", "resource"],
        "must_mention": ["lock", "counter"],
    },
    {
        "question": "When would you use an LSM tree over a B-tree?",
        "intent": "knowledge", "topic": "technical_concept",
        "reference_answer": (
            "An LSM tree turns random writes into sequential ones and merges them "
            "in the background, so it wins on write-heavy workloads like event "
            "logs, while a B-tree is better for read-heavy workloads with range "
            "queries."),
        "key_terms": ["write", "sequential", "merge", "read", "range"],
        "must_mention": ["write", "read"],
    },
    {
        "question": "What is eventual consistency?",
        "intent": "knowledge", "topic": "architecture",
        "reference_answer": (
            "Eventual consistency means replicas converge to the same value once "
            "updates stop, so a read immediately after a write may see stale data "
            "but reads eventually return the latest value."),
        "key_terms": ["replicas", "converge", "stale", "latest"],
        "must_mention": ["replicas", "converge"],
    },
    # ---------------- design --------------------------------------------- #
    {
        "question": "How would you design a URL shortener?",
        "intent": "design", "topic": "architecture",
        "reference_answer": (
            "I would start with the requirements: read-heavy traffic, a mapping "
            "from short code to long URL, and low latency redirects. I would "
            "generate a unique code with base62 encoding, store it in a key-value "
            "store with a cache in front, and shard by code hash. The trade-off "
            "is availability over strict consistency, because a redirect being "
            "briefly stale is acceptable."),
        "key_terms": ["requirements", "cache", "shard", "trade-off", "latency"],
        "must_mention": ["cache", "trade-off"],
    },
    {
        "question": "How would you design a rate limiter?",
        "intent": "design", "topic": "architecture",
        "reference_answer": (
            "I would use a token bucket per client stored in Redis with a TTL, "
            "check and decrement atomically on the hot path, and fail open with a "
            "local in-process limit if Redis is unavailable. The trade-off is "
            "exactness against availability."),
        "key_terms": ["token bucket", "redis", "atomic", "fail open", "trade-off"],
        "must_mention": ["redis", "trade-off"],
    },
    {
        "question": "How would you design a chat system for a million users?",
        "intent": "design", "topic": "architecture",
        "reference_answer": (
            "I would separate presence, message fan-out and history. Messages go "
            "through a queue so delivery is durable, fan-out is per-recipient "
            "inbox lists, and history is paginated from a partitioned store. The "
            "trade-off is ordering guarantees against throughput."),
        "key_terms": ["queue", "fan-out", "history", "partition", "trade-off"],
        "must_mention": ["queue", "trade-off"],
    },
    {
        "question": "How would you design a news feed for a hundred million users?",
        "intent": "design", "topic": "architecture",
        "reference_answer": (
            "The read path dominates, so I would precompute timelines into a "
            "cache, fan out on write for normal users and pull on read for "
            "celebrities, and shard the timeline store by user id. The trade-off "
            "is freshness against read latency."),
        "key_terms": ["precompute", "cache", "fan-out", "shard", "trade-off"],
        "must_mention": ["cache", "trade-off"],
    },
    # ---------------- coding / troubleshooting --------------------------- #
    {
        "question": "How would you find the first duplicate in an array?",
        "intent": "coding", "topic": "algorithms",
        "reference_answer": (
            "I would use a hash set and scan once, returning the first element "
            "already in the set. That is O(n) time and O(n) space, and it handles "
            "the edge case where there is no duplicate by returning null."),
        "key_terms": ["hash set", "scan", "O(n)", "edge case"],
        "must_mention": ["hash set"],
    },
    {
        "question": "This service is returning 500 errors, how would you debug it?",
        "intent": "troubleshooting", "topic": "reliability",
        "reference_answer": (
            "I would first narrow the blast radius by checking whether all "
            "requests or a subset fail, then look at the service's own metrics, "
            "logs and traces for the failing dependency. I would check recent "
            "deploys and the dependency's error budget before changing anything, "
            "and roll back if a deploy correlates."),
        "key_terms": ["blast radius", "metrics", "logs", "deploy", "roll back"],
        "must_mention": ["logs", "deploy"],
    },
    {
        "question": "A page load got slower after a release, how do you find the cause?",
        "intent": "troubleshooting", "topic": "reliability",
        "reference_answer": (
            "I would compare the release window against the latency percentiles, "
            "profile the hot path to find the new bottleneck, and check whether a "
            "new query or dependency was added. Then I would fix or roll back and "
            "add a latency alert so it does not recur."),
        "key_terms": ["percentiles", "profile", "bottleneck", "alert"],
        "must_mention": ["bottleneck", "alert"],
    },
    # ---------------- behavioural ---------------------------------------- #
    {
        "question": "Tell me about a time you showed leadership.",
        "intent": "people", "topic": "leadership",
        "reference_answer": (
            "In one project two teams disagreed on an ownership boundary that "
            "blocked a migration. I wrote both options as a short RFC with the "
            "cost of each, got both leads to review it, and we shipped the smaller "
            "change first. It unblocked the migration in two weeks."),
        "key_terms": ["disagreed", "rfc", "leads", "shipped", "unblocked"],
        "must_mention": ["rfc", "shipped"],
    },
    {
        "question": "Tell me about a time you dealt with a difficult colleague.",
        "intent": "people", "topic": "collaboration",
        "reference_answer": (
            "A teammate and I disagreed about a design direction. I asked them to "
            "walk me through their reasoning, wrote both options down with the "
            "trade-offs, and we agreed on the smaller change with a review point. "
            "We shipped on time and the relationship was better afterwards."),
        "key_terms": ["disagreed", "reasoning", "trade-offs", "shipped"],
        "must_mention": ["disagreed", "trade-offs"],
    },
    {
        "question": "Tell me about a time you failed.",
        "intent": "past_experience", "topic": "self",
        "reference_answer": (
            "I shipped a migration without a rollback plan and it took down a "
            "secondary service for forty minutes. I owned the incident, wrote the "
            "postmortem, and we added a canary step and a rollback rehearsal to "
            "the release checklist."),
        "key_terms": ["migration", "rollback", "postmortem", "canary", "checklist"],
        "must_mention": ["postmortem", "rollback"],
    },
    {
        "question": "Describe a challenging bug you had to debug.",
        "intent": "past_experience", "topic": "reliability",
        "reference_answer": (
            "We had a race condition that only appeared under load. I added "
            "targeted logging, reproduced it in staging, and found that two "
            "workers were mutating the same record without a lock. Adding a mutex "
            "and a regression test fixed it and the error rate went to zero."),
        "key_terms": ["race condition", "reproduced", "lock", "regression test"],
        "must_mention": ["race condition", "lock"],
    },
    # ---------------- self assessment ------------------------------------ #
    {
        "question": "What is your greatest weakness?",
        "intent": "self_assessment", "topic": "self",
        "reference_answer": (
            "I tend to go deep on a problem before checking whether it is the "
            "right problem. I now timebox the exploration and write down the "
            "decision criteria first, which has made me faster without losing the "
            "depth."),
        "key_terms": ["timebox", "decision criteria", "faster"],
        "must_mention": ["timebox"],
    },
    {
        "question": "What is your greatest strength?",
        "intent": "self_assessment", "topic": "self",
        "reference_answer": (
            "I am good at turning a vague production problem into a short list of "
            "measurable hypotheses and then eliminating them quickly with data."),
        "key_terms": ["hypotheses", "measurable", "data"],
        "must_mention": ["hypotheses"],
    },
    # ---------------- motivation / culture ------------------------------- #
    {
        "question": "Why do you want to work here?",
        "intent": "motivation", "topic": "company",
        "reference_answer": (
            "Two reasons: the payments problem is one where correctness is not "
            "optional, which is the kind of constraint I do my best work under, "
            "and the team is at a scale where individual contributions still move "
            "the product."),
        "key_terms": ["payments", "correctness", "scale", "contributions"],
        "must_mention": ["scale"],
    },
    {
        "question": "What kind of work environment do you do your best work in?",
        "intent": "culture", "topic": "culture",
        "reference_answer": (
            "I do my best work where disagreement is cheap: I can push back with "
            "evidence and expect the same in return, and where the team writes "
            "decisions down so we do not relitigate them."),
        "key_terms": ["disagreement", "evidence", "decisions", "push back"],
        "must_mention": ["evidence"],
    },
    {
        "question": "Where do you see yourself in five years?",
        "intent": "goals", "topic": "career",
        "reference_answer": (
            "I want to be the person a team brings hard architectural problems "
            "to, which for me means going deeper on distributed systems and "
            "getting better at writing decisions down so they scale beyond me."),
        "key_terms": ["architectural", "distributed systems", "decisions"],
        "must_mention": ["distributed systems"],
    },
    # ---------------- meta / logistics ----------------------------------- #
    {
        "question": "Could you repeat that please?",
        "intent": "meta", "topic": "meta",
        "reference_answer": "Of course -- which part would you like me to go over again?",
        "key_terms": ["part", "again"],
        "must_mention": [],
    },
    {
        "question": "Do you have any questions for us?",
        "intent": "meta", "topic": "meta",
        "reference_answer": (
            "Yes -- what does the team measure success on this quarter, and what "
            "is the hardest problem the role is expected to own?"),
        "key_terms": ["measure", "success", "hardest problem"],
        "must_mention": ["success"],
    },
    {
        "question": "What are your salary expectations?",
        "intent": "logistics", "topic": "logistics",
        "reference_answer": (
            "I am looking for something in the senior band for this market, and I "
            "am flexible depending on the equity mix and the scope of the role."),
        "key_terms": ["senior band", "equity", "flexible"],
        "must_mention": [],
    },
    {
        "question": "When could you start?",
        "intent": "logistics", "topic": "logistics",
        "reference_answer": (
            "I can give four weeks' notice, so realistically about a month from an "
            "offer, and I am happy to be flexible if that helps."),
        "key_terms": ["notice", "month", "flexible"],
        "must_mention": ["notice"],
    },
    # ---------------- self introduction ---------------------------------- #
    {
        "question": "Tell me about yourself.",
        "intent": "self_introduction", "topic": "self",
        "reference_answer": (
            "I am a backend engineer with about six years on payment and "
            "infrastructure systems. Most recently I owned our delivery pipeline, "
            "and before that I was on the platform team that rebuilt our job "
            "queue. I care most about reliability and about making the people "
            "around me faster."),
        "key_terms": ["backend", "payments", "pipeline", "reliability"],
        "must_mention": ["reliability"],
    },
    {
        "question": "Walk me through your resume.",
        "intent": "background", "topic": "career",
        "reference_answer": (
            "I started on a platform team, moved into payments where I owned the "
            "retry path, and then took over delivery tooling. Each move was "
            "towards owning a larger slice of the production surface."),
        "key_terms": ["platform", "payments", "delivery", "owned"],
        "must_mention": ["owned"],
    },
    {
        "question": "How do you measure the success of a project?",
        "intent": "measurement", "topic": "measurement",
        "reference_answer": (
            "I pick the metric before the work starts, take a baseline, and agree "
            "the target with the people who asked for it. For reliability work "
            "that is usually error rate and p99 latency; for product work it is "
            "the activation or conversion number the team already tracks."),
        "key_terms": ["metric", "baseline", "target", "error rate", "p99"],
        "must_mention": ["baseline", "target"],
    },
    {
        "question": "What would you do if you disagreed with your manager's decision?",
        "intent": "hypothetical", "topic": "judgement",
        "reference_answer": (
            "I would ask what information they have that I do not, put my "
            "objection in writing with the specific risk and a cheaper "
            "alternative, and if they still disagree I would commit and make the "
            "decision work while logging what to watch for."),
        "key_terms": ["information", "objection", "risk", "commit", "logging"],
        "must_mention": ["commit"],
    },
]


def load_cases(path: str) -> List[BenchmarkCase]:
    """Load cases from a JSON file, falling back to the built-in set."""
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
            return [BenchmarkCase.from_dict(c) for c in cases]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            LOG.warning("could not load benchmark cases from %s: %s", path, exc)
    return [BenchmarkCase.from_dict(c) for c in CASES]


def default_cases() -> List[BenchmarkCase]:
    return [BenchmarkCase.from_dict(c) for c in CASES]


def stats() -> Dict[str, Any]:
    """Distribution of the built-in dataset."""
    by_intent: Dict[str, int] = {}
    by_topic: Dict[str, int] = {}
    for case in CASES:
        by_intent[case["intent"]] = by_intent.get(case["intent"], 0) + 1
        by_topic[case["topic"]] = by_topic.get(case["topic"], 0) + 1
    return {"cases": len(CASES), "by_intent": dict(sorted(by_intent.items())),
            "by_topic": dict(sorted(by_topic.items()))}
