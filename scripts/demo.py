#!/usr/bin/env python3
"""A scripted end-to-end demonstration of InterviewGenie.

    python3 scripts/demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewgenie.factory import build_demo_genie  # noqa: E402

QUESTIONS = [
    "Tell me about yourself.",
    "Walk me through your resume.",
    "How would you design a URL shortener?",
    "What is the difference between TCP and UDP?",
    "Tell me about a time you showed leadership.",
    "Tell me about a time you debugged a production issue.",
    "This service is returning 500 errors, how would you debug it?",
    "What is your greatest weakness?",
    "Why do you want to work here?",
    "How do you measure the success of a project?",
    "Do you have any questions for us?",
    "Could you repeat that please?",
]


def main() -> int:
    genie = build_demo_genie()
    summary = genie.start()
    graph = summary["graph"]
    print("=" * 78)
    print("  InterviewGenie — live interview copilot demonstration")
    print("=" * 78)
    print(f"knowledge graph : {graph['nodes']} nodes, {graph['edges']} edges, "
          f"{graph['aliases']} aliases")
    print(f"asr provider    : {summary['asr'].get('provider', 'local')}")
    print(f"nlp             : {summary['nlp']['intents']} labelled training examples")
    print(f"profile         : {summary['profile']}")
    print()

    for index, question in enumerate(QUESTIONS, start=1):
        response = genie.ask(question)
        card = response.scorecard
        print(f"[{index:02d}] Q  {question}")
        print(f"     A  {response.text}")
        print(f"         intent={genie.dialogue.state.slots.get('last_intent')} "
              f"topic={genie.dialogue.state.topic()} "
              f"strategy={response.strategy} tone={response.tone} "
              f"register={response.register}")
        print(f"         overall={card.overall():.3f} accuracy={card.accuracy:.3f} "
              f"relevance={card.relevance:.3f} engagement={card.engagement:.3f} "
              f"personalization={card.personalization:.3f} error={card.error_rate:.3f} "
              f"({response.latency_ms:.0f} ms)")
        print()

    report = genie.stop()
    print("=" * 78)
    print("  Session report")
    print("=" * 78)
    metrics = report["metrics"]
    for key, value in metrics["means"].items():
        print(f"  {key:16s} {value:.4f}")
    print(f"  {'trend':16s} {metrics['trend']['direction']}")
    print(f"  {'recovery':16s} {report['recovery']}")
    print(f"  {'learning':16s} signals={report['learning']['signals']} "
          f"kg_updates={report['learning']['kg_updates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
