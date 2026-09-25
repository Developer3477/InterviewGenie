#!/usr/bin/env python3
"""Run the full evaluation benchmark and print a report.

    python3 scripts/benchmark.py [--limit N] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewgenie.evaluation.dataset import default_cases, stats as dataset_stats  # noqa: E402
from interviewgenie.evaluation.metrics import Benchmark  # noqa: E402
from interviewgenie.factory import build_demo_genie  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="InterviewGenie benchmark")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print(json.dumps(dataset_stats(), indent=2))
    genie = build_demo_genie()
    genie.start()
    benchmark = Benchmark(cases=default_cases())
    result = benchmark.run(genie.nlp, genie.composer, genie.retriever, limit=args.limit)
    genie.stop()

    print()
    print(result.summary())
    print()
    for intent, means in result.by_intent.items():
        print(f"  {intent:18s} overall={means['overall']:.3f} "
              f"accuracy={means['accuracy']:.3f} relevance={means['relevance']:.3f} "
              f"engagement={means['engagement']:.3f} "
              f"personalization={means['personalization']:.3f} "
              f"error={means['error_rate']:.3f}")
    if args.json:
        print()
        print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
