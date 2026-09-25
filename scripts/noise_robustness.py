#!/usr/bin/env python3
"""Measure how much the noise-suppression front-end actually helps.

    python3 scripts/noise_robustness.py

Delegates to :func:`interviewgenie.cli.noise_report` so this script and
``python3 -m interviewgenie noise`` can never drift apart.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewgenie.cli import noise_report  # noqa: E402


def main() -> int:
    print(f"{'noise':10s} {'snr':>5s} {'before':>8s} {'after':>8s} {'gain':>7s}")
    print("-" * 44)
    for row in noise_report():
        print(f"{row['noise']:10s} {row['snr']:4d}dB {row['before']:8.2f} "
              f"{row['after']:8.2f} {row['gain']:+7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
