"""InterviewGenie — a real-time, dependency-free interview copilot.

The package is deliberately built on the Python standard library only, so it runs
on a bare interpreter.  Optional accelerators (spaCy, NLTK, Vosk, cloud ASR,
LLM endpoints) are probed lazily and fall back to the built-in implementations.

Modules
-------
``nlp``             tokenisation, tagging, intent, sentiment, embeddings, NER, coref
``asr``             DSP front-end, noise suppression, VAD, DTW decoder, cloud adapters
``knowledge``       property graph, retrieval, common sense
``context``         dialogue state, memory, emotional intelligence
``generation``      retrieval-augmented composer, LLM clients, validator
``personalization`` interviewee profile and style adaptation
``learning``        feedback, online updates, metric tracking
``evaluation``      the five required metrics plus a benchmark harness
``api``             stdlib asyncio HTTP + WebSocket server
``orchestrator``    the top-level coordinator implementing the interview lifecycle
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"
__all__ = ["InterviewGenie", "build_default_genie", "build_demo_genie", "__version__"]

_LAZY = {
    "InterviewGenie": ("interviewgenie.orchestrator", "InterviewGenie"),
    "build_default_genie": ("interviewgenie.factory", "build_default_genie"),
    "build_demo_genie": ("interviewgenie.factory", "build_demo_genie"),
}


def __getattr__(name: str) -> Any:
    """Import the heavy orchestration objects lazily.

    This keeps ``import interviewgenie.nlp`` cheap and avoids a circular import
    between the package root and the orchestrator.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))
