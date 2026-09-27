"""Command line interface.

::

    python -m interviewgenie demo                 # scripted end-to-end demo
    python -m interviewgenie ask "…"              # one question, one answer
    python -m interviewgenie serve [--port 8420]  # live cockpit + API
    python -m interviewgenie benchmark [--limit]  # evaluation benchmark
    python -m interviewgenie listen [--phrases …] # ASR pipeline on synthetic audio
    python -m interviewgenie evaluate             # NLP-layer evaluation
    python -m interviewgenie profile --show       # inspect the personalisation model
    python -m interviewgenie graph --stats        # knowledge-graph statistics
    python -m interviewgenie noise                # noise-robustness measurement

Log verbosity comes from the first of these that is set: the ``--log-level``
flag, the ``INTERVIEWGENIE_LOG_LEVEL`` environment variable, then
``system.log_level`` in the configuration file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from .errors import InterviewGenieError
from .logging import configure_logging, get_logger

LOG = get_logger("cli")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_demo(args: argparse.Namespace) -> int:
    genie = _build_genie(args.demo_profile)
    summary = genie.start()
    _print_header("InterviewGenie demo")
    print(f"knowledge graph : {summary['graph']['nodes']} nodes, "
          f"{summary['graph']['edges']} edges, {summary['graph']['aliases']} aliases")
    print(f"asr provider    : {summary['asr'].get('provider', 'local')}")
    print()

    for question in args.questions:
        response = genie.ask(question)
        _print_turn(question, response)
        if args.feedback:
            result = genie.feedback(args.feedback)
            print(f"   ↳ feedback applied: {result.get('action')} "
                  f"(reward {result.get('reward')})")

    report = genie.stop()
    _print_header("Session report")
    print(json.dumps(report["metrics"], indent=2))
    print()
    print(f"dialogue  : {json.dumps(report['dialogue'], ensure_ascii=False)}")
    print(f"recovery  : {json.dumps(report['recovery'], ensure_ascii=False)}")
    return 0


def _build_genie(demo_profile: bool) -> Any:
    # Lazy import keeps ``--help`` and parser construction fast.
    from .factory import build_default_genie, build_demo_genie

    return build_demo_genie() if demo_profile else build_default_genie()


def cmd_ask(args: argparse.Namespace) -> int:
    genie = _build_genie(args.demo_profile)
    genie.start()
    response = genie.ask(" ".join(args.question))
    print(response.text)
    if args.verbose:
        print()
        print(json.dumps({
            "scorecard": response.scorecard.to_dict() if response.scorecard else None,
            "plan": response.plan,
            "strategy": response.strategy,
            "evidence": [e.to_dict() for e in response.evidence],
            "validation": response.validation,
            "metrics": response.metrics,
        }, indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api.server import create_server

    server = create_server(host=args.host, port=args.port)
    print(f"InterviewGenie cockpit on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.run()
    return 0


def cmd_desktop(args: argparse.Namespace) -> int:
    """Launch the always-on-top overlay driven by live speech."""
    from .desktop.app import main as desktop_main

    argv = ["--port", str(args.port), "--debounce", str(args.debounce)]
    if args.no_auto:
        argv.append("--no-auto")
    if args.no_browser:
        argv.append("--no-browser")
    if args.headless:
        argv.append("--headless")
    return desktop_main(argv)


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .evaluation.dataset import default_cases
    from .evaluation.metrics import Benchmark
    from .factory import build_demo_genie

    genie = build_demo_genie()
    genie.start()
    benchmark = Benchmark(cases=default_cases())
    result = benchmark.run(genie.nlp, genie.composer, genie.retriever, limit=args.limit)
    _print_header("Benchmark")
    print(result.summary())
    print()
    for intent, means in result.by_intent.items():
        print(f"  {intent:18s} overall={means['overall']:.3f} "
              f"accuracy={means['accuracy']:.3f} relevance={means['relevance']:.3f} "
              f"engagement={means['engagement']:.3f} personalization={means['personalization']:.3f} "
              f"error={means['error_rate']:.3f}")
    if args.json:
        print()
        print(json.dumps(result.to_dict(), indent=2))
    if args.failures:
        print()
        for failure in result.failures[:10]:
            print(f"  ✗ {failure.get('question', '?')} -> {failure.get('overall', '?')}")
            if "answer" in failure:
                print(f"      {failure['answer'][:140]}")
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    """Exercise the full local ASR pipeline on synthetic audio."""
    from .asr.denoise import add_noise, measure_snr
    from .asr.local import LocalStreamingASR
    from .asr.synth import FormantSynthesizer

    asr = LocalStreamingASR(noise_suppression=not args.no_denoise)
    synth = FormantSynthesizer(sample_rate=asr.sample_rate)
    # ``--phrases`` defaults to None, so guard before iterating: the empty
    # case must fall back to the template bank, not raise.
    phrases = (list(args.phrases) if args.phrases
                else list(asr.supported_phrases[:4]))
    _print_header("Local ASR pipeline")
    print(f"sample rate     : {asr.sample_rate} Hz")
    print(f"noise suppression: {'on' if asr.noise_suppression else 'off'}")
    print(f"phrases in bank : {len(asr.supported_phrases)}")
    print()

    correct = 0
    total = 0
    for phrase in phrases:
        audio = synth.synthesize(phrase, asr.sample_rate)
        for noise_kind, snr in (("clean", None), ("babble", 10), ("white", 5)):
            test_audio = audio if noise_kind == "clean" else add_noise(
                audio, snr, noise_kind, seed=args.seed)
            asr.reset()
            started = time.perf_counter()
            chunk_size = int(asr.sample_rate * args.chunk_ms / 1000)
            for offset in range(0, len(test_audio.samples), chunk_size):
                asr.accept_audio(test_audio.samples[offset:offset + chunk_size],
                                 asr.sample_rate)
            asr.flush()
            chunks = asr.poll()
            elapsed = time.perf_counter() - started
            final = next((c for c in reversed(chunks) if c.is_final), None)
            text = final.text if final else ""
            confidence = final.confidence if final else 0.0
            ok = text.strip().lower() == phrase.lower()
            correct += int(ok)
            total += 1
            label = f"{noise_kind}@{snr}dB" if snr is not None else noise_kind
            print(f"  {'✓' if ok else '✗'} {label:14s} snr={measure_snr(test_audio):6.2f}dB "
                  f"conf={confidence:.3f} rtf={elapsed / max(0.01, test_audio.duration):.2f} "
                  f"| {text!r}")
    print()
    print(f"accuracy: {correct}/{total} = {correct / max(1, total):.1%}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .nlp.intent import IntentClassifier

    classifier = IntentClassifier()
    _print_header("NLP evaluation")
    print(json.dumps(classifier.evaluate(), indent=2))

    _print_header("Holdout (leak-free: models fitted on the training split only)")
    holdout = classifier.evaluate_holdout()
    print(f"  rules + model    intent={holdout['intent_accuracy']:.3f}  "
          f"topic={holdout['topic_accuracy']:.3f}   "
          f"({int(holdout['seeds'])} seeds, n={int(holdout['support'])})")
    print(f"  model only       intent={holdout['model_only_intent_accuracy']:.3f}  "
          f"topic={holdout['model_only_topic_accuracy']:.3f}")
    print()
    print("  per seed:")
    for row in holdout["per_seed"]:
        print(f"    seed={row['seed']}  intent={row['intent']:.3f} "
              f"topic={row['topic']:.3f}  "
              f"(model-only intent={row['model_only_intent']:.3f} "
              f"topic={row['model_only_topic']:.3f})")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    from .factory import build_demo_genie

    genie = build_demo_genie()
    if args.show:
        print(json.dumps(genie.profile.to_dict(), indent=2))
    if args.feedback:
        print()
        print(json.dumps(genie.feedback(args.feedback), indent=2))
        print(json.dumps(genie.profile.style, indent=2))
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from .knowledge.graph import graph_from_config
    from .config import load_config

    graph = graph_from_config(load_config())
    if args.stats:
        print(json.dumps(graph.stats(), indent=2))
    if args.search:
        for node in graph.search(args.search, limit=10):
            print(f"  {node.label():12s} {node.get('name', node.id):32s} "
                  f"{node.get('description', '')[:80]}")
    return 0


#: Noise conditions swept by the robustness measurement.
NOISE_KINDS = ("white", "pink", "babble", "hum")
NOISE_SNRS = (20, 10, 5, 0)
NOISE_PHRASE = "tell me about yourself"


def noise_report(seed: int = 7, kinds: Sequence[str] = NOISE_KINDS,
                 snrs: Sequence[int] = NOISE_SNRS,
                 phrase: str = NOISE_PHRASE) -> List[Dict[str, Any]]:
    """Measure the noise-suppression gain for every (noise, SNR) condition.

    A fresh suppressor is built for each case so no state leaks between
    measurements, which is what makes the numbers reproducible.
    """
    from .asr.denoise import NoiseSuppressor, add_noise, measure_snr
    from .asr.dsp import Audio
    from .asr.synth import FormantSynthesizer

    synth = FormantSynthesizer()
    rows: List[Dict[str, Any]] = []
    for kind in kinds:
        for snr in snrs:
            audio: Audio = synth.synthesize(phrase, 16000)
            noisy = add_noise(audio, snr, kind, seed=seed)
            suppressor = NoiseSuppressor(sample_rate=16000, enabled=True)
            clean = suppressor.process_audio(noisy)
            before, after = measure_snr(noisy), measure_snr(clean)
            rows.append({"noise": kind, "snr": int(snr), "before": round(before, 2),
                         "after": round(after, 2), "gain": round(after - before, 2)})
    return rows


def cmd_noise(args: argparse.Namespace) -> int:
    """Measure how much noise suppression actually helps."""
    _print_header("Noise robustness")
    print(f"{'noise':10s} {'snr':>5s} {'before':>8s} {'after':>8s} {'gain':>7s}")
    for row in noise_report(seed=args.seed):
        print(f"{row['noise']:10s} {row['snr']:4d}dB {row['before']:8.2f} "
              f"{row['after']:8.2f} {row['gain']:+7.2f}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    from .config import load_config

    config = load_config(args.path)
    print(json.dumps(config.to_dict(), indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _print_turn(question: str, response: Any) -> None:
    print(f"Q  {question}")
    print(f"A  {response.text}")
    if response.scorecard:
        card = response.scorecard
        print(f"   overall={card.overall():.3f} accuracy={card.accuracy:.3f} "
              f"relevance={card.relevance:.3f} engagement={card.engagement:.3f} "
              f"personalization={card.personalization:.3f} error={card.error_rate:.3f}")
    print(f"   strategy={response.strategy} tone={response.tone} "
          f"register={response.register} confidence={response.confidence:.3f} "
          f"latency={response.latency_ms:.0f}ms")
    print()


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interviewgenie",
        description="InterviewGenie — a real-time, dependency-free interview copilot.")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="log verbosity (default: system.log_level from the "
                             "configuration file, or INFO)")
    parser.add_argument("--json-logs", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="scripted end-to-end demonstration")
    demo.add_argument("questions", nargs="*",
                      default=["Tell me about yourself.",
                               "How would you design a URL shortener?",
                               "Tell me about a time you showed leadership.",
                               "What is your greatest weakness?",
                               "Why do you want to work here?"])
    demo.add_argument("--feedback", default=None,
                      help="apply this feedback after every turn")
    demo.add_argument("--no-demo", dest="demo_profile", action="store_false",
                      help="start from an empty candidate profile instead of "
                           "the built-in demo profile (Priya)")
    demo.set_defaults(func=cmd_demo, demo_profile=True)

    ask = subparsers.add_parser("ask", help="answer a single question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--verbose", "-v", action="store_true")
    ask.add_argument("--no-demo", dest="demo_profile", action="store_false",
                     help="start from an empty candidate profile instead of "
                          "the built-in demo profile (Priya)")
    ask.set_defaults(func=cmd_ask, demo_profile=True)

    serve = subparsers.add_parser("serve", help="run the API server and web cockpit")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8420)
    serve.set_defaults(func=cmd_serve)

    desktop = subparsers.add_parser(
        "desktop", help="run the always-on-top live overlay")
    desktop.add_argument("--port", type=int, default=8422)
    desktop.add_argument("--no-auto", action="store_true",
                         help="require a click before answering")
    desktop.add_argument("--debounce", type=int, default=700,
                         help="ms of silence that marks a question as finished")
    desktop.add_argument("--no-browser", action="store_true",
                         help="do not open the audio-capture page")
    desktop.add_argument("--headless", action="store_true",
                         help="run without the GUI (tests, servers, CI)")
    desktop.set_defaults(func=cmd_desktop)

    benchmark = subparsers.add_parser("benchmark", help="run the evaluation benchmark")
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument("--json", action="store_true")
    benchmark.add_argument("--failures", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)

    listen = subparsers.add_parser("listen", help="exercise the local ASR pipeline")
    listen.add_argument("--phrases", nargs="*", default=None)
    listen.add_argument("--chunk-ms", type=int, default=320)
    listen.add_argument("--seed", type=int, default=7)
    listen.add_argument("--no-denoise", action="store_true")
    listen.set_defaults(func=cmd_listen)

    evaluate = subparsers.add_parser("evaluate", help="evaluate the NLP layer")
    evaluate.set_defaults(func=cmd_evaluate)

    profile = subparsers.add_parser("profile", help="inspect the personalisation model")
    profile.add_argument("--show", action="store_true")
    profile.add_argument("--feedback", default=None)
    profile.set_defaults(func=cmd_profile)

    graph = subparsers.add_parser("graph", help="inspect the knowledge graph")
    graph.add_argument("--stats", action="store_true")
    graph.add_argument("--search", default=None)
    graph.set_defaults(func=cmd_graph)

    noise = subparsers.add_parser("noise", help="measure noise-suppression gain")
    noise.add_argument("--seed", type=int, default=7)
    noise.set_defaults(func=cmd_noise)

    config = subparsers.add_parser("config", help="print the resolved configuration")
    config.add_argument("--path", default=None)
    config.set_defaults(func=cmd_config)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    configure_logging(level=args.log_level, json_logs=args.json_logs, force=True)
    try:
        return int(args.func(args) or 0)
    except InterviewGenieError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
