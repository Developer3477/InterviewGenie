"""Factory helpers: build a fully-wired InterviewGenie from configuration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import DEFAULT_CONFIG, Config, load_config
from .errors import ErrorCode, InterviewGenieError
from .logging import configure_logging, get_logger, level_pinned
from .orchestrator import InterviewGenie

LOG = get_logger("factory")


def build_default_genie(config: Optional[Config | Dict[str, Any] | str] = None,
                        **overrides: Any) -> InterviewGenie:
    """Construct a ready-to-run :class:`InterviewGenie`.

    ``config`` may be a :class:`Config`, a dict of overrides, a path to a JSON
    file, or ``None`` (which loads ``interviewgenie.json`` if present and then
    applies environment overrides).
    """
    if config is None:
        config = load_config()
    elif isinstance(config, str):
        config = load_config(config)
    elif isinstance(config, dict):
        config = Config(defaults=DEFAULT_CONFIG, overrides=config)
    if overrides:
        config = config.with_overrides(**overrides)
    config.validate()

    # A CLI ``--log-level`` flag or the ``INTERVIEWGENIE_LOG_LEVEL`` environment
    # variable is an explicit choice and must win over the configuration file.
    if level_pinned():
        configure_logging(json_logs=bool(config.get("system.json_logs", False)),
                          fmt=str(config.get("system.log_format", "auto")), force=True)
    else:
        configure_logging(level=str(config.get("system.log_level", "INFO")),
                          json_logs=bool(config.get("system.json_logs", False)),
                          fmt=str(config.get("system.log_format", "auto")), force=True)
    LOG.debug("building InterviewGenie", context={"overrides": list(overrides)})
    return InterviewGenie(config=config)


def build_demo_genie(**overrides: Any) -> InterviewGenie:
    """A genie pre-loaded with a realistic candidate profile.

    Used by the CLI demo, the benchmark and the web cockpit so the output is
    meaningful without any manual setup.
    """
    genie = build_default_genie(**overrides)
    genie.profile.update_facts(
        name="Priya",
        target_role="Senior Software Engineer",
        target_company="Stripe",
        seniority="senior",
        years_experience=6,
        strengths=["debugging distributed systems", "mentoring engineers",
                   "turning vague problems into measurable hypotheses"],
        weaknesses=["going deep before checking whether it is the right problem"],
        skills=["Python", "PostgreSQL", "Kubernetes", "Kafka", "Redis",
                "System design", "Observability"],
        goals=["move into a staff-level role"],
    )
    genie.profile.add_story(
        "the payments retry storm",
        "our webhook deliveries were timing out during a provider outage",
        "keep payments flowing even when the provider was down",
        "I added an idempotent retry queue with exponential backoff and a "
        "circuit breaker, then a provider health check",
        "delivery success went from 91 to 99.6 percent and we cut p99 latency "
        "by 400ms",
        tags=["reliability", "distributed systems"],
        learning="I would have added the provider health check from day one "
                 "instead of discovering the failure mode in production",
    )
    genie.profile.add_story(
        "the schema migration standoff",
        "a redesign of the orders table was blocked because two teams disagreed "
        "on the ownership boundary",
        "unblock the migration without forcing a decision nobody owned",
        "I wrote the two options as a short RFC with the cost of each, got both "
        "leads to review it, and we picked the smaller change first",
        "we shipped the migration in two weeks and the second team adopted the "
        "pattern a quarter later",
        tags=["collaboration", "leadership", "conflict"],
        learning="writing the disagreement down as options, not opinions, is "
                 "what actually moves people",
    )
    genie.profile.add_story(
        "the flaky test epidemic",
        "our CI went from green to unusable because a third of the suite was "
        "flaky",
        "make the pipeline trustworthy again without deleting coverage",
        "I quarantined the flaky tests, fixed the shared fixture that caused "
        "them, and added a retry budget with an alert",
        "CI failure rate dropped from 34 percent to under 2 percent and review "
        "turnaround halved",
        tags=["delivery", "testing", "ownership"],
        learning="quarantining first and fixing the root cause second is what "
                 "kept the team trusting the suite",
    )
    return genie


__all__ = ["build_default_genie", "build_demo_genie"]
