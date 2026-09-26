"""Typed configuration for InterviewGenie.

Configuration is layered, lowest priority first:

1. built-in :data:`DEFAULT_CONFIG`
2. a JSON file (``config/interviewgenie.json`` by default)
3. environment variables prefixed with ``INTERVIEWGENIE_`` (``__`` = nesting)
4. explicit keyword overrides

Values are exposed through attribute access with dotted-path lookup so callers
can do ``cfg.asr.provider`` or ``cfg.get("asr.provider")``.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .errors import InterviewGenieError, ErrorCode
from .logging import get_logger

LOG = get_logger("config")


class Section(dict):
    """Dictionary with attribute access and dotted ``get``."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - ergonomics
            raise AttributeError(item) from exc

    def get(self, path: str, default: Any = None) -> Any:  # type: ignore[override]
        node: Any = self
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {
        "name": "InterviewGenie",
        "version": "1.0.0",
        "log_level": "INFO",
        "log_format": "auto",             # auto | pretty | json
        "json_logs": False,
        "seed": 20260925,
        "timezone": "UTC",
    },
    "asr": {
        "provider": "local",             # local | google | azure | ibm | mock
        "sample_rate": 16000,
        "chunk_ms": 320,
        "language": "en-US",
        "min_confidence": 0.45,
        "silence_timeout_ms": 900,
        "partial_interval_ms": 240,
        "noise_suppression": True,
        "denoise_strength": 0.75,        # over-subtraction factor
        "vad": {
            "energy_threshold": 0.012,
            "zcr_max": 0.35,
            "hangover_ms": 260,
            "min_speech_ms": 180,
        },
        "credentials": {},               # per-provider api keys / endpoints
        "model_path": "",                # optional Vosk model directory
        "phrase_list": [],               # constrained decoding vocabulary
    },
    "nlp": {
        "backend": "builtin",            # builtin | spacy | nltk
        "spacy_model": "en_core_web_sm",
        "max_tokens": 512,
        "embedding_dim": 96,
        "intent_threshold": 0.30,
        "enable_coref": True,
        "keyword_count": 12,
    },
    "knowledge": {
        "backend": "memory",             # memory | graphdb | neptune
        "seed_file": "config/knowledge_graph.json",
        "endpoint": "",
        "graph": "interviewgenie",
        "max_hops": 2,
        "max_evidence": 8,
        "min_edge_weight": 0.15,
        "enable_external_sources": False,
        "external_sources": [],
        "persist_path": "data/knowledge_graph_snapshot.json",
    },
    "context": {
        "max_turns": 40,
        "topic_decay": 0.85,
        "emotion_smoothing": 0.35,
        "empathy_level": 0.7,
        "track_tone": True,
    },
    "generation": {
        "backend": "auto",               # auto | composer | llm
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "endpoint": "",
            "api_key": "",
            "temperature": 0.4,
            "max_tokens": 220,
        },
        "max_words": 90,
        "min_words": 12,
        "hedging": 0.35,
        "use_evidence": True,
        "validate": True,
    },
    "personalization": {
        "enabled": True,
        "profile_file": "data/profile.json",
        "learning_rate": 0.25,
        "adapt_register": True,
    },
    "learning": {
        "enabled": True,
        "buffer_size": 500,
        "min_feedback_to_adapt": 3,
        "persist_path": "data/learning_state.json",
    },
    "evaluation": {
        "dataset": "config/interview_qa.json",
        "auto_evaluate": True,
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8765,
        "cors_origins": ["*"],
        "static_dir": "web",
        "max_ws_connections": 16,
    },
}


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(raw: str) -> Any:
    low = raw.strip().lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def env_overrides(prefix: str = "INTERVIEWGENIE_") -> Dict[str, Any]:
    """Translate ``INTERVIEWGENIE_ASR__PROVIDER=google`` into nested overrides."""
    out: Dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        key = raw_key[len(prefix):].lower()
        if not key:
            continue
        node = out
        for part in key.split("__")[:-1]:
            node = node.setdefault(part, {})
        node[key.split("__")[-1]] = _coerce(raw_value)
    return out


@dataclass
class Config:
    """Layered, validated configuration object."""

    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    source: str = "defaults"

    @classmethod
    def load(cls, path: Optional[str] = None, **overrides: Any) -> "Config":
        """Load configuration from ``path`` (optional) then apply overrides."""
        data = copy.deepcopy(DEFAULT_CONFIG)
        source = "defaults"
        path = path or os.environ.get("INTERVIEWGENIE_CONFIG")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    file_data = json.load(fh)
                data = deep_merge(data, file_data)
                source = path
            except (OSError, json.JSONDecodeError) as exc:
                raise InterviewGenieError(
                    f"cannot read config {path}: {exc}",
                    code=ErrorCode.CONFIG_INVALID,
                    recoverable=False,
                ) from exc
        elif path:
            # an explicitly requested config file that does not exist is a
            # configuration error, not a reason to silently fall back
            raise InterviewGenieError(
                f"config file not found: {path}",
                code=ErrorCode.CONFIG_INVALID,
                recoverable=False,
            )

        data = deep_merge(data, env_overrides())
        data = deep_merge(data, overrides)
        cfg = cls(data=data, source=source)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        asr = self.data.get("asr", {})
        provider = asr.get("provider", "local")
        if provider not in {"local", "google", "azure", "ibm", "mock"}:
            raise InterviewGenieError(
                f"unknown asr provider {provider!r}",
                code=ErrorCode.CONFIG_INVALID,
                recoverable=False,
            )
        gen = self.data.get("generation", {})
        if gen.get("backend") not in {"auto", "composer", "llm"}:
            raise InterviewGenieError(
                f"unknown generation backend {gen.get('backend')!r}",
                code=ErrorCode.CONFIG_INVALID,
                recoverable=False,
            )
        if int(gen.get("max_words", 90)) <= 0:
            raise InterviewGenieError(
                "generation.max_words must be positive",
                code=ErrorCode.CONFIG_INVALID,
                recoverable=False,
            )
        for section in ("system", "asr", "nlp", "knowledge", "context", "generation"):
            if not isinstance(self.data.get(section), dict):
                raise InterviewGenieError(
                    f"config section {section!r} must be an object",
                    code=ErrorCode.CONFIG_INVALID,
                    recoverable=False,
                )

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def section(self, name: str) -> Section:
        return Section(copy.deepcopy(self.data.get(name, {})))

    def __getattr__(self, item: str) -> Any:
        try:
            return self.__dict__["data"][item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def with_overrides(self, **overrides: Any) -> "Config":
        cfg = Config(data=deep_merge(self.to_dict(), overrides), source=self.source)
        cfg.validate()
        return cfg


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    """Convenience wrapper around :meth:`Config.load`."""
    return Config.load(path, **overrides)
