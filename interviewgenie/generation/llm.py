"""Pluggable LLM backends and the response validator.

``LLMClient`` abstracts a chat-completions endpoint.  Two implementations ship
with the system:

* :class:`OpenAICompatibleClient` -- talks to any OpenAI-compatible
  ``/chat/completions`` endpoint (OpenAI, Azure OpenAI, vLLM, Ollama, ...)
  using only the standard library, and
* :class:`DeterministicLLM` -- a template-backed stand-in used in tests and when
  no credentials are configured.

:class:`ResponseValidator` is the gate every candidate answer passes through
before it reaches the candidate: it checks groundedness, plausibility, length,
duplication against earlier answers, and internal consistency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import GenerationError, LLMUnavailableError, UnsupportedClaimError
from ..logging import get_logger
from ..types import Response

LOG = get_logger("generation.llm")

SYSTEM_PROMPT = """You are InterviewGenie, an expert interview coach embedded in a live interview.
You hear the interviewer's question and draft the answer the candidate should give.

Rules:
- Lead with the answer, then the reasoning. Interviewers reward being able to stop you at any point.
- Be specific: name technologies, numbers and outcomes. Never invent facts the candidate did not give you.
- Keep it to {max_words} words or fewer.
- Match the requested register ({register}) and emotional tone ({tone}).
- If the question is ambiguous, ask one short clarifying question instead of guessing.
- Use the knowledge-graph evidence provided; do not contradict it.
"""


@dataclass
class LLMClient:
    """Interface for language-model backends."""

    name: str = "abstract"

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 220,
                 temperature: float = 0.4) -> str:
        raise NotImplementedError

    def available(self) -> bool:  # pragma: no cover - overridden
        return True


@dataclass
class OpenAICompatibleClient(LLMClient):
    """Minimal OpenAI-compatible chat-completions client."""

    model: str = "gpt-4o-mini"
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    api_key: str = ""
    temperature: float = 0.4
    max_tokens: int = 220
    timeout: float = 30.0
    name: str = "openai"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.endpoint:
            self.endpoint = os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, prompt: str, *, system: str = "", max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        if not self.available():
            raise LLMUnavailableError("no API key configured for the LLM backend")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or SYSTEM_PROMPT.format(
                    max_words=self.max_tokens // 2, register="professional", tone="neutral")},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                data = json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMUnavailableError(f"LLM request failed: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(f"unexpected LLM response shape: {exc}") from exc


@dataclass
class DeterministicLLM(LLMClient):
    """A deterministic stand-in that renders a grounded prompt into text.

    It reports itself as *unavailable* so the orchestrator keeps using the
    retrieval-augmented composer unless a real model endpoint is configured;
    it is mainly useful for inspecting the prompt that *would* be sent.
    """

    name: str = "deterministic"

    def available(self) -> bool:
        return False

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 220,
                 temperature: float = 0.4) -> str:
        # strip the scaffolding and keep the substance
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        body = [ln for ln in lines if not ln.lower().startswith(
            ("question:", "evidence:", "knowledge", "constraints:", "tone:", "register:"))]
        text = " ".join(body)
        return text[: max_tokens * 4].strip()


@dataclass
class ResponseValidator:
    """Gate candidate answers before they reach the candidate."""

    max_words: int = 110
    min_words: int = 8
    max_repetition: float = 0.45

    def validate(self, response: Response, history: Optional[Sequence[str]] = None,
                 min_confidence: float = 0.35) -> Dict[str, Any]:
        text = response.text.strip()
        issues: List[Dict[str, Any]] = []
        words = len(text.split())

        if not text:
            issues.append({"code": "empty", "severity": "error"})
        if words < self.min_words:
            issues.append({"code": "too_short", "severity": "warning", "words": words})
        if words > self.max_words:
            issues.append({"code": "too_long", "severity": "warning", "words": words})

        repetition = _repetition_ratio(text)
        if repetition > self.max_repetition:
            issues.append({"code": "repetitive", "severity": "warning",
                           "ratio": round(repetition, 3)})

        if history:
            for earlier in history[-6:]:
                similarity = _jaccard(text.lower().split(), earlier.lower().split())
                if similarity > 0.7:
                    issues.append({"code": "duplicate_answer", "severity": "warning",
                                   "similarity": round(similarity, 3)})
                    break

        unsupported = response.validation.get("violations", [])
        if unsupported:
            issues.append({"code": "unsupported_claim", "severity": "warning",
                           "rules": unsupported})

        if response.confidence < min_confidence:
            issues.append({"code": "low_confidence", "severity": "warning",
                           "confidence": response.confidence})

        blocking = [i for i in issues if i["severity"] == "error"]
        return {
            "ok": not blocking,
            "issues": issues,
            "words": words,
            "repetition": round(repetition, 3),
            "blocking": blocking,
        }

    def repair(self, response: Response, issues: Sequence[Dict[str, Any]]) -> Response:
        """Apply a best-effort repair for the detected issues."""
        text = response.text
        for issue in issues:
            code = issue.get("code")
            if code == "too_long":
                words = text.split()
                text = " ".join(words[: self.max_words])
                boundary = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
                if boundary > len(text) * 0.5:
                    text = text[: boundary + 1]
                else:
                    text = text.rstrip(",;: ") + "."
            elif code == "repetitive":
                seen: set[str] = set()
                sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")]
                kept = [s for s in sentences if s and not (s.lower() in seen or seen.add(s.lower()))]
                text = ". ".join(kept) + "."
            elif code == "duplicate_answer":
                text = (f"To add to what I said earlier: {text[0].lower() + text[1:]}"
                        if text else text)
        response.text = text
        response.validation = {**response.validation, "repaired": True,
                               "repaired_issues": [i.get("code") for i in issues]}
        return response


def _repetition_ratio(text: str) -> float:
    words = [w.lower() for w in text.split() if len(w) > 3]
    if not words:
        return 0.0
    counts: Dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(words)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def llm_from_config(section: Dict[str, Any]) -> LLMClient:
    provider = str(section.get("provider", "openai")).lower()
    if provider in {"openai", "azure", "vllm", "ollama", "compatible"}:
        client = OpenAICompatibleClient(
            model=section.get("model", "gpt-4o-mini"),
            endpoint=section.get("endpoint", ""),
            api_key=section.get("api_key", ""),
            temperature=float(section.get("temperature", 0.4)),
            max_tokens=int(section.get("max_tokens", 220)),
        )
        if client.available():
            return client
        LOG.warning("LLM credentials missing; falling back to the deterministic composer")
        return DeterministicLLM()
    return DeterministicLLM()
