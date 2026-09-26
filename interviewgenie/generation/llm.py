"""Pluggable LLM backends, streaming, and the response validator.

``LLMClient`` abstracts a chat-completion endpoint and adds a ``stream()``
generator so the cockpit can render an answer as it is produced rather than
waiting for the whole thing -- during a live interview the first token is what
matters, not the last.

Four implementations ship with the system, all driven by the standard library:

* :class:`OpenAICompatibleClient` -- OpenAI, Azure OpenAI, Groq, Together,
  vLLM, Ollama, DeepSeek and anything else speaking ``/chat/completions``.
* :class:`AnthropicClient` -- the Claude Messages API.
* :class:`GoogleClient` -- the Gemini ``streamGenerateContent`` API.
* :class:`DeterministicLLM` -- an offline stand-in that renders the grounded
  prompt into text.  It reports itself as *unavailable* so the orchestrator
  keeps using the retrieval-augmented composer unless a real endpoint is
  configured; it is mainly useful for inspecting the prompt that *would* be
  sent, and for running the whole system with no credentials at all.

:class:`ResponseValidator` is the gate every candidate answer passes through
before it reaches the candidate: groundedness, plausibility, length,
duplication against earlier answers, and internal consistency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..errors import GenerationError, LLMUnavailableError, UnsupportedClaimError
from ..logging import get_logger
from ..types import Response

LOG = get_logger("generation.llm")

#: Providers that speak the OpenAI chat-completions dialect.
OPENAI_DIALECT = {"openai", "azure", "groq", "together", "vllm", "ollama",
                  "deepseek", "compatible", "openrouter", "mistral", "xai"}

#: Environment variables probed for a key when none is configured explicitly.
KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "azure": ("AZURE_OPENAI_API_KEY",),
}

DEFAULT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "together": "https://api.together.xyz/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "ollama": "http://localhost:11434/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "google": "https://generativelanguage.googleapis.com/v1beta/models",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-2.0-flash",
}

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


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str],
               timeout: float) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def _iter_sse(response: Any) -> Iterator[Dict[str, Any]]:
    """Yield decoded ``data:`` payloads from a server-sent-event stream.

    Works against ``http.client.HTTPResponse`` (what ``urlopen`` returns), which
    is line-buffered enough for SSE.
    """
    data_lines: List[str] = []
    for raw in response:
        line = raw.decode("utf-8", "replace").rstrip("\r\n") if isinstance(raw, bytes) else raw.rstrip("\r\n")
        if line == "":
            if data_lines:
                yield _decode_sse("\n".join(data_lines))
                data_lines = []
            continue
        if line.startswith(":"):        # comment / keep-alive
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield _decode_sse("\n".join(data_lines))


def _decode_sse(payload: str) -> Dict[str, Any]:
    if payload.strip() in ("[DONE]", "DONE"):
        return {"__done__": True}
    try:
        decoded = json.loads(payload)
    except (ValueError, TypeError):
        return {"__raw__": payload}
    return decoded if isinstance(decoded, dict) else {"__raw__": payload}


def _open_stream(url: str, payload: Dict[str, Any], headers: Dict[str, str],
                 timeout: float) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream", **headers}, method="POST")
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
@dataclass
class LLMClient:
    """Interface for language-model backends."""

    name: str = "abstract"

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 220,
                 temperature: float = 0.4) -> str:
        """Blocking completion: collects the whole stream."""
        return "".join(self.stream(prompt, system=system, max_tokens=max_tokens,
                                   temperature=temperature)).strip()

    def stream(self, prompt: str, *, system: str = "", max_tokens: int = 220,
               temperature: float = 0.4) -> Iterator[str]:
        raise NotImplementedError

    def available(self) -> bool:  # pragma: no cover - overridden
        return True

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name, "available": self.available()}


@dataclass
class OpenAICompatibleClient(LLMClient):
    """Any endpoint speaking the OpenAI ``/chat/completions`` dialect.

    Covers OpenAI, Azure OpenAI, Groq, Together, OpenRouter, Mistral, xAI,
    DeepSeek, vLLM and Ollama.  Streaming is requested with ``stream: true``
    and the ``data:`` frames are parsed incrementally.
    """

    model: str = "gpt-4o-mini"
    endpoint: str = ""
    api_key: str = ""
    temperature: float = 0.4
    max_tokens: int = 220
    timeout: float = 30.0
    name: str = "openai"
    provider: str = "openai"

    def __post_init__(self) -> None:
        if not self.api_key:
            for env in KEY_ENV.get(self.provider, ()):
                self.api_key = os.environ.get(env, "")
                if self.api_key:
                    break
        if not self.endpoint:
            self.endpoint = os.environ.get(
                "OPENAI_BASE_URL") or DEFAULT_ENDPOINTS.get(
                    self.provider, DEFAULT_ENDPOINTS["openai"])

    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == "azure":
            headers["api-key"] = self.api_key
        return headers

    def _payload(self, prompt: str, system: str, max_tokens: int,
                 temperature: float, stream: bool) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": system or SYSTEM_PROMPT.format(
                     max_words=max_tokens // 2, register="professional",
                     tone="neutral")},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def stream(self, prompt: str, *, system: str = "", max_tokens: int = 220,
               temperature: float = 0.4) -> Iterator[str]:
        if not self.available():
            raise LLMUnavailableError("no API key configured for the LLM backend")
        payload = self._payload(prompt, system, max_tokens, temperature, True)
        try:
            response = _open_stream(self.endpoint, payload, self._headers(),
                                    self.timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMUnavailableError(f"LLM request failed: {exc}") from exc
        try:
            for frame in _iter_sse(response):
                if frame.get("__done__"):
                    break
                try:
                    delta = frame["choices"][0].get("delta") or {}
                except (KeyError, IndexError, TypeError):
                    continue
                piece = delta.get("content")
                if piece:
                    yield piece
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001 - closing must never mask output
                pass


@dataclass
class AnthropicClient(LLMClient):
    """The Claude Messages API, with server-sent-event streaming."""

    model: str = "claude-sonnet-4-5"
    endpoint: str = ""
    api_key: str = ""
    temperature: float = 0.4
    max_tokens: int = 220
    timeout: float = 30.0
    name: str = "anthropic"
    provider: str = "anthropic"
    version: str = "2023-06-01"

    def __post_init__(self) -> None:
        if not self.api_key:
            for env in KEY_ENV["anthropic"]:
                self.api_key = os.environ.get(env, "")
                if self.api_key:
                    break
        if not self.endpoint:
            self.endpoint = DEFAULT_ENDPOINTS["anthropic"]

    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": self.version}

    def _payload(self, prompt: str, system: str, max_tokens: int,
                 temperature: float, stream: bool) -> Dict[str, Any]:
        return {
            "model": self.model,
            "system": system or SYSTEM_PROMPT.format(
                max_words=max_tokens // 2, register="professional", tone="neutral"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def stream(self, prompt: str, *, system: str = "", max_tokens: int = 220,
               temperature: float = 0.4) -> Iterator[str]:
        if not self.available():
            raise LLMUnavailableError("no Anthropic API key configured")
        payload = self._payload(prompt, system, max_tokens, temperature, True)
        try:
            response = _open_stream(self.endpoint, payload, self._headers(),
                                    self.timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMUnavailableError(f"Anthropic request failed: {exc}") from exc
        try:
            for frame in _iter_sse(response):
                if frame.get("__done__"):
                    break
                if frame.get("type") == "content_block_delta":
                    piece = (frame.get("delta") or {}).get("text")
                    if piece:
                        yield piece
                elif frame.get("type") == "error":
                    raise LLMUnavailableError(
                        f"Anthropic error: {frame.get('error')}")
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass


@dataclass
class GoogleClient(LLMClient):
    """The Gemini ``streamGenerateContent`` API (SSE via ``alt=sse``)."""

    model: str = "gemini-2.0-flash"
    endpoint: str = ""
    api_key: str = ""
    temperature: float = 0.4
    max_tokens: int = 220
    timeout: float = 30.0
    name: str = "google"
    provider: str = "google"

    def __post_init__(self) -> None:
        if not self.api_key:
            for env in KEY_ENV["google"]:
                self.api_key = os.environ.get(env, "")
                if self.api_key:
                    break

    def available(self) -> bool:
        return bool(self.api_key)

    def _url(self, stream: bool) -> str:
        base = self.endpoint or DEFAULT_ENDPOINTS["google"]
        base = base.rstrip("/")
        method = "streamGenerateContent" if stream else "generateContent"
        return f"{base}/{self.model}:{method}?key={self.api_key}" + (
            "&alt=sse" if stream else "")

    def _payload(self, prompt: str, system: str, max_tokens: int,
                 temperature: float) -> Dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system or SYSTEM_PROMPT.format(
                max_words=max_tokens // 2, register="professional", tone="neutral")}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens},
        }

    def stream(self, prompt: str, *, system: str = "", max_tokens: int = 220,
               temperature: float = 0.4) -> Iterator[str]:
        if not self.available():
            raise LLMUnavailableError("no Google API key configured")
        payload = self._payload(prompt, system, max_tokens, temperature)
        try:
            response = _open_stream(self._url(True), payload, {}, self.timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMUnavailableError(f"Google request failed: {exc}") from exc
        try:
            for frame in _iter_sse(response):
                if frame.get("__done__"):
                    break
                try:
                    parts = frame["candidates"][0]["content"]["parts"]
                except (KeyError, IndexError, TypeError):
                    continue
                for part in parts:
                    piece = part.get("text")
                    if piece:
                        yield piece
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass


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

    #: Metadata scaffolding that carries no answer substance on its own.
    _DROP = ("evidence:", "knowledge", "constraints:", "tone:", "register:",
             "intent:", "topic:", "strategy:", "candidate ", "emotion",
             "draft the answer")

    def stream(self, prompt: str, *, system: str = "", max_tokens: int = 220,
               temperature: float = 0.4) -> Iterator[str]:
        """Render the grounded prompt back as readable text, offline.

        The question itself is kept (it is the substance); only the retrieval
        scaffolding and the trailing instruction are dropped.  If that leaves
        nothing, fall back to the whole prompt so the caller always gets
        something to render.
        """
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        kept = [ln for ln in lines
                if not any(ln.lower().startswith(p) for p in self._DROP)]
        body = " ".join(kept).strip()
        if len(body.split()) < 4:
            body = " ".join(lines).strip()
        text = body[: max_tokens * 4].strip()
        # emit in chunks so the streaming renderer is exercised offline too
        for start in range(0, len(text), 24):
            yield text[start:start + 24]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
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
                seen: set = set()
                sentences = [s.strip() for s in
                             text.replace("!", ".").replace("?", ".").split(".")]
                kept = [s for s in sentences
                        if s and not (s.lower() in seen or seen.add(s.lower()))]
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


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
_CLIENT_TYPES = {
    "openai": OpenAICompatibleClient,
    "azure": OpenAICompatibleClient,
    "groq": OpenAICompatibleClient,
    "together": OpenAICompatibleClient,
    "vllm": OpenAICompatibleClient,
    "ollama": OpenAICompatibleClient,
    "deepseek": OpenAICompatibleClient,
    "openrouter": OpenAICompatibleClient,
    "mistral": OpenAICompatibleClient,
    "xai": OpenAICompatibleClient,
    "compatible": OpenAICompatibleClient,
    "anthropic": AnthropicClient,
    "claude": AnthropicClient,
    "google": GoogleClient,
    "gemini": GoogleClient,
}


def client_for(provider: str, section: Optional[Dict[str, Any]] = None) -> LLMClient:
    """Build the client for ``provider`` without any credential fallback."""
    section = section or {}
    cls = _CLIENT_TYPES.get(str(provider).lower())
    if cls is None:
        raise GenerationError(f"unknown LLM provider: {provider!r}")
    return cls(
        model=section.get("model") or DEFAULT_MODELS.get(str(provider).lower(), ""),
        endpoint=section.get("endpoint", ""),
        api_key=section.get("api_key", ""),
        temperature=float(section.get("temperature", 0.4)),
        max_tokens=int(section.get("max_tokens", 220)),
        provider=str(provider).lower(),
    )


def available_providers(section: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every provider that currently has credentials, best-first.

    Lets the system pick a working backend automatically instead of failing
    because the configured one has no key -- during a live interview a slightly
    slower model beats no answer at all.
    """
    section = section or {}
    preferred = str(section.get("provider", "openai")).lower()
    order = [preferred] + [p for p in DEFAULT_MODELS if p != preferred]
    found: List[Dict[str, Any]] = []
    for provider in order:
        try:
            client = client_for(provider, section)
        except GenerationError:
            continue
        if client.available():
            found.append({"provider": provider, "model": client.model,
                          "client": client})
    return found


def llm_from_config(section: Dict[str, Any]) -> LLMClient:
    """Resolve an LLM client from configuration.

    Prefers the configured provider, then any other provider that happens to
    have credentials in the environment, and finally falls back to the
    deterministic stand-in so the system always runs.
    """
    if not section:
        section = {}
    provider = str(section.get("provider", "openai")).lower()

    try:
        client = client_for(provider, section)
    except GenerationError as exc:
        LOG.warning("unknown LLM provider %r (%s); using the composer", provider, exc)
        return DeterministicLLM()

    if client.available():
        return client

    for candidate in available_providers(section):
        if candidate["provider"] != provider:
            LOG.info("provider %r has no key; using %r instead",
                     provider, candidate["provider"])
            return candidate["client"]

    LOG.warning("no LLM credentials found; using the deterministic composer")
    return DeterministicLLM()
