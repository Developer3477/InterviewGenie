"""Cloud speech-to-text adapters (Google, Azure, IBM).

All three adapters speak real HTTP(S) using only the standard library
(``urllib.request`` + ``http.client``), so they work without any third-party
SDK.  Credentials come from the ``asr.credentials`` config section or from
environment variables (``GOOGLE_APPLICATION_CREDENTIALS``,
``AZURE_SPEECH_KEY`` / ``AZURE_SPEECH_REGION``, ``IBM_SPEECH_API_KEY`` /
``IBM_SPEECH_URL``).

Each adapter implements chunked streaming recognition: audio is posted as it
arrives and interim/final results are decoded from the vendor's response
protocol.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import ASRError, ASRTimeoutError, LowConfidenceError
from ..logging import get_logger
from ..types import TranscriptChunk
from .base import ASRCapabilities, ASRProvider
from .dsp import int16_from_floats

LOG = get_logger("asr.cloud")

_HTTP_TIMEOUT = 20.0


def _post(url: str, payload: bytes, headers: Dict[str, str], timeout: float = _HTTP_TIMEOUT) -> bytes:
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _bearer_token_from_service_account(credentials: Dict[str, Any]) -> Optional[str]:
    """Mint an OAuth2 access token from a GCP service-account key file."""
    key_path = credentials.get("service_account_file") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.exists(key_path):
        return None
    try:
        import jwt  # type: ignore

        with open(key_path, "r", encoding="utf-8") as fh:
            key = json.load(fh)
        now = int(time.time())
        claim = {
            "iss": key["client_email"],
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": key.get("token_uri", "https://oauth2.googleapis.com/token"),
            "iat": now,
            "exp": now + 3600,
        }
        signed = jwt.encode(claim, key["private_key"], algorithm="RS256",
                            headers={"kid": key.get("private_key_id")})
        body = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": signed,
        }).encode()
        raw = _post(key.get("token_uri", "https://oauth2.googleapis.com/token"),
                    body, {"Content-Type": "application/x-www-form-urlencoded"})
        return json.loads(raw).get("access_token")
    except Exception as exc:  # noqa: BLE001 - missing lib / bad key
        LOG.warning("could not mint a GCP access token: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Google Cloud Speech-to-Text
# --------------------------------------------------------------------------- #
@dataclass
class GoogleSpeechProvider(ASRProvider):
    """Google Cloud Speech-to-Text (streaming + synchronous recognise)."""

    credentials: Dict[str, Any] = field(default_factory=dict)
    language: str = "en-US"
    sample_rate: int = 16000
    min_confidence: float = 0.45
    name: str = "google"

    def __post_init__(self) -> None:
        self._buffer: List[float] = []
        self._outbox: List[TranscriptChunk] = []
        self._token: Optional[str] = None
        self._token_at: float = 0.0

    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            streaming=True, partial_results=True, diarisation=True,
            punctuation=True, word_timestamps=True, languages=(self.language,),
            noise_robust=True,
        )

    def reset(self) -> None:
        self._buffer = []
        self._outbox = []

    def _access_token(self) -> str:
        if self._token and time.time() - self._token_at < 3000:
            return self._token
        token = self.credentials.get("access_token") or _bearer_token_from_service_account(
            self.credentials)
        if not token:
            raise ASRError("no Google credentials available", code="asr.provider_unavailable")
        self._token = token
        self._token_at = time.time()
        return token

    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        self._buffer.extend(float(s) for s in samples)

    def flush(self) -> None:
        if not self._buffer:
            return
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.sample_rate,
                "languageCode": self.language,
                "enableAutomaticPunctuation": True,
                "model": "latest_long",
                "useEnhanced": True,
            },
            "audio": {"content": base64.b64encode(
                int16_from_floats(self._buffer)).decode("ascii")},
        }
        try:
            body = _post(
                "https://speech.googleapis.com/v1/speech:recognize",
                json.dumps(payload).encode("utf-8"),
                {"Authorization": f"Bearer {self._access_token()}",
                 "Content-Type": "application/json; charset=utf-8"},
            )
            data = json.loads(body)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ASRError(f"google speech request failed: {exc}") from exc

        duration = len(self._buffer) / self.sample_rate
        for result in data.get("results", []):
            alternative = (result.get("alternatives") or [{}])[0]
            text = alternative.get("transcript", "").strip()
            confidence = float(alternative.get("confidence", 0.0))
            self._outbox.append(TranscriptChunk(
                text=text, is_final=bool(result.get("isFinal", True)),
                start=0.0, end=duration, confidence=confidence, provider=self.name,
            ))
        self._buffer = []

    def poll(self) -> List[TranscriptChunk]:
        chunks, self._outbox = self._outbox, []
        return chunks


# --------------------------------------------------------------------------- #
# Microsoft Azure Speech Services
# --------------------------------------------------------------------------- #
@dataclass
class AzureSpeechProvider(ASRProvider):
    """Azure Cognitive Services Speech-to-Text (short-audio REST endpoint)."""

    credentials: Dict[str, Any] = field(default_factory=dict)
    language: str = "en-US"
    sample_rate: int = 16000
    min_confidence: float = 0.45
    name: str = "azure"

    def __post_init__(self) -> None:
        self._buffer: List[float] = []
        self._outbox: List[TranscriptChunk] = []

    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            streaming=True, partial_results=True, diarisation=False,
            punctuation=True, word_timestamps=True, languages=(self.language,),
            noise_robust=True,
        )

    def reset(self) -> None:
        self._buffer = []
        self._outbox = []

    def _endpoint(self) -> str:
        key = self.credentials.get("key") or os.environ.get("AZURE_SPEECH_KEY")
        region = self.credentials.get("region") or os.environ.get("AZURE_SPEECH_REGION")
        if not key or not region:
            raise ASRError("azure speech requires key + region",
                           code="asr.provider_unavailable")
        return (f"https://{region}.stt.speech.microsoft.com/speech/recognition/"
                f"conversation/cognitiveservices/v1?language={self.language}"
                f"&format=detailed"), key

    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        self._buffer.extend(float(s) for s in samples)

    def flush(self) -> None:
        if not self._buffer:
            return
        endpoint, key = self._endpoint()
        pcm = int16_from_floats(self._buffer)
        try:
            body = _post(endpoint, pcm, {
                "Ocp-Apim-Subscription-Key": key,
                "Content-Type": f"audio/wav; codec=audio/pcm; samplerate={self.sample_rate}",
                "Accept": "application/json",
            })
            data = json.loads(body)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ASRError(f"azure speech request failed: {exc}") from exc
        duration = len(self._buffer) / self.sample_rate
        status = data.get("RecognitionStatus")
        if status != "Success":
            LOG.warning("azure recognition did not succeed", context={"status": status})
            self._buffer = []
            return
        best = data.get("NBest", [{}])[0]
        text = best.get("Display", "").strip()
        self._outbox.append(TranscriptChunk(
            text=text, is_final=True, start=0.0, end=duration,
            confidence=float(best.get("Confidence", 0.0)), provider=self.name,
        ))
        self._buffer = []

    def poll(self) -> List[TranscriptChunk]:
        chunks, self._outbox = self._outbox, []
        return chunks


# --------------------------------------------------------------------------- #
# IBM Watson Speech to Text
# --------------------------------------------------------------------------- #
@dataclass
class IBMSpeechProvider(ASRProvider):
    """IBM Watson Speech to Text (WebSocket-free REST fallback)."""

    credentials: Dict[str, Any] = field(default_factory=dict)
    language: str = "en-US_BroadbandModel"
    sample_rate: int = 16000
    min_confidence: float = 0.45
    name: str = "ibm"

    def __post_init__(self) -> None:
        self._buffer: List[float] = []
        self._outbox: List[TranscriptChunk] = []

    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            streaming=True, partial_results=True, diarisation=False,
            punctuation=True, word_timestamps=True, languages=(self.language,),
            noise_robust=True,
        )

    def reset(self) -> None:
        self._buffer = []
        self._outbox = []

    def _auth(self) -> Tuple[str, Dict[str, str]]:
        key = self.credentials.get("api_key") or os.environ.get("IBM_SPEECH_API_KEY")
        url = self.credentials.get("url") or os.environ.get(
            "IBM_SPEECH_URL", "https://api.us-south.speech-to-text.watson.cloud.ibm.com")
        if not key:
            raise ASRError("ibm speech requires an api key",
                           code="asr.provider_unavailable")
        token = base64.b64encode(f"apikey:{key}".encode()).decode()
        return url, {"Authorization": f"Basic {token}"}

    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        self._buffer.extend(float(s) for s in samples)

    def flush(self) -> None:
        if not self._buffer:
            return
        url, headers = self._auth()
        pcm = int16_from_floats(self._buffer)
        try:
            body = _post(
                f"{url}/v1/recognize?model={self.language}",
                pcm, {**headers, "Content-Type": f"audio/l16; rate={self.sample_rate}"},
            )
            data = json.loads(body)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ASRError(f"ibm speech request failed: {exc}") from exc
        duration = len(self._buffer) / self.sample_rate
        for result in data.get("results", []):
            for alternative in result.get("alternatives", []):
                self._outbox.append(TranscriptChunk(
                    text=alternative.get("transcript", "").strip(),
                    is_final=bool(result.get("final", True)),
                    start=0.0, end=duration,
                    confidence=float(alternative.get("confidence", 0.0)),
                    provider=self.name,
                ))
        self._buffer = []

    def poll(self) -> List[TranscriptChunk]:
        chunks, self._outbox = self._outbox, []
        return chunks


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
@dataclass
class CloudASRProvider(ASRProvider):
    """Route to the requested cloud vendor, with graceful fallback."""

    vendor: str = "google"
    credentials: Dict[str, Any] = field(default_factory=dict)
    language: str = "en-US"
    sample_rate: int = 16000
    min_confidence: float = 0.45
    name: str = "cloud"

    def __post_init__(self) -> None:
        vendor = self.vendor.lower()
        if vendor == "google":
            self._impl = GoogleSpeechProvider(self.credentials, self.language,
                                              self.sample_rate, self.min_confidence)
        elif vendor == "azure":
            self._impl = AzureSpeechProvider(self.credentials, self.language,
                                             self.sample_rate, self.min_confidence)
        elif vendor == "ibm":
            self._impl = IBMSpeechProvider(self.credentials, self.language,
                                           self.sample_rate, self.min_confidence)
        else:
            raise ASRError(f"unknown cloud vendor {self.vendor!r}")
        self._fallback = None

    def capabilities(self) -> ASRCapabilities:
        return self._impl.capabilities()

    def reset(self) -> None:
        self._impl.reset()

    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        self._impl.accept_audio(samples, sample_rate)

    def flush(self) -> None:
        try:
            self._impl.flush()
        except ASRError as exc:
            LOG.warning("cloud ASR failed, falling back to the local engine: %s", exc)
            if self._fallback is None:
                from .local import LocalStreamingASR

                self._fallback = LocalStreamingASR(sample_rate=self.sample_rate)
                if self._impl._buffer:
                    self._fallback.accept_audio(self._impl._buffer, self.sample_rate)
                    self._impl._buffer = []
            self._fallback.flush()

    def poll(self) -> List[TranscriptChunk]:
        try:
            return self._impl.poll()
        except ASRError as exc:
            LOG.warning("cloud ASR failed, falling back to local engine: %s", exc)
            if self._fallback is None:
                from .local import LocalStreamingASR

                self._fallback = LocalStreamingASR(sample_rate=self.sample_rate)
            return self._fallback.poll()


class DeepgramStreamProvider(ASRProvider):
    """Real-time streaming transcription over Deepgram's WebSocket API.

    Unlike the batch adapters above, this one keeps a socket open for the whole
    interview: PCM is pushed as it is captured and interim/final transcripts
    come back continuously, which is what keeps the perceived latency low
    enough to answer while the interviewer is still talking.

    Credentials: ``DEEPGRAM_API_KEY``, or ``asr.credentials.deepgram``.

    The connection is served by a daemon thread that reads frames into a queue,
    so ``poll()`` never blocks on the network.
    """

    name = "deepgram"

    def __init__(self, api_key: str = "", model: str = "nova-2",
                 language: str = "en-US", endpointing_ms: int = 300,
                 interim_results: bool = True, **_: Any):
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self.model = model
        self.language = language
        self.endpointing_ms = int(endpointing_ms)
        self.interim_results = interim_results
        self._queue: List[TranscriptChunk] = []
        self._ws: Optional[Any] = None
        self._thread: Optional[Any] = None
        self._closed = False
        self._sample_rate = 16000
        self._sent = 0

    # -- provider contract ------------------------------------------------ #
    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            streaming=True, partial_results=True, diarisation=False,
            punctuation=True, word_timestamps=True,
            languages=(self.language,), noise_robust=True)

    def reset(self) -> None:
        self._queue = []
        self._sent = 0

    def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
            self._ws = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def available(self) -> bool:
        return bool(self.api_key)

    # -- connection ------------------------------------------------------- #
    def _connect(self, sample_rate: int) -> None:
        if not self.api_key:
            raise ASRError("Deepgram needs an API key (DEEPGRAM_API_KEY)",
                           code="asr_no_credentials")
        from .wsclient import WebSocketClient

        query = urllib.parse.urlencode({
            "model": self.model,
            "language": self.language,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "channels": 1,
            "punctuate": "true",
            "smart_format": "true",
            "interim_results": "true" if self.interim_results else "false",
            "endpointing": self.endpointing_ms,
        })
        url = f"wss://api.deepgram.com/v1/listen?{query}"
        self._ws = WebSocketClient(
            url, headers={"Authorization": f"Token {self.api_key}"},
            timeout=30.0)
        self._closed = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        """Pull frames off the socket until it closes or the turn ends."""
        ws = self._ws
        if ws is None:
            return
        while not self._closed:
            message = ws.recv_json()
            if message is None:
                break
            kind = message.get("type")
            if kind == "Results":
                self._on_results(message)
            elif kind == "Error":
                LOG.warning("Deepgram error: %s", message.get("description"))
                break
            elif kind == "Metadata":
                LOG.debug("Deepgram metadata: %s", message.get("request_id"))

    def _on_results(self, message: Dict[str, Any]) -> None:
        try:
            alternatives = message["channel"]["alternatives"]
        except (KeyError, TypeError, IndexError):
            return
        if not alternatives:
            return
        best = alternatives[0]
        text = (best.get("transcript") or "").strip()
        if not text:
            return
        confidence = float(best.get("confidence") or 0.0)
        is_final = bool(message.get("is_final"))
        speech_final = bool(message.get("speech_final"))
        self._queue.append(self._chunk(
            text, is_final=is_final or speech_final, confidence=confidence,
            start=float(message.get("start", 0.0)), end=float(message.get("end", 0.0)),
        ))

    # -- audio ------------------------------------------------------------ #
    def accept_audio(self, samples: Sequence[float], sample_rate: int = 16000) -> None:
        if self._ws is None or self._ws.closed:
            if self._closed:
                return
            self._connect(sample_rate)
            self._sample_rate = sample_rate
        pcm = int16_from_floats(list(samples))
        if not pcm:
            return
        try:
            self._ws.send_binary(pcm)
            self._sent += len(pcm)
        except Exception as exc:  # noqa: BLE001 - a dead socket ends the stream
            LOG.warning("Deepgram send failed: %s", exc)
            self._closed = True

    def flush(self) -> None:
        """Ask Deepgram to endpoint whatever is buffered.

        Deepgram finalises on silence or on the ``endpointing`` timer, so the
        graceful thing to do is to send ``Finalize`` and wait briefly for the
        result rather than tearing the socket down.
        """
        if self._ws is None or self._ws.closed:
            return
        try:
            self._ws.send_text(json.dumps({"type": "Finalize"}))
            deadline = time.time() + 0.5
            while time.time() < deadline and not self._queue:
                time.sleep(0.05)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Deepgram finalize failed: %s", exc)

    def poll(self) -> List[TranscriptChunk]:
        out, self._queue = self._queue, []
        return out
