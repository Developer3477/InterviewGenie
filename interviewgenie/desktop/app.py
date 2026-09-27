"""The desktop application: an always-on-top overlay driven by live speech.

Run it with::

    python3 -m interviewgenie desktop

What happens:

  1. the InterviewGenie engine starts (knowledge graph, NLP, composer/LLM)
  2. a local HTTP + WebSocket server starts on 127.0.0.1
  3. the browser opens a small *capture* page that asks for tab or screen
     audio — the browser is the only cross-platform way to grab the
     interviewer's voice out of Zoom, Meet or Teams
  4. audio is streamed to the recogniser; complete questions are detected and
     answered automatically
  5. every event is pushed to the tkinter overlay through a queue

The overlay is the product. The browser page is a microphone: it can be
minimised or left on another monitor, and nothing about the interview depends
on looking at it.

Requires: Python 3.9+ with tkinter (bundled with the standard CPython
installer on Windows and macOS; on Debian/Ubuntu it is ``python3-tk``).
"""

from __future__ import annotations

import queue
import threading
import time
import webbrowser
from typing import Any, Dict, Optional

from ..logging import get_logger
from ..live import LiveConfig, LiveSession

LOG = get_logger("desktop.app")


class DesktopApp:
    """Owns the engine, the local server, the live loop and the overlay."""

    def __init__(self, *, port: int = 8422, auto: bool = True,
                 debounce_ms: int = 700, open_browser: bool = True,
                 headless: bool = False):
        self.port = port
        self.auto = auto
        self.debounce_ms = debounce_ms
        self.open_browser = open_browser
        self.headless = headless

        self.events: "queue.Queue" = queue.Queue()
        self.genie = None
        self.live: Optional[LiveSession] = None
        self.server = None
        self._loop = None
        self._stop = threading.Event()
        self._last_question = ""

    # -- wiring ---------------------------------------------------------- #
    def _emit(self, event: Dict[str, Any]) -> None:
        self.events.put(event)

    def build(self) -> None:
        """Construct the engine and the live loop (no UI, no server yet)."""
        from ..factory import build_demo_genie

        self.genie = build_demo_genie()
        self.genie.start()
        self.live = LiveSession(
            self.genie,
            LiveConfig(auto_answer=self.auto, debounce_ms=self.debounce_ms),
        )
        # So audio arriving over the socket reaches this same live session
        # rather than being answered fragment by fragment.
        self.genie.attach_live(self.live)
        LOG.info("engine ready: %s", type(self.genie.llm).__name__)

    # -- the live loop --------------------------------------------------- #
    def run_live_loop(self) -> None:
        """Background thread: polls for a paused question and answers it."""
        while not self._stop.is_set():
            try:
                if self.live is not None:
                    for event in self.live.tick(time.time()):
                        self._emit(event)
            except Exception as exc:  # noqa: BLE001 - never kill the loop
                LOG.warning("live loop error: %s", exc)
            self._stop.wait(0.1)

    # -- speech in ------------------------------------------------------- #
    def feed_transcript(self, chunk: Any) -> None:
        """Feed one recogniser emission into the live loop."""
        if self.live is None:
            return
        for event in self.live.feed_transcript(chunk, time.time()):
            self._emit(event)

    def submit(self, text: str) -> None:
        """Answer a question the candidate typed."""
        if self.live is None or not text or not text.strip():
            return
        self._last_question = text.strip()
        for event in self.live.submit(text):
            self._emit(event)

    # -- commands from the overlay --------------------------------------- #
    def command(self, name: str) -> None:
        if name == "quit":
            self.shutdown()
            return
        if name == "clear":
            self.events.put({"type": "question", "text": ""})
            self.events.put({"type": "response", "text": "", "scorecard": {}})
            return
        if name == "auto":
            if self.live is not None:
                self.live.set_auto(not self.live.auto)
                state = "on" if self.live.auto else "off"
                self.events.put({"type": "status", "state": "listening",
                                 "text": f"auto-answer {state}"})
            return
        if name == "regenerate":
            question = (self.live.pending_question if self.live else "") \
                or self._last_question
            if question:
                self.submit(question)
            return
        if name in ("accept", "reject"):
            if self.genie is not None:
                try:
                    self.genie.accept(edited=False, rejected=(name == "reject"))
                except Exception as exc:  # noqa: BLE001
                    LOG.debug("accept failed: %s", exc)
            self.events.put({"type": "status", "state": "listening",
                             "text": "noted"})
            return
        if name == "type":
            self.events.put({"type": "status", "state": "listening",
                             "text": "type in the browser page"})

    # -- the local server ------------------------------------------------ #
    def start_server(self) -> None:
        """Serve the capture page and accept audio over WebSocket.

        The server is handed ``lambda: self.genie`` as its session factory, so
        the audio that arrives over the socket reaches the *same* engine the
        overlay is reading from — otherwise the overlay would show answers for
        a different session than the one being spoken to.

        ``serve_forever`` is a coroutine, so it needs its own event loop on a
        background thread; the overlay owns the main thread.
        """
        import asyncio
        import os

        from ..api.server import APIServer

        web_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "web")
        self.server = APIServer(factory=lambda: self.genie,
                                host="127.0.0.1", port=self.port,
                                web_root=web_root)
        self._loop = asyncio.new_event_loop()

        def _serve() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.server.serve_forever())

        threading.Thread(target=_serve, daemon=True).start()
        LOG.info("local server on http://127.0.0.1:%s", self.port)

    def open_capture_page(self) -> None:
        url = f"http://127.0.0.1:{self.port}/capture"
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 - a missing browser is not fatal
            LOG.warning("could not open the browser: %s", exc)
            LOG.info("open %s manually", url)

    # -- lifecycle ------------------------------------------------------- #
    def run(self) -> int:
        self.build()
        if self.headless:
            self._run_headless()
            return 0

        try:
            from .overlay import Overlay
        except ImportError as exc:
            LOG.error("the overlay needs tkinter: %s", exc)
            print("InterviewGenie's overlay needs tkinter, which is missing.")
            print("  Windows / macOS: it ships with the Python installer.")
            print("  Debian / Ubuntu : sudo apt install python3-tk")
            print("  Fedora         : sudo dnf install python3-tkinter")
            print("Or run without the GUI:  --headless")
            return 1

        self.start_server()
        threading.Thread(target=self.run_live_loop, daemon=True).start()
        self._emit({"type": "status", "state": "listening",
                    "text": "starting…"})
        if self.open_browser:
            # let the server bind before the browser asks for audio
            threading.Timer(1.0, self.open_capture_page).start()

        overlay = Overlay(self.events, on_command=self.command)
        try:
            overlay.start()
        finally:
            self.shutdown()
        return 0

    def _run_headless(self) -> None:
        """No GUI: run the live loop until interrupted (for tests and CI)."""
        self.start_server()
        threading.Thread(target=self.run_live_loop, daemon=True).start()
        LOG.info("headless mode; press Ctrl+C to stop")
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        server, loop = self.server, self._loop
        self.server = None
        if server is None:
            return

        async def _stop() -> None:
            try:
                await server.stop()
            except Exception as exc:  # noqa: BLE001
                LOG.debug("server stop: %s", exc)

        if loop is not None and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(_stop(), loop).result(timeout=5)
            except Exception as exc:  # noqa: BLE001
                LOG.debug("server shutdown: %s", exc)
            finally:
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    pass


def main(argv: Optional[list] = None) -> int:
    """CLI entry point: ``python3 -m interviewgenie desktop``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="interviewgenie-desktop",
        description="Always-on-top interview overlay.")
    parser.add_argument("--port", type=int, default=8422)
    parser.add_argument("--no-auto", action="store_true",
                        help="require a click before answering")
    parser.add_argument("--debounce", type=int, default=700,
                        help="ms of silence that marks a question as finished")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the capture page automatically")
    parser.add_argument("--headless", action="store_true",
                        help="run without the GUI (tests, servers, CI)")
    args = parser.parse_args(argv)

    app = DesktopApp(port=args.port, auto=not args.no_auto,
                     debounce_ms=args.debounce,
                     open_browser=not args.no_browser,
                     headless=args.headless)
    return app.run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
