"""The always-on-top overlay — the screen the candidate actually looks at.

Pure standard library (``tkinter`` ships with CPython), so the desktop app has
no dependencies to install. The overlay owns the main thread, as tkinter
requires; the speech pipeline and HTTP server run on background threads and
push events through a queue that the UI drains on a timer.

Design rules for a screen you read while talking:

  * the answer dominates; everything else is chrome
  * never steal focus, so typing in the interview keeps working
  * one glance gives you: what was asked, the answer, and whether it is current
  * everything is reachable from the keyboard
"""

from __future__ import annotations

import queue
import tkinter as tk
from typing import Any, Callable, Dict, Optional

from ..logging import get_logger

LOG = get_logger("desktop.overlay")

BG = "#0b0d12"
PANEL = "#14171f"
TEXT = "#e8ecf4"
DIM = "#8b93a7"
GOOD = "#35c37d"
WARN = "#e8b339"
BAD = "#e5544b"

FONT_ANSWER = ("Segoe UI", 15)
FONT_UI = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 9)


class Overlay:
    """A compact, always-on-top, borderless answer window."""

    WIDTH = 460
    HEIGHT = 420

    def __init__(self, events: "queue.Queue",
                 on_command: Optional[Callable[[str], None]] = None):
        self.events = events
        self.on_command = on_command
        self.answer_text = ""
        self.question_text = ""
        self._meta_text = ""
        self._drag = {"x": 0, "y": 0}
        self._build()

    # -- construction ---------------------------------------------------- #
    def _build(self) -> None:
        self.root = tk.Tk()
        self.root.title("InterviewGenie")
        self.root.configure(bg=BG)
        self.root.overrideredirect(True)          # borderless
        self.root.attributes("-topmost", True)    # always on top
        self.root.attributes("-alpha", 0.96)
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}+80+80")
        self.root.minsize(360, 260)

        # ── title bar: drag to move, buttons on the right ─────────────── #
        bar = tk.Frame(self.root, bg=PANEL, height=30)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        bar.bind("<ButtonPress-1>", self._press)
        bar.bind("<B1-Motion>", self._motion)

        grip = tk.Label(bar, text="⠿", bg=PANEL, fg=DIM, font=FONT_UI)
        grip.pack(side="left", padx=(9, 4))
        grip.bind("<ButtonPress-1>", self._press)
        grip.bind("<B1-Motion>", self._motion)

        title = tk.Label(bar, text="InterviewGenie", bg=PANEL, fg=TEXT,
                         font=("Segoe UI Semibold", 10))
        title.pack(side="left")
        title.bind("<ButtonPress-1>", self._press)
        title.bind("<B1-Motion>", self._motion)

        self.status = tk.Label(bar, text="idle", bg=PANEL, fg=DIM, font=FONT_UI)
        self.status.pack(side="left", padx=(10, 0))

        for label, cmd, colour in (("—", "minimise", DIM), ("✕", "quit", BAD)):
            btn = tk.Label(bar, text=label, bg=PANEL, fg=colour,
                           font=FONT_UI, cursor="hand2", padx=7)
            btn.pack(side="right")
            btn.bind("<Button-1>", lambda _e, c=cmd: self._command(c))

        # ── what the interviewer said ─────────────────────────────────── #
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(head, text="INTERVIEWER", bg=BG, fg=DIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.question = tk.Label(head, text="waiting for audio…", bg=BG,
                                 fg="#c3cadb", font=FONT_UI,
                                 wraplength=self.WIDTH - 30, justify="left",
                                 anchor="w")
        self.question.pack(fill="x", pady=(2, 0))

        # ── the answer ────────────────────────────────────────────────── #
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(4, 6))
        self.answer = tk.Label(body, text="", bg=BG, fg=TEXT, font=FONT_ANSWER,
                               wraplength=self.WIDTH - 30, justify="left",
                               anchor="nw")
        self.answer.pack(fill="both", expand=True)

        # ── footer: metrics + controls ────────────────────────────────── #
        foot = tk.Frame(self.root, bg=PANEL)
        foot.pack(fill="x", side="bottom")
        self.meta = tk.Label(foot, text="", bg=PANEL, fg=DIM, font=FONT_MONO,
                             anchor="w")
        self.meta.pack(fill="x", padx=12, pady=(6, 2))

        controls = tk.Frame(foot, bg=PANEL)
        controls.pack(fill="x", padx=10, pady=(0, 8))
        for label, cmd, tip in (("↻", "regenerate", "Regenerate  (Space)"),
                                ("✓", "accept", "Used it  (A)"),
                                ("✗", "reject", "Missed  (R)"),
                                ("⌨", "type", "Type the question  (T)"),
                                ("◐", "auto", "Toggle auto-answer"),
                                ("✕", "clear", "Clear  (Esc)")):
            btn = tk.Label(controls, text=label, bg=PANEL, fg=TEXT,
                           font=FONT_UI, cursor="hand2", padx=8, pady=2)
            btn.pack(side="left", padx=(0, 2))
            btn.bind("<Button-1>", lambda _e, c=cmd: self._command(c))
            btn.bind("<Enter>", lambda _e, t=tip: self.meta.config(text=t))
            btn.bind("<Leave>", lambda _e: self.meta.config(text=self._meta_text))

        self.root.bind("<space>", lambda _e: self._command("regenerate"))
        self.root.bind("<a>", lambda _e: self._command("accept"))
        self.root.bind("<r>", lambda _e: self._command("reject"))
        self.root.bind("<t>", lambda _e: self._command("type"))
        self.root.bind("<Escape>", lambda _e: self._command("clear"))

    # -- dragging -------------------------------------------------------- #
    def _press(self, event: Any) -> None:
        self._drag["x"] = event.x
        self._drag["y"] = event.y

    def _motion(self, event: Any) -> None:
        x = self.root.winfo_x() + event.x - self._drag["x"]
        y = self.root.winfo_y() + event.y - self._drag["y"]
        self.root.geometry(f"+{x}+{y}")

    # -- commands -------------------------------------------------------- #
    def _command(self, name: str) -> None:
        if name == "quit":
            self.root.quit()
            return
        if name == "minimise":
            self.root.iconify()
            return
        if self.on_command:
            self.on_command(name)

    # -- rendering ------------------------------------------------------- #
    def set_status(self, text: str, colour: str = DIM) -> None:
        self.status.config(text=text, fg=colour)

    def set_question(self, text: str) -> None:
        self.question_text = text
        self.question.config(text=text or "waiting for audio…")

    def append_answer(self, fragment: str) -> None:
        if not fragment:
            return
        self.answer_text += fragment
        self.answer.config(text=self.answer_text)

    def set_answer(self, text: str) -> None:
        self.answer_text = text or ""
        self.answer.config(text=self.answer_text)

    def set_meta(self, text: str) -> None:
        self._meta_text = text or ""
        self.meta.config(text=self._meta_text)

    # -- event loop ------------------------------------------------------ #
    def start(self) -> None:
        self._drain()
        self.root.mainloop()

    def _drain(self) -> None:
        """Render every queued event, then re-arm. Never blocks the UI."""
        try:
            while True:
                self.render(self.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(40, self._drain)

    def render(self, event: Dict[str, Any]) -> None:
        kind = event.get("type")

        if kind == "status":
            colour = {"listening": GOOD, "thinking": WARN,
                      "error": BAD}.get(event.get("state"), DIM)
            self.set_status(event.get("text", ""), colour)

        elif kind == "partial":
            self.set_question(event.get("text", ""))

        elif kind == "question":
            self.set_question(event.get("text", ""))
            self.set_answer("")
            self.set_status("thinking…", WARN)
            self.set_meta("")

        elif kind == "analysis":
            bits = [event.get("strategy") or "", event.get("topic") or ""]
            self.set_meta(" · ".join(b for b in bits if b))

        elif kind == "delta":
            self.append_answer(event.get("text", ""))

        elif kind == "response":
            self.set_answer(event.get("text", ""))
            scorecard = event.get("scorecard") or {}
            score = scorecard.get("overall")
            latency = event.get("latency_ms")
            bits = []
            if event.get("strategy"):
                bits.append(event["strategy"])
            if score is not None:
                bits.append(f"score {score:.2f}")
            if latency:
                bits.append(f"{latency:.0f}ms")
            self.set_meta(" · ".join(bits))
            self.set_status("ready", GOOD)

        elif kind == "error":
            self.set_status("error", BAD)
            self.set_meta(str(event.get("message", ""))[:90])
