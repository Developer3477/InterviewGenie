/* InterviewGenie — Live overlay.
 *
 * The interface used *during* an interview.  Design rules:
 *   - the answer owns the screen; everything else is chrome
 *   - answering is automatic — no click between the question and the answer
 *   - the interviewer's words stay visible so you can re-read what was asked
 *   - every path degrades: no speech API -> type it, no model -> composer
 */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    ws: null,
    sessionId: null,
    started: false,
    auto: true,
    recording: false,
    recognition: null,
    mediaStream: null,
    // transcript accumulation for question detection
    buffer: "",
    finals: [],
    lastFinalAt: 0,
    detectTimer: null,
    questionAt: 0,
    firstTokenAt: 0,
    streaming: false,
    answered: 0
  };

  // ─────────────────────────────────────────────────────────────── helpers
  function setPill(id, text, kind) {
    var el = $(id);
    el.textContent = text;
    el.className = "pill" + (kind ? " pill-" + kind : "");
  }

  function log(line) {
    var t = $("transcript");
    if (t.querySelector(".dim")) { t.innerHTML = ""; }
    var div = document.createElement("div");
    div.className = "line";
    div.textContent = line;
    t.appendChild(div);
    t.scrollTop = t.scrollHeight;
    while (t.children.length > 40) { t.removeChild(t.firstChild); }
  }

  function showAnswer(text, streaming) {
    var a = $("answer");
    if (!streaming && a.dataset.full === text) { return; }
    a.textContent = text;
    if (!streaming) { a.dataset.full = text; }
  }

  function clearAnswer() {
    var a = $("answer");
    a.textContent = "";
    delete a.dataset.full;
    $("meta").textContent = "";
  }

  function nowMs() { return Date.now(); }

  // ─────────────────────────────────────────────────────────────── socket
  function connect() {
    var proto = location.protocol === "https:" ? "wss://" : "ws://";
    var ws = new WebSocket(proto + location.host + "/ws");
    state.ws = ws;

    ws.onopen = function () {
      setPill("conn", "connected", "on");
      send({ type: "start", profile: {} });
    };
    ws.onclose = function () {
      setPill("conn", "reconnecting…", "off");
      setTimeout(connect, 1200);
    };
    ws.onerror = function () { setPill("conn", "error", "off"); };
    ws.onmessage = function (event) { handle(JSON.parse(event.data)); };
  }

  function send(payload) {
    if (state.ws && state.ws.readyState === 1) { state.ws.send(JSON.stringify(payload)); }
  }

  // ─────────────────────────────────────────────────────── inbound frames
  function handle(message) {
    switch (message.type) {
      case "session":
        state.sessionId = message.session && message.session.id;
        state.started = true;
        setPill("conn", "connected", "on");
        break;

      case "analysis":
        state.firstTokenAt = nowMs();
        var bits = [];
        if (message.intent) { bits.push(message.intent); }
        if (message.topic) { bits.push(message.topic); }
        if (message.strategy) { bits.push(message.strategy); }
        if (message.emotion && message.emotion.dominant) {
          bits.push(message.emotion.dominant);
        }
        $("meta").textContent = bits.join(" · ");
        clearAnswer();
        break;

      case "delta":
        if (!state.streaming) {
          state.streaming = true;
          var a = $("answer");
          a.textContent = "";
          delete a.dataset.full;
        }
        showAnswer($("answer").textContent + (message.text || ""), true);
        break;

      case "response":
        state.streaming = false;
        state.answered += 1;
        var r = message.response || message;
        showAnswer(r.text || "", false);
        if (r.scorecard) {
          var s = r.scorecard;
          var overall = (s.overall !== undefined && s.overall !== null)
            ? (typeof s.overall === "number" ? s.overall.toFixed(2) : s.overall)
            : null;
          $("meta").textContent =
            (r.strategy ? r.strategy + " · " : "") +
            (r.latency_ms ? Math.round(r.latency_ms) + "ms · " : "") +
            (overall !== null ? "score " + overall : "");
        }
        if (state.questionAt && state.firstTokenAt) {
          var ttft = state.firstTokenAt - state.questionAt;
          setPill("latency", ttft + "ms to first words", ttft < 2500 ? "on" : "warn");
        }
        break;

      case "event":
        break;

      case "error":
        showAnswer("⚠ " + (message.message || "something went wrong"), false);
        break;
    }
  }

  // ───────────────────────────────────────────────── question detection
  // A question is "complete" when the transcript ends in a question mark, or
  // when it opens with a question word and the interviewer has paused.  This
  // is what removes the manual "Start answering" click that every competitor
  // still asks for.
  var QUESTION_OPENERS = [
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "tell me", "walk me", "describe", "explain", "can you", "could you",
    "would you", "do you", "did you", "have you", "are you", "is there",
    "talk me", "give me", "share", "let's", "imagine", "suppose"
  ];

  function looksLikeQuestion(text) {
    var t = text.trim();
    if (!t) { return false; }
    if (t.indexOf("?") !== -1) { return true; }
    var low = t.toLowerCase();
    for (var i = 0; i < QUESTION_OPENERS.length; i += 1) {
      if (low.indexOf(QUESTION_OPENERS[i]) === 0) { return true; }
    }
    return false;
  }

  function onFinal(text) {
    state.lastFinalAt = nowMs();
    state.buffer = (state.buffer + " " + text).replace(/\s+/g, " ").trim();
    log(state.buffer);
    if (state.detectTimer) { clearTimeout(state.detectTimer); }
    // wait for a pause, then decide whether we have a whole question
    state.detectTimer = setTimeout(maybeAsk, 700);
  }

  function maybeAsk() {
    if (!state.auto || !state.buffer) { return; }
    var question = state.buffer;
    if (!looksLikeQuestion(question)) { return; }
    var words = question.split(/\s+/).length;
    if (words < 3) { return; }              // too short to be a real question
    state.buffer = "";
    state.questionAt = nowMs();
    state.firstTokenAt = 0;
    setPill("latency", "thinking…", "warn");
    send({ type: "question", text: question, confidence: 0.9 });
  }

  // ───────────────────────────────────────────────────────── audio setup
  function stopRecognition() {
    if (state.recognition) {
      try { state.recognition.stop(); } catch (e) { /* already stopped */ }
    }
    state.recognition = null;
    state.recording = false;
    if (state.mediaStream) {
      state.mediaStream.getTracks().forEach(function (t) { t.stop(); });
      state.mediaStream = null;
    }
    setPill("src", "no audio", "muted");
  }

  function startRecognition(stream, label) {
    var Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      setPill("src", "speech API unavailable", "off");
      $("dlg-type").showModal();
      return;
    }
    state.mediaStream = stream;
    var recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;

    recognition.onresult = function (event) {
      for (var i = event.resultIndex; i < event.results.length; i += 1) {
        var result = event.results[i];
        var text = (result[0].transcript || "").trim();
        if (!text) { continue; }
        if (result.isFinal) {
          onFinal(text);
        } else {
          var t = $("transcript");
          var last = t.lastElementChild;
          if (last && last.dataset.interim) {
            last.textContent = state.buffer + " " + text;
          } else {
            var div = document.createElement("div");
            div.className = "line dim";
            div.dataset.interim = "1";
            div.textContent = state.buffer + " " + text;
            t.appendChild(div);
          }
        }
      }
    };

    recognition.onerror = function (event) {
      // "no-speech" and "aborted" are routine; anything else is worth showing
      if (event.error !== "no-speech" && event.error !== "aborted") {
        setPill("src", label + " · " + event.error, "warn");
      }
    };

    recognition.onend = function () {
      // Chrome stops after a silence; keep listening for the whole interview
      if (state.recording && state.recognition === recognition) {
        try { recognition.start(); } catch (e) { /* restart race */ }
      }
    };

    state.recognition = recognition;
    state.recording = true;
    try {
      recognition.start();
      setPill("src", label + " · listening", "on");
    } catch (e) {
      setPill("src", "could not start", "off");
    }
  }

  function useMic() {
    stopRecognition();
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
      startRecognition(stream, "microphone");
    }).catch(function (err) {
      setPill("src", "mic denied", "off");
      showAnswer("⚠ Microphone access was denied (" + err.name + "). " +
                 "Use “⌨ Type” or pick tab audio instead.", false);
    });
  }

  function useDisplayAudio() {
    stopRecognition();
    if (!navigator.mediaDevices.getDisplayMedia) {
      setPill("src", "tab audio unsupported", "off");
      showAnswer("⚠ This browser cannot capture tab audio. Use the microphone, " +
                 "or open the page in Chrome or Edge.", false);
      return;
    }
    navigator.mediaDevices.getDisplayMedia({ audio: true, video: true })
      .then(function (stream) {
        // drop the video track: we only want the interviewer's voice
        stream.getVideoTracks().forEach(function (t) { t.stop(); stream.removeTrack(t); });
        if (stream.getAudioTracks().length === 0) {
          stream.getTracks().forEach(function (t) { t.stop(); });
          setPill("src", "no audio shared", "off");
          showAnswer("⚠ No audio was shared. Re-open “🎧 Audio source”, pick the " +
                     "meeting tab, and tick “Share tab audio”.", false);
          return;
        }
        startRecognition(stream, "tab audio");
      })
      .catch(function (err) {
        setPill("src", "capture cancelled", "off");
      });
  }

  // ─────────────────────────────────────────────────────── user actions
  function ask(text) {
    state.questionAt = nowMs();
    state.firstTokenAt = 0;
    setPill("latency", "thinking…", "warn");
    send({ type: "question", text: text, confidence: 1.0 });
  }

  function wire() {
    $("btn-src").addEventListener("click", function () { $("dlg-src").showModal(); });
    $("src-mic").addEventListener("click", function () {
      $("dlg-src").close(); useMic();
    });
    $("src-display").addEventListener("click", function () {
      $("dlg-src").close(); useDisplayAudio();
    });
    $("src-cancel").addEventListener("click", function () { $("dlg-src").close(); });

    $("btn-type").addEventListener("click", function () {
      $("dlg-type").showModal(); $("typed").focus();
    });
    $("type-close").addEventListener("click", function () { $("dlg-type").close(); });
    $("type-send").addEventListener("click", function () {
      var text = $("typed").value.trim();
      if (!text) { return; }
      $("typed").value = "";
      $("dlg-type").close();
      log(text);
      ask(text);
    });

    $("btn-auto").addEventListener("click", function () {
      state.auto = !state.auto;
      setPill("auto", state.auto ? "auto-answer on" : "auto-answer off",
              state.auto ? "on" : "muted");
    });

    $("btn-reask").addEventListener("click", function () {
      var last = $("transcript").textContent.trim();
      if (last) { ask(last.slice(-400)); }
    });
    $("btn-accept").addEventListener("click", function () {
      send({ type: "accept", edited: false, rejected: false });
      setPill("latency", "noted", "on");
    });
    $("btn-reject").addEventListener("click", function () {
      send({ type: "accept", edited: false, rejected: true });
      setPill("latency", "noted", "warn");
    });
    $("btn-clear").addEventListener("click", function () {
      clearAnswer(); state.buffer = ""; $("transcript").innerHTML = "";
    });

    $("btn-pop").addEventListener("click", function () {
      var w = window.open(location.href, "iglive",
        "popup=yes,width=760,height=520");
      if (w) { w.focus(); }
    });

    // keyboard: everything reachable without the mouse
    document.addEventListener("keydown", function (event) {
      if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT") {
        if (event.key === "Escape") { event.target.blur(); }
        return;
      }
      if (event.key === " ") { event.preventDefault(); $("btn-reask").click(); }
      else if (event.key === "a" || event.key === "A") { $("btn-accept").click(); }
      else if (event.key === "r" || event.key === "R") { $("btn-reject").click(); }
      else if (event.key === "Escape") { $("btn-clear").click(); }
      else if (event.key === "t" || event.key === "T") { $("btn-type").click(); }
    });

    window.addEventListener("beforeunload", stopRecognition);
  }

  // ────────────────────────────────────────────────────────────── startup
  wire();
  connect();

  // Ask for the microphone immediately: the candidate should not have to
  // click anything once the interview starts.
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    setTimeout(useMic, 400);
  } else {
    $("dlg-type").showModal();
  }
})();
