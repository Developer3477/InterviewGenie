/* InterviewGenie live cockpit.
 *
 * Talks to the stdlib asyncio API server over a WebSocket. Falls back to plain
 * HTTP POSTs when WebSockets are unavailable so the UI still works.
 */
(function () {
  "use strict";

  var state = {
    ws: null,
    connected: false,
    sessionId: null,
    lastResponse: null,
    recognition: null,
    recording: false,
    audioCtx: null,
    samples: [],
  };

  var SAMPLE_QUESTIONS = [
    "Tell me about yourself.",
    "Walk me through your resume.",
    "How would you design a URL shortener?",
    "What is the difference between TCP and UDP?",
    "Tell me about a time you showed leadership.",
    "Tell me about a time you dealt with a difficult colleague.",
    "This service is returning 500 errors, how would you debug it?",
    "What is your greatest weakness?",
    "Why do you want to work here?",
    "How would you design a rate limiter?",
    "Where do you see yourself in five years?",
    "What are the tradeoffs between microservices and a monolith?",
    "How do you measure the success of a project?",
    "Do you have any questions for us?"
  ];

  // ---------------------------------------------------------------- helpers
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }
  function setPill(id, text, on) {
    var node = $(id);
    if (!node) { return; }
    node.textContent = text;
    node.classList.toggle("pill-on", !!on);
    node.classList.toggle("pill-off", !on);
  }
  function fmt(value, digits) {
    if (value === undefined || value === null || isNaN(value)) { return "–"; }
    return Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  // ---------------------------------------------------------------- socket
  function connect() {
    var protocol = location.protocol === "https:" ? "wss://" : "ws://";
    var url = protocol + location.host + "/ws";
    try {
      state.ws = new WebSocket(url);
    } catch (err) {
      setPill("conn", "no websocket", false);
      return;
    }
    state.ws.onopen = function () {
      state.connected = true;
      setPill("conn", "connected", true);
      send({ type: "describe" });
    };
    state.ws.onclose = function () {
      state.connected = false;
      setPill("conn", "disconnected", false);
      setTimeout(connect, 2500);
    };
    state.ws.onerror = function () { setPill("conn", "error", false); };
    state.ws.onmessage = function (event) {
      var message;
      try { message = JSON.parse(event.data); } catch (err) { return; }
      handle(message);
    };
  }

  function send(payload) {
    if (state.ws && state.ws.readyState === 1) {
      state.ws.send(JSON.stringify(payload));
      return true;
    }
    return post(payload.type, payload);
  }

  // HTTP fallback for the non-duplex paths
  function post(type, payload) {
    var map = {
      question: "/api/question",
      feedback: "/api/feedback",
      accept: "/api/accept",
      report: "/api/report"
    };
    var url = map[type];
    if (!url) { return false; }
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (r) { return r.json(); }).then(handle).catch(function () {});
    return true;
  }

  // ---------------------------------------------------------------- inbound
  function handle(message) {
    switch (message.type) {
      case "session":
        state.sessionId = message.session && message.session.id;
        setPill("phase", message.session && message.session.phase || "ready", true);
        if (message.summary) { renderSystem(message.summary); }
        break;
      case "describe":
        renderSystem(message.describe);
        break;
      case "partial":
        $("partial").textContent = "… " + (message.text || "");
        break;
      case "analysis":
        renderAnalysis(message);
        break;
      case "evidence":
        renderEvidence(message);
        break;
      case "response":
        renderResponse(message);
        break;
      case "report":
        $("report").textContent = JSON.stringify(message.report, null, 2);
        break;
      case "stopped":
        setPill("phase", "stopped", false);
        break;
      case "error":
        $("partial").textContent = "error: " + (message.message || "unknown");
        break;
      default:
        break;
    }
  }

  // ---------------------------------------------------------------- render
  function renderSystem(describe) {
    if (!describe) { return; }
    var parts = [];
    var graph = describe.graph || {};
    parts.push("knowledge graph: <b>" + (graph.nodes || 0) + "</b> nodes, <b>" +
      (graph.edges || 0) + "</b> edges, <b>" + (graph.aliases || 0) + "</b> aliases");
    var asr = describe.asr || {};
    parts.push("asr: <b>" + (asr.provider || asr.backend || "?") + "</b>" +
      (asr.phrases ? " · " + asr.phrases + " phrases" : "") +
      (asr.noise_suppression ? " · noise suppression on" : ""));
    parts.push("llm: <b>" + (describe.llm || "composer") + "</b>");
    parts.push("profile: <b>" + (describe.profile && describe.profile.name || "–") + "</b>");
    var stats = describe.profile || {};
    parts.push("stories: <b>" + ((stats.story_bank || []).length) + "</b> · skills: <b>" +
      ((stats.skills || []).length) + "</b>");
    $("system").innerHTML = parts.map(function (p) { return "<div>" + p + "</div>"; }).join("");
    $("footer-stats").textContent =
      (graph.nodes || 0) + " nodes · " + (graph.edges || 0) + " edges · " +
      (asr.phrases || 0) + " phrases";
  }

  function renderAnalysis(message) {
    $("m-intent").textContent = message.intent || "–";
    $("m-topic").textContent = message.topic || "–";
    $("m-conf").textContent = fmt(message.confidence);
    if (message.emotion) {
      $("m-emotion").textContent = message.emotion.dominant || "–";
    }
    if (message.sentiment) {
      $("m-sentiment").textContent = fmt(message.sentiment.polarity);
    }
    var box = $("entities");
    box.innerHTML = "";
    (message.entities || []).forEach(function (entity) {
      var chip = el("span", "chip", (entity.canonical || entity.text) +
        (entity.label ? " · " + entity.label : ""));
      box.appendChild(chip);
    });
  }

  function renderEvidence(message) {
    var box = $("evidence");
    box.innerHTML = "";
    if (!message.entities || !message.entities.length) {
      box.appendChild(el("p", "placeholder", "No knowledge entities matched this question."));
    } else {
      (message.entities || []).slice(0, 6).forEach(function (name) {
        box.appendChild(el("div", "ev-item", name));
      });
    }
    (message.evidence || []).slice(0, 6).forEach(function (item) {
      var node = el("div", "ev-item", item.text || "");
      node.appendChild(el("span", "src",
        (item.kind || "evidence") + " · score " + fmt(item.score)));
      box.appendChild(node);
    });
  }

  function renderResponse(message) {
    var box = $("answer");
    box.textContent = message.text || "";
    box.classList.remove("appear");
    void box.offsetWidth;
    box.classList.add("appear");
    state.lastResponse = message;

    setPill("phase", message.strategy || "answered", true);
    setPill("latency", fmt(message.latency_ms, 0) + " ms", false);

    var card = $("scorecard");
    card.innerHTML = "";
    var card2 = message.scorecard || {};
    var metrics = message.metrics || {};
    var keys = ["accuracy", "relevance", "engagement", "personalization",
                "groundedness", "fluency", "empathy"];
    keys.forEach(function (key) {
      var value = card2[key] !== undefined ? card2[key] : metrics[key];
      if (value === undefined) { return; }
      var cls = "metric" + (value >= 0.6 ? " hi" : (value < 0.3 ? " low" : ""));
      var node = el("span", cls, key + " ");
      node.appendChild(el("b", null, fmt(value)));
      card.appendChild(node);
    });
    if (card2.overall !== undefined) {
      var node = el("span", "metric", "overall ");
      node.appendChild(el("b", null, fmt(card2.overall)));
      card.appendChild(node);
    }
  }

  // ---------------------------------------------------------------- actions
  function profileFromForm() {
    var skills = $("p-skills").value.split(",").map(function (s) { return s.trim(); })
      .filter(Boolean);
    var strengths = $("p-strengths").value.split(",").map(function (s) { return s.trim(); })
      .filter(Boolean);
    return {
      name: $("p-name").value,
      target_role: $("p-role").value,
      target_company: $("p-company").value,
      years_experience: parseInt($("p-years").value, 10) || null,
      skills: skills,
      strengths: strengths
    };
  }

  function startSession() {
    send({ type: "start", profile: profileFromForm() });
    setPill("phase", "listening", true);
  }

  function ask() {
    var text = $("question").value.trim();
    if (!text) { return; }
    send({ type: "question", text: text, confidence: 1.0 });
  }

  // ---------------------------------------------------------------- speech
  function initSpeech() {
    var Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      $("btn-mic").textContent = "🎙 speech recognition unavailable";
      $("btn-mic").disabled = true;
      return;
    }
    var recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = function (event) {
      var interim = "";
      for (var i = event.resultIndex; i < event.results.length; i += 1) {
        var result = event.results[i];
        if (result.isFinal) {
          $("question").value = result[0].transcript.trim();
          send({ type: "question", text: result[0].transcript.trim(),
                 confidence: result[0].confidence || 0.9 });
        } else {
          interim += result[0].transcript;
        }
      }
      if (interim) { $("partial").textContent = "… " + interim; }
    };
    recognition.onerror = function (event) {
      $("partial").textContent = "mic error: " + event.error;
    };
    recognition.onend = function () {
      if (state.recording) { recognition.start(); }
    };
    state.recognition = recognition;
  }

  function toggleMic() {
    if (!state.recognition) { return; }
    var button = $("btn-mic");
    if (state.recording) {
      state.recording = false;
      state.recognition.stop();
      button.textContent = "🎙 Start microphone";
      button.classList.remove("recording");
    } else {
      state.recording = true;
      state.recognition.start();
      button.textContent = "⏹ Stop microphone";
      button.classList.add("recording");
    }
  }

  // ---------------------------------------------------------------- wiring
  function wire() {
    $("btn-start").addEventListener("click", startSession);
    $("btn-ask").addEventListener("click", ask);
    $("btn-random").addEventListener("click", function () {
      $("question").value = SAMPLE_QUESTIONS[
        Math.floor(Math.random() * SAMPLE_QUESTIONS.length)];
      ask();
    });
    $("btn-feedback").addEventListener("click", function () {
      var text = $("feedback").value.trim();
      if (!text) { return; }
      send({ type: "feedback", text: text });
      $("feedback").value = "";
      $("learning").innerHTML = "<div>applied feedback: <b>" + text + "</b></div>";
    });
    $("btn-report").addEventListener("click", function () {
      send({ type: "report" });
    });
    $("btn-bench").addEventListener("click", function () {
      fetch("/api/benchmark?limit=12").then(function (r) { return r.json(); })
        .then(function (data) {
          $("report").textContent = JSON.stringify(data, null, 2);
        }).catch(function () {});
    });
    $("btn-mic").addEventListener("click", toggleMic);
    document.querySelectorAll("[data-accept]").forEach(function (button) {
      button.addEventListener("click", function () {
        var mode = button.getAttribute("data-accept");
        send({ type: "accept", edited: mode === "edited", rejected: mode === "rejected" });
        $("learning").innerHTML = "<div>recorded outcome: <b>" + mode + "</b></div>";
      });
    });
    $("question").addEventListener("keydown", function (event) {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { ask(); }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    wire();
    initSpeech();
    connect();
  });
})();
