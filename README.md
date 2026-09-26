# InterviewGenie

**A real-time, dependency-free interview copilot.**

InterviewGenie listens to a live interview, understands what the interviewer is
really asking, grounds an answer in a knowledge graph plus common sense, and
composes a personalised, emotionally appropriate response the candidate can
deliver — all in the same breath.

```bash
# zero dependencies: Python 3.9+ standard library only
python3 -m interviewgenie demo          # scripted end-to-end demonstration
python3 -m interviewgenie ask "How would you design a URL shortener?"
python3 -m interviewgenie serve         # live cockpit at http://localhost:8420
python3 -m interviewgenie benchmark     # evaluation metrics
python3 -m unittest discover -s tests -t .   # 289 tests
```

Log verbosity comes from the first of these that is set: the `--log-level` flag,
the `INTERVIEWGENIE_LOG_LEVEL` environment variable, then `system.log_level` in
`config/interviewgenie.json` (default `INFO`). Add `--log-level CRITICAL` to
silence the pipeline trace, or `DEBUG` to see every phase.

---



## Why this exists

An interview is a high-pressure, multi-turn, real-time conversation in which the
questions are ambiguous, the feedback is non-verbal, and the cost of a bad answer
is high. InterviewGenie treats that as an engineering problem and decomposes it
into the ten capabilities the product requires:

| # | Capability | Where it lives |
|---|------------|----------------|
| 1 | Advanced NLP | `interviewgenie/nlp/` |
| 2 | Real-time speech recognition, noise robust | `interviewgenie/asr/` |
| 3 | Contextual understanding (tone, intent, implied meaning) | `nlp/`, `context/` |
| 4 | Knowledge-graph integration (GraphDB / Neptune / Google Cloud KG) | `knowledge/` |
| 5 | Common sense & world knowledge | `knowledge/commonsense.py` |
| 6 | Emotional intelligence | `context/emotion.py`, `nlp/sentiment.py` |
| 7 | Personalisation | `personalization/` |
| 8 | Multi-turn dialogue management | `context/dialogue.py`, `context/memory.py` |
| 9 | Error handling & recovery | `errors.py`, `orchestrator.py` |
| 10 | Continuous learning | `learning/` |

Everything runs on a bare CPython interpreter. Optional accelerators (spaCy,
NLTK, Vosk, Google/Azure/IBM speech, an OpenAI-compatible LLM) are probed lazily
and degrade gracefully when absent — see
[Extending](#extending-with-optional-accelerators).

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   microphone ─────▶│  asr/  dsp → denoise → vad → decoder → text  │
                    └───────────────────────┬──────────────────────┘
                                            │ TranscriptChunk
                    ┌───────────────────────▼──────────────────────┐
                    │  nlp/  textnorm → tagger → intent → entities │
                    │        → sentiment/emotion → coref → keywords│
                    └───────────────────────┬──────────────────────┘
                                            │ Analysis
       ┌────────────────────────────────────┼────────────────────────────┐
       │                                    │                            │
┌──────▼───────┐   ┌─────────────────┐   ┌──▼──────────────┐   ┌─────────▼────────┐
│ knowledge/   │   │ context/        │   │ generation/     │   │ personalization/ │
│ graph        │──▶│ dialogue state  │──▶│ composer        │──▶│ profile + style  │
│ retrieval    │   │ memory          │   │ validator       │   │ adapters         │
│ commonsense  │   │ emotional IQ    │   │ llm (optional)  │   │                  │
└──────┬───────┘   └─────────────────┘   └────────┬────────┘   └─────────────────┘
       │                                          │
       │            ┌─────────────────────────────▼──────────────┐
       └───────────▶│ evaluation/  metrics → benchmark → report  │
                    └─────────────────────────────┬──────────────┘
                                                  │ reward
                    ┌─────────────────────────────▼──────────────┐
                    │ learning/  feedback → online updates → KG  │
                    └────────────────────────────────────────────┘
```

### The turn loop

`InterviewGenie.answer()` implements the workflow phases from the requirements
end to end. Every stage is guarded, event-emitting and measurable:

| Phase | What happens | Failure mode → recovery |
|-------|--------------|-------------------------|
| **Pre-interview** (`start`) | load config + KG, init NLP/ASR/context/EI, restore profile | config error → `ESCALATE` |
| **Interview** (`ingest` → `handle_transcript`) | audio in, ASR out, transcripts streamed | low ASR confidence → ask for clarification |
| **Analysis** | normalise, tag, classify intent + topic, extract + link entities, sentiment, emotion, coref | parse failure → `CLARIFY` |
| **Retrieval** | link entities to the KG, expand weighted paths, rank evidence | no evidence → `DEGRADE` |
| **Dialogue** | update topic stack, slots, emotion, rapport; pick a strategy | contradiction → `REPHRASE` |
| **Generation** | compose a grounded, register-matched answer | no plan → `ESCALATE` |
| **Validation** | length, repetition, duplication, unsupported claims | warnings → auto-repair |
| **Post-response** | score the answer, update memory + KG weights, feed the bandit | – |

---

## 1 · Advanced NLP (`nlp/`)

| Module | What it does |
|--------|--------------|
| `textnorm.py` | NFKC normalisation, contraction expansion, abbreviation-aware sentence splitting, a regex tokeniser with character offsets, n-grams, Jaccard, Levenshtein |
| `stemmer.py` | full Porter stemmer (steps 1a–5b) plus a rule-based lemmatiser with an irregular-form dictionary |
| `lexicon.py` | POS tag inventory, closed-class sets, verb-form tables; `build_pos_lexicon()` generates the unigram lexicon |
| `tagger.py` | transformation-based (Brill-lite) tagger: lexicon-seeded, 13 contextual rewrite rules, morphology fallback, optional spaCy/NLTK delegation |
| `intent.py` | multinomial Naïve Bayes over word uni/bi-grams with add-α smoothing, plus ordered high-precision regex rules evaluated **before** the statistical model. Two-axis taxonomy: 16 coarse intents × 16 fine topics |
| `sentiment.py` | VADER-style rule engine (valence lexicon, negation window, intensifier/dampener scaling, punctuation and caps emphasis) → Plutchik-8 emotion scores → empathy tone |
| `embeddings.py` | TF-IDF index, LSA via power iteration on a sparsely-accumulated Gram matrix, cosine / sparse cosine, BM25, TextRank keyword extraction |
| `entities.py` | rule + gazetteer NER (dates, durations, versions, emails, URLs, tech terms, capitalised runs) and fuzzy entity linking with an alias index |
| `coref.py` | Hobbs-style pronoun resolution with gender/number agreement and role scoring |
| `pipeline.py` | `NLPPipeline.analyze()` composes all of the above into an `Analysis` |

**Measured accuracy** on the labelled holdout — models fitted on the training
split only, scored on unseen rows, averaged over seeds 1–5
(`config/intent_dataset.json`, 290 examples, 16 coarse intents + 16 fine topics):

| Axis | Rules + model | Model only |
|------|---------------|------------|
| Coarse intent | **0.923** | 0.578 |
| Fine topic | **0.638** | 0.534 |

Reproduce with `python3 -m interviewgenie evaluate`.

The taxonomy is deliberately **two-axis**. An earlier single flat taxonomy of 23
labels stalled at ~0.50 holdout accuracy because *behavioural*, *conflict*,
*leadership*, *teamwork*, *strength* and *weakness* all share the same surface
form ("tell me about a time when…"). Separating "what kind of question is this?"
from "what is it about?" is what made the problem learnable.

---

## 2 · Real-time speech recognition (`asr/`)

The front-end is a textbook MFCC pipeline implemented from scratch:

```
PCM → pre-emphasis (0.97) → 25 ms frames / 10 ms hop → Hamming window
    → 512-point FFT → power spectrum → 26 mel filters → log → DCT → 13 MFCC
    → delta + delta-delta → cepstral mean/variance normalisation
```

| Module | What it does |
|--------|--------------|
| `dsp.py` | radix-2 FFT/IFFT (verified against the direct DFT), mel filterbank, DCT, WAV/PCM I/O, resampling, framing, frame descriptors |
| `denoise.py` | minimum-statistics noise estimation, spectral subtraction with a spectral floor, Wiener gain, synthetic noise generators (white/pink/babble/hum) |
| `vad.py` | hysteresis VAD combining energy, zero-crossing rate and spectral flatness, with an asymmetric (fast-down / slow-up) noise-floor tracker |
| `templates.py` | acoustic template bank with averaging, persistence and enrollment |
| `decoder.py` | DTW decoder with a Sakoe-Chiba band; confidence from the best/second-best margin |
| `synth.py` | compact cascade-formant synthesiser + rule-based G2P, used to bootstrap templates and to speak prompts |
| `local.py` | `LocalStreamingASR`: streaming partial + final hypotheses, SNR-aware confidence, runtime speaker calibration |
| `cloud.py` | Google / Azure / IBM adapters over `urllib`, with automatic local fallback |
| `base.py`, `mock.py` | provider contract and a deterministic test double |

### Measured noise robustness

`python3 scripts/noise_robustness.py` (or `python3 -m interviewgenie noise`):

```
noise        snr     before     after      gain
white       20dB      25.60     28.52    + 2.92
white       10dB      18.12     20.82    + 2.70
white        5dB      14.55     17.36    + 2.81
white        0dB      11.84     14.53    + 2.69
pink        20dB      23.56     26.15    + 2.59
pink        10dB      17.24     19.80    + 2.56
pink         5dB      14.29     16.59    + 2.30
pink         0dB      12.58     15.02    + 2.44
babble      20dB      27.04     29.25    + 2.21
babble      10dB      20.35     22.76    + 2.41
babble       5dB      17.09     19.66    + 2.57
babble       0dB      14.67     17.18    + 2.51
hum         20dB      22.82     25.05    + 2.23
hum         10dB      13.73     19.20    + 5.47
hum          5dB       9.48     14.77    + 5.29
hum          0dB       5.62     12.15    + 6.53
```

These are reproducible: the synthesiser's excitation source used to be seeded
with Python's `hash()`, which is randomised per process, so every run produced
different numbers. It now uses a deterministic FNV-1a mix, so the table above is
exactly what you will see.

The VAD segments speech correctly down to **0 dB SNR** across white, pink,
babble and hum noise — the asymmetric noise-floor tracker is what makes that
work (a symmetric smoother climbs into the speech and never recovers).

Recognition of a known phrase degrades gracefully: clean confidence 0.58 → 0.41
at 0 dB babble, which crosses the low-confidence threshold and triggers the
clarification path rather than a wrong answer.

---

## 3 · Contextual understanding

* **Intent + topic** — the two-axis classifier plus WH-type detection
  (`how` / `what` / `why` / `tell-me-about` …) and a clarification flag.
* **Tone** — smoothed Plutchik-8 emotion trajectory, sentiment trend, and an
  interviewer *rapport* classifier (`warming`, `cooling`, `probing`, `rushing`,
  `supportive`) that changes how the answer is framed.
* **Implied meaning** — `knowledge/commonsense.py` carries ~15 conceptual rules
  ("any distributed design must say what happens when a node fails", "migrations
  need a rollback plan", "conflict answers should show empathy *and* a path to a
  decision") plus a category ontology that lets the system reason about entities
  it has never seen.
* **Coreference** — "it", "that project", "the same approach" resolve against
  the running turn history.

---

## 4 · Knowledge-graph integration

`knowledge/graph.py` is a labelled, weighted, directed property graph with
secondary indices, BFS and uniform-cost traversal, a declarative pattern-query
language, JSON/triple serialisation, and a `RemoteGraphBackend` that speaks
SPARQL to **GraphDB** or **Amazon Neptune** over REST.

The seed graph (`knowledge/seed.py`) contains **195 nodes / 293 edges / 246
aliases** across skills, concepts, roles, companies, competencies and metrics,
with typed relations (`requires`, `used_by`, `related_to`, `alternative_to`,
`contrasts_with`, `prerequisite_of`, `implements`, `measured_by`,
`demonstrated_in`, `uses`). Edge weights are priors that the learning loop
adjusts at runtime.

`knowledge/retrieval.py` turns a question into ranked evidence: link entities →
expand weighted neighbourhoods → rank paths by edge weight, hop count and
question-term overlap → add prose evidence from a TF-IDF/LSA corpus index.

---

## 5 · Common sense & world knowledge

`CommonsenseEngine` provides the general knowledge an interviewer expects:

* **forward chaining** (`implications`) — "if you mention a cache, you must
  address invalidation and staleness";
* **a plausibility score** that penalises answers which violate a rule;
* **contradiction detection** across the candidate's own earlier answers;
* **an ontology** so `is_a("database", "thing")` holds transitively.

---

## 6 · Emotional intelligence

`context/emotion.py` maps detected emotion → response register → concrete
linguistic moves:

| Emotion | Register | Opening move |
|---------|----------|--------------|
| fear | reassuring | "That is a fair thing to ask." |
| anger | calm | "That is a legitimate frustration." |
| joy | warm | "Great question." |
| sadness | supportive | "I appreciate you asking." |
| surprise | engaged | "Interesting angle." |

Rapport signals add moves like "lead with the evidence, not the narrative"
(probing), "shorten and land the headline in the first sentence" (cooling) and
"be candid about what went wrong" (supportive).

---

## 7 · Personalisation

`IntervieweeProfile` is an interpretable preference model: static facts (name,
target role, skills, strengths, weaknesses, a **story bank** with STAR
structure), seven style dimensions tracked with exponential moving averages,
topic affinities and evidence weights. `StyleAdapter` converts the profile into
concrete generator parameters (word budget, register, sentence style).

Both explicit feedback ("too long", "more numbers", "too technical") and
implicit feedback (accepted / edited / rejected) move the style vector.

---

## 8 · Multi-turn dialogue management

`context/dialogue.py` owns the topic stack (exponential decay), a slot store, an
emotion smoother, an interviewer-style model, coreference chains, and a
**contextual bandit** over seven response strategies (`direct`, `structured`,
`example_first`, `clarify`, `bridge`, `question_back`, `hedge`) with an
ε-greedy exploration bonus.

`context/memory.py` provides three tiers: working (last 8 turns), episodic
(salience-weighted retrieval of earlier exchanges) and semantic (durable facts).

---

## 9 · Error handling & recovery

`errors.py` defines 22 typed error codes, a `RecoveryEngine` that maps each code
to a bounded recovery action (`retry`, `rephrase`, `clarify`,
`fallback_provider`, `degrade`, `escalate`, `ignore`, `reset_context`), and a
`guard` decorator that converts unexpected exceptions into typed errors.

Every failure is recorded, counted, and reported in the session report's
`recovery` block — which is also where the **Error Rate** evaluation metric
comes from.

---

## 10 · Continuous learning

Three loops run during and after an interview:

* **feedback** — explicit and implicit signals are normalised to a reward and
  applied to the style model, the dialogue bandit and the KG edge weights;
* **online** — the intent classifier takes `partial_fit`-style corrections and
  the KG weights are reinforced/weakened by the evidence that supported (or
  failed to support) an answer;
* **metrics** — `LearningMetrics` keeps a rolling window of scorecards and
  reports whether the system is actually improving.

---

## Evaluation metrics

`evaluation/` implements the five required metrics and scores every response:

| Metric | How it is computed |
|--------|--------------------|
| **Response Accuracy** | reference-free: 0.6 × commonsense plausibility + 0.4 × groundedness; with a reference: token-F1 + ROUGE-L + LSA cosine |
| **Response Relevance** | overlap between question and answer terms, plus KG entity hits |
| **Engagement** | empathy markers + concrete-example markers + questions back + numbers, minus verbosity penalty |
| **Personalisation** | profile facts present + style-fit of length and hedging |
| **Error Rate** | empty answers, commonsense violations, repairs, low confidence, clarifications |

The composite `Scorecard.overall()` weights accuracy .25, relevance .25,
engagement .15, groundedness .15, fluency .10, personalisation .10, minus
0.10 × error rate.

```
$ python3 -m interviewgenie benchmark
cases: 29
  accuracy         0.2929
  relevance        0.2338
  engagement       0.4648
  personalization  0.2253
  groundedness     0.4506
  fluency          0.9638
  error_rate       0.2328
  overall          0.3646
```

> **Read this honestly.** Those numbers are for the built-in *composer* backend,
> which composes answers from templates + the KG + the candidate's story bank
> with no language model involved. Fluency is high (0.97) and groundedness is
> moderate, but accuracy against hand-written reference answers is low because a
> template composer cannot reproduce specific facts it was never told.
> Configure an OpenAI-compatible endpoint (`generation.llm.*` in
> `config/interviewgenie.json`) and the same pipeline drives a real model with
> the retrieval-augmented prompt, falling back to the composer if the model is
> unavailable.

---

## The live cockpit

```bash
python3 -m interviewgenie serve --port 8420
```

Open `http://localhost:8420`. The cockpit is a single-page app served by a
**pure-stdlib asyncio HTTP server with a hand-written RFC 6455 WebSocket
implementation** — no Node, no framework, no build step.

It shows the suggested answer with its scorecard, the interviewer understanding
panel (intent, topic, confidence, emotion, sentiment, rapport, entities), the
retrieved KG evidence, a live-speech panel that uses the browser's speech
recognition API when available, and the feedback/learning controls.

### The live overlay (`/live`)

This is the screen you look at *during* an interview. Open
`http://localhost:8420/live` — ideally in a separate window on a second monitor,
which is the honest way to use a copilot without hiding anything from the
interviewer.

```
 ┌─ connected · tab audio · auto-answer on · 340ms to first words ─────────┐
 │ interviewer                                                             │
 │ "tell me about a time you had to push back on a deadline…"              │
 ├─────────────────────────────────────────────────────────────────────────┤
 │                                                                         │
 │  I owned the orders migration. The deadline was fixed by a partner      │
 │  launch, so I cut scope to the read path first and shipped that, then   │
 │  landed the write path a week later. The launch went out on time and    │
 │  we had no rollback.                                                    │
 │                                                                         │
 ├─ structured · 412ms · score 0.62   [↻ Regenerate] [✓ Used] [✗ Missed] ─┤
 └─────────────────────────────────────────────────────────────────────────┘
```

What it does differently:

- **Answers automatically.** The client watches the transcript, detects when a
  question is complete (a question mark, or a question word followed by a
  pause), and fires immediately — no "Start answering" click between the
  question and your answer. That dead beat is the single most criticised
  weakness of the tools this competes with.
- **Streams the answer word by word**, so you can start reading the first
  sentence while the rest is still being written. Time-to-first-words is shown
  in the status bar.
- **Listens to the meeting, not the room.** `🎧 Audio source → Tab / screen
  audio` captures the interviewer's voice straight out of Zoom, Meet or Teams
  via `getDisplayMedia`, which is far more accurate than a microphone in a
  noisy room. The browser asks you to pick a tab and tick *"Share tab audio"*.
- **Recovers from anything.** No speech API? Type the question. No model key?
  The offline composer still answers. Socket drops? It reconnects.
- **Keyboard first.** `Space` regenerate · `A` used it · `R` missed ·
  `Esc` clear · `T` type — you never need the mouse mid-answer.

### Deployment

```bash
cp .env.example .env        # add one LLM key and, optionally, a STT key
docker compose up --build   # http://localhost:8420
```

There is no build step and no dependency to install: the image is
`python:3.11-slim` plus this repository. Configuration is entirely
environment-variable driven (`INTERVIEWGENIE_` prefix, `__` for nesting), so
the same image runs in dev and production. See `.env.example` for every knob.

```bash
# or without Docker
python3 -m interviewgenie serve --host 0.0.0.0 --port 8420
```

---

## Extending with optional accelerators

| Capability | How to enable | Fallback |
|------------|---------------|----------|
| spaCy / NLTK tagging | `pip install spacy` then `nlp.tagger_backend = "spacy"` | built-in Brill-lite tagger |
| Vosk offline ASR | `pip install vosk`, set `asr.model_path` | MFCC + DTW template decoder |
| Google / Azure / IBM STT | `asr.provider = "google"` + credentials | local engine (automatic) |
| LLM generation | `generation.backend = "auto"` (default) + any `*_API_KEY` | retrieval-augmented composer |
| Deepgram streaming STT | `DEEPGRAM_API_KEY` + `asr.provider = "deepgram"` | browser speech API, then local engine |
| GraphDB / Neptune | `knowledge.backend = "graphdb"` + `knowledge.endpoint` | in-memory property graph |

Accelerators are probed with `importlib` and `try/except`, so a missing package
or unreachable endpoint is a log line, never a crash.

---

## Repository layout

```
interviewgenie/
  __init__.py __main__.py cli.py factory.py orchestrator.py
  config.py errors.py events.py logging.py types.py
  nlp/       textnorm stemmer lexicon tagger intent sentiment
             embeddings entities coref pipeline
  asr/       dsp denoise vad templates decoder synth local cloud base mock
  knowledge/ graph retrieval commonsense seed
  context/   dialogue memory emotion
  generation/ composer templates llm
  personalization/ profile
  learning/  online
  evaluation/ metrics dataset
  api/       server routes ws protocol
config/      interviewgenie.json mock_asr.json cloud_asr.example.json
             intent_dataset.json
scripts/     demo.py benchmark.py noise_robustness.py
tests/       test_nlp test_asr test_knowledge test_generation
             test_api test_infra base.py
web/         index.html app.js style.css
docs/        ARCHITECTURE.md EVALUATION.md
```

---

## Design decisions worth knowing

1. **Standard library only.** No third-party package is required to run, test or
   serve the system. This is a deliberate constraint, not a limitation of
   ambition: every algorithm (FFT, MFCC, Porter stemmer, Naïve Bayes, LSA, DTW,
   Hobbs coref, TF-IDF, TextRank, WebSocket) is implemented in the repository.
2. **Rules before statistics, most specific first.** The intent classifier
   evaluates ordered high-precision regex rules before the Naïve Bayes model;
   rule *ordering* was the single biggest source of error until it was made
   explicit.
3. **Two-axis taxonomy.** Coarse intent and fine topic are separate axes,
   because they are separate questions.
4. **Asymmetric noise tracking.** Fast-down / slow-up minimum statistics is what
   stops a VAD from locking onto speech.
5. **Graceful degradation everywhere.** Every optional dependency, every remote
   call and every stage failure has a defined fallback, and the fallback is
   reported rather than hidden.
6. **Metrics are computed, not asserted.** Groundedness, plausibility, fluency
   and concision are real measurements over the generated text, and the
   benchmark runs the same scoring over a labelled dataset so numbers are
   comparable between runs.

## License

MIT.
