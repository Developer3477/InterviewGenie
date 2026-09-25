# Architecture

This document describes how InterviewGenie is put together and why each piece is
shaped the way it is. It assumes you have read the [README](../README.md).

---

## 1 · Layering

The system is a strict pipeline with a feedback edge. Data flows one way during
a turn; learning flows back between turns.

```
Layer          Owns                                    Talks to
-------------  -------------------------------------  --------------------
speech         audio buffers, features, hypotheses   orchestrator
language       tokens, tags, intents, entities        speech, knowledge
knowledge      the world model and evidence           language, generation
context        dialogue state and emotional state     language, knowledge
generation     the answer                             all of the above
evaluation     the scorecard                           generation
learning       the reward and the updates             evaluation, knowledge
```

The orchestrator is the only module that knows about all of the others. Each
subsystem exposes a small, testable surface:

| Subsystem | Entry point | Returns |
|-----------|-------------|---------|
| ASR | `ASRProvider.accept_audio()` / `.poll()` | `TranscriptChunk` |
| NLP | `NLPPipeline.analyze(text)` | `Analysis` |
| Knowledge | `KnowledgeRetriever.retrieve(analysis)` | `RetrievalResult` |
| Dialogue | `DialogueManager.observe(turn)` / `.choose_strategy(analysis)` | events / strategy |
| Generation | `ResponseComposer.compose(...)` | `Response` |
| Evaluation | `Evaluator.score(analysis, response)` | `Scorecard` |
| Learning | `OnlineLearner.learn(feedback, turn)` | applied changes |

---

## 2 · The turn loop in detail

`InterviewGenie.answer(turn)` is the whole product in one method. Annotated:

```python
# 1. ANALYSIS -------------------------------------------------------------
analysis = self.nlp.analyze(turn.text)          # tokens, tags, intent, topic,
                                                # entities, sentiment, emotion
# 2. RETRIEVAL ------------------------------------------------------------
retrieval = self.retriever.retrieve(analysis)   # linked entities + ranked
                                                # KG paths + prose evidence
# 3. DIALOGUE -------------------------------------------------------------
events = self.dialogue.observe(turn)            # topic stack, slots, emotion,
                                                # rapport, contradictions
strategy = self.dialogue.choose_strategy(analysis)
# 4. GENERATION -----------------------------------------------------------
response = self._generate(turn, analysis, retrieval, strategy)
# 5. VALIDATION -----------------------------------------------------------
validation = self.validator.validate(response, history=...)
# 6. EVALUATION -----------------------------------------------------------
scorecard = get_evaluator().score(analysis, response)
# 7. MEMORY + LEARNING ----------------------------------------------------
self.dialogue.record_answer(turn, response)     # must precede observe():
self.memory.observe(turn)                       # otherwise there is nothing
                                                # worth remembering yet
```

The ordering constraint in step 7 is a real invariant: the episodic memory only
stores turns that already carry a response.

---

## 3 · NLP internals

### 3.1 Tokenisation and normalisation

`normalize()` applies NFKC, expands ~60 contractions, collapses whitespace and
normalises dashes/quotes. `split_sentences()` is abbreviation-aware (`Dr.`,
`Inc.`, `e.g.` do not end a sentence). The tokeniser is a single regex with a
character-offset view (`spans()`), so entity spans can be mapped back onto the
original text — which is what makes the web cockpit able to highlight them.

### 3.2 Tagging

`PosTagger` is transformation-based:

1. seed every token from `POS_LEXICON` (built by `build_pos_lexicon()` from the
   closed-class sets and verb-form tables in `lexicon.py`);
2. unknown tokens get an `_initial_guess` from suffix and shape rules;
3. thirteen contextual rewrite rules run in order (punctuation short-circuit,
   protected closed-class tags, capitalisation after a noun, `be`-form
   distribution, `wh`-word handling, …).

Because the seed is a lexicon and the rules are ordered, the tagger is
deterministic and debuggable — every tag can be traced to a rule number.

### 3.3 Intent classification

Two independent classifiers, both multinomial Naïve Bayes with add-α smoothing
over word uni/bi-grams:

```
classifier        : text -> coarse intent   (16 classes)
topic_classifier  : text -> fine topic      (16 classes)
```

Both are *shadowed* by ordered regex rules:

```python
def classify(self, text):
    label, confidence, scores = self.rule_match(text)      # rules first
    if label is None:
        label, confidence, scores = self.classifier.predict(text)
    return label, confidence, scores
```

The `_RULES` list is ordered most-specific-first
(`clarification/meta` → `self_assessment` → `people` → `motivation` → … →
`knowledge`) and the **first match wins**. That ordering is load-bearing: a
generic `why are …` pattern placed before the motivation rule misclassifies
"why are you interested in this role?" as troubleshooting.

Topic classification uses the same shape (`_TOPIC_RULES`), falling back to the
statistical model.

### 3.4 Sentiment and emotion

`SentimentAnalyzer` is a VADER-style rule engine:

```
valence = Σ lexicon weights
        × intensifier/dampener scaling
        × negation-window flip
        + punctuation emphasis
        + caps emphasis
```

Emotion scores are Plutchik-8 activations accumulated from `EMOTION_LEXICON`,
then normalised. `EmpathyTone` maps the dominant emotion to a response register.

### 3.5 Embeddings

`TfidfIndex` supports sparse cosine, BM25 and a dense LSA view. The LSA basis is
computed by power iteration on the document–document Gram matrix, which is
accumulated sparsely (one outer product per non-zero term) so the cost is
`O(nnz · n_docs)` rather than `O(n_docs² · n_terms)`. Basis vectors are mapped
back into term space, which is what lets an arbitrary query string be projected
without refitting.

---

## 4 · Speech internals

### 4.1 Why a template matcher

No pre-trained acoustic model can be downloaded in the target environment
(`alphacephei.com`, `raw.githubusercontent.com` and `huggingface.co` are all
unreachable), so the offline engine is an isolated-phrase DTW matcher:

* templates are **seeded** by the bundled formant synthesiser so the recogniser
  works out of the box, and
* **re-calibrated at runtime** by speaking each phrase once, which adapts the
  templates to the real speaker.

Production deployments can point `asr.model_path` at a Vosk model directory or
set `asr.provider` to `google` / `azure` / `ibm`; the orchestrator is unchanged.

### 4.2 The noise floor

The VAD's noise estimator is the piece that decides whether the whole front-end
works. It must not climb into speech:

```
if energy < floor:   floor ← attack·energy + (1−attack)·floor   # fast down
else:                floor ← floor·(1 + release)                # very slow up
```

With symmetric smoothing the floor rises during speech and the detector
silences itself. With asymmetric smoothing the detector stays locked on to
speech from +20 dB all the way down to 0 dB SNR.

### 4.3 Confidence

The decoder's raw confidence is `(1 − normalised_cost) × template_confidence ×
(1 + margin × margin_weight)`, where the margin is the gap to the runner-up. The
orchestrator then multiplies by an SNR factor and compares against
`asr.min_confidence`; below the threshold the system asks for clarification
instead of guessing.

---

## 5 · Knowledge internals

### 5.1 Graph

`PropertyGraph` keeps four indices: nodes by id, outgoing edges, incoming edges,
and nodes by label, plus an alias map. Traversal is BFS for bounded expansion
and uniform-cost (Dijkstra) for shortest paths. Edges returned by a
neighbourhood lookup are re-oriented by `_as_traversed()` so every `Path` reads
left-to-right regardless of the stored direction.

### 5.2 Retrieval

```
Analysis ──▶ link entities ──▶ expand neighbourhoods ──▶ rank paths ──▶ Evidence
                 │                                                        ▲
                 └── keyword/BM25/LSA fallback ◀───────────────────────────┘
```

Ranking combines edge weight (geometric mean along the path), a hop penalty, the
overlap between the path text and the question's keywords, and a bonus for paths
that end on a well-described node.

Entity linking is deliberately strict: an exact alias match is required, and the
text-search fallback demands either the full node name in the question or ≥ 75 %
word overlap. A single shared word is *not* enough to claim the question is about
an entity — that mistake produced nonsense subjects like "Solutions Architect"
for "how would you architect a news feed".

When nothing links, the retriever falls back to a per-intent concept cluster
(`INTENT_CONCEPTS`), surfaced as *inferred* entities, so a design question with
no named technology still gets System design, Load balancing, Caching and
Sharding to reason about.

### 5.3 Common sense

`CommonsenseEngine.check()` scores an answer against conceptual rules. A rule
fires when its trigger terms appear in the question *or* the answer, and is
satisfied when at least one of its `requires` terms also appears. Violations
subtract a per-rule penalty from a plausibility score in `[0, 1]`.

Rules are also *meta-aware*: implications that are coaching notes about the
candidate ("the candidate's personal contribution should be identifiable") are
filtered out before they can leak into a generated answer.

---

## 6 · Generation internals

Generation is a three-stage NLG pipeline:

```
planning      ResponsePlanner.plan()          -> List[PlanStep]
sentence      (per-kind slot filling)         -> propositions
realisation   SurfaceRealiser.realise()       -> text
```

Content comes from three sources, in priority order:

1. the interviewee profile (real strengths, skills, STAR stories),
2. the knowledge graph (definitions, relationships, alternatives),
3. common-sense implications (what a competent answer must mention).

Each intent has an `AnswerSkeleton` with headline / support / trade-off patterns
whose slots are filled from the analysis, the retrieval result and the profile.
`_fill()` refuses to render a pattern when any referenced slot is empty, and
`_dedupe_sentences()` drops repeats — both were needed because an empty slot
produced sentences like "The pieces around it are, and I would keep them…".

Scoring is computed on the *generated text*, not on the plan:

* `groundedness` — fraction of the answer's sentences backed by retrieved
  evidence (per-sentence, so unused evidence does not count against the answer);
* `accuracy` — plausibility + groundedness (or F1/ROUGE-L/LSA against a
  reference when one exists);
* `engagement` / `empathy` — marker-based, register-aware;
* `personalization` — profile facts present plus style-fit;
* `fluency` — sentence-length variance and repetition;
* `concision` — word budget.

---

## 7 · Dialogue internals

`DialogueState` is a plain dataclass so it can be serialised and inspected. The
topic stack decays multiplicatively each turn and is boosted by the new
utterance's keywords and its intent topic; a change of subject is recorded as an
event, **not** raised as an error — topic shifts are normal in an interview.

The strategy bandit is ε-greedy over a prior-ranked candidate list:

```
expected_reward(strategy) = mean_reward + 0.25 / sqrt(pulls)
```

The optimism bonus decays as evidence accumulates, so exploration happens early
and exploitation later.

---

## 8 · Error handling internals

`RecoveryEngine` maps each of 22 error codes to a recovery action and records
every attempt. `execute()` increments the plan's attempt counter and books the
outcome as `recovered` or `escalated` once `max_attempts` is reached, so retries
are bounded and the error rate is a real measurement rather than a guess.

`guard` converts arbitrary exceptions into `InterviewGenieError` with
`ErrorCode.INTERNAL`, preserving the original as `__cause__`.

---

## 9 · API internals

The server is `asyncio.start_server` with a hand-written HTTP/1.1 response
builder and a hand-written RFC 6455 WebSocket implementation (`api/ws.py`):
the `Sec-WebSocket-Accept` handshake, masked client frames, fragmented-message
reassembly, close and ping/pong.

Because the request line and headers are consumed by the HTTP dispatcher before
the upgrade is detected, they are replayed into a buffered `StreamReader` that
the handshake consumes before continuing on the live socket — with a pump task
that forwards the remaining bytes.

---

## 10 · Testing strategy

211 `unittest` tests, grouped by layer:

| Module | Focus |
|--------|-------|
| `test_nlp.py` | normalisation idempotence, tokenizer offsets, Porter stems, POS invariants, intent accuracy, sentiment polarity, emotion normalisation, TF-IDF ranking, NER patterns, coref determinism, leak-free holdout, fallback-sample labelling |
| `test_asr.py` | FFT vs the direct DFT, IFFT round-trip, filterbank normalisation, MFCC determinism, CMVN, noise-suppression SNR gain across four noise types, VAD under 0 dB noise, DTW alignment, template persistence, streaming recognition, calibration, reproducibility of the noise report, WebSocket short reads |
| `test_knowledge.py` | graph CRUD, duplicate-edge merging, alias lookup, shortest paths, pattern queries, serialisation round-trips, triple export, retrieval strictness, ontology transitivity, rule violations, contradiction detection |
| `test_generation.py` | every intent produces an answer, word budget, entity grounding, story selection by topic, scorecard bounds, validator repair, feedback normalisation, style clamping, bandit learning, metric trends |
| `test_api.py` | protocol encode/decode, router routes, path traversal, static assets, WebSocket frame codec and handshake, a live HTTP round-trip, a live WebSocket round-trip, full orchestrator lifecycle, error recovery |
| `test_infra.py` | config layering and validation, context logging and level gating, explicit-log-level precedence over the config file, CLI argument defaults (including the `listen` phrase fallback), event bus wildcards and handler isolation, error hierarchy and recovery bounds, type serialisation |

The suite shares a single `InterviewGenie` instance across classes (built once
in `shared_genie()`), which keeps the whole run under a minute.
