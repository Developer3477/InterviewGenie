# Evaluation

How InterviewGenie is measured, what the numbers mean, and how to reproduce
them.

---

## 1 · The five required metrics

| Metric | Definition | Implementation |
|--------|------------|----------------|
| **Response Accuracy** | Is the answer factually consistent with the knowledge graph, free of commonsense violations, and (when a reference exists) close to it? | `Evaluator._accuracy` |
| **Response Relevance** | Does it answer the question that was actually asked? | `Evaluator._relevance` + `ResponseComposer._relevance` |
| **Engagement** | Is it concrete, specific and interesting rather than generic? | `EmotionalIntelligence.engagement_score` |
| **Personalisation** | Does it use *this* candidate's material and match their style? | `IntervieweeProfile.personalisation_score` |
| **Error Rate** | How often does it fail, need repair, or produce a low-confidence answer? | `Evaluator._error_rate` + `RecoveryEngine.error_rate` |

### Composite score

```python
overall = 0.25·accuracy + 0.25·relevance + 0.15·engagement
        + 0.10·personalization + 0.15·groundedness + 0.10·fluency
        − 0.10·error_rate
```

---

## 2 · Metric definitions in detail

### Response Accuracy

*Reference-free* (the normal case, since a live interview has no gold answer):

```
accuracy = 0.6 · plausibility + 0.4 · groundedness
```

`plausibility` comes from `CommonsenseEngine.check()` — the fraction of the
conceptual rules that the answer satisfies. `groundedness` is the fraction of the
answer's sentences that share content words with retrieved evidence.

*With a reference* (the benchmark):

```
accuracy = 0.5 · token-F1 + 0.3 · ROUGE-L + 0.2 · LSA cosine
```

### Response Relevance

```
relevance = min(1, 0.6 · |question_terms ∩ answer_terms| / |question_terms|
                    + 0.1 · min(3, KG entity hits))
```

Content words longer than four characters only, so function words do not
dominate.

### Engagement

```
score = min(0.5, 0.18 · empathy_markers)
      + min(0.5, 0.12 · engagement_markers)
      + 0.15 · has_question_back
      + 0.10 · contains_a_number
      − 0.10 · over_130_words
```

Empathy markers include *appreciate, understand, fair, that makes sense, good
question*. Engagement markers include *for example, specifically, in practice,
we shipped, the result, the trade-off*.

### Personalisation

```
score = 0.15·name  + 0.15·target_role  + 0.10·target_company
      + 0.08·skill_hits  + 0.08·strength_hits  + 0.06·goal_hits
      + 0.20·length_fit  + 0.10·hedging_fit
```

`length_fit` compares the answer's word count to the length implied by the
candidate's verbosity setting; `hedging_fit` compares hedge density to their
hedging setting.

### Error Rate

```
errors = 1.0 · empty_answer
       + 0.25 · min(4, commonsense_violations)
       + 0.50 · repaired
       + 0.50 · confidence < 0.35
       + 0.25 · requires_clarification
error_rate = min(1.0, errors)
```

The session-level error rate additionally counts unrecovered
`RecoveryEngine` events.

---

## 3 · Reproducing the numbers

```bash
# the labelled benchmark (29 cases with reference answers)
python3 -m interviewgenie benchmark
python3 -m interviewgenie benchmark --failures --json

# the same thing from a script
python3 scripts/benchmark.py --limit 10

# NLP-layer accuracy (intent + topic, holdout)
python3 -m interviewgenie evaluate

# noise-suppression gain
python3 -m interviewgenie noise
python3 scripts/noise_robustness.py

# the ASR pipeline on synthetic audio, clean and noisy
python3 -m interviewgenie listen
```

### Benchmark dataset

`interviewgenie/evaluation/dataset.py` ships 29 labelled cases covering all 16
coarse intents, each with a hand-written reference answer, expected intent and
topic, key terms and must-mention terms. `load_cases()` falls back to the
built-in set if the configured JSON file is missing, so the benchmark always
runs.

### Intent dataset

`config/intent_dataset.json` holds **290** labelled interviewer utterances
across 16 coarse intents and 16 fine topics. An 18-sample embedded fallback set
is substituted only when the file cannot be read; it is *not* appended to a
dataset that loaded, so the training set is exactly those 290 rows.

`IntentClassifier.evaluate()` fits fresh models on 75 % and scores the held-out
25 % — those are the **statistical-model-only** numbers:

```
{'intent_accuracy': 0.589, 'topic_accuracy': 0.5753, 'support': 73.0, 'train_size': 217.0}
```

`IntentClassifier.evaluate_holdout()` goes further and scores the **deployed**
rules+model path on the same split, with the models fitted on the training rows
only, averaged over seeds 1–5:

```
rules + model    intent=0.923  topic=0.638
model only       intent=0.578  topic=0.534
```

> **Why these are lower than an earlier revision of this document.** The first
> version of `evaluate` reported a perfect 1.000 by building the holdout
> classifier and assigning `.examples` without ever re-fitting the models, so
> the statistical models stayed trained on the full dataset — including the
> rows being scored — and the ordered rules were applied on top. That was
> leakage, not skill. The numbers above are leak-free and are what the CLI
> prints.

---

## 4 · Current results

### NLP layer

| Metric | Value |
|--------|-------|
| Intent accuracy (rules + model, leak-free holdout, seeds 1–5) | **0.923** |
| Topic accuracy (rules + model, leak-free holdout, seeds 1–5) | **0.638** |
| Intent accuracy (statistical model only) | 0.578 |
| Topic accuracy (statistical model only) | 0.534 |
| Training examples | 290 (16 intents × 16 topics) |

### ASR layer

| Measurement | Value |
|-------------|-------|
| SNR gain, white noise @ 0 dB | +2.69 dB |
| SNR gain, white noise @ 20 dB | +2.92 dB |
| SNR gain, hum @ 0 dB | +6.53 dB |
| SNR gain, hum @ 10 dB | +5.47 dB |
| VAD segmentation @ 0 dB SNR | correct (1 segment, ±0.15 s) |
| Phrase recognition, clean | correct, confidence 0.58 |
| Phrase recognition, 0 dB babble | correct, confidence 0.41 |
| Streaming partials | emitted every ~240 ms of speech |
| Real-time factor | ≈ 1.2× on CPython 3.11 |

### Generation layer (composer backend, 29 benchmark cases)

| Metric | Value |
|--------|-------|
| Accuracy | 0.293 |
| Relevance | 0.234 |
| Engagement | 0.465 |
| Personalisation | 0.225 |
| Groundedness | 0.451 |
| Fluency | 0.964 |
| Error rate | 0.233 |
| **Overall** | **0.365** |

### Generation layer (session report, 5-question demo)

| Metric | Value |
|--------|-------|
| Accuracy | 0.620 |
| Relevance | 0.240 |
| Engagement | 0.392 |
| Personalisation | 0.375 |
| Groundedness | 0.580 |
| Fluency | 0.950 |
| Error rate | 0.350 |
| **Overall** | **0.458** |

Note the accuracy gap between the two tables: the demo has no reference answers,
so accuracy is plausibility + groundedness (both high). The benchmark scores
against hand-written references, which a template composer cannot reproduce
exactly. Both numbers are honest measurements of what the built-in backend does.

---

## 5 · Known limitations

1. **The composer is not a language model.** It composes fluent, well-structured,
   grounded answers from templates, the KG and the candidate's story bank. It
   cannot state a fact it was never given. Point `generation.llm.*` at an
   OpenAI-compatible endpoint to get a real model behind the same retrieval and
   validation pipeline.
2. **The offline ASR is an isolated-phrase matcher.** It recognises the phrases
   in its template bank (seeded with 12 standard interview questions, extendable
   and re-calibratable at runtime). It is not a large-vocabulary continuous
   speech recogniser; that is what the Vosk and cloud backends are for.
3. **Topic accuracy trails intent accuracy.** Fine topics are inferred from
   keyword rules with a Naïve Bayes fallback; 0.64 is the honest number,
   against 0.92 for coarse intent.
4. **Relevance is the weakest metric.** It measures lexical overlap, which
   penalises a good answer that legitimately reframes the question. A
   reference-based relevance score would be better; it is computed when a
   reference is available.
5. **The formant synthesiser is robotic.** It is intelligible and perfectly
   adequate for seeding acoustic templates and for spoken prompts; it is not a
   production TTS voice.

---

## 6 · Adding your own cases

Append to `CASES` in `interviewgenie/evaluation/dataset.py`, or point
`evaluation.dataset` at a JSON file:

```json
{
  "cases": [
    {
      "question": "How would you design a rate limiter?",
      "intent": "design",
      "topic": "architecture",
      "reference_answer": "I would use a token bucket per client in Redis with a TTL…",
      "key_terms": ["token bucket", "redis", "fail open"],
      "must_mention": ["redis"]
    }
  ]
}
```

The benchmark reports per-intent means and the worst-scoring cases, so a
regression in one intent shows up immediately.
