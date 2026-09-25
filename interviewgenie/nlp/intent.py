"""Text classification primitives and interview intent recognition.

Two classifiers are trained from the bundled labelled dataset
(``config/intent_dataset.json``):

* a **coarse intent** classifier -- *what kind of answer does the interviewer
  want* (a STAR story, an architecture sketch, a definition, ...), and
* a **fine topic** classifier -- *what is it about* (algorithms, reliability,
  collaboration, ...).

The underlying model is a multinomial Naive Bayes classifier over word
n-grams and character n-grams (the character features make it robust to
speech-recognition noise).  ``partial_fit`` supports online updates from the
continuous-learning loop.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..logging import get_logger
from .textnorm import char_ngrams, tokenize

LOG = get_logger("nlp.intent")


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def word_features(text: str) -> List[str]:
    """Unigrams + bigrams of the lower-cased token stream."""
    toks = [t.lower() for t in tokenize(text, keep_punct=False)]
    feats = list(toks)
    feats.extend(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
    return feats


def char_features(text: str, n: int = 3) -> List[str]:
    """Character n-grams, which make the model robust to ASR noise."""
    return [f"#{g}" for g in char_ngrams(text.lower(), n)]


def extract_features(text: str, use_char: bool = True) -> List[str]:
    feats = word_features(text)
    if use_char:
        feats.extend(char_features(text))
    return feats


# --------------------------------------------------------------------------- #
# Multinomial Naive Bayes
# --------------------------------------------------------------------------- #
@dataclass
class NaiveBayesTextClassifier:
    """Multinomial Naive Bayes with add-alpha smoothing."""

    alpha: float = 0.2
    use_char: bool = True
    classes: List[str] = field(default_factory=list)
    _log_prior: Dict[str, float] = field(default_factory=dict, repr=False)
    _log_likelihood: Dict[str, Dict[str, float]] = field(default_factory=dict, repr=False)
    _default_ll: Dict[str, float] = field(default_factory=dict, repr=False)
    _doc_count: Counter = field(default_factory=Counter, repr=False)
    _feat_count: Counter = field(default_factory=Counter, repr=False)
    _trained_on: int = 0

    # -- training ---------------------------------------------------------- #
    def partial_fit(self, samples: Sequence[Tuple[str, str]]) -> "NaiveBayesTextClassifier":
        for text, label in samples:
            self._doc_count[label] += 1
            self._trained_on += 1
            for feat in set(extract_features(text, self.use_char)):
                self._feat_count[(label, feat)] += 1
        self._recompute()
        return self

    def fit(self, samples: Sequence[Tuple[str, str]]) -> "NaiveBayesTextClassifier":
        self.classes = []
        self._log_prior = {}
        self._log_likelihood = {}
        self._default_ll = {}
        self._doc_count = Counter()
        self._feat_count = Counter()
        self._trained_on = 0
        return self.partial_fit(samples)

    def _recompute(self) -> None:
        self.classes = sorted(self._doc_count)
        total_docs = max(1, sum(self._doc_count.values()))
        vocab = {feat for (_label, feat) in self._feat_count}
        v = max(1, len(vocab))
        for label in self.classes:
            self._log_prior[label] = math.log(self._doc_count[label] / total_docs)
            total = sum(cnt for (lbl, _f), cnt in self._feat_count.items() if lbl == label)
            denom = total + self.alpha * v
            self._default_ll[label] = math.log(self.alpha / denom)
            self._log_likelihood[label] = {
                feat: math.log((cnt + self.alpha) / denom)
                for (lbl, feat), cnt in self._feat_count.items() if lbl == label
            }

    # -- inference --------------------------------------------------------- #
    def predict_log_proba(self, text: str) -> Dict[str, float]:
        feats = set(extract_features(text, self.use_char))
        scores: Dict[str, float] = {}
        for label in self.classes:
            table = self._log_likelihood[label]
            default = self._default_ll[label]
            score = self._log_prior[label]
            for feat in feats:
                score += table.get(feat, default)
            scores[label] = score
        return scores

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Return ``(label, confidence, normalised_scores)``."""
        if not self.classes:
            return "unknown", 0.0, {}
        log_scores = self.predict_log_proba(text)
        if not log_scores:
            return "unknown", 0.0, {}
        hi = max(log_scores.values())
        exps = {lbl: math.exp(v - hi) for lbl, v in log_scores.items()}
        total = sum(exps.values()) or 1.0
        probs = {lbl: e / total for lbl, e in exps.items()}
        best = max(probs, key=probs.get)
        return best, probs[best], probs

    def top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        _, _, probs = self.predict(text)
        return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]

    @property
    def trained_on(self) -> int:
        return self._trained_on


# --------------------------------------------------------------------------- #
# High precision linguistic rules (run *before* the statistical model)
#
# Rules are evaluated in order and the FIRST match wins, so the list is ordered
# from most specific to most general.
# --------------------------------------------------------------------------- #
_RULES: List[Tuple[str, str, float]] = [
    # --- clarification / meta ------------------------------------------- #
    (r"\brepeat that\b|\bdid not quite catch\b|\bwhat did you say\b|\bclarify what you mean\b"
     r"|\bwhich .{0,25}do you mean\b|\bcan you be more specific\b|\bi missed the last part\b"
     r"|\baudio cut out\b|\bconnection (?:cut|dropped)\b|\bsay that again\b|\bcome again\b",
     "meta", 0.95),
    (r"\bquestions for (?:us|me)\b|\banything else you would like\b|\bfinal thoughts\b"
     r"|\bwe are out of time\b|\bbefore we wrap up\b|\blast questions\b|\bthanks for your time\b"
     r"|\banything i have not asked\b|\bwhat would you like to know\b",
     "meta", 0.95),
    (r"\bhow is your day\b|\bhow was your commute\b|\bdo you need a break\b|\bfun plans\b"
     r"|\bnice weather\b|\bare you local\b|\bhow are you doing today\b|\bfound the office\b"
     r"|\bconnection is working\b|\bhope the connection\b",
     "meta", 0.95),

    # --- self assessment ------------------------------------------------ #
    (r"\bwhat (?:is|are) your (?:greatest |biggest |main |current )?(?:strength|superpower)"
     r"|\bwhere do you struggle\b|\bmistake you made\b|\bdropped the ball\b"
     r"|\barea you (?:know you )?need to improve\b|\bnot good at yet\b"
     r"|\bwhat would you do differently\b|\bbiggest professional regret\b"
     r"|\bwhat part of the job do you find hardest\b|\bskill are you (?:currently )?working on\b",
     "self_assessment", 0.95),

    # --- people / leadership / teamwork --------------------------------- #
    (r"\b(?:tell|walk|describe|give)\b[^.?!]{0,60}\ba time\b[^.?!]{0,60}"
     r"\b(?:led|lead|mentor|mentored|helped|escalat|ownership|technical direction|"
     r"grew|grown|major change|through a crunch)\b",
     "people", 0.95),
    (r"\b(?:tell|walk|describe|give)\b[^.?!]{0,60}\ba time\b[^.?!]{0,60}"
     r"\b(?:disagree\w*|pushed back|conflict|difficult|rude|dismissive|credit for your work)\b",
     "people", 0.95),
    (r"\b(?:conflict|disagree|disagreement|difficult (?:colleague|coworker|stakeholder|person|situation)"
     r"|pushed back|escalate a|rude|dismissive|not pulling their weight|credit for your work"
     r"|handle conflict|cannot agree|could not agree)\b",
     "people", 0.9),
    (r"\b(?:mentor|mentored|mentoring|leadership|technical direction|took ownership|"
     r"delegate|motivate a team|grew other engineers|grow other engineers|"
     r"drive a decision|set direction|recover after a bad incident)\b",
     "people", 0.85),
    (r"\b(?:collaborate|collaboration|cross[- ]functional|working relationship|"
     r"share knowledge|code review|build trust|make sure everyone is aligned|"
     r"role do you (?:usually )?play on a team|work with other teams)\b",
     "people", 0.85),

    # --- motivation ----------------------------------------------------- #
    (r"\bwhy (?:do you want|are you interested|did you apply|should we hire you|"
     r"are you looking|this role|our company)\b|\bwhat excites you\b|\bwhat drew you\b"
     r"|\bwhat made you reach out\b|\bwhat would make you join\b|\bwhat is it about this position\b"
     r"|\bwhat do you hope to get out of\b",
     "motivation", 0.95),

    # --- goals ---------------------------------------------------------- #
    (r"\bwhere do you see yourself\b|\blong[- ]term career\b|\bcareer goals?\b"
     r"|\bnext step you want\b|\bcareer success\b|\bmove into management\b"
     r"|\bhoping to achieve in your next role\b|\bkind of role are you aiming for\b"
     r"|\bindividual contributor track\b|\bwhere do you want your career\b",
     "goals", 0.95),

    # --- culture -------------------------------------------------------- #
    (r"\bwork environment\b|\bteam culture\b|\bcore values at work\b"
     r"|\bpsychologically safe\b|\bwork life balance\b|\bkind of manager\b"
     r"\bhow do you recharge\b|\bremotely or in an office\b"
     r"|\bhow do you like to receive feedback\b|\bhow do you handle ambiguity\b"
     r"|\bwhat does a good team culture\b",
     "culture", 0.95),

    # --- measurement ---------------------------------------------------- #
    (r"\bhow would you measure\b|\bwhat metrics\b|\bwhat kpis?\b|\bhow do you know when\b"
     r"|\bwhat data would you\b|\bmeasure the success\b|\bwhat does good look like\b"
     r"|\bquantify the impact\b|\bmeasure code quality\b|\bwhether an experiment worked\b"
     r"|\bjustify a rewrite\b",
     "measurement", 0.95),

    # --- logistics ------------------------------------------------------ #
    (r"\bsalary\b|\bcompensation\b|\bnotice period\b|\bwhen (?:could|can) you start\b"
     r"|\brelocation\b|\bequity\b|\bvisa sponsorship\b|\bsign an offer\b"
     r"|\binterviewing elsewhere\b|\bavailability for the next round\b"
     r"|\bflexible are you on the base\b|\bwhat are you looking for in terms of\b",
     "logistics", 0.95),

    # --- background / experience / education ---------------------------- #
    (r"\bwhat experience\b|\bhow much experience\b|\bhave you worked with\b"
     r"|\bhow long have you been\b|\byour background with\b|\beducational background\b"
     r"|\bhow do you keep your skills\b|\bcourses or certifications\b"
     r"|\bdid you finish your degree\b|\bformal training\b|\bhow do you learn\b"
     r"|\bhow much of your work\b|\bexposure have you had\b|\bowned a service\b"
     r"|\bwhat was the last thing you taught yourself\b|\bwhat did you study\b"
     r"|\bfavourite course\b|\bfavorite course\b",
     "background", 0.95),

    # --- system design -------------------------------------------------- #
    (r"\bhow would you (?:design|architect|scale|build)\b|\bdesign (?:a|an|the)\b"
     r"|\bwalk me through the architecture\b|\barchitect a\b|\bdata model for\b"
     r"|\bstorage layer for\b|\bapi gateway\b|\bmulti tenant\b|\bmonolithic web application\b",
     "design", 0.95),

    # --- coding --------------------------------------------------------- #
    (r"\b(?:write|implement|code|coding)\b[^.?!]{0,50}\b(?:function|query|code|solution|"
     r"program|class|stack|queue|cache|tree|list|search|anagram|palindrome|fibonacci)\b"
     r"|\bcode up\b|\bimplement an?\b|\btwo sum\b|\bdepth first search\b|\bbinary search\b",
     "coding", 0.95),

    # --- troubleshooting ------------------------------------------------ #
    (r"\b(?:debug|troubleshoot|investigate|diagnose|flaky|500 errors|error rate|"
     r"memory leak|race condition|cannot reproduce|can't reproduce)\b"
     r"|\bwhy (?:is|are) (?:this|it|the|my|our|that)\b|\bwhy is .{0,20}(?:slow|failing|broken)\b"
     r"|\bfailing\b|\bdeployment is failing\b|\bworks locally\b|\bintermittent\b",
     "troubleshooting", 0.9),

    # --- hypothetical --------------------------------------------------- #
    (r"\bwhat would you do if\b|\bwhat would you do\b|\bsuppose\b"
     r"|\bif you (?:joined|had|were|found|realised|realized|were asked)\b",
     "hypothetical", 0.92),

    # --- self introduction --------------------------------------------- #
    (r"\b(?:tell|talk|walk|give|summarise|summarize)\b[^.?!]{0,50}"
     r"\b(?:about yourself|your background|your resume|your cv|your story|your journey|"
     r"your career|me through your)\b|\btell me your story\b|\bhow would you introduce yourself\b"
     r"|\bwhat should i know about you\b|\bgive me a sense of your professional\b",
     "self_introduction", 0.95),

    # --- past experience (STAR) ---------------------------------------- #
    (r"\b(?:tell|walk|describe|give)\b[^.?!]{0,60}\ba time\b"
     r"|\b(?:tell|walk|describe|give)\b[^.?!]{0,60}\ban example\b"
     r"|\btell me about a project\b|\bchallenging bug you\b|\bhardest technical problem\b"
     r"|\btell me about your (?:last|most recent|previous) (?:project|role)\b",
     "past_experience", 0.95),

    # --- knowledge (generic fallback for explain/compare) --------------- #
    (r"\bexplain\b|\bwhat is the difference\b|\bwhat are the\b|\bwhat is an?\b"
     r"|\bhow does .{0,40}work\b|\bcompare\b|\btradeoffs?\b|\bpros and cons\b"
     r"|\bversus\b|\bwhen would you not use\b|\bpros and cons\b"
     r"|\bwould you rather\b|\bis it better to\b|\bshould we use\b|\bwould you choose between\b",
     "knowledge", 0.85),
]

_COMPILED_RULES = [(re.compile(pat, re.I), intent, conf) for pat, intent, conf in _RULES]

#: keyword cues -> fine topic (evaluated after intent, most specific first)
_TOPIC_RULES: List[Tuple[str, str]] = [
    # Conversational management first: these phrases are about the interview
    # itself (clarifying, closing, small talk) rather than its subject matter,
    # and several would otherwise be swallowed by the content rules below.
    (r"\b(?:repeat that|say (?:that|it) again|did ?n[o']?t quite catch|"
     r"didn'?t catch|come again|pardon|clarify|more specific|"
     r"audio cut|cut out|missed the last|go back|referring to exactly)\b", "meta"),
    (r"\b(?:questions? for (?:us|me)|questions do you have|like to know about|"
     r"anything else|last questions?|final thoughts|wrap ?up|all from me|"
     r"out of time|highlight before)\b", "meta"),
    (r"\b(?:how is your day|your day going|find the office|need a break|"
     r"some water|nice weather|how was your commute|how are you doing|"
     r"fun plans|for the weekend|like the area|connection is working)\b", "meta"),
    # Motivation about *this* employer must be resolved before the culture rule,
    # which also contains the word "company" (as in "company culture").
    (r"\b(?:why did you apply|why this (?:company|role|team|position)|"
     r"why do you want to (?:work|join)|why (?:are you|do you) (?:here|interested)|"
     r"what (?:excites|draws|interests) you about)\b", "company"),
    (r"\b(?:algorithms?|linked lists?|binary search|sort(?:ing)?|hash(?:es|ing)?|"
     r"trees?|graphs?|dynamic programming|big o|complexity|fibonacci|"
     r"anagrams?|palindromes?)\b", "algorithms"),
    (r"\b(?:design(?:ing|s)?|architect(?:ure|ing|s)?|scal(?:e|ing)|shard(?:s|ing)?|"
     r"replicat(?:e|es|ion|ing)|load balanc(?:e|es|ing)|cach(?:e|es|ing)|queues?|"
     r"microservices?|monoliths?|latency|throughput|availability|partition(?:s|ing)?)\b", "architecture"),
    (r"\b(?:debug(?:s|ging)?|bugs?|incidents?|outages?|errors?|flaky|"
     r"memory leaks?|races?|latency spikes?|on call|troubleshoot(?:s|ing)?|"
     r"production)\b", "reliability"),
    (r"\b(?:python|java|javascript|typescript|golang|rust|sql|nosql|"
     r"databases?|index(?:es|ing)?|tcp|udp|http|apis?|closures?|"
     r"garbage collection|cap theorem|acid|kubernetes|docker|terraform|"
     r"aws|azure|gcp|cloud|ci|cd|pipelines?|deploy(?:s|ed|ing|ment)?|"
     r"machine learning|models?|training|datasets?|neural|embedding(?:s)?|"
     r"llms?|nlp)\b", "technical_concept"),
    (r"\b(?:degrees?|universit(?:y|ies)|colleges?|courses?|certifications?|"
     r"majors?|study(?:ing)?|studied|self[- ]taught|bootcamps?|learn(?:s|ed|ing)?)\b", "learning"),
    (r"\b(?:mentor(?:s|ed|ing)?|led|lead(?:s|ing)?|leadership|ownership|"
     r"delegate(?:s|d)?|direction|influence|took charge|drove|initiative|"
     r"step(?:ped)? up)\b", "leadership"),
    (r"\b(?:conflicts?|disagree(?:s|d|ment|ments)?|collaborat(?:e|es|ed|ing|ion)?|"
     r"teammates?|stakeholders?|cross[- ]functional|review(?:s|ed)?|align(?:s|ed|ment)?|"
     r"trust|push(?:ed)? back|managers?|colleagues?|difficult)\b", "collaboration"),
    (r"\b(?:deadlines?|ship(?:s|ped|ping)?|deliver(?:s|ed|ing|y)?|scope|"
     r"prioriti[sz](?:e|es|ed|ing)|projects?|migrat(?:e|es|ed|ing|ion)|"
     r"launch(?:es|ed|ing)?|release(?:s|d)?)\b", "delivery"),
    (r"\b(?:strengths?|weakness(?:es)?|improve(?:s|d|ment)?|struggl(?:e|es|ed|ing)|"
     r"feedback|mistakes?|regret(?:s|ted)?|myself|yourself)\b", "self"),
    (r"\b(?:salar(?:y|ies)|compensation|equity|notice|relocat(?:e|es|ed|ing|ion)|"
     r"visas?|start dates?|offers?)\b", "logistics"),
    (r"\b(?:metrics?|kpis?|measure(?:s|d|ment)?|success|impact|quantif(?:y|ies|ied)|"
     r"data would you)\b", "measurement"),
    (r"\b(?:company|product|team cultures?|values|environment|remote|culture|balance|mission)\b", "culture"),
    (r"\b(?:career|five years|long term|management|goals?|ambition|"
     r"next role|growth)\b", "career"),
    (r"\b(?:why|excite(?:s|d)?|interest(?:s|ed|ing)?|motivat(?:e|es|ed|ing|ion)|"
     r"draw(?:s|n)?|apply|applied|hire|hiring)\b", "company"),
    (r"\b(?:trade[- ]?offs?|compare|versus|choose|decid(?:e|es|ed|ing)|"
     r"prioriti[sz]e|options?)\b", "judgement"),
]
_COMPILED_TOPIC_RULES = [(re.compile(pat, re.I), topic) for pat, topic in _TOPIC_RULES]

#: WH-word -> question semantics
WH_TYPES = {
    "what": "what", "why": "why", "how": "how", "when": "when", "where": "where",
    "who": "who", "which": "which", "whose": "whose", "whom": "who",
}

DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "intent_dataset.json",
)


@dataclass
class IntentClassifier:
    """Two-axis interview intent recogniser (rules + statistical model)."""

    dataset_path: Optional[str] = None
    threshold: float = 0.30
    alpha: float = 0.22
    classifier: NaiveBayesTextClassifier = field(default_factory=NaiveBayesTextClassifier)
    topic_classifier: NaiveBayesTextClassifier = field(default_factory=NaiveBayesTextClassifier)
    descriptions: Dict[str, str] = field(default_factory=dict)
    topic_descriptions: Dict[str, str] = field(default_factory=dict)
    examples: List[Dict[str, str]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fit_from_dataset(self.dataset_path)

    # -- data -------------------------------------------------------------- #
    def load_dataset(self, path: Optional[str] = None) -> List[Dict[str, str]]:
        path = path or self.dataset_path or DEFAULT_DATASET
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.descriptions = data.get("classes", {})
            self.topic_descriptions = data.get("topics", {})
            examples = [
                {"text": ex["text"], "intent": ex["intent"],
                 "topic": ex.get("topic", "meta"), "fine": ex.get("fine", ex["intent"])}
                for ex in data.get("examples", [])
            ]
            if not examples:
                raise ValueError("dataset contains no examples")
            return examples
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            LOG.warning("intent dataset unavailable (%s); using embedded fallback", exc)
            self.descriptions = {k: v for k, v in _FALLBACK_DESCRIPTIONS.items()}
            self.topic_descriptions = {}
            return [{"text": t, "intent": i, "topic": topic, "fine": i}
                    for t, i, topic in _FALLBACK_SAMPLES]

    def fit_from_dataset(self, path: Optional[str] = None,
                         use_fallback: bool = True) -> "IntentClassifier":
        # ``load_dataset`` already substitutes the embedded fallback samples
        # when the file is missing or unreadable, so they must NOT be appended
        # on top of a dataset that loaded successfully: doing so duplicated
        # rows and mislabelled every fallback sample's topic as "meta".
        examples = self.load_dataset(path)
        self.examples = examples
        self.classifier.alpha = self.alpha
        self.topic_classifier.alpha = self.alpha
        self.classifier.fit([(e["text"], e["intent"]) for e in examples])
        self.topic_classifier.fit([(e["text"], e["topic"]) for e in examples])
        LOG.info("intent classifiers trained", context={
            "examples": len(examples),
            "intents": len(self.classifier.classes),
            "topics": len(self.topic_classifier.classes),
        })
        return self

    def learn(self, text: str, intent: str, topic: Optional[str] = None) -> None:
        """Online update from a corrected / human-labelled interaction."""
        self.classifier.partial_fit([(text, intent)])
        if topic:
            self.topic_classifier.partial_fit([(text, topic)])
        self.examples.append({"text": text, "intent": intent,
                              "topic": topic or "meta", "fine": intent})
        self.history.append({"text": text, "intent": intent, "topic": topic})
        if len(self.history) > 500:
            self.history = self.history[-500:]

    # -- inference --------------------------------------------------------- #
    def rule_match(self, text: str) -> Optional[Tuple[str, float]]:
        for pattern, intent, confidence in _COMPILED_RULES:
            if pattern.search(text):
                return intent, confidence
        return None

    def classify(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Return ``(intent, confidence, scores)`` for ``text``."""
        rule = self.rule_match(text)
        if rule is not None:
            intent, confidence = rule
            _, _, probs = self.classifier.predict(text)
            probs[intent] = max(probs.get(intent, 0.0), confidence)
            return intent, confidence, probs
        return self.classifier.predict(text)

    def classify_topic(self, text: str) -> Tuple[str, float]:
        """Fine topic: keyword rules first, statistical model as fallback."""
        for pattern, topic in _COMPILED_TOPIC_RULES:
            if pattern.search(text):
                return topic, 0.85
        label, confidence, _ = self.topic_classifier.predict(text)
        return label, confidence

    def wh_type(self, text: str) -> Optional[str]:
        for tok in tokenize(text, keep_punct=False)[:6]:
            low = tok.lower()
            if low in WH_TYPES:
                return WH_TYPES[low]
        return None

    # -- evaluation -------------------------------------------------------- #
    def evaluate(self, holdout: float = 0.25, seed: int = 7) -> Dict[str, float]:
        """Train/test split evaluation of both classifiers."""
        examples = self.examples or self.load_dataset()
        if len(examples) < 12:
            return {"intent_accuracy": 0.0, "topic_accuracy": 0.0,
                    "support": float(len(examples))}
        rng = random.Random(seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * (1 - holdout)))
        train, test = shuffled[:cut], shuffled[cut:]

        intent_clf = NaiveBayesTextClassifier(alpha=self.classifier.alpha,
                                              use_char=self.classifier.use_char)
        intent_clf.fit([(e["text"], e["intent"]) for e in train])
        topic_clf = NaiveBayesTextClassifier(alpha=self.topic_classifier.alpha,
                                             use_char=self.topic_classifier.use_char)
        topic_clf.fit([(e["text"], e["topic"]) for e in train])

        intent_ok = topic_ok = 0
        for ex in test:
            pi, _, _ = intent_clf.predict(ex["text"])
            intent_ok += int(pi == ex["intent"])
            pt, _, _ = topic_clf.predict(ex["text"])
            topic_ok += int(pt == ex["topic"])
        return {
            "intent_accuracy": round(intent_ok / len(test), 4),
            "topic_accuracy": round(topic_ok / len(test), 4),
            "support": float(len(test)),
            "train_size": float(len(train)),
        }

    # -- fitting ----------------------------------------------------------- #
    def _fit_examples(self, examples: List[Dict[str, str]]) -> "IntentClassifier":
        """Fit both statistical models on an explicit example list."""
        self.examples = examples
        self.classifier.alpha = self.alpha
        self.topic_classifier.alpha = self.alpha
        self.classifier.fit([(e["text"], e["intent"]) for e in examples])
        self.topic_classifier.fit([(e["text"], e["topic"]) for e in examples])
        return self

    def _clone_for_training(self) -> "IntentClassifier":
        """An unfitted twin, built without touching the dataset.

        ``__post_init__`` would otherwise fit on the *whole* dataset, which is
        exactly the leakage a holdout split exists to rule out.
        """
        twin = object.__new__(IntentClassifier)
        twin.dataset_path = self.dataset_path
        twin.threshold = self.threshold
        twin.alpha = self.alpha
        twin.classifier = NaiveBayesTextClassifier(alpha=self.alpha)
        twin.topic_classifier = NaiveBayesTextClassifier(alpha=self.alpha)
        twin.descriptions = self.descriptions
        twin.topic_descriptions = self.topic_descriptions
        twin.examples = []
        twin.history = []
        return twin

    def evaluate_holdout(self, holdout: float = 0.25,
                         seeds: Sequence[int] = (1, 2, 3, 4, 5)
                         ) -> Dict[str, Any]:
        """Leak-free holdout evaluation of the *deployed* rules+model path.

        The statistical models are fitted on the training split only, then the
        full ``classify``/``classify_topic`` pipeline — the one used at
        inference time — is scored on the unseen split.  Reported alongside the
        model-only figures so the contribution of the hand-written rules is
        visible rather than hidden.
        """
        examples = self.examples or self.load_dataset()
        if len(examples) < 12:
            return {"intent_accuracy": 0.0, "topic_accuracy": 0.0,
                    "support": float(len(examples)), "seeds": 0}

        per_seed: List[Dict[str, float]] = []
        for seed in seeds:
            rng = random.Random(seed)
            shuffled = list(examples)
            rng.shuffle(shuffled)
            cut = max(1, int(len(shuffled) * (1 - holdout)))
            train, test = shuffled[:cut], shuffled[cut:]

            # Deployed path: rules + model fitted on TRAIN ONLY.
            probe = self._clone_for_training()._fit_examples(train)
            intent_ok = sum(probe.classify(e["text"])[0] == e["intent"] for e in test)
            topic_ok = sum(probe.classify_topic(e["text"])[0] == e["topic"] for e in test)

            # Same split, statistical model alone, for comparison.
            model_intent = NaiveBayesTextClassifier(alpha=self.alpha,
                                                    use_char=self.classifier.use_char)
            model_topic = NaiveBayesTextClassifier(alpha=self.alpha,
                                                   use_char=self.topic_classifier.use_char)
            model_intent.fit([(e["text"], e["intent"]) for e in train])
            model_topic.fit([(e["text"], e["topic"]) for e in train])
            m_intent = sum(model_intent.predict(e["text"])[0] == e["intent"] for e in test)
            m_topic = sum(model_topic.predict(e["text"])[0] == e["topic"] for e in test)

            n = len(test)
            per_seed.append({
                "intent": intent_ok / n,
                "topic": topic_ok / n,
                "model_only_intent": m_intent / n,
                "model_only_topic": m_topic / n,
                "support": float(n),
            })

        def _mean(key: str) -> float:
            return sum(s[key] for s in per_seed) / len(per_seed)

        return {
            "intent_accuracy": round(_mean("intent"), 4),
            "topic_accuracy": round(_mean("topic"), 4),
            "model_only_intent_accuracy": round(_mean("model_only_intent"), 4),
            "model_only_topic_accuracy": round(_mean("model_only_topic"), 4),
            "support": float(sum(s["support"] for s in per_seed) / len(per_seed)),
            "seeds": float(len(per_seed)),
            "per_seed": [
                {"seed": seed, "intent": round(s["intent"], 4), "topic": round(s["topic"], 4),
                 "model_only_intent": round(s["model_only_intent"], 4),
                 "model_only_topic": round(s["model_only_topic"], 4)}
                for seed, s in zip(seeds, per_seed)
            ],
        }


# --------------------------------------------------------------------------- #
# Embedded fallback (used only when the dataset file cannot be read)
# --------------------------------------------------------------------------- #
_FALLBACK_DESCRIPTIONS = {
    "self_introduction": "Self introduction",
    "past_experience": "Past experience question",
    "knowledge": "Domain knowledge question",
    "design": "Architecture design question",
    "coding": "Write code",
    "motivation": "Why this role",
    "self_assessment": "Strengths and weaknesses",
    "logistics": "Logistics and salary",
    "meta": "Clarification, closing and small talk",
}

_FALLBACK_SAMPLES: List[Tuple[str, str, str]] = [
    # (text, intent, topic) - used only when the dataset file cannot be read.
    ("tell me about yourself", "self_introduction", "self"),
    ("walk me through your resume", "self_introduction", "self"),
    ("tell me about a time when you led a team", "past_experience", "leadership"),
    ("describe a situation where you failed", "past_experience", "self"),
    ("what is your greatest weakness", "self_assessment", "self"),
    ("what is your greatest strength", "self_assessment", "self"),
    ("explain the cap theorem", "knowledge", "technical_concept"),
    ("what is the difference between tcp and udp", "knowledge", "technical_concept"),
    ("how would you design a url shortener", "design", "architecture"),
    ("design a rate limiter", "design", "architecture"),
    ("write a function that reverses a linked list", "coding", "algorithms"),
    ("implement an lru cache", "coding", "architecture"),
    ("why do you want to work here", "motivation", "company"),
    ("why are you interested in this role", "motivation", "company"),
    ("what are your salary expectations", "logistics", "logistics"),
    ("when can you start", "logistics", "logistics"),
    ("do you have any questions for us", "meta", "meta"),
    ("could you repeat that please", "meta", "meta"),
]
