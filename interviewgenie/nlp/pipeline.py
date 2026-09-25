"""The NLP pipeline facade.

One call to :meth:`NLPPipeline.analyze` runs the whole linguistic stack and
returns a single :class:`~interviewgenie.types.Analysis` object that the rest of
the system consumes:

    normalise -> sentence split -> tokenise -> POS tag -> lemmatise
    -> NER + KG linking -> coreference -> keywords -> intent
    -> sentiment + emotion -> topic / embedding projection
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..logging import get_logger
from ..types import Analysis, EmotionScores, Entity, Intent, PosToken, Sentiment
from . import lexicon as LX
from .coref import CoreferenceResolver
from .embeddings import TfidfIndex, extract_keywords
from .entities import EntityExtractor, EntityLinker
from .intent import IntentClassifier
from .sentiment import SentimentAnalyzer
from .tagger import PosTagger
from .textnorm import split_questions, split_sentences, normalize

LOG = get_logger("nlp.pipeline")


@dataclass
class NLPPipeline:
    """Composes every linguistic analyser into a single entry point."""

    tagger: PosTagger = field(default_factory=PosTagger)
    extractor: EntityExtractor = field(default_factory=EntityExtractor)
    intent_classifier: IntentClassifier = field(default_factory=IntentClassifier)
    sentiment: SentimentAnalyzer = field(default_factory=SentimentAnalyzer)
    coref: CoreferenceResolver = field(default_factory=CoreferenceResolver)
    linker: Optional[EntityLinker] = None
    index: Optional[TfidfIndex] = None
    keyword_count: int = 12
    enable_coref: bool = True
    history: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    # -- wiring ------------------------------------------------------------ #
    def attach_linker(self, linker: EntityLinker) -> None:
        """Install a KG linker so entities are resolved to graph nodes."""
        self.linker = linker
        self.extractor.gazetteer = lambda span: self._gazetteer_lookup(span)

    def _gazetteer_lookup(self, span: str) -> List[Tuple[str, str, float]]:
        if self.linker is None:
            return []
        hits: List[Tuple[str, str, float]] = []
        for alias, (node_id, label) in self.linker.alias_index.items():
            if alias == span:
                hits.append((alias, label, 0.95))
            elif len(span) > 4 and (alias.startswith(span) or span.startswith(alias)):
                hits.append((alias, label, 0.6))
        return hits[:4]

    def attach_index(self, index: TfidfIndex) -> None:
        self.index = index

    def reset(self) -> None:
        """Clear per-session state (history and counters)."""
        self.history = []
        self.stats = {"analyses": 0, "total_ms": 0.0}

    # -- main entry point -------------------------------------------------- #
    def analyze(self, text: str, *, update_history: bool = True) -> Analysis:
        """Run the full linguistic analysis of one utterance."""
        started = time.perf_counter()
        normalized = normalize(text)
        analysis = Analysis(text=text, normalized=normalized)
        if not normalized:
            analysis.duration = time.perf_counter() - started
            return analysis

        analysis.sentences = split_sentences(normalized)
        analysis.questions = split_questions(normalized)

        tagged = self.tagger.tag_text(normalized)
        analysis.tokens = [
            PosToken(text=t.text, lemma=t.lemma, tag=t.tag, index=t.index, is_stop=t.is_stop)
            for t in tagged
        ]

        entities = self.extractor.extract(normalized)
        if self.linker is not None:
            entities = self.linker.link(entities)
        analysis.entities = entities

        if self.enable_coref:
            try:
                self._coref = self.coref.resolve(normalized, list(self.history))
            except Exception as exc:  # noqa: BLE001 - coref must never break analysis
                LOG.debug("coref failed: %s", exc)
                self._coref = []

        analysis.keywords = extract_keywords(normalized, top_k=self.keyword_count)
        analysis.intent = self._classify_intent(normalized)
        analysis.sentiment = self.sentiment.analyze(normalized)
        analysis.emotion = self.sentiment.emotions(normalized)

        if self.index is not None and self.index.fitted:
            analysis.embedding = self.index.embed(normalized)
            analysis.topics = self._topics(normalized)

        if update_history:
            self.history.append(normalized)
            if len(self.history) > 60:
                self.history = self.history[-60:]

        analysis.duration = time.perf_counter() - started
        self.stats = {
            "last_duration_ms": round(analysis.duration * 1000, 3),
            "tokens": len(analysis.tokens),
            "entities": len(analysis.entities),
            "history": len(self.history),
        }
        return analysis

    # -- sub-steps --------------------------------------------------------- #
    def _classify_intent(self, text: str) -> Intent:
        label, confidence, scores = self.intent_classifier.classify(text)
        topic, topic_conf = self.intent_classifier.classify_topic(text)
        wh = self.intent_classifier.wh_type(text)
        low = text.lower()
        is_question = text.rstrip().endswith("?") or bool(wh) or low.startswith((
            "tell", "describe", "explain", "walk", "give", "how", "what", "why",
            "when", "where", "who", "which", "design", "write", "implement",
            "code", "compare", "summarise", "summarize",
        ))
        ambiguous = confidence < self.intent_classifier.threshold
        low = text.lower().strip()
        asks_for_repetition = label == "meta" and any(
            cue in low for cue in (
                "repeat", "say that again", "come again", "did not catch",
                "didn't catch", "not sure i understood", "sorry", "pardon",
                "clarify", "what do you mean", "could you rephrase"))
        asks_for_disambiguation = ambiguous and is_question and "?" in low
        intent = Intent(
            name=label, wh_type=wh, confidence=round(confidence, 4),
            scores={k: round(v, 4) for k, v in scores.items()},
            is_question=is_question,
            requires_clarification=bool(asks_for_repetition or asks_for_disambiguation),
        )
        # the fine-grained topic is carried on the intent's score map for the UI
        intent.scores[f"topic:{topic}"] = round(topic_conf, 4)
        intent.topic = topic
        intent.topic_confidence = round(topic_conf, 4)
        return intent

    def _topics(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Map an utterance onto its nearest indexed knowledge topics."""
        if self.index is None or not self.index.fitted:
            return []
        hits = self.index.search(text, k=top_k, use_lsa=True)
        out: List[Tuple[str, float]] = []
        for idx, score in hits:
            doc = self.index.documents[idx]
            label = doc.split("|")[0].strip() or doc[:60]
            out.append((label, round(score, 4)))
        return out

    # -- introspection ----------------------------------------------------- #
    def describe(self) -> Dict[str, Any]:
        return {
            "tagger_backend": self.tagger.backend,
            "external_tagger": self.tagger._external is not None,
            "intent_classes": len(self.intent_classifier.classifier.classes),
            "intent_training_examples": self.intent_classifier.classifier.trained_on,
            "lexicon_size": len(LX.POS_LEXICON),
            "index": self.index.stats() if self.index and self.index.fitted else None,
        }


#: module level singleton
_PIPELINE: Optional[NLPPipeline] = None


def get_pipeline(**kwargs: Any) -> NLPPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = NLPPipeline(**kwargs)
    return _PIPELINE
