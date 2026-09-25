"""Part-of-speech tagging.

A transformation-based (Brill-lite) tagger seeded by the unigram lexicon in
:mod:`interviewgenie.nlp.lexicon`.  Unknown words are handled by suffix /
shape rules.  When spaCy or NLTK is importable the tagger delegates to it and
only falls back to the built-in rules otherwise, so accuracy improves
automatically in richer environments without any code change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import lexicon as LX
from .stemmer import lemmatize
from .textnorm import Tokenizer, is_acronym

_ADJECTIVE_SUFFIXES = (
    "able", "ible", "al", "ial", "ical", "ous", "ful", "less", "ive", "ic",
    "ish", "ary", "ory", "ant", "ent", "ern", "est",
)
_ADVERB_SUFFIXES = ("ly", "ward", "wards", "wise", "ways")
_NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship", "hood",
    "ism", "ist", "ology", "graphy", "ery", "age", "dom", "acy",
)
_VERB_SUFFIXES = ("ize", "ise", "ate", "ify", "en")


@dataclass
class TaggedWord:
    """One token with its assigned tag and lemma."""

    text: str
    tag: str
    lemma: str
    index: int
    is_stop: bool = False
    shape: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "tag": self.tag,
            "lemma": self.lemma,
            "index": self.index,
            "is_stop": self.is_stop,
        }


@dataclass
class PosTagger:
    """Transformation-based POS tagger with optional external backend."""

    backend: str = "builtin"
    _external: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._external = _load_external(self.backend)

    def tag(self, tokens: Sequence[str]) -> List[TaggedWord]:
        """Tag a sequence of raw tokens."""
        if not tokens:
            return []
        if self._external is not None:
            return self._external(tokens)
        return _rule_based_tag(tokens)

    def tag_text(self, text: str) -> List[TaggedWord]:
        return self.tag(Tokenizer().tokenize(text))


# --------------------------------------------------------------------------- #
# Backend discovery
# --------------------------------------------------------------------------- #
def _load_external(backend: str) -> Optional[object]:
    """Try to obtain a spaCy/NLTK tagger; return ``None`` when unavailable."""
    if backend == "spacy":
        try:
            import spacy  # type: ignore

            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:  # noqa: BLE001 - model not installed
                return None
            return _SpacyAdapter(nlp)
        except Exception:  # noqa: BLE001
            return None
    if backend == "nltk":
        try:
            import nltk  # type: ignore

            try:
                nltk.pos_tag(["test"])
            except Exception:  # noqa: BLE001 - corpora missing
                return None
            return _NltkAdapter(nltk.pos_tag)
        except Exception:  # noqa: BLE001
            return None
    return None


class _SpacyAdapter:
    def __init__(self, nlp: object) -> None:
        self._nlp = nlp

    def __call__(self, tokens: Sequence[str]) -> List[TaggedWord]:
        doc = self._nlp(" ".join(tokens))  # type: ignore[operator]
        return [
            TaggedWord(
                text=tok.text, tag=tok.tag_,
                lemma=tok.lemma_.lower() or tok.text.lower(),
                index=i, is_stop=tok.is_stop,
            )
            for i, tok in enumerate(doc)
        ]


class _NltkAdapter:
    def __init__(self, tagger: object) -> None:
        self._tagger = tagger

    def __call__(self, tokens: Sequence[str]) -> List[TaggedWord]:
        pairs = self._tagger(list(tokens))  # type: ignore[operator]
        return [
            TaggedWord(
                text=word, tag=tag, lemma=lemmatize(word.lower(), tag), index=i,
                is_stop=word.lower() in LX.STOP_WORDS,
            )
            for i, (word, tag) in enumerate(pairs)
        ]


# --------------------------------------------------------------------------- #
# Rule based tagging
# --------------------------------------------------------------------------- #
def _shape(word: str) -> str:
    """Coarse word shape: ``Xxxx``, ``XXXX``, ``dddd`` ..."""
    out = []
    for ch in word:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(out))


def _initial_guess(word: str) -> str:
    """Morphology driven guess for words absent from the lexicon."""
    if not word[0].isalnum():
        return LX.TAG_PUNCT
    low = word.lower()
    if is_acronym(word) or (
        len(word) > 1 and word[0].isupper() and word[1:].islower() and low not in LX.POS_LEXICON
    ):
        return LX.TAG_PROPER_SINGULAR
    if word.isdigit() or re.fullmatch(r"\d+([.,]\d+)?%?", word):
        return LX.TAG_CARDINAL
    for suffix in _ADVERB_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix) + 2:
            return LX.TAG_ADV
    for suffix in _ADJECTIVE_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix) + 1:
            return LX.TAG_ADJ
    for suffix in _VERB_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix) + 1:
            return LX.TAG_VERB_BASE
    for suffix in _NOUN_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix) + 1:
            return LX.TAG_NOUN_SINGULAR
    if low.endswith("s") and len(low) > 3:
        return LX.TAG_NOUN_PLURAL
    return LX.TAG_NOUN_SINGULAR


def _rule_based_tag(tokens: Sequence[str]) -> List[TaggedWord]:
    tags: List[str] = []
    for word in tokens:
        low = word.lower()
        if low in LX.POS_LEXICON:
            tags.append(LX.POS_LEXICON[low])
        else:
            tags.append(_initial_guess(word))

    def prev(i: int) -> Optional[str]:
        return tags[i - 1] if i > 0 else None

    def prev_word(i: int) -> Optional[str]:
        return tokens[i - 1].lower() if i > 0 else None

    for i, word in enumerate(tokens):
        # punctuation is never re-tagged
        if not word[0].isalnum():
            tags[i] = LX.TAG_PUNCT
            continue
        # closed-class words keep their lexicon tag
        if tags[i] in LX._PROTECTED_TAGS:
            continue

        lw = word.lower()

        # 1. capitalised token following a noun -> proper noun
        if i > 0 and prev(i) in LX.NOUN_TAGS and word[0].isupper() and lw not in LX.POS_LEXICON:
            if tags[i] in (LX.TAG_NOUN_SINGULAR, LX.TAG_NOUN_PLURAL, LX.TAG_PROPER_SINGULAR):
                tags[i] = LX.TAG_PROPER_SINGULAR

        # 2. determiner followed by adjective/adverb/verb -> noun
        if prev(i) == LX.TAG_DET and tags[i] in (LX.TAG_ADJ, LX.TAG_ADV, LX.TAG_VERB_BASE):
            tags[i] = LX.TAG_NOUN_SINGULAR

        # 3. possessive pronoun followed by a verb -> noun
        if prev(i) == LX.TAG_POSS_PRONOUN and tags[i] in LX.VERB_TAGS:
            tags[i] = LX.TAG_NOUN_SINGULAR

        # 4. modal / "to" followed by a content word -> base verb
        if prev(i) in (LX.TAG_MODAL, LX.TAG_TO) and lw not in LX.PERSONAL_PRONOUNS \
                and tags[i] not in LX.VERB_TAGS and tags[i] != LX.TAG_NOUN_SINGULAR:
            tags[i] = LX.TAG_VERB_BASE

        # 5. "have/has/had" + X -> past participle
        if prev(i) in ("VBP", "VBZ", "VBD") and prev_word(i) in {"have", "has", "had", "having"}:
            if tags[i] in (LX.TAG_VERB_BASE, LX.TAG_VERB_PAST, LX.TAG_NOUN_SINGULAR):
                tags[i] = LX.TAG_VERB_PAST_PART

        # 6. adverb right after a determiner is really a noun
        if tags[i] == LX.TAG_ADV and prev(i) == LX.TAG_DET:
            tags[i] = LX.TAG_NOUN_SINGULAR

        # 7. "than" + adjective -> comparative
        if prev(i) == "IN" and prev_word(i) == "than" and lw in LX.ADJECTIVES:
            tags[i] = LX.TAG_ADJ_COMP

        # 8. noun + noun compound
        if prev(i) in LX.NOUN_TAGS and tags[i] == LX.TAG_VERB_BASE and lw in LX.NOUNS:
            tags[i] = LX.TAG_NOUN_SINGULAR

        # 9. wh-word starting a clause
        if i == 0 and lw in LX.WH_PRONOUNS:
            tags[i] = LX.TAG_WH_PRONOUN
        if i == 0 and lw in LX.WH_ADVERBS:
            tags[i] = LX.TAG_WH_ADV

        # 10. cardinal followed by a singular noun -> plural
        if prev(i) == LX.TAG_CARDINAL and tags[i] == LX.TAG_NOUN_SINGULAR:
            tags[i] = LX.TAG_NOUN_PLURAL

        # 11. gerund after a proper noun is usually a noun
        if prev(i) in (LX.TAG_PROPER_SINGULAR, LX.TAG_PROPER_PLURAL) and tags[i] == LX.TAG_VERB_GERUND:
            tags[i] = LX.TAG_NOUN_SINGULAR

        # 12. coordinated verbs: "X and Y" where X is a verb -> Y is a verb
        if prev(i) == LX.TAG_CONJ and i >= 2 and tags[i - 2] in LX.VERB_TAGS:
            if tags[i] in (LX.TAG_NOUN_SINGULAR, LX.TAG_VERB_BASE):
                tags[i] = LX.TAG_VERB_BASE

        # 13. "not/never" + unknown noun -> verb (adjectives are preserved)
        if prev(i) == "RB" and prev_word(i) in {"not", "never", "n't"}:
            if tags[i] == LX.TAG_NOUN_SINGULAR and lw not in LX.ADJECTIVES:
                tags[i] = LX.TAG_VERB_BASE

    # ---- final pass: shape driven fixes --------------------------------- #
    for i, word in enumerate(tokens):
        if not word[0].isalnum():
            tags[i] = LX.TAG_PUNCT
            continue
        lw = word.lower()
        if lw in LX.POS_LEXICON:
            continue
        if word.isdigit() and tags[i] != LX.TAG_CARDINAL:
            tags[i] = LX.TAG_CARDINAL
        if i > 0 and tokens[i - 1][:1].isupper() and word[:1].isupper():
            if tags[i] in (LX.TAG_NOUN_SINGULAR, LX.TAG_NOUN_PLURAL):
                tags[i] = LX.TAG_PROPER_SINGULAR

    return [
        TaggedWord(
            text=word, tag=tag, lemma=lemmatize(word.lower(), tag), index=i,
            is_stop=word.lower() in LX.STOP_WORDS, shape=_shape(word),
        )
        for i, (word, tag) in enumerate(zip(tokens, tags))
    ]


def tag_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Token-level tagging accuracy."""
    if not gold:
        return 0.0
    n = min(len(predicted), len(gold))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if predicted[i] == gold[i]) / n
