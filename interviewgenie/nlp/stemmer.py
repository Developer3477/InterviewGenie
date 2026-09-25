"""A faithful implementation of the Porter (1980) stemming algorithm.

Used for keyword extraction, index terms and lexical matching.  The classic
test vectors are embedded in ``tests/test_nlp.py``.
"""

from __future__ import annotations

import re
from typing import List

_VOWELS = "aeiou"


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's *m*: the number of VC sequences in ``stem``."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        if _is_consonant(stem, i):
            if prev_vowel:
                m += 1
            prev_vowel = False
        else:
            prev_vowel = True
    return m


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and _is_consonant(word, len(word) - 1)
    )


def _ends_cvc(word: str) -> bool:
    """True when the last three letters are consonant-vowel-consonant (not w/x/y)."""
    if len(word) < 3:
        return False
    if not (_is_consonant(word, len(word) - 3) and not _is_consonant(word, len(word) - 2)
            and _is_consonant(word, len(word) - 1)):
        return False
    return word[-1] not in "wxy"


class PorterStemmer:
    """The Porter stemming algorithm (step 1a-5b)."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _replace(word: str, suffix: str, replacement: str, condition: bool) -> str | None:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if condition(stem):
                return stem + replacement
        return None

    # -- steps ------------------------------------------------------------- #
    def _step1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s") and len(word) > 1:
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        if word.endswith("eed"):
            return word[:-1] if _measure(word[:-3]) > 0 else word
        for suffix, repl in (("ing", ""), ("ed", "")):
            if word.endswith(suffix) and _contains_vowel(word[: -len(suffix)]):
                stem = word[: -len(suffix)]
                if stem.endswith(("at", "bl", "iz")):
                    return stem + "e"
                if _ends_double_consonant(stem) and stem[-1] not in "lsz":
                    return stem[:-1]
                if _measure(stem) == 1 and _ends_cvc(stem):
                    return stem + "e"
                return stem
        return word

    def _step1c(self, word: str) -> str:
        if word.endswith("y") and _contains_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    def _step2(self, word: str) -> str:
        pairs = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
        ]
        for suffix, repl in pairs:
            out = self._replace(word, suffix, repl, lambda s: _measure(s) > 0)
            if out is not None:
                return out
        return word

    def _step3(self, word: str) -> str:
        pairs = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
            ("ical", "ic"), ("ful", ""), ("ness", ""),
        ]
        for suffix, repl in pairs:
            out = self._replace(word, suffix, repl, lambda s: _measure(s) > 0)
            if out is not None:
                return out
        return word

    def _step4(self, word: str) -> str:
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ]
        # "ion" only strips when preceded by s or t
        if word.endswith("ion"):
            stem = word[:-3]
            if _measure(stem) > 1 and stem.endswith(("s", "t")):
                return stem
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if _measure(stem) > 1:
                    return stem
        return word

    def _step5a(self, word: str) -> str:
        if word.endswith("e"):
            stem = word[:-1]
            m = _measure(stem)
            if m > 1 or (m == 1 and not _ends_cvc(stem)):
                return stem
        return word

    def _step5b(self, word: str) -> str:
        if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
            return word[:-1]
        return word

    # -- public ------------------------------------------------------------ #
    def stem(self, word: str) -> str:
        """Return the Porter stem of a single (already lower-cased) word."""
        if not word or not word.isalpha():
            return word
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        if len(word) <= 2:
            self._cache[word] = word
            return word
        out = word
        for step in (self._step1a, self._step1b, self._step1c, self._step2,
                     self._step3, self._step4, self._step5a, self._step5b):
            out = step(out)
        self._cache[word] = out
        return out

    def stem_tokens(self, words: List[str]) -> List[str]:
        return [self.stem(w) for w in words]


_DEFAULT_STEMMER = PorterStemmer()


def stem(word: str) -> str:
    """Module-level convenience wrapper."""
    return _DEFAULT_STEMMER.stem(word)


# --------------------------------------------------------------------------- #
# Light-weight lemmatisation
# --------------------------------------------------------------------------- #
#: irregular inflections -> lemma
IRREGULAR: dict[str, str] = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "being": "be", "has": "have", "had": "have", "having": "have", "does": "do",
    "did": "do", "doing": "do", "done": "do", "goes": "go", "went": "go",
    "gone": "go", "made": "make", "makes": "make", "said": "say", "says": "say",
    "ran": "run", "runs": "run", "led": "lead", "leads": "lead", "built": "build",
    "builds": "build", "built": "build", "saw": "see", "seen": "see", "sees": "see",
    "took": "take", "taken": "take", "takes": "take", "got": "get", "gotten": "get",
    "wrote": "write", "written": "write", "writes": "write", "taught": "teach",
    "teaches": "teach", "thought": "think", "thinks": "think", "found": "find",
    "finds": "find", "gave": "give", "given": "give", "gives": "give",
    "knew": "know", "known": "know", "knows": "know", "grew": "grow",
    "grown": "grow", "grows": "grow", "chose": "choose", "chosen": "choose",
    "brought": "bring", "broughts": "bring", "spent": "spend", "spends": "spend",
    "felt": "feel", "feels": "feel", "kept": "keep", "keeps": "keep",
    "left": "leave", "leaves": "leave", "met": "meet", "meets": "meet",
    "paid": "pay", "pays": "pay", "sold": "sell", "sells": "sell",
    "sent": "send", "sends": "send", "won": "win", "wins": "win",
    "understood": "understand", "children": "child", "men": "man",
    "women": "woman", "people": "person", "feet": "foot", "teeth": "tooth",
    "data": "data", "analyses": "analysis", "theses": "thesis", "crises": "crisis",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
}

#: suffixes ordered so that the longest match wins
_LEMMA_SUFFIXES = (
    ("ies", "y"), ("ves", "f"), ("ses", "s"), ("xes", "x"), ("ches", "ch"),
    ("shes", "sh"), ("men", "man"), ("women", "woman"),
)


def lemmatize(word: str, pos: str | None = None) -> str:
    """Rule-based lemmatiser with an irregular-form dictionary."""
    if not word:
        return word
    low = word.lower()
    if low in IRREGULAR:
        return IRREGULAR[low]
    if len(low) <= 3:
        return low
    for suffix, repl in _LEMMA_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix):
            stem_word = low[: -len(suffix)] + repl
            break
    else:
        if low.endswith("s") and not low.endswith(("ss", "us", "is")) and len(low) > 3:
            stem_word = low[:-1]
        else:
            stem_word = low
    # verbs ending in -ing / -ed
    if low.endswith("ing") and len(low) > 5:
        stem_word = low[:-3]
        if stem_word.endswith(stem_word[-1:] * 2) and len(stem_word) > 3:
            stem_word = stem_word[:-1]
        if stem_word + "e" in _KNOWN_BASE or len(stem_word) > 3:
            return stem_word
    if low.endswith("ed") and len(low) > 4:
        stem_word = low[:-2]
        if stem_word.endswith(stem_word[-1:] * 2) and len(stem_word) > 3:
            stem_word = stem_word[:-1]
        return stem_word
    return stem_word


#: a small vocabulary used to decide whether a stripped verb form is plausible
_KNOWN_BASE = {
    "make", "take", "build", "design", "write", "lead", "manage", "develop",
    "test", "deploy", "ship", "run", "use", "work", "learn", "teach", "plan",
    "review", "improve", "reduce", "increase", "handle", "support", "deliver",
}
