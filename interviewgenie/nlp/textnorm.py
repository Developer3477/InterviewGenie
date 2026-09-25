"""Text normalisation, sentence segmentation and tokenisation.

These primitives are intentionally dependency free (no NLTK/spaCy required) but
are written to be swappable: :class:`Tokenizer` and :func:`split_sentences` are
the only entry points used elsewhere in the codebase.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: common abbreviations that must not terminate a sentence
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc",
    "ltd", "co", "corp", "dept", "univ", "eg", "ie", "al", "fig", "no",
    "vol", "pp", "ed", "eds", "est", "approx", "apt", "gov", "sen", "rep",
    "gen", "col", "lt", "capt", "rev", "hon", "pres", "ceo", "cto", "vp",
}

#: contraction expansion table
CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "n't": " not", "'ll": " will", "'re": " are", "'ve": " have",
    "'m": " am", "'d": " would", "let's": "let us", "that's": "that is",
    "it's": "it is", "i'm": "i am", "i've": "i have", "i'd": "i would",
    "i'll": "i will", "you're": "you are", "you've": "you have",
    "we're": "we are", "we've": "we have", "they're": "they are",
    "what's": "what is", "who's": "who is", "how's": "how is",
    "where's": "where is", "there's": "there is", "here's": "here is",
}

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,]\d+)*%?|[^\sA-Za-z\d]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_WS_RE = re.compile(r"\s+")
_PUNCT_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalize(text: str, *, expand_contractions: bool = True, lowercase: bool = False) -> str:
    """Unicode-normalise, tidy whitespace and optionally expand contractions."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = _WS_RE.sub(" ", text).strip()
    if expand_contractions:
        text = expand_contraction(text)
    if lowercase:
        text = text.lower()
    return text


def expand_contraction(text: str) -> str:
    """Expand English contractions in place (case preserving)."""
    if not text:
        return text

    def _repl(match: re.Match) -> str:
        word = match.group(0)
        low = word.lower()
        if low in CONTRACTIONS:
            expansion = CONTRACTIONS[low]
        else:
            suffix = low[-3:] if len(low) >= 3 else low
            if suffix in CONTRACTIONS:
                expansion = word[: len(word) - 3] + " " + CONTRACTIONS[suffix]
            else:
                return word
        if word[0].isupper():
            expansion = expansion[0].upper() + expansion[1:]
        return expansion

    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", _repl, text)


def strip_punctuation(token: str) -> str:
    """Remove leading/trailing punctuation from a token."""
    return _PUNCT_STRIP_RE.sub("", token)


def is_acronym(token: str) -> bool:
    return len(token) >= 2 and token.isupper() and token.isalpha()


# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> List[str]:
    """Split text into sentences, honouring abbreviations and ellipses."""
    if not text or not text.strip():
        return []
    text = _WS_RE.sub(" ", text.strip())
    raw_parts = _SENT_SPLIT_RE.split(text)
    sentences: List[str] = []
    buffer = ""
    for part in raw_parts:
        if not part:
            continue
        candidate = f"{buffer} {part}".strip() if buffer else part
        tail = candidate.rstrip("\"')\]]").split(" ")[-1].rstrip(".").lower()
        # keep appending when the "sentence" ends with a known abbreviation
        if tail in ABBREVIATIONS and not candidate.rstrip().endswith(("!", "?")):
            buffer = candidate
            continue
        sentences.append(candidate.strip())
        buffer = ""
    if buffer:
        sentences.append(buffer.strip())
    return [s for s in sentences if s]


def split_questions(text: str) -> List[str]:
    """Return the interrogative sub-spans of a (possibly compound) utterance."""
    out: List[str] = []
    for sentence in split_sentences(text):
        if sentence.rstrip().endswith("?"):
            out.append(sentence.rstrip("? ").strip())
        elif "?" in sentence:
            for piece in sentence.split("?"):
                if piece.strip():
                    out.append(piece.strip())
    return out


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
class Tokenizer:
    """Regex tokeniser with optional POS-aware metadata.

    ``keep_punct`` controls whether punctuation is emitted as its own token.
    The tokeniser is stateless and thread safe.
    """

    __slots__ = ("keep_punct", "_word_re")

    def __init__(self, keep_punct: bool = True) -> None:
        self.keep_punct = keep_punct
        self._word_re = _WORD_RE

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = self._word_re.findall(text)
        if not self.keep_punct:
            tokens = [t for t in tokens if t[0].isalnum()]
        return tokens

    def spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Return ``(token, start, end)`` triples with character offsets."""
        return [(m.group(0), m.start(), m.end()) for m in self._word_re.finditer(text)]

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


_DEFAULT_TOKENIZER = Tokenizer()


def tokenize(text: str, keep_punct: bool = True) -> List[str]:
    return Tokenizer(keep_punct=keep_punct).tokenize(text)


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Return the n-grams of ``tokens`` as tuples."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def char_ngrams(text: str, n: int = 3) -> List[str]:
    """Character n-grams of the word-boundary padded text (robust to typos)."""
    padded = f" {text.strip()} "
    if len(padded) <= n:
        return [padded] if padded.strip() else []
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity between two collections of tokens."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def edit_distance(a: str, b: str, cap: int = 64) -> int:
    """Levenshtein distance with early exit."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """Normalised string similarity in ``[0, 1]`` based on edit distance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = edit_distance(a.lower(), b.lower())
    return 1.0 - dist / max(len(a), len(b))
