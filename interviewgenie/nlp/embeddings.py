"""Vector space retrieval: TF-IDF, LSA (truncated SVD), BM25 and cosine.

Pure Python / standard library only.  Latent Semantic Analysis is implemented
with power iteration on the term-document matrix, which is plenty accurate for
the few-thousand-document corpora InterviewGenie deals with and avoids a NumPy
dependency.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..logging import get_logger
from .stemmer import PorterStemmer
from .textnorm import tokenize

LOG = get_logger("nlp.embeddings")

_STEMMER = PorterStemmer()


def _terms(text: str, *, use_stem: bool = True, min_len: int = 2) -> List[str]:
    out = []
    for tok in tokenize(text, keep_punct=False):
        low = tok.lower()
        if len(low) < min_len:
            continue
        out.append(_STEMMER.stem(low) if use_stem else low)
    return out


# --------------------------------------------------------------------------- #
# Cosine similarity
# --------------------------------------------------------------------------- #
def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two dense vectors."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        av = a[i]
        bv = b[i]
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return [0.0] * len(vector)
    return [v / norm for v in vector]


def sparse_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse term->weight mappings."""
    if not a or not b:
        return 0.0
    if len(b) < len(a):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# TF-IDF index
# --------------------------------------------------------------------------- #
@dataclass
class TfidfIndex:
    """A TF-IDF index with LSA projection and BM25 scoring."""

    use_stem: bool = True
    dim: int = 48
    documents: List[str] = field(default_factory=list)
    _doc_terms: List[List[str]] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _vocab: Dict[str, int] = field(default_factory=dict, repr=False)
    _idf: Dict[str, float] = field(default_factory=dict, repr=False)
    _vectors: List[Dict[str, float]] = field(default_factory=list, repr=False)
    _dense: List[List[float]] = field(default_factory=list, repr=False)
    _lsa_basis: List[List[float]] = field(default_factory=list, repr=False)
    _avg_len: float = 0.0
    _fitted: bool = False

    # -- fitting ----------------------------------------------------------- #
    def fit(self, documents: Sequence[str]) -> "TfidfIndex":
        self.documents = list(documents)
        self._doc_terms = [_terms(d, use_stem=self.use_stem) for d in self.documents]
        self._df = Counter()
        for terms in self._doc_terms:
            for term in set(terms):
                self._df[term] += 1
        n = max(1, len(self.documents))
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in self._df.items()}
        self._vocab = {t: i for i, t in enumerate(sorted(self._idf))}
        self._vectors = [self._tfidf(t) for t in self._doc_terms]
        self._avg_len = (sum(len(t) for t in self._doc_terms) / n) if n else 0.0
        self._build_lsa()
        self._fitted = True
        LOG.debug("tfidf index fitted", context={"docs": n, "vocab": len(self._vocab)})
        return self

    def _tfidf(self, terms: Sequence[str]) -> Dict[str, float]:
        counts = Counter(terms)
        vec: Dict[str, float] = {}
        for term, count in counts.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            vec[term] = (1.0 + math.log(count)) * idf
        return vec

    # -- LSA --------------------------------------------------------------- #
    def _build_lsa(self) -> None:
        """Project documents onto the top-``dim`` right singular vectors.

        Power iteration is run on the document-document Gram matrix, which is
        built sparsely (one outer product per non-zero term) so the cost is
        ``O(nnz * n_docs)`` rather than ``O(n_docs^2 * n_terms)``.
        """
        self._lsa_basis = []
        self._lsa_vocab: List[str] = []
        self._lsa_vindex: Dict[str, int] = {}
        if not self._vectors or self.dim <= 0:
            return

        vocab = sorted(self._vocab, key=lambda t: self._vocab[t])
        v_index = {t: i for i, t in enumerate(vocab)}
        m = len(vocab)
        n = len(self._vectors)
        # sparse column per document: {term_index: weight}
        columns: List[Dict[int, float]] = []
        for vec in self._vectors:
            column = {v_index[t]: w for t, w in vec.items() if t in v_index}
            norm = math.sqrt(sum(w * w for w in column.values()))
            if norm > 0:
                column = {k: w / norm for k, w in column.items()}
            columns.append(column)

        # Gram matrix (n x n), accumulated sparsely: one outer product per term
        postings: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for d, column in enumerate(columns):
            for t, w in column.items():
                postings[t].append((d, w))
        gram = [[0.0] * n for _ in range(n)]
        for entries in postings.values():
            for a in range(len(entries)):
                da, wa = entries[a]
                for b in range(a, len(entries)):
                    db, wb = entries[b]
                    gram[da][db] += wa * wb
                    if da != db:
                        gram[db][da] += wa * wb

        k = min(self.dim, n)
        rng = random.Random(1234)
        work = [row[:] for row in gram]
        basis: List[List[float]] = []
        for _ in range(k):
            if all(abs(v) < 1e-9 for row in work for v in row):
                break
            v = normalize([rng.random() + 0.1 for _ in range(n)])
            previous = 0.0
            for _iteration in range(15):
                nv = [sum(work[i][j] * v[j] for j in range(n)) for i in range(n)]
                norm = math.sqrt(sum(x * x for x in nv))
                if norm < 1e-12:
                    break
                v = [x / norm for x in nv]
                if abs(norm - previous) < 1e-6 * max(1.0, norm):
                    break
                previous = norm
            # map the document-space singular vector back into term space
            term_vec = [0.0] * m
            for d in range(n):
                coeff = v[d]
                if coeff == 0.0:
                    continue
                for t, w in columns[d].items():
                    term_vec[t] += coeff * w
            term_vec = normalize(term_vec)
            if math.sqrt(sum(x * x for x in term_vec)) < 1e-9:
                continue
            basis.append(term_vec)
            # deflate
            wv = [sum(work[i][j] * v[j] for j in range(n)) for i in range(n)]
            lam = sum(v[i] * wv[i] for i in range(n))
            for i in range(n):
                if v[i] == 0.0:
                    continue
                for j in range(n):
                    work[i][j] -= lam * v[i] * v[j]

        self._lsa_basis = basis
        self._lsa_vocab = vocab
        self._lsa_vindex = v_index
        self._dense = [
            [sum(column.get(t, 0.0) * b[t] for t in column) for b in basis]
            for column in columns
        ]

    def _to_dense(self, vec: Dict[str, float], v_index: Dict[str, int], m: int) -> List[float]:
        out = [0.0] * m
        for term, weight in vec.items():
            idx = v_index.get(term)
            if idx is not None:
                out[idx] = weight
        return out

    # -- encoding ---------------------------------------------------------- #
    def transform(self, text: str) -> Dict[str, float]:
        """Sparse TF-IDF vector for ``text`` (OOV terms are dropped)."""
        return self._tfidf(_terms(text, use_stem=self.use_stem))

    def transform_dense(self, text: str) -> List[float]:
        """Dense LSA vector for ``text`` (dimension ``self.dim``)."""
        if not self._lsa_basis:
            return []
        vec = self.transform(text)
        column = {self._lsa_vindex[t]: w for t, w in vec.items() if t in self._lsa_vindex}
        return [sum(w * b[t] for t, w in column.items()) for b in self._lsa_basis]

    def embed(self, text: str) -> List[float]:
        """L2-normalised dense embedding, ready for cosine similarity."""
        return normalize(self.transform_dense(text))

    # -- retrieval --------------------------------------------------------- #
    def search(self, query: str, k: int = 5, use_lsa: bool = True) -> List[Tuple[int, float]]:
        """Return ``[(doc_index, score)]`` best matches for ``query``."""
        if not self.documents:
            return []
        if use_lsa and self._lsa_basis:
            q = self.embed(query)
            scored = [(i, cosine(q, normalize(doc))) for i, doc in enumerate(self._dense)]
        else:
            qv = self.transform(query)
            scored = [(i, sparse_cosine(qv, v)) for i, v in enumerate(self._vectors)]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 1e-9]

    def bm25(self, query: str, k: int = 5, k1: float = 1.4, b: float = 0.72) -> List[Tuple[int, float]]:
        """Okapi BM25 ranking over the indexed documents."""
        if not self.documents:
            return []
        q_terms = _terms(query, use_stem=self.use_stem)
        scores: List[Tuple[int, float]] = []
        n = len(self.documents)
        for i, terms in enumerate(self._doc_terms):
            counts = Counter(terms)
            dl = len(terms) or 1
            score = 0.0
            for term in set(q_terms):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * dl / (self._avg_len or 1))
                score += idf * (tf * (k1 + 1)) / denom
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda kv: kv[1], reverse=True)
        return scores[:k]

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self._vocab, key=lambda t: self._vocab[t])

    def stats(self) -> Dict[str, object]:
        return {
            "documents": len(self.documents),
            "vocabulary": len(self._vocab),
            "lsa_dim": len(self._lsa_basis),
            "avg_doc_len": round(self._avg_len, 2),
        }


# --------------------------------------------------------------------------- #
# Keyword extraction (TextRank)
# --------------------------------------------------------------------------- #
def extract_keywords(text: str, top_k: int = 12, window: int = 4,
                     use_stem: bool = True) -> List[Tuple[str, float]]:
    """TextRank keyword extraction over a co-occurrence graph."""
    from . import lexicon as LX

    words = _terms(text, use_stem=use_stem, min_len=3)
    words = [w for w in words if w not in LX.STOP_WORDS]
    if not words:
        return []
    if len(words) == 1:
        return [(words[0], 1.0)]

    # build co-occurrence graph
    edges: Dict[str, Counter] = defaultdict(Counter)
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + window, len(words))):
            if word == words[j]:
                continue
            edges[word][words[j]] += 1
            edges[words[j]][word] += 1

    scores = {w: 1.0 for w in set(words)}
    damping = 0.85
    for _ in range(30):
        new_scores: Dict[str, float] = {}
        for word in scores:
            rank = 0.0
            for neighbour, weight in edges.get(word, {}).items():
                out_sum = sum(edges[neighbour].values()) or 1.0
                rank += (weight / out_sum) * scores[neighbour]
            new_scores[word] = (1 - damping) + damping * rank
        total = sum(new_scores.values()) or 1.0
        scores = {w: v / total for w, v in new_scores.items()}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(w, round(s, 5)) for w, s in ranked[:top_k]]
