"""Subtitle-line scorers: the VSI dense encoder, alternative encoders and BM25.
"""
import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np

VSI_STOPWORDS = {"a", "an", "the", "is", "are", "in", "on", "at", "of", "and", "or", "with"}


def vsi_filter(text: str) -> str:
    return " ".join(w for w in text.lower().split() if w not in VSI_STOPWORDS)


class DenseScorer:
    """Cosine similarity from a sentence-transformers bi-encoder."""

    def __init__(self, model_name: str, device: str = "cpu", query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self._cache: Dict[str, np.ndarray] = {}

    def _encode(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            embs = self.model.encode(missing, batch_size=128, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
            self._cache.update(zip(missing, embs))
        return np.stack([self._cache[t] for t in texts])

    def score(self, query: str, lines: List[str]) -> np.ndarray:
        q = self.model.encode(self.query_prefix + vsi_filter(query), normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return self._encode([vsi_filter(t) for t in lines]) @ q


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in VSI_STOPWORDS]


class BM25Scorer:
    """Okapi BM25 with the subtitle lines of one video as the collection."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def score(self, query: str, lines: List[str]) -> np.ndarray:
        docs = [tokenize(t) for t in lines]
        n = len(docs)
        avgdl = max(1e-9, sum(map(len, docs)) / max(n, 1))
        df = Counter(w for d in docs for w in set(d))
        scores = np.zeros(n)
        q_terms = tokenize(query)
        for i, d in enumerate(docs):
            tf = Counter(d)
            s = 0.0
            for w in q_terms:
                if w not in tf:
                    continue
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(d) / avgdl))
            scores[i] = s
        return scores


def to_unit_range(scores: np.ndarray) -> np.ndarray:
    """Divide unbounded scores (BM25) by their per-video maximum."""
    m = scores.max() if len(scores) else 0.0
    return scores / m if m > 0 else np.zeros_like(scores)


def rrf(score_lists: List[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion (Cormack et al., 2009) over line rankings."""
    out = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s))
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + ranks)
    return out
