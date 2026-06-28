"""Booleanisation of text into Tsetlin literals.

A Tsetlin Machine consumes a Boolean feature vector.  For the content guard we
turn a piece of text (a prompt, a tool argument, a model output) into a fixed
Boolean vector of *token-presence* features.  The vocabulary is selected on the
TRAINING split only, never the test split, using a simple, transparent
class-discriminative score so the resulting features (and the clauses the TM
learns over them) are human-readable: every feature is literally "the token
``ignore`` appears in the text".

No embeddings, no neural tokeniser: an auditor can read exactly which words a
TRACE content clause keys on.  This is the property neural guardrails give up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, keep unigrams."""
    return _TOKEN_RE.findall(text.lower())


def _with_bigrams(tokens: Sequence[str]) -> List[str]:
    grams = list(tokens)
    grams += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return grams


@dataclass
class BooleanVocabulizer:
    """Fit a discriminative token vocabulary; transform text -> Boolean vector.

    Parameters
    ----------
    max_features : int
        Vocabulary size (= number of Boolean features).
    use_bigrams : bool
        Also consider adjacent token bigrams (``ignore_previous``).
    min_df : int
        A token must appear in at least this many training documents.
    """

    max_features: int = 256
    use_bigrams: bool = True
    min_df: int = 2
    vocab_: List[str] = field(default_factory=list)
    index_: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.vocab_)

    def _doc_terms(self, text: str) -> set:
        toks = tokenize(text)
        if self.use_bigrams:
            toks = _with_bigrams(toks)
        return set(toks)

    def fit(self, texts: Sequence[str], labels: Sequence[int] | None = None) -> "BooleanVocabulizer":
        """Select the vocabulary.  If ``labels`` are given (binary), rank terms
        by a class-discriminative score; otherwise rank by document frequency."""
        docs = [self._doc_terms(t) for t in texts]
        n = len(docs)
        df: dict = {}
        for d in docs:
            for term in d:
                df[term] = df.get(term, 0) + 1
        terms = [t for t, c in df.items() if c >= self.min_df]

        if labels is not None:
            y = np.asarray(labels, dtype=np.int8)
            n_pos = max(1, int((y == 1).sum()))
            n_neg = max(1, int((y == 0).sum()))
            pos_df: dict = {}
            neg_df: dict = {}
            for d, yi in zip(docs, y):
                tgt = pos_df if yi == 1 else neg_df
                for term in d:
                    tgt[term] = tgt.get(term, 0) + 1
            # Discriminative score: |P(term|pos) - P(term|neg)|, with a small
            # frequency prior so ultra-rare terms do not dominate.
            def score(t: str) -> float:
                p = pos_df.get(t, 0) / n_pos
                q = neg_df.get(t, 0) / n_neg
                freq_prior = min(1.0, df[t] / n)
                return abs(p - q) * (0.5 + 0.5 * freq_prior)
            terms.sort(key=score, reverse=True)
        else:
            terms.sort(key=lambda t: df[t], reverse=True)

        self.vocab_ = terms[: self.max_features]
        self.index_ = {t: i for i, t in enumerate(self.vocab_)}
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        X = np.zeros((len(texts), self.n_features), dtype=np.int8)
        for i, text in enumerate(texts):
            for term in self._doc_terms(text):
                j = self.index_.get(term)
                if j is not None:
                    X[i, j] = 1
        return X

    def fit_transform(self, texts, labels=None) -> np.ndarray:
        return self.fit(texts, labels).transform(texts)

    def feature_name(self, literal_index: int) -> str:
        """Map a literal index (0..2F-1) back to a readable term.

        Literals 0..F-1 are positive (``token``); F..2F-1 are negated
        (``NOT token``)."""
        F = self.n_features
        if literal_index < F:
            return self.vocab_[literal_index]
        return f"NOT {self.vocab_[literal_index - F]}"
