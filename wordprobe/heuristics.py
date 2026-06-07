"""Low-level scoring math helpers used by future heuristic modules."""
import string

from math import log2

from typing import Sequence

from .constraints import WORD_SIZE

__all__ = ("bin_entropy", )


def bin_entropy(p: float) -> float:
    """Return the binary entropy of probability `p`"""
    if p <= 0 or p >= 1:
        return 0.0

    return -(p * log2(p) + (np := 1 - p) * log2(np))


def map_token_scores(
        candidates: Sequence[str],
        exclude_tokens: set[str] | None = None,
        entropize: bool = True,
        ) -> dict[str, float]:
    """
    Mapping of token occurence probabilities. Optionally convert to binary
    entropy or probability.
    """
    if exclude_tokens is None:
        exclude_tokens = set()

    total = len(candidates)
    tokens = string.ascii_lowercase
    counter = {t: 0 for t in tokens}
    for word in candidates:
        for token in set(word) - exclude_tokens:
            counter[token] += 1

    if entropize:
        return {k: bin_entropy(v / total) for k, v in counter.items()}
    return {k: (v / total) for k, v in counter.items()}


def map_token_index_scores(
        candidates: Sequence[str],
        exclude_indices: set[str] | None = None,
        entropize: bool = True,
        ) -> dict[str, list[float]]:
    """
    Mapping of token index probabilities. Optionally convert to binary
    entropy or probability.
    """
    if exclude_indices is None:
        exclude_indices = set()

    total = len(candidates)
    tokens = string.ascii_lowercase
    counter = {t: [0] * WORD_SIZE for t in tokens}
    for word in candidates:
        for idx, token in enumerate(word):
            if idx not in exclude_indices:
                counter[token][idx] += 1

    if entropize:
        return {
            k: [bin_entropy(r / total) for r in rates]
            for k, rates in counter.items()
        }
    return {k: [(r / total) for r in rates] for k, rates in counter.items()}


def word_scores(
        candidates: Sequence[str],
        exclude_tokens: set[str] | None = None,
        entropize: bool = True,
        ) -> list[float]:
    mapping = map_token_scores(candidates, exclude_tokens, entropize)
    scores = []
    for word in candidates:
        score = sum(mapping[token] for token in set(word))
        score /= entropize or len(word)
        scores.append(score)

    return scores


def word_index_scores(
        candidates: Sequence[str],
        exclude_indices: set[int] | None = None,
        entropize: bool = True,
        ) -> list[float]:
    mapping = map_token_index_scores(candidates, exclude_indices, entropize)
    scores = []
    for word in candidates:
        score = sum(mapping[token][idx] for idx, token in enumerate(word))
        score /= entropize or len(word)
        scores.append(score)

    return scores


def composite_scores(
        candidates: Sequence[str],
        exclude_tokens: set[str] | None = None,
        exclude_indices: set[int] | None = None,
        entropize: bool = True,
        ) -> float:
    x_mapping = map_token_scores(candidates, exclude_tokens, entropize)
    y_mapping = map_token_index_scores(candidates, exclude_indices, entropize)
    scores = []
    for word in candidates:
        x = sum(x_mapping[token] for token in set(word))
        y = sum(y_mapping[token][idx] for idx, token in enumerate(word))
        score = (x + y) / 2
        scores.append(score)

    return scores
