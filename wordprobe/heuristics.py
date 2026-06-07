"""Low-level scoring math helpers used by future heuristic modules."""

from math import log2

from .constraints import WORD_SIZE

__all__ = ("bin_entropy", )


def bin_entropy(p: float) -> float:
    """Return the binary entropy of probability `p`"""
    if p <= 0 or p >= 1:
        return 0.0

    return -(p * log2(p) + (np := 1 - p) * log2(np))
