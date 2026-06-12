"""
NumPy-backed scoring, rate, and ranking helpers.

This module is the scoring boundary between the word-oriented engine and the
matrix-oriented heuristic implementation.

Core terminology:
    candidates:
        Current possible answers. These are always the basis for rates.

    pool:
        Words being scored/ranked. If None, pool defaults to candidates.

    scores:
        NumPy array aligned to pool.

Important invariant:
    Rates/weights come from candidates.
    Scores align to pool.
"""
import numpy as np

from .constants import ALPHABET_SIZE, TOKEN_OFFSET, WORDSIZE
from .transforms import tokenize

__all__ = ()
