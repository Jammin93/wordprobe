from __future__ import annotations

import string

import numpy as np

from collections import Counter, defaultdict
from functools import cached_property

from .abstract import WordList, NGramArray
from .constants import ALPHABET_SIZE, NULL, TokenFlags, WORDSIZE
from .transforms import (
    bin_entropy,
    compose_token_mask,
    encode_token_set,
    encode_tokens,
    token_presence,
)


class State:
    __slots__ = ("_fixed", "_banned", "_min_counts", "_max_counts")

    def __init__(self):
        self._fixed = np.full(WORDSIZE, NULL, dtype=int)
        self._banned = np.zeros((ALPHABET_SIZE, WORDSIZE), dtype=bool)
        self._min_counts = np.zeros(ALPHABET_SIZE, dtype=int)
        self._max_counts = np.full(ALPHABET_SIZE, WORDSIZE, dtype=int)

    @property
    def fixed(self):
        return self._fixed.copy()

    @property
    def free(self):
        return set(np.flatnonzero(self._fixed == NULL))

    @property
    def known(self):
        return set(np.flatnonzero(self._min_counts > 0))

    def update(self, guess, mask):
        fixed = np.fromiter(
            (flag == TokenFlags.FIXED for flag in mask),
            dtype=bool,
            count=WORDSIZE,
        )
        floating = np.fromiter(
            (flag == TokenFlags.FLOAT for flag in mask),
            dtype=bool,
            count=WORDSIZE,
        )
        missed = np.fromiter(
            (flag == TokenFlags.MISS for flag in mask),
            dtype=bool,
            count=WORDSIZE,
        )
        present = fixed | floating

        self._update_fixed(guess, fixed)
        self._update_banned(guess, floating, missed)
        self._update_counts(guess, present, missed)
        return self

    def match(self, ngrams):
        ids = np.arange(len(ngrams))

        fixed = self._match_fixed(ngrams)
        ids = ids[fixed]
        if len(ids) == 0:
            return fixed

        banned = self._match_banned(ngrams[ids])
        ids = ids[banned]
        if len(ids) == 0:
            matches = np.zeros(len(ngrams), dtype=bool)
            return matches

        counts = self._token_counts(ngrams[ids])

        minimum = self._match_min_counts(counts)
        ids = ids[minimum]
        if len(ids) == 0:
            matches = np.zeros(len(ngrams), dtype=bool)
            return matches

        counts = counts[minimum]

        maximum = self._match_max_counts(counts)
        ids = ids[maximum]

        matches = np.zeros(len(ngrams), dtype=bool)
        matches[ids] = True
        return matches

    def reset(self):
        self._fixed[:] = NULL
        self._banned[:] = False
        self._min_counts[:] = 0
        self._max_counts[:] = WORDSIZE
        return self

    def _match_fixed(self, ngrams):
        fixed = self._fixed != NULL
        if not np.any(fixed):
            return np.ones(len(ngrams), dtype=bool)

        return np.all(ngrams[:, fixed] == self._fixed[fixed], axis=1)

    def _match_banned(self, ngrams):
        banned = self._banned[ngrams, np.arange(WORDSIZE)]
        return ~np.any(banned, axis=1)

    def _match_min_counts(self, counts):
        required = self._min_counts > 0
        if not np.any(required):
            return np.ones(len(counts), dtype=bool)

        return np.all(
            counts[:, required] >= self._min_counts[required],
            axis=1,
        )

    def _match_max_counts(self, counts):
        constrained = self._max_counts < WORDSIZE
        if not np.any(constrained):
            return np.ones(len(counts), dtype=bool)

        return np.all(
            counts[:, constrained] <= self._max_counts[constrained],
            axis=1,
        )

    def _token_counts(self, ngrams):
        counts = np.zeros((len(ngrams), ALPHABET_SIZE), dtype=int)
        rows = np.arange(len(ngrams))
        one = 1
        for i in range(WORDSIZE):
            np.add.at(
                counts,
                (rows, ngrams[:, i]),
                one,
            )

        return counts

    def _update_fixed(self, guess, fixed):
        self._fixed[fixed] = guess[fixed]

    def _update_banned(self, guess, floating, missed):
        banned = floating | missed

        for i in np.flatnonzero(banned):
            self._banned[guess[i], i] = True

    def _update_counts(self, guess, present, missed):
        for token in np.unique(guess):
            token_mask = guess == token

            hits = int(np.sum(token_mask & present))
            misses = int(np.sum(token_mask & missed))

            self._min_counts[token] = max(
                self._min_counts[token],
                hits,
            )

            if misses:
                self._max_counts[token] = min(
                    self._max_counts[token],
                    hits,
                )


class SearchSpace:
    __slots__ = ("_candidates", "_playable", "_state")

    def __init__(self, answers, guesses):
        self._state = State()
        self._playable = WordList(guesses)
        self._candidates = WordList(answers)

    @property
    def state(self):
        return self._state

    @property
    def playable(self):
        return self._playable

    @property
    def candidates(self):
        return self._candidates

    @property
    def empty(self):
        return len(self._candidates) == 0

    def update(self, guess, mask):
        guess_tokens = encode_tokens(guess)

        self._state.update(guess_tokens, mask)
        self._eliminate(guess)

        self._candidates.keep(
            lambda words: self._state.match(words.ngrams())
        )
        return self

    def reset(self):
        self._state.reset()
        self._playable.reset()
        self._candidates.reset()
        return self

    def _eliminate(self, guess):
        self._candidates.discard(guess)
        self._playable.discard(guess)

    def __len__(self):
        return len(self._candidates)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"candidates={len(self._candidates)}, "
            f"playable={len(self._playable)})"
        )
