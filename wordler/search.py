"""Masked word-pool containers and filtering helpers for candidate search."""

from __future__ import annotations

import re

from contextlib import contextmanager
from itertools import islice
from functools import partial, wraps
from typing import Callable, Iterator, Sequence

from .descriptors import SlottedDataDescriptor

ScoreKey = Callable[[str], float]


class WordListMask(SlottedDataDescriptor):
    """
    Descriptor for the active-word mask.

    Ensures that any assigned mask has the same length as the underlying word
    pool, then stores a copied list so callers cannot mutate the mask through
    an external reference.
    """

    def __set__(self, instance, value: Sequence[bool]):
        if len(value) != len(instance._pool):
            raise ValueError(
                f"mask length mismatch: {len(value)} != {len(instance._pool)}"
            )

        super().__set__(instance, list(value))


def check_for_empty_pool(method: Callable) -> Callable:
    """
    Decorator that prevents operations on an empty active search space.

    Raises RuntimeError before calling the wrapped method when no active words
    remain.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.empty():
            raise RuntimeError("search space is empty")

        return method(self, *args, **kwargs)
    return wrapper


# noinspection PyUnresolvedReferences
class SearchSpace:
    """
    Mutable active-word pool.

    Tracks the full word pool, an active/inactive mask, and submitted guesses.
    Filtering operations mutate the active mask while preserving the original
    pool, allowing the search space to be reset or temporarily remasked.
    """
    __slots__ = (
        "_pool",
        "_index",
        "_mask__slot",
        "_guesses",
    )
    _mask = WordListMask()

    def __init__(self, words: Sequence[str]):
        """
        Initialize the search space from a sequence of words.

        Duplicate words are removed. All remaining words start as active
        candidates, and no guesses are recorded.
        """
        self._pool = tuple(set(words))
        self._index: dict[str, int] = {
            word: idx for idx, word in enumerate(self._pool)
        }
        self._mask = [True] * len(self._pool)
        self._guesses = []

    @property
    def guesses(self):
        """List of all guesses that have been submitted."""
        return list(self._guesses)

    @property
    def eliminated(self) -> list[str]:
        """The list of eliminated words."""
        return [
            word for word, active in zip(self._pool, self._mask)
            if not active
        ]

    @check_for_empty_pool
    def best_guess(self, key: ScoreKey) -> str:
        """
        Return the highest-scoring active word.
        Does not mutate the search space.
        """
        return max(self, key=key)

    def ranked_by(self, key: ScoreKey, **kwargs) -> Iterator[str]:
        """
        Return ranked active words without mutating the search space. It may
        be useful to submit a partials function.
        """
        key = partial(key, **kwargs)
        yield from sorted(self, key=key, reverse=True)

    def top_n_candidates(
            self,
            n: int,
            key: ScoreKey,
            **kwargs,
            ) -> Iterator[str]:
        """Iterator which produces the top `n` number of candidates."""
        yield from islice(self.ranked_by(key, **kwargs), n)

    def submit_guess(self, guess: str):
        """Push a guess onto the the list of tracked guesses."""
        if guess not in self._index:
            raise ValueError(f"invalid guess: {guess!r}")

        self._guesses.append(guess)

        idx = self._index[guess]
        self._mask[idx] = False
        return self

    @contextmanager
    def mask(self, mask: Sequence[bool] | None) -> Iterator[SearchSpace]:
        """
        Temporarily replace the active mask.

        If mask is None, all answers are temporarily active.
        """
        saved = self._mask.copy()
        try:
            if mask is None:
                self._mask = [True] * len(self._pool)
            else:
                self.where(mask)

            yield self
        finally:
            self._mask = saved

    @contextmanager
    def heterograms(self) -> Iterator[SearchSpace]:
        """Temporarily restrict the search space to heterograms."""
        mask = [
            active and len(word) == len(set(word))
            for word, active in zip(self._pool, self._mask)
        ]
        with self.mask(mask):
            yield self

    def where(self, mask: list[bool]) -> SearchSpace:
        """
        Replace the active mask.

        The supplied mask must align one-to-one with the original word pool.
        Active words are represented by True values, eliminated words by False
        values.
        """
        self._mask = mask
        return self

    def keep_where(self, predicate: Callable[[str], bool]) -> SearchSpace:
        """
        Keep currently active words that satisfy a predicate.

        Previously eliminated words remain eliminated. The predicate is only
        allowed to preserve or remove active words; it does not reactivate
        inactive words.
        """
        self._mask = [
            active and predicate(word)
            for word, active in zip(self._pool, self._mask)
        ]
        return self

    def pattern_count(self, pattern: re.Pattern[str]) -> int:
        """
        Count active words that fully match a compiled regex pattern.
        """
        return len([x for x in self if pattern.fullmatch(x)])

    def pattern_percentage(self, pattern: re.Pattern[str]) -> float:
        """
        Return the fraction of active words that fully match a pattern.
        """
        return self.pattern_count(pattern) / len(self)

    def reset(self) -> SearchSpace:
        """
        Restore all words to active status and clear submitted guesses.
        """
        self._clear_mask()
        self._guesses.clear()
        return self

    def empty(self) -> bool:
        """Return True if no active words remain."""
        return len(self) == 0

    def _clear_mask(self) -> None:
        """Reactivate every word in the original pool."""
        self._mask = [True] * len(self._pool)

    def __len__(self):
        return sum(self._mask)

    def __contains__(self, word: str):
        idx = self._index.get(word)
        if idx is None:
            return False

        return self._mask[idx]

    def __iter__(self):
        for word, active in zip(self._pool, self._mask):
            if active:
                yield word
