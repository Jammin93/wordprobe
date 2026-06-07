"""Masked word-pool containers and filtering helpers for candidate search."""

from __future__ import annotations

import re

from contextlib import contextmanager
from typing import Callable, Iterator, overload, Sequence

from .descriptors import SlottedDataDescriptor

__all__ = ("SearchSpace", )

SortKey = Callable[[str], float]
MaskLike = Sequence[bool] | Callable[[str], bool]


class WordListMask(SlottedDataDescriptor):
    """
    Descriptor for the active-word mask.

    Ensures that any assigned mask has the same length as the underlying word
    pool, then stores a copied list so callers cannot mutate the mask through
    an external reference.
    """

    def __set__(self, instance, value: Sequence[bool]):
        if len(value) != len(instance._words):
            raise ValueError(
                f"mask length mismatch: {len(value)} != {len(instance._words)}"
            )

        super().__set__(instance, list(value))


class SearchPool:
    __slots__ = (
        "_words",
        "_index",
        "_mask__slot",
    )
    _mask = WordListMask()

    def __init__(self, words: Sequence[str]):
        self._words = words
        self._index = {word: idx for idx, word in enumerate(words)}
        self._mask = [True] * len(words)

    @property
    def eliminated(self) -> list[str]:
        """The list of eliminated words."""
        return [
            word for word, active in self._word_mask_pairs()
            if not active
        ]

    @contextmanager
    def mask(self, mask: MaskLike) -> Iterator[bool]:
        """Temporarily replace the active mask."""
        saved = self._mask.copy()
        try:
            if callable(mask):
                self._mask = [
                    active and mask(word)
                    for word, active in self._word_mask_pairs()
                ]
            else:
                self._mask = [
                    active and selected
                    for active, selected in zip(self._mask, mask)
                ]

            yield self
        finally:
            self._mask = saved

    @contextmanager
    def matching(
            self,
            *conditions: Callable[[str], bool],
            ) -> Iterator[SearchPool]:
        """
        Temporarily restrict active words to those matching all predicates.
        """
        mask = [
            active and all(predicate(word) for predicate in conditions)
            for word, active in self._word_mask_pairs()
        ]
        with self.mask(mask):
            yield self

    @contextmanager
    def excluding(
            self,
            *conditions: Callable[[str], bool],
            ) -> Iterator[SearchPool]:
        """
        Temporarily restrict active words to those not matching any of
        the predicates.
        """
        mask = [
            active and not any(predicate(word) for predicate in conditions)
            for word, active in self._word_mask_pairs()
        ]
        with self.mask(mask):
            yield self

    @contextmanager
    def heterograms(self) -> Iterator[SearchPool]:
        """Temporarily restrict the search space to heterograms."""
        mask = [
            len(word) == len(set(word))
            for word in self._active_words()
        ]
        with self.mask(mask):
            yield self

    def pattern_count(self, pattern: re.Pattern[str]) -> int:
        """Count active words that match the compiled regex pattern."""
        return len([x for x in self if pattern.match(x)])

    def pattern_rate(self, pattern: re.Pattern[str]) -> int:
        """
        Return the active fraction of words that match the compiled pattern.
        """
        return self.pattern_count(pattern) / len(self)

    def _keep_where(self, predicate: Callable[[str], bool]) -> SearchPool:
        """
        Keep currently active words that satisfy a predicate.

        Previously eliminated words remain eliminated. The predicate is only
        allowed to preserve or remove active words; it does not reactivate
        inactive words.
        """
        self._mask = [
            active and predicate(word)
            for word, active in self._word_mask_pairs()
        ]
        return self

    def empty(self) -> bool:
        """Indicates whether the search space is empty."""
        return len(self) == 0

    def _eliminate(self, word: str) -> SearchPool:
        """Remove a word from the pool's visible set."""
        if word not in self:
            raise ValueError(f"invalid word: {word!r}")

        idx = self._index[word]
        self._mask[idx] = False
        return self

    def _clear_mask(self) -> SearchPool:
        """Remove all active words from the search space."""
        self._mask = [True] * len(self._words)
        return self

    def _reset(self) -> SearchPool:
        """Restore all words to active status."""
        self._clear_mask()
        return self

    def _word_mask_pairs(self):
        yield from zip(self._words, self._mask)

    def _active_words(self):
        yield from (
            word for word, active in self._word_mask_pairs() if active
        )

    def _inactive_words(self):
        yield from (
            word for word, active in self._word_mask_pairs() if not active
        )

    def __len__(self) -> int:
        return sum(self._mask)

    def __iter__(self):
        yield from self._active_words()

    def __contains__(self, word: str):
        idx = self._index.get(word)
        if idx is None:
            return False

        return self._mask[idx]

    @overload
    def __getitem__(self, index: int) -> str:
        ...

    @overload
    def __getitem__(self, index: slice) -> SearchPool:
        ...

    def __getitem__(self, index: int | slice) -> str | SearchPool:
        active = self._active_words()
        if isinstance(index, slice):
            return SearchPool(list(active)[index])
        return list(active)[index]

    def __str__(self) -> str:
        return list(self).__str__()


class SearchSpace(SearchPool):
    __slots__ = ("_pool", "_guesses")

    def __init__(self, words: Sequence[str]):
        # Remove duplicates but preserve order.
        words = tuple(dict.fromkeys(words))
        super().__init__(words)
        self._pool = SearchPool(self._words)

    @property
    def pool(self) -> SearchPool:
        """
        The underlying search pool of playable candidates, whose mask only
        tracks eliminated guesses.
        """
        return self._pool

    def _eliminate(self, word: str) -> SearchSpace:
        super()._eliminate(word)
        self._pool._eliminate(word)
        return self

    def reset(self) -> SearchSpace:
        """
        Restore all words to active status and clear submitted guesses.
        """
        self._reset()
        self._pool._reset()
        return self
