from __future__ import annotations

import operator

import numpy as np

from abc import ABC, abstractmethod
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

from .transforms import tokenize

if TYPE_CHECKING:
    from .core import SearchSpace


class NGramArray:
    __slots__ = ("_ngrams", "_mask")

    def __init__(self, ngrams: np.ndarray):
        self._ngrams = np.array(ngrams, copy=True)
        self._mask = np.ones(len(self._ngrams), dtype=bool)

        self._ngrams.setflags(write=False)

    @classmethod
    def _view(cls, ngrams, mask: np.ndarray):
        obj = cls.__new__(cls)
        obj._ngrams = ngrams
        obj._mask = mask
        return obj

    def _view_from_mask(self, mask):
        return self._view(self._ngrams, mask)

    @property
    def _ids(self):
        return np.flatnonzero(self._mask)

    @property
    def empty(self):
        return len(self) == 0

    def hgrams(self):
        mask = np.fromiter(
            (len(ng) == len(set(ng)) for ng in self._ngrams),
            dtype=bool,
            count=len(self._ngrams),
        )
        return self._view_from_mask(self._mask & mask)

    def apply(self, func, **kwargs):
        values = func(self, **kwargs)
        if len(values) != len(self):
            raise ValueError("applied result must be same length as array")

        return values

    def where(self, condition, **kwargs):
        condition = self._resolve_condition(condition, **kwargs)
        return type(self)(self._selected()[condition].copy())

    def keep(self, condition, **kwargs):
        condition = self._resolve_condition(condition, **kwargs)

        ids = self._ids
        self._mask[ids] &= condition
        return self

    def min(self, key=None, **kwargs):
        if key is None:
            return self._selected().min()

        values = self.apply(key, **kwargs)
        return self[np.argmin(values)]

    def max(self, key=None, **kwargs):
        if key is None:
            return self._selected().max()

        values = self.apply(key, **kwargs)
        return self[np.argmax(values)]

    def discard(self, idx: int):
        self._mask[idx] = False
        return self

    def reset(self):
        self._mask[:] = True
        return self

    def copy(self):
        return type(self)(self._selected().copy())

    def _resolve_condition(self, condition, **kwargs):
        if callable(condition):
            condition = condition(self, **kwargs)

        if len(condition) != len(self):
            raise ValueError("condition must align to length of array")

        return condition

    def __array__(self, dtype=None):
        return np.asarray(self._selected(), dtype=dtype)

    def _selected(self):
        return self._ngrams[self._mask]

    def _validate(self, other):
        if not isinstance(other, NGramArray):
            raise TypeError("expected NGramArray")

        if self._ngrams is not other._ngrams:
            raise ValueError(
                "cannot combine NGramArrays from different sources"
            )

    def __repr__(self):
        return f"{type(self).__name__}(count={len(self)})"

    def __and__(self, other: NGramArray):
        self._validate(other)
        return self._view_from_mask(self._mask & other._mask)

    def __or__(self, other: NGramArray):
        self._validate(other)
        return self._view_from_mask(self._mask | other._mask)

    def __invert__(self):
        return self._view_from_mask(~self._mask)

    def __sub__(self, other: NGramArray):
        if isinstance(other, NGramArray):
            self._validate(other)
            return self._view_from_mask(self._mask & ~other._mask)

        return operator.sub(self._selected(), other)

    def __add__(self, other):
        return operator.add(self._selected(), other)

    def __radd__(self, other):
        return operator.add(other, self._selected())

    def __mul__(self, other):
        return operator.mul(self._selected(), other)

    def __rmul__(self, other):
        return operator.mul(other, self._selected())

    def __matmul__(self, other):
        return self._selected() @ other

    def __rmatmul__(self, other):
        return other @ self._selected()

    def __eq__(self, other):
        return self._selected() == other

    def __ne__(self, other):
        return self._selected() != other

    def __lt__(self, other):
        return self._selected() < other

    def __le__(self, other):
        return self._selected() <= other

    def __gt__(self, other):
        return self._selected() > other

    def __ge__(self, other):
        return self._selected() >= other

    def __len__(self):
        return int(self._mask.sum())

    def __iter__(self):
        for i in self._ids:
            yield self._ngrams[i]

    def __getitem__(self, item):
        return self._selected()[item]

    def __getattr__(self, name):
        return getattr(self._selected(), name)

    def __copy__(self):
        return self.copy()


class WordList:
    __slots__ = ("_words", "_index", "_ngrams")

    def __init__(self, words):
        self._words = tuple(dict.fromkeys(words))
        self._index = MappingProxyType({
            word: i for i, word in enumerate(self._words)
        })
        self._ngrams = NGramArray(tokenize(self._words))

    @classmethod
    def _view(cls, words, index, ngrams):
        obj = cls.__new__(cls)
        obj._words = words
        obj._index = index
        obj._ngrams = ngrams
        return obj

    def _view_from_ngrams(self, ngrams):
        return self._view(self._words, self._index, ngrams)

    @property
    def empty(self):
        return len(self) == 0

    def ngrams(self) -> NGramArray:
        return self._ngrams

    def hgrams(self):
        return self._view_from_ngrams(self._ngrams.hgrams())

    def apply(self, func, **kwargs):
        values = func(self, **kwargs)
        if len(values) != len(self):
            raise ValueError("applied result must be same length as array")

        return values

    def where(self, condition, **kwargs):
        condition = self._resolve_condition(condition, **kwargs)
        words = tuple(word for word, keep in zip(self, condition) if keep)
        return type(self)(words)

    def keep(self, condition, **kwargs):
        condition = self._resolve_condition(condition, **kwargs)
        self._ngrams.keep(condition)
        return self

    def min(self, key=None, **kwargs):
        if key is None:
            return min(iter(self))

        values = self.apply(key, **kwargs)
        return self[np.argmin(values)]

    def max(self, key=None, **kwargs):
        if key is None:
            return max(iter(self))

        values = self.apply(key, **kwargs)
        return self[np.argmax(values)]

    def discard(self, word: str):
        idx = self._index.get(word)
        if idx is None:
            return self

        self._ngrams.discard(idx)
        return self

    def reset(self):
        self._ngrams.reset()
        return self

    def copy(self):
        return type(self)(tuple(self))

    def _resolve_condition(self, condition, **kwargs):
        if callable(condition):
            condition = condition(self, **kwargs)

        if len(condition) != len(self):
            raise ValueError("condition must align to length of array")

        return condition

    def _validate(self, other):
        if not isinstance(other, WordList):
            raise TypeError("expected WordList")

        if self._words is not other._words:
            raise ValueError("cannot combine WordLists from different sources")

    def __and__(self, other):
        self._validate(other)
        return self._view_from_ngrams(self._ngrams & other._ngrams)

    def __or__(self, other):
        self._validate(other)
        return self._view_from_ngrams(self._ngrams | other._ngrams)

    def __invert__(self):
        return self._view_from_ngrams(~self._ngrams)

    def __sub__(self, other):
        self._validate(other)
        return self._view_from_ngrams(self._ngrams - other._ngrams)

    def __len__(self):
        return len(self._ngrams)

    def __contains__(self, item):
        idx = self._index.get(item)
        return idx is not None and bool(self._ngrams._mask[idx])

    def __iter__(self):
        for i in self._ngrams._ids:
            yield self._words[i]

    def __getitem__(self, item):
        idx = self._ngrams._ids[item]

        if isinstance(item, slice):
            return type(self)(self._words[i] for i in idx)

        return self._words[idx]

    def __copy__(self):
        return self.copy()

    def __str__(self):
        return "\n".join(self)

    def __repr__(self):
        return f"{type(self).__name__}(count={len(self)})"


class AbstractStrategy(ABC):

    def __init__(self):
        self._search_space = None

    @property
    def game(self) -> SearchSpace:
        if self._search_space is None:
            raise RuntimeError("search space not set")

        return self._search_space

    def _set_search_space(self, search_space: SearchSpace) -> AbstractStrategy:
        self._search_space = search_space
        return self

    @cached_property
    def opener(self) -> str:
        return self.select_opener()

    def select_opener(self) -> str:
        return None

    @abstractmethod
    def next_guess(self, turn: int) -> str:
        raise NotImplementedError


class GameEngine:

    def __init__(self, answers, guesses, strategy: AbstractStrategy):
        from .core import SearchSpace

        self._search_space = SearchSpace(answers, guesses)
        self._strategy = strategy
        self._turn = 1

        self._strategy._set_search_space(self._search_space)

    def next_guess(self) -> str:
        return self._strategy.next_guess(self._turn)

    def submit(self, guess: str, mask: str) -> GameEngine:
        self._search_space.update(guess, mask)
        self._turn += 1
        return self

    def reset(self):
        self._search_space.reset()
        self._turn = 1
        self._strategy._set_search_space(self._search_space)
        return self
