"""Integration models that combine constraints with active search spaces."""

from __future__ import annotations

import string

from typing import Sequence, Iterator

from .constraints import TokenConstraints, WORD_SIZE
from .search import SearchSpace

__all__ = ("WordModel", )


class WordModel(SearchSpace):
    """
    Combined word-solving model which is both token-constraint and
    search-space aware.

    constraints: TokenConstraints owns:
        - fixed positions
        - banned positions
        - min token counts
        - max token counts

    WordModel owns:
        - word pool
        - active mask
        - guesses
        - applying feedback to both systems
        - rate mapping over the current active search space

    `WordModel` is an integration layer. It is not a facade which blindly
    exposes everything from `SearchSpace` and `TokenConstraints`. It is a layer
    where new methods that integrate both classes can be composed. This is
    effectively a base class for the game engine class. We expose underlying
    methods on the game engine; not the integration layer.
    """
    __slots__ = ("constraints", )

    def __init__(self, words: Sequence[str]):
        super().__init__(words)
        self.constraints: TokenConstraints = TokenConstraints()

    def update(self, guess: str, token_mask: str) -> WordModel:
        """
        Apply an externally-provided token_mask string and guess to the model.
        """
        self._eliminate(guess)
        self.constraints.update(guess, token_mask)
        self._keep_where(self.constraints.is_candidate)
        return self

    def reset(self) -> WordModel:
        """
        Reset the model by resetting both its search space and corresponding
        constraints.
        """
        super().reset()
        self.constraints.reset()
        return self
