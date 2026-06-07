"""Integration models that combine constraints with active search spaces."""

from __future__ import annotations

import string

from typing import Sequence, Iterator

from .constraints import TokenConstraints, WORD_SIZE
from .search import SearchSpace


class WordModel:
    """
    Combined word-solving model which is both token-constraint and
    search-space aware.

    sp: SearchSpace owns:
        - word pool
        - active mask
        - guesses
        - ranking helpers

    tcs: TokenConstraints owns:
        - fixed positions
        - banned positions
        - min token counts
        - max token counts

    WordModel owns:
        - applying feedback to both systems
        - rate mapping over the current active search space

    `WordModel` is an integration layer. It is not a facade which blindly
    exposes everything from `SearchSpace` and `TokenConstraints`. It is a layer
    where new methods that integrate both classes can be composed. This is
    effectively a base class for the game engine class. We expose underlying
    methods on the game engine; not the integration layer.
    """
    __slots__ = ("search_space", "constraints")

    def __init__(self, words: Sequence[str]):
        self.search_space = SearchSpace(words)
        self.constraints = TokenConstraints()

    def update(self, guess: str, token_mask: str) -> WordModel:
        """
        Apply an externally-provided token_mask string and guess to the model.
        """
        self.search_space.submit_guess(guess)
        self.constraints.update(guess, token_mask)
        self.search_space.keep_where(self.constraints.is_candidate)
        return self

    def reset(self) -> WordModel:
        """
        Reset the model by resetting both its search space and corresponding
        constraints.
        """
        self.search_space.reset()
        self.constraints.reset()
        return self

    def token_rates(self) -> Iterator[tuple[str, float]]:
        """
        Iterator which produces global token rates.

        Each rate represents the number of times a token is seen across the
        entire search space. It does not count duplicate word tokens.
        """
        total = len(self.search_space)
        if total == 0:
            return

        tokens = string.ascii_lowercase
        counter = {t: 0 for t in tokens}
        for word in self.search_space:
            for token in set(word):
                counter[token] += 1

        yield from ((token, count / total) for token, count in counter.items())

    def token_index_rates(self) -> Iterator[tuple[str, list[float]]]:
        """
        Iterator which produces per-index token rates.

        Each rate represents the number of times a token is seen at a given
        index in a word, across the entire search space. It does not ignore
        duplicate word tokens.
        """
        total = len(self.search_space)
        if total == 0:
            return

        tokens = string.ascii_lowercase
        counter = {t: [0] * WORD_SIZE for t in tokens}
        for word in self.search_space:
            for idx, token in enumerate(word):
                counter[token][idx] += 1

        yield from (
            (token, [ct / total for ct in counts])
            for token, counts in counter.items()
        )
