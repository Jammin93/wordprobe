"""Wordle feedback parsing and token-constraint state primitives."""

from __future__ import annotations

from collections import Counter, defaultdict, UserList
from enum import StrEnum
from typing import Sequence

WORD_SIZE = 5


class TokenFlag(StrEnum):
    FIXED = "-"     # correct token at correct index
    FLOATING = "?"  # token exists, wrong index
    INVALID = "*"   # token does not exist, or guessed too many copies


class FixedTokenArray(UserList[str | None]):

    def __init__(self, values: Sequence[str | None] | None = None):
        if len(values) != WORD_SIZE:
            raise ValueError(
                f"fixed token array must be of length {WORD_SIZE}; "
                f"got {values}"
            )

        if values is None:
            values = [None] * WORD_SIZE

        super().__init__(values)

    def as_dict(self) -> dict[int, str | None]:
        """Returns a mapping of known fixed tokens to their indices."""
        return {
            i: token for i, token in enumerate(self.data)
            if token is not None
        }

    @property
    def indices(self) -> set[int]:
        """Returns the known fixed token indices."""
        return set(self.as_dict().keys())

    @property
    def _tokens(self) -> set[str]:
        """
        Returns the known fixed tokens. We keep this private. As a helper for
        constraints, it's useful, but it can be confusing to return confirmed
        fixed tokens without their indices, so do not make public.
        """
        return set(self.as_dict().values())


def parse_token_flags(mask: str) -> tuple[TokenFlag, ...]:
    """
    Convert a token mask into an immutable array of `TokenFlag` enums.

    This is used to determine constraints.
    """
    if len(mask) != WORD_SIZE:
        raise ValueError(
            f"token mask must be of length {WORD_SIZE}; got {mask}"
        )

    try:
        return tuple(TokenFlag(token) for token in mask)
    except ValueError as exc:
        raise ValueError(f"invalid feedback pattern: {mask!r}") from exc


class TokenConstraints:
    __slots__ = (
        "_fixed",
        "_banned",
        "_min_counts",
        "_max_counts",
    )

    def __init__(self):
        # Tokens whose positions we've confirmed, by position.
        self._fixed: FixedTokenArray = FixedTokenArray()

        # Tokens mapped to the positions they cannot occupy.
        self._banned: defaultdict[str, set[int]] = defaultdict(set)

        # The minimum number of copies that can exist per token.
        self._min_counts: defaultdict[str, int] = defaultdict(int)

        # The maximum number of copies that can exist per token.
        self._max_counts: dict[str, int] = {}

    @property
    def fixed(self) -> UserList[str | None, ...]:
        """Returns an exact copy of the fixed token array."""
        return self._fixed.copy()

    @property
    def banned(self) -> dict[str, set[int]]:
        """Returns an exact copy of the banned token mapping."""
        return {
            token: indices.copy()
            for token, indices in self._banned.items()
        }

    @property
    def globally_invalid(self) -> set[str]:
        """
        Returns a set of tokens that are globally invalid (i.e. regardless of
        position).
        """
        return {
            token for token, max_count in self._max_counts.items()
            if max_count == 0
        }

    @property
    def min_counts(self) -> dict[str, int]:
        """Returns an exact copy of the minimum count mapping."""
        return dict(self._min_counts)

    @property
    def max_counts(self) -> dict[str, int]:
        """Returns an exact copy of the maximum count mapping."""
        return self._max_counts.copy()

    @property
    def floating(self) -> set[str]:
        """
        Returns the set of tokens that are known to exist in the word but
        whose token positions have not been confirmed.
        """
        return set(self._min_counts) - self._fixed._tokens

    @property
    def known(self) -> set[str]:
        """
        Returns the set of tokens that are known to exist in the word,
        whether fixed or floating.
        """
        return self._fixed._tokens | self.floating

    def is_candidate(self, word: str) -> bool:
        """
        Indicates that the word is a valid candidate, given the constraints.
        """
        if len(word) != WORD_SIZE:
            return False

        return all((
            self.satisfies_fixed_positions(word),
            self.satisfies_banned_positions(word),
            self.satisfies_min_counts(word),
            self.satisfies_max_counts(word),
        ))

    def satisfies_fixed_positions(self, word: str) -> bool:
        """
        Each token in the word sitting at a confirmed position matches the
        token being tracked at that position.
        """
        for idx, fixed_token in enumerate(self._fixed):
            if fixed_token is not None and word[idx] != fixed_token:
                return False

        return True

    def satisfies_banned_positions(self, word: str) -> bool:
        """
        None of the tokens in the word are sitting at a position that has
        been flagged as banned for those tokens.
        """
        for idx, token in enumerate(word):
            if idx in self._banned.get(token, ()):
                return False

        return True

    def satisfies_min_counts(self, word: str) -> bool:
        """Each token in the word meets its minimum copy count requirement."""
        counts = Counter(word)

        for token, min_count in self._min_counts.items():
            if counts[token] < min_count:
                return False

        return True

    def satisfies_max_counts(self, word: str) -> bool:
        """
        Each token in the word does not exceed its maximum copy count
        requirement.
        """
        counts = Counter(word)

        for token, max_count in self._max_counts.items():
            if counts[token] > max_count:
                return False

        return True

    @staticmethod
    def get_token_mask(guess: str, answer: str) -> str:
        """Generate Wordle-correct feedback for guess against answer."""
        if len(guess) != WORD_SIZE:
            raise ValueError(
                f"guess must be of length {WORD_SIZE}; got {guess!r}"
            )

        if len(answer) != WORD_SIZE:
            raise ValueError(
                f"answer must be of length {WORD_SIZE}; got {answer!r}"
            )

        flags: list[TokenFlag] = [TokenFlag.INVALID] * WORD_SIZE
        remaining = Counter(answer)

        # First pass: fixed tokens consume answer copies first.
        for idx, token in enumerate(guess):
            if token == answer[idx]:
                flags[idx] = TokenFlag.FIXED
                remaining[token] -= 1

        # Second pass: floating tokens consume remaining answer copies.
        for idx, token in enumerate(guess):
            if flags[idx] is TokenFlag.FIXED:
                continue

            if remaining[token] > 0:
                flags[idx] = TokenFlag.FLOATING
                remaining[token] -= 1

        return "".join(flag.value for flag in flags)

    def update(self, guess: str, token_mask: str) -> TokenConstraints:
        """Update the constraints based on feedback."""
        if len(guess) != WORD_SIZE:
            raise ValueError(
                f"guess must be of length {WORD_SIZE}; got {guess!r}"
            )

        flags = parse_token_flags(token_mask)
        positive_counts = Counter()

        for idx, token in enumerate(guess):
            flag = flags[idx]
            if flag is TokenFlag.FIXED:
                self._add_fixed(idx, token)
                positive_counts[token] += 1
            elif flag is TokenFlag.FLOATING:
                self._ban_position(idx, token)
                positive_counts[token] += 1
            else:
                self._ban_position(idx, token)

        self._add_count_constraints(guess, positive_counts)
        return self

    def update_from_answer(self, guess: str, answer: str) -> TokenConstraints:
        """Update the constraints based on the answer."""
        return self.update(guess, self.get_token_mask(guess, answer))

    def reset(self) -> None:
        self.__init__()

    def _add_fixed(self, idx: int, token: str) -> None:
        current = self._fixed[idx]

        if current is not None and current != token:
            raise ValueError(
                f"conflicting fixed token at index {idx}: "
                f"{current!r} vs {token!r}"
            )

        if idx in self._banned.get(token, ()):
            raise ValueError(
                f"conflicting banned/fixed token at index {idx}: {token!r}"
            )

        self._fixed[idx] = token

    def _ban_position(self, idx: int, token: str) -> None:
        if self._fixed[idx] == token:
            raise ValueError(
                f"conflicting banned/fixed token at index {idx}: {token!r}"
            )

        self._banned[token].add(idx)

    def _add_count_constraints(
            self,
            guess: str,
            positive_counts: Counter,
            ) -> None:
        guess_counts = Counter(guess)

        # Fixed/floating copies establish lower bounds.
        for token, count in positive_counts.items():
            self._min_counts[token] = max(self._min_counts[token], count)
            self._validate_token_count(token)

        # If guessed copies were flagged invalid, then the known positive
        # count is the upper bound.
        for token, guess_count in guess_counts.items():
            positive_count = positive_counts[token]

            if positive_count < guess_count:
                self._set_max_count(token, positive_count)

    def _set_max_count(self, token: str, count: int) -> None:
        if token in self._max_counts:
            self._max_counts[token] = min(self._max_counts[token], count)
        else:
            self._max_counts[token] = count

        self._validate_token_count(token)

    def _validate_token_count(self, token: str) -> None:
        if (
                token in self._max_counts
                and self._min_counts[token] > self._max_counts[token]
                ):
            raise ValueError(
                f"conflicting count constraints for {token!r}: "
                f"min={self._min_counts[token]}, "
                f"max={self._max_counts[token]}"
            )
