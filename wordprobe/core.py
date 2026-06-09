import string
import random

from collections import Counter, defaultdict

from .constants import WORDSIZE


def compose_token_mask(guess, answer):
    mask = ["*"] * WORDSIZE
    remaining = Counter(answer)
    for i, token in enumerate(guess):
        if token == answer[i]:
            mask[i] = "-"
            remaining[token] -= 1

    for i, token in enumerate(guess):
        if mask[i] == "-":
            continue

        if remaining[token] > 0:
            mask[i] = "?"
            remaining[token] -= 1

    return "".join(mask)


def validate_word(word):
    if len(word) != WORDSIZE:
        raise ValueError(
            f"guesses must be of length {WORDSIZE}; got {len(word)}"
        )

    tokens = string.ascii_lowercase
    if not set(word).issubset(tokens):
        raise ValueError(
            f"invalid token in guess '{word}'; valid tokens: {tokens}"
        )


def validate_token_mask(mask):
    if (size := len(mask)) != WORDSIZE:
        raise ValueError(
            f"token masks must be of length {WORDSIZE}; got {size}"
        )

    if not set(mask).issubset(set("*?-")):
        raise ValueError(
            f"invalid token flag in token mask '{mask}';" 
            f"valid flags: *, ?, -"
        )


class State:
    __slots__ = ("_fixed", "_banned", "_min_counts", "_max_counts")

    def __init__(self):
        self._fixed = [None] * WORDSIZE
        self._banned = defaultdict(set)
        self._min_counts = defaultdict(int)
        self._max_counts = {}

    @property
    def fixed(self):
        return list(self._fixed)

    @property
    def banned(self):
        return {
            token: list(indices)
            for token, indices in self._banned.items()
        }

    @property
    def invalid(self):
        return {
            token for token, mx in self._max_counts.items()
            if mx == 0
        }

    @property
    def min_counts(self):
        return dict(self._min_counts)

    @property
    def max_counts(self):
        return dict(self._max_counts)

    @property
    def floating(self):
        fixed_counts = Counter(filter(lambda x: x), self._fixed)
        return {
            token for token, minimum in self._min_counts.items()
            if minimum > fixed_counts[token]
        }

    @property
    def known(self):
        return self.floating.union(filter(lambda x: x), self._fixed)

    @property
    def free(self):
        return set(range(WORDSIZE)) - {
            i for i, token in enumerate(self._fixed)
            if token is not None
        }

    def is_candidate(self, word):
        return not any((
            self.has_invalid_tokens(word),
            not self.satisfies_banned_positions(word),
            not self.satisfies_fixed_positions(word),
            not self.satisfies_min_counts(word),
            not self.satisfies_max_counts(word),
        ))

    def has_invalid_tokens(self, word):
        return bool(self.invalid.intersection(word))

    def satisfies_fixed_positions(self, word):
        for idx, fixed in enumerate(self._fixed):
            if fixed is not None and word[idx] != fixed:
                return False

        return True

    def satisfies_banned_positions(self, word):
        for idx, token in enumerate(word):
            if idx in self._banned.get(token, ()):
                return False

        return True

    def satisfies_min_counts(self, word):
        counts = Counter(word)
        for token, minimum in self._min_counts.items():
            if counts[token] < minimum:
                return False

        return True

    def satisfies_max_counts(self, word):
        counts = Counter(word)
        for token, mx in self._max_counts.items():
            if counts[token] > mx:
                return False

        return True

    def update(self, guess, mask):
        validate_word(guess)
        validate_token_mask(mask)

        positive_counts = Counter()
        for i, token in enumerate(guess):
            flag = mask[i]
            if flag == "-":
                self._add_fixed(i, token)
                positive_counts[token] += 1
            elif flag == "?":
                self._ban_position(i, token)
                positive_counts[token] += 1
            else:
                self._ban_position(i, token)

        self._add_count_constraints(guess, positive_counts)
        return self

    def update_from_answer(self, guess, answer):
        validate_word(answer)
        return self.update(guess, compose_token_mask(guess, answer))

    def reset(self):
        self.__init__()
        return self

    def _add_fixed(self, idx, token):
        active = self._fixed[idx]
        if active is not None and active != token:
            raise ValueError(
                f"fixed token conflict at index {idx}: {active!r} vs {token!r}"
            )

        if idx in self._banned.get(token, ()):
            raise ValueError(
                f"banned token conflict at index {idx}: {token!r}"
            )

        self._fixed[idx] = token

    def _ban_position(self, idx, token):
        if self._fixed[idx] == token:
            raise ValueError(f"fixed token conflict at index {idx}: {token!r}")

        self._banned[token].add(idx)

    def _add_count_constraints(self, guess, positive_counts):
        guess_counts = Counter(guess)
        for token, count in positive_counts.items():
            self._min_counts[token] = max(self._min_counts[token], count)
            self._validate_token_count(token)

        for token, count in guess_counts.items():
            if (pct := positive_counts[token]) < count:
                self._set_max_count(token, pct)

    def _set_max_count(self, token, count):
        if token in self._max_counts:
            self._max_counts[token] = min(self._max_counts[token], count)
        else:
            self._max_counts[token] = count

        self._validate_token_count(token)

    def _validate_token_count(self, token):
        if (
                    token in self._max_counts
                and self._min_counts[token] > self._max_counts[token]
                ):
            raise ValueError(
                f"conflicting count constraints for {token!r}"
                f"min={self._min_counts[token]}, "
                f"max={self._max_counts[token]}"
            )


class SearchSpace:
    __slots__ = ("_pool", "_playable", "_candidates", "_state")

    def __init__(self, words):
        self._pool = tuple(dict.fromkeys(words))
        self._playable = list(dict.fromkeys(words))
        self._candidates = list(dict.fromkeys(words))
        self._state = State()

    @property
    def candidates(self):
        return tuple(self._candidates)

    @property
    def playable(self):
        return tuple(self._playable)

    @property
    def eliminated(self):
        active = set(self._candidates)
        return tuple(word for word in self._pool if word not in active)

    @property
    def state(self):
        return self._state

    @property
    def empty(self):
        return len(self._candidates) == 0

    def where(self, *predicates):
        return [
            word for word in self._candidates
            if all(predicate(word) for predicate in predicates)
        ]

    def update(self, guess, mask):
        self.state.update(guess, mask)
        self._candidates = [
            word for word in self._candidates
            if word != guess and self.state.is_candidate(word)
        ]
        self._playable = [
            word for word in self._playable
            if word != guess
        ]
        return self

    def reset(self):
        self._playable = list(self._pool)
        self._candidates = list(self._pool)
        self._state.reset()
        return self


class GameEngine(SearchSpace):
    __slots__ = ("_guesses", "_turn")

    def __init__(self, words):
        super().__init__(words)
        self._guesses = []
        self._turn = 1

    @property
    def guesses(self):
        return tuple(self._guesses)

    @property
    def turn(self):
        return self._turn

    @property
    def last_guess(self):
        if len(self._guesses) == 0:
            raise RuntimeError("no guesses submitted")

        return self._guesses[-1]

    @property
    def solved(self):
        return len(self._candidates) == 1

    def submit_guess(self, guess, feedback_mask):
        if guess in self._guesses:
            raise ValueError(f"duplicate guess: {guess!r}")

        self.update(guess, feedback_mask)
        self._guesses.append(guess)
        self._turn += 1
        return self

    def reset(self):
        self._turn = 1
        self._guesses.clear()
        super().reset()
        return self


class Simulator(GameEngine):
    __slots__ = ("_answer", )

    def __init__(self, words):
        super().__init__(words)
        self._answer = self.choose_answer()

    @property
    def answer(self):
        return self._answer

    def choose_answer(self):
        return random.choice(self._pool)

    def submit_guess(self, guess):
        mask = compose_token_mask(guess, self._answer)
        super().submit_guess(guess, mask)
        return self

    def play_game(self):
        ...
        self.reset()

    def reset(self):
        super().reset()
        self._answer = self.choose_answer()
        return self
