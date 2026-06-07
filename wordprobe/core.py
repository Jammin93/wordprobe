"""Public orchestration layer placeholder for the evolving solver API."""
from __future__ import annotations

import random

from collections import Counter
from functools import partial
from typing import Sequence, Any

from .abstract import (
    AbstractInteractiveEngine,
    AbstractSimulatorEngine,
    increment_turn,
    AbstractSimulator,
)
from .constraints import TokenConstraints, WORD_SIZE
from .models import WordModel
from .heuristics import word_scores, word_index_scores, composite_scores
from .search import SearchPool, SearchSpace

Model = WordModel | SearchSpace | SearchPool


class GameEngineBase:

    def __init__(self, model: WordModel[str]):
        self._model = model
        self._turn = 1
        self._guesses = []

    @property
    def candidates(self) -> WordModel:
        return self._model

    @property
    def playable(self) -> WordModel:
        return self._model.pool

    @property
    def state(self) -> TokenConstraints:
        return self._model.constraints

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def guesses(self) -> list[str]:
        return self._guesses.copy()

    @property
    def last_guess(self) -> str:
        if len(self._guesses) == 0:
            return None
        return self._guesses[-1]

    def reset(self) -> GameEngine:
        self._model.reset()
        self._turn = 1
        self._guesses.clear()
        return self


class GameEngine(GameEngineBase, AbstractInteractiveEngine):

    @increment_turn
    def submit_guess(self, guess: str, feedback: str) -> GameEngine:
        self._model.update(guess, feedback)
        self._guesses.append(guess)
        return self


class SimulatorEngine(GameEngineBase, AbstractSimulatorEngine):

    def __init__(self, model: WordModel[str]):
        super().__init__(model)
        self._answer = None

    @property
    def answer(self) -> str:
        return self._answer

    def set_answer(self, value: str) -> None:
        self._answer = value

    @increment_turn
    def submit_guess(self, guess: str) -> SimulatorEngine:
        if self._answer is None:
            raise RuntimeError("answer must be set before submitting guesses")

        token_mask = self._model.constraints.get_token_mask(guess, self._answer)
        self._model.update(guess, token_mask)
        self._guesses.append(guess)
        return self

    def reset(self) -> GameEngine:
        self._answer = None
        return super().reset()


class Simulator(AbstractSimulator):

    def __init__(self, engine: SimulatorEngine):
        self._engine = engine
        self._state = engine.state
        self._answers: Sequence[str] = None

    @property
    def engine(self) -> SimulatorEngine:
        return self._engine

    @property
    def candidates(self) -> WordModel:
        return self.engine.candidates

    @property
    def answers(self) -> list[str]:
        return self._answers

    def set_answers(self, answers: Sequence[str]) -> Simulator:
        self._answers = answers
        return self

    def choose_answer(self) -> str:
        random.shuffle(list(self._answers))
        return random.choice(self._answers)

    def run_game(self) -> None:
        answer = self.choose_answer()
        self.engine.set_answer(answer)
        stats = {}
        while True:
            print(self.candidates)
            turn = self.engine.turn
            if len(self.candidates) == 1:
                guess = self.candidates[0]
            elif turn == 1:
                guess = self.get_initial_guess()
            elif turn == 2:
                with self.candidates.heterograms() as pool:
                    if len(pool) == 0:
                        guess = self.get_entropy_guess(self.candidates)
                    else:
                        guess = self.get_entropy_guess(pool)
            elif turn == 3 or turn == 4:
                if self.should_probe(self.candidates):
                    guess = self.probe(self.candidates)
                else:
                    with self.candidates.heterograms() as pool:
                        if len(pool) == 0:
                            guess = self.get_entropy_guess(self.candidates)
                        else:
                            guess = self.get_entropy_guess(pool)
            elif turn == 5:
                guess = self.get_entropy_guess(self.candidates)
            elif turn == 6:
                guess = self.get_probability_guess(self.candidates)
            else:
                guess = self.get_entropy_guess(self.candidates)

            self.engine.submit_guess(guess)
            if guess == answer:
                print("Solved in", turn, "turns!")
                print("Answer:", answer)
                break
            elif len(self.candidates) == 0:
                print("No candidates left!")
                break

    def simulate(self, n: int) -> dict[str, Any]:
        ...

    def fixed_custer_ratio(self, candidates: Model) -> float:
        if len(candidates) == 0:
            return 0.0

        condition = self._state.satisfies_fixed_positions
        with candidates.matching(condition) as pool:
            return len(pool) / len(pool)

    def should_probe(
            self,
            candidates: Model,
            *,
            min_candidates: int = 4,
            min_known_token_count: int = 3,
            min_cluster_ratio: float = 0.75,
            ):
        if len(candidates) < min_candidates:
            return False

        if len(self._state.known) < min_known_token_count:
            return False

        if self.fixed_custer_ratio(candidates) < min_cluster_ratio:
            return False

        return True

    def free_position_token_rates(self, words):
        free = self._state.free_indices
        counts = Counter()
        for word in words:
            counts.update({word[idx] for idx in free})

        return {token: count / len(words) for token, count in counts.items()}

    def probe(self, candidates: Model) -> list[str]:
        print("probing ...")
        if len(self.candidates) == 1:
            return self.candidates[0]

        condition = self._state.satisfies_fixed_positions
        with candidates.matching(condition) as pool:
            matching = pool

        free_tokens = set().union(*[
            {word[i] for i in range(WORD_SIZE) if i in self._state.free_indices}
            for word in matching
        ])
        candidates = [
            x for x in self.engine.playable if x not in matching
            and x not in self.candidates.eliminated
        ]
        if len(candidates) == 0:
            return self.get_entropy_guess(self.candidates)

        rates = self.free_position_token_rates(candidates)

        def score(word: str) -> tuple[int, int, float]:
            floating_overlap = self._state.floating.intersection(word)
            free_overlap = free_tokens.intersection(word)
            return (
                len(floating_overlap),
                len(free_overlap),
                sum(rates[token] for token in free_overlap),
            )

        return sorted(candidates, key=score, reverse=True)[0]

    def best_guess(self, candidates: Model, key, **kwargs) -> str:
        if len(self.candidates) == 1:
            return self.candidates[0]

        key = partial(key, **kwargs)
        return max(candidates, key=key)

    def get_initial_guess(self) -> str:
        candidates = self.engine.candidates
        with candidates.heterograms() as pool:
            guess = self.best_guess(pool, composite_scores)
            return guess

    def get_probability_guess(self, candidates: Model) -> str:
        return self.best_guess(candidates, word_index_scores, entropize=False)

    def get_entropy_guess(self, candidates: Model) -> str:
        return self.best_guess(
            candidates,
            composite_scores,
            exclude_tokens=self._state.known
        )
