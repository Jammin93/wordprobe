"""Public orchestration layer placeholder for the evolving solver API."""
from __future__ import annotations

from typing import Any

from .abstract import (
    AbstractInteractiveEngine,
    AbstractSimulatorEngine,
    increment_turn,
    AbstractSimulator,
)
from .constraints import TokenConstraints
from .models import WordModel


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
        return self._guesses[-1]

    def reset(self) -> GameEngine:
        self._model.reset()
        self._turn = 1
        self._guesses.clear()
        return self


class GameEngine(GameEngineBase, BaseInteractiveEngine):

    @increment_turn
    def submit_guess(self, guess: str, feedback: str) -> GameEngine:
        self._model.update(guess, feedback)
        self._guesses.append(guess)
        return self


class SimulatorEngine(GameEngineBase, BaseSimulatorEngine):

    def __init__(self, model: WordModel[str]):
        super().__init__(model)
        self._answer = None

    @property
    def answer(self) -> str:
        return self._answer

    @answer.setter
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
