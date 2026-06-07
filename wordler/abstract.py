from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Self, Sequence

from .constraints import TokenConstraints
from .models import WordModel
from .search import SearchPool

__all__ = (
    "AbstractInteractiveEngine",
    "AbstractSimulatorEngine",
    "AbstractSimulator",
)


def increment_turn(method: callable) -> callable:

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._turn += 1
        return result
    return wrapper


class AbstractBaseEngine(ABC):
    _model: WordModel[str]
    _turn: int
    _guesses: list

    @property
    @abstractmethod
    def candidates(self) -> WordModel:
        raise NotImplementedError

    @property
    @abstractmethod
    def playable(self) -> SearchPool:
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self) -> TokenConstraints:
        raise NotImplementedError

    @property
    @abstractmethod
    def turn(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def guesses(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def last_guess(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Self:
        raise NotImplementedError


class AbstractInteractiveEngine(AbstractBaseEngine):

    @abstractmethod
    def submit_guess(self, guess: str, feedback: str) -> Any:
        raise NotImplementedError


class AbstractSimulatorEngine(AbstractBaseEngine):
    _answer: str | None

    @property
    @abstractmethod
    def answer(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def set_answer(self, value: str) -> None:
        self._answer = value

    @abstractmethod
    def submit_guess(self, guess: str) -> Any:
        raise NotImplementedError


class AbstractSimulator(ABC):
    _engine: AbstractSimulatorEngine
    _answers: Sequence[str]

    @property
    @abstractmethod
    def engine(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def answers(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def choose_answer(self):
        raise NotImplementedError

    @abstractmethod
    def run_game(self):
        raise NotImplementedError

    @abstractmethod
    def simulate(self, n: int) -> Any:
        raise NotImplementedError
