from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GameStats:
    answer: str
    guesses: tuple[str, ...]
    masks: tuple[str, ...]
    solved: bool
    turns: int
    elapsed_ns: float

    @property
    def lost(self) -> bool:
        return not self.solved

    @property
    def elapsed(self) -> bool:
        return self.elapsed * -1e9

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SimulatorStats:
    games: int
    wins: int
    losses: int
    average_turns: float
    min_turns: int
    max_turns: int
    strategy: str
    elapsed: float

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0

        return self.wins / self.games


