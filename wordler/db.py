"""Local dataset loading and historical Wordle API synchronization helpers."""

import pathlib
import time

import pandas as pd

from datetime import datetime, timedelta
from typing import Any

from .utils import debrine, to_iso_date

__all__ = ("get_answers", "get_past_games")

CWD = str(pathlib.Path(__file__).parent.resolve())

API_URL = "https://wordlehints.co.uk/wp-json/wordlehint/v1"
ANSWERS_ALL = API_URL + "/answers"
ANSWERS_LATEST = API_URL + "/answers/latest"


def get_answers() -> tuple[str, ...]:
    """Return the list of all valid answers recognized by Wordle."""
    cwd = pathlib.Path(CWD)
    return debrine(cwd.parent / "data" / "db" / "valid_answers.pkl")


def get_past_games() -> pd.DataFrame:
    """Return the list of all past Wordle games that are tracked by the API."""
    cwd = pathlib.Path(CWD)
    db = debrine(cwd.parent / "data" / "db" / "past_games.pkl")
    return update(db)


def standardize_games_list(games) -> pd.DataFrame:
    """
    Remove unnecessary columns, coerce the dataframe into the shape we need,
    and convert date strings to proper Pandas timestamp objects.
    """
    games = (
        pd.DataFrame(games)
          .drop(columns=["day_name", "difficulty"])
          .rename(columns={
                "game": "wordle_number",
                "answer": "word",
          }
        )
    )[["date", "wordle_number", "word"]]
    games["date"] = pd.to_datetime(games["date"])
    games["wordle_number"] = games["wordle_number"].astype(int)
    return games


def get_games_since(date: datetime) -> list[dict[str, Any]]:
    """Get the list of past games since a given date."""
    # Lazy import because we only need this object if we are making a request
    # over the network.
    from .sessions import CachedLimiterSession

    cwd = pathlib.Path(CWD).parent
    cache_dir = cwd / "data" / "http"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = CachedLimiterSession(
        cache_name=cache_dic / "cache",
        urls_expire_after={API_URL + "/*": timedelta(days=1)},
        allowable_methods=["GET"],
        allowable_codes=[200],
        stale_if_error=True,
        per_second=10,
        burst=2,
    )
    params = {"from": to_iso_date(date), "per_page": "200", "page": 1}
    games = []
    # The data from the API is paginated, so continue requesting more data
    # until the API tells us there is no more data to consume.
    while True:
        r = session.get(ANSWERS_ALL, params=params).json()
        games.extend(r["results"])
        if not r["has_more"]:
            break

        params["page"] += 1
        time.sleep(0.05)

    return games


def update(db: pd.DataFrame) -> pd.DataFrame:
    """Update the database we loaded locally with new data from the API."""
    last = len(db) - 1
    latest_game: pd.Timestamp = db["date"].max()
    latest_game = latest_game.date()
    # The API never has today's game, so we subtract one day. This is a good
    # check because we defer importing the session object until it is needed.
    if latest_game >= datetime.now().date() - timedelta(days=1):
        return db

    games = get_games_since(latest_game + timedelta(days=1))
    # If we didn't receive any new data, return the original db, unmodified.
    if len(games) == 0:
        return db

    games = [x for x in games if x["game"] not in db["wordle_number"].values]
    games = standardize_games_list(games)

    updated = pd.concat([db, games]).sort_values("wordle_number")
    updated = updated.dropna().reset_index(drop=True)
    updated.to_pickle(
        pathlib.Path(CWD).parent / "data" / "db" / "past_games.pkl"
    )
    return updated
