"""General utility helpers shared across the Wordler package."""

import gc
import pathlib
import pickle

from datetime import datetime

from typing import Any


def debrine(pkl_file: pathlib.Path) -> Any:
    """
    Dedicated function for speeding up load times for pickled objects. See
    this StackOverflow question for details: https://shorturl.at/dWSJZ
    """
    gc.disable()
    obj = pickle.loads(pkl_file.read_bytes())
    gc.enable()
    return obj


def to_iso_date(date: datetime.date) -> str:
    return date.strftime("%Y-%m-%d")
