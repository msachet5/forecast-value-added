from datetime import date, timedelta

import polars as pl
import pytest


def make_series(uid: str, values: list[float], start: date = date(2024, 1, 1), step_days: int = 7):
    return pl.DataFrame(
        {
            "unique_id": [uid] * len(values),
            "ds": [start + timedelta(days=step_days * i) for i in range(len(values))],
            "y": [float(v) for v in values],
        }
    )


@pytest.fixture
def series():
    return make_series


@pytest.fixture
def small_panel():
    smooth = make_series("smooth", [10, 11, 9, 10, 12, 10, 9, 11, 10, 10, 11, 9] * 3)
    inter = make_series("intermittent", [0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5] * 3)
    lumpy = make_series("lumpy", [0, 0, 1, 0, 0, 20, 0, 0, 2, 0, 0, 30] * 3)
    erratic = make_series("erratic", [1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10] * 3)
    return pl.concat([smooth, inter, lumpy, erratic])
