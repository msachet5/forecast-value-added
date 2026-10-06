import numpy as np
import polars as pl
import pytest

from forecast_value_added import backtest


def test_shapes_and_cutoffs(small_panel):
    bt = backtest(
        small_panel, models=["naive", "seasonal_naive"], h=3, n_windows=4, season_length=12
    )
    assert bt.height == 4 * 4 * 3
    per = bt.group_by("unique_id").agg(pl.col("cutoff").n_unique())
    assert per["cutoff"].to_list() == [4, 4, 4, 4]
    assert bt["horizon"].max() == 3
    assert (bt["cutoff"] < bt["ds"]).all()


def test_no_leakage(series):
    df = series("a", list(range(1, 21)))
    seen = []

    def spy(y, h, season_length):
        seen.append(y[-1])
        return np.full(h, y[-1])

    bt = backtest(df, models={"spy": spy}, h=2, n_windows=3)
    # each cutoff's training data must end strictly before its first forecast date
    for cutoff, first_y in zip(seen, bt.filter(pl.col("horizon") == 1)["y"].to_list(), strict=True):
        assert cutoff == first_y - 1


def test_forecasts_non_negative(series):
    df = series("a", [10, 8, 6, 4, 2, 0, 0, 0, 0, 0])
    bt = backtest(df, models=["drift"], h=2, n_windows=2)
    assert (bt["drift"] >= 0).all()


def test_too_short_raises(series):
    with pytest.raises(ValueError, match="No backtest windows"):
        backtest(series("a", [1, 2]), h=4, n_windows=2, season_length=52)


def test_custom_callable(series):
    df = series("a", [1.0] * 30)

    def two(y, h, season_length):
        return np.full(h, 2.0)

    bt = backtest(df, models=[two], h=1, n_windows=2)
    assert "two" in bt.columns and bt["two"].to_list() == [2.0, 2.0]
