"""Rolling-origin backtests.

A single holdout period tells you how a method did in one month. A rolling
origin (several cutoffs, each followed by an h-period forecast) tells you
how it does as a process, which is what FVA is about.

    history .......................|cutoff 1| h periods
    history ...........................|cutoff 2| h periods
    history ...............................|cutoff 3| h periods
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl

from ._frame import DS, ID, Y, validate
from .models import Model, get_model


def _resolve(models: Sequence[str | Model] | dict[str, Model]) -> dict[str, Model]:
    if isinstance(models, dict):
        return dict(models)
    resolved: dict[str, Model] = {}
    for m in models:
        if isinstance(m, str):
            resolved[m] = get_model(m)
        elif callable(m):
            resolved[getattr(m, "__name__", f"model_{len(resolved)}")] = m
        else:
            raise TypeError(f"Model must be a name or a callable, got {m!r}")
    return resolved


def backtest(
    df: Any,
    models: Sequence[str | Model] | dict[str, Model] = (
        "naive",
        "seasonal_naive",
        "ses",
        "croston_sba",
    ),
    h: int = 4,
    n_windows: int = 6,
    step_size: int | None = None,
    season_length: int = 1,
    min_train: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Rolling-origin backtest of numpy models over every series.

    Parameters
    ----------
    df:
        Long-format actuals (unique_id, ds, y) on a complete calendar.
    models:
        Built-in model names (see `models.BUILTIN_MODELS`), callables with the
        signature (y, h, season_length) -> forecast, or a {name: callable} dict.
    h:
        Forecast horizon in periods (for example 4 weeks, or 3 months).
    n_windows:
        Number of cutoffs. Six is a reasonable minimum for an FVA review.
    step_size:
        Periods between cutoffs. Defaults to h (non-overlapping windows).
    season_length:
        52 for weekly, 12 for monthly, 7 for daily data with a weekly cycle.
    min_train:
        Minimum training periods before a cutoff is used. Defaults to
        season_length + 1 so seasonal naive has a full season.

    Returns
    -------
    Long frame: unique_id, ds, cutoff, horizon (1..h), y and one column per model.
    """
    data = validate(df)
    fns = _resolve(models)
    step = step_size or h
    min_train = min_train if min_train is not None else max(season_length + 1, 2)

    cols: dict[str, list[np.ndarray]] = {k: [] for k in (ID, DS, "cutoff", "horizon", Y)}
    for name in fns:
        cols[name] = []
    parts = data.partition_by(ID, maintain_order=True, as_dict=False)
    for i, part in enumerate(parts):
        y = part[Y].to_numpy().astype(float)
        days = part[DS].to_physical().to_numpy()  # days since epoch, int32
        n = y.size
        uid = part[ID][0]
        for w in range(n_windows):
            start = n - h - (n_windows - 1 - w) * step
            if start < min_train or start + h > n:
                continue
            train, test = y[:start], y[start : start + h]
            cols[ID].append(np.full(h, uid, dtype=object))
            cols[DS].append(days[start : start + h])
            cols["cutoff"].append(np.full(h, days[start - 1]))
            cols["horizon"].append(np.arange(1, h + 1))
            cols[Y].append(test)
            for name, fn in fns.items():
                fc = np.asarray(fn(train, h, season_length), dtype=float)
                if fc.shape != (h,):
                    raise ValueError(f"Model {name} returned shape {fc.shape}, expected ({h},)")
                cols[name].append(np.maximum(fc, 0.0))  # demand forecasts are non-negative
        if progress:
            progress(i + 1, len(parts))

    if not cols[Y]:
        raise ValueError(
            "No backtest windows fit the data. Lower n_windows, h or min_train, "
            "or provide longer history."
        )
    out = pl.DataFrame(
        {
            ID: pl.Series(ID, np.concatenate(cols[ID]).tolist(), dtype=pl.Utf8),
            DS: pl.Series(DS, np.concatenate(cols[DS]), dtype=pl.Int32).cast(pl.Date),
            "cutoff": pl.Series("cutoff", np.concatenate(cols["cutoff"]), dtype=pl.Int32).cast(
                pl.Date
            ),
            "horizon": pl.Series("horizon", np.concatenate(cols["horizon"]), dtype=pl.Int64),
            Y: np.concatenate(cols[Y]),
            **{name: np.concatenate(cols[name]) for name in fns},
        }
    )
    return out


def backtest_statsforecast(
    df: Any,
    models: Sequence[Any],
    h: int,
    n_windows: int,
    freq: str,
    step_size: int | None = None,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Rolling-origin backtest using Nixtla statsforecast models (optional extra).

    Example
    -------
    >>> from statsforecast.models import AutoETS, CrostonSBA, SeasonalNaive
    >>> cv = backtest_statsforecast(
    ...     sales, [SeasonalNaive(52), AutoETS(season_length=52), CrostonSBA()],
    ...     h=4, n_windows=6, freq="W-MON")

    Output columns match `backtest`: unique_id, ds, cutoff, horizon, y, <models>.
    """
    try:
        from statsforecast import StatsForecast  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "statsforecast is an optional extra: pip install 'forecast-value-added[statsforecast]'"
        ) from exc

    data = validate(df).to_pandas()
    sf = StatsForecast(models=list(models), freq=freq, n_jobs=n_jobs)
    cv = sf.cross_validation(df=data, h=h, n_windows=n_windows, step_size=step_size or h)
    out = pl.from_pandas(cv.reset_index() if ID not in cv.columns else cv)
    out = out.with_columns(pl.col(DS).cast(pl.Date), pl.col("cutoff").cast(pl.Date))
    out = out.with_columns(
        (pl.col(DS).rank("ordinal").over(ID, "cutoff")).cast(pl.Int64).alias("horizon")
    )
    model_cols = [c for c in out.columns if c not in {ID, DS, "cutoff", "horizon", Y}]
    out = out.with_columns([pl.col(c).clip(lower_bound=0) for c in model_cols])
    return out.select(ID, DS, "cutoff", "horizon", Y, *model_cols)


__all__ = ["backtest", "backtest_statsforecast"]
