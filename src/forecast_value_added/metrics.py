"""Forecast error metrics the way demand planners use them.

Sign conventions (stated once, used everywhere):

* error          = forecast - actual  (positive = over-forecast)
* bias_pct       = sum(forecast - actual) / sum(actual)
* tracking signal = sum(actual - forecast) / MAD, the textbook (APICS) form,
                   so a large positive value means persistent UNDER-forecast.

WAPE is the volume-weighted error planners report (sum |e| / sum actual).
MASE scales MAE by the in-sample MAE of a seasonal naive forecast, so a
value below 1 means "better than seasonal naive on history" and series of
different sizes can be averaged fairly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from ._frame import DS, ID, Y, to_polars, validate

METRICS = ("wape", "mae", "rmse", "bias_pct", "mase", "accuracy")


def mase_scale(history: Any, season_length: int = 1) -> pl.DataFrame:
    """In-sample MAE of the seasonal naive forecast per series (the MASE denominator).

    Series whose history is constant, or shorter than one season plus one
    period, get a null scale and are excluded from MASE.
    """
    data = validate(history)
    m = max(int(season_length), 1)
    scale = (
        data.with_columns((pl.col(Y) - pl.col(Y).shift(m).over(ID)).abs().alias("_d"))
        .group_by(ID)
        .agg(pl.col("_d").mean().alias("scale"))
        .with_columns(
            pl.when(pl.col("scale") > 0).then(pl.col("scale")).otherwise(None).alias("scale")
        )
    )
    return scale


def melt_forecasts(df: Any, models: Sequence[str], keep: Sequence[str] = ()) -> pl.DataFrame:
    """Wide (one column per model) to long (model, forecast) format."""
    data = to_polars(df)
    missing = [m for m in models if m not in data.columns]
    if missing:
        raise KeyError(f"Forecast columns not found: {missing}")
    index = [c for c in data.columns if c not in models]
    out = data.unpivot(on=list(models), index=index, variable_name="model", value_name="forecast")
    if keep:
        out = out.select(*keep, "model", "forecast")
    return out


def _agg_exprs() -> list[pl.Expr]:
    err = pl.col("forecast") - pl.col(Y)
    exprs = [
        pl.len().alias("n"),
        pl.col(Y).sum().alias("actual"),
        pl.col("forecast").sum().alias("forecast_total"),
        err.abs().mean().alias("mae"),
        (err**2).mean().sqrt().alias("rmse"),
        (err.abs().sum() / pl.col(Y).sum()).alias("wape"),
        (err.sum() / pl.col(Y).sum()).alias("bias_pct"),
    ]
    return exprs


def error_table(
    df: Any,
    models: Sequence[str],
    by: Sequence[str] | str | None = None,
    scale: Any | None = None,
) -> pl.DataFrame:
    """Error metrics for each model, optionally within groups.

    Parameters
    ----------
    df:
        Rows of (unique_id, ds, y, <model columns>). Extra columns such as
        demand_class or abc can be used in `by`.
    models:
        Names of the forecast columns to evaluate.
    by:
        Column(s) to group by, for example "demand_class" or ["abc", "xyz"].
        None gives one overall row per model.
    scale:
        Output of `mase_scale`. When given, MASE is added.

    Returns
    -------
    One row per (group, model) with n, actual, forecast_total, mae, rmse,
    wape, bias_pct, accuracy (1 - WAPE, floored at 0) and mase.
    """
    by = [by] if isinstance(by, str) else list(by or [])
    long = melt_forecasts(df, models)
    long = long.filter(pl.col("forecast").is_not_null() & pl.col(Y).is_not_null())
    exprs = _agg_exprs()
    if scale is not None:
        sc = to_polars(scale).select(ID, "scale")
        long = long.join(sc, on=ID, how="left")
        exprs.append(
            ((pl.col("forecast") - pl.col(Y)).abs() / pl.col("scale")).mean().alias("mase")
        )
    out = long.group_by([*by, "model"], maintain_order=True).agg(exprs)
    out = out.with_columns(
        pl.when(pl.col("actual") > 0)
        .then((1 - pl.col("wape")).clip(lower_bound=0))
        .otherwise(None)
        .alias("accuracy"),
        pl.when(pl.col("actual") > 0).then(pl.col("wape")).otherwise(None).alias("wape"),
        pl.when(pl.col("actual") > 0).then(pl.col("bias_pct")).otherwise(None).alias("bias_pct"),
    )
    order = {m: i for i, m in enumerate(models)}
    out = (
        out.with_columns(pl.col("model").replace_strict(order, return_dtype=pl.Int32).alias("_o"))
        .sort([*by, "_o"])
        .drop("_o")
    )
    return out


def series_errors(df: Any, models: Sequence[str], scale: Any | None = None) -> pl.DataFrame:
    """Error metrics per series and model: the input to per-item FVA."""
    return error_table(df, models, by=ID, scale=scale)


def trigg_limit(alpha: float = 0.1, confidence: float = 0.95) -> float:
    """Control limit for Trigg's smoothed tracking signal under unbiased normal errors.

    With smoothing constant alpha, the smoothed error has standard deviation
    sigma * sqrt(alpha / (2 - alpha)) and the smoothed absolute error tends to
    sigma * sqrt(2 / pi), so the signal's standard deviation is their ratio.
    """
    from statistics import NormalDist  # noqa: PLC0415

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    return z * (alpha / (2 - alpha)) ** 0.5 / (2 / 3.141592653589793) ** 0.5


def tracking_signal(
    df: Any,
    model: str,
    method: str = "trigg",
    threshold: float | None = None,
    alpha: float = 0.1,
    window: int | None = None,
    full: bool = False,
) -> pl.DataFrame:
    """Tracking signal per series: is the forecast biased, not just noisy?

    method="trigg" (default)
        Trigg's smoothed signal: exponentially smoothed (actual - forecast)
        divided by smoothed |actual - forecast|, alpha 0.1. Bounded in
        [-1, 1]; the default threshold is the 95% limit for unbiased errors
        (about 0.56 at alpha 0.1, see `trigg_limit`).
    method="rsfe"
        The textbook running sum of forecast errors divided by MAD, over the
        last `window` periods (all periods if None). Default threshold 4.
        Note that for unbiased errors its spread grows with the square root
        of the window, so long windows raise false alarms.

    Positive values mean persistent UNDER-forecast. Returns the last value
    per series with a `flag` ("under-forecast", "over-forecast" or "ok");
    `full=True` returns every period.
    """
    data = to_polars(df).filter(pl.col(model).is_not_null()).sort(ID, DS)
    data = data.with_columns((pl.col(Y) - pl.col(model)).alias("error"))
    if method == "trigg":
        threshold = trigg_limit(alpha) if threshold is None else threshold
        data = data.with_columns(
            pl.col("error").ewm_mean(alpha=alpha, adjust=False).over(ID).alias("smoothed_error"),
            pl.col("error").abs().ewm_mean(alpha=alpha, adjust=False).over(ID).alias("mad"),
        ).with_columns(
            pl.when(pl.col("mad") > 0)
            .then(pl.col("smoothed_error") / pl.col("mad"))
            .otherwise(0.0)
            .alias("tracking_signal")
        )
        extra = ["smoothed_error", "mad"]
    elif method == "rsfe":
        threshold = 4.0 if threshold is None else threshold
        if window:
            rsfe = pl.col("error").rolling_sum(window, min_samples=1).over(ID)
            mad = pl.col("error").abs().rolling_mean(window, min_samples=1).over(ID)
        else:
            rsfe = pl.col("error").cum_sum().over(ID)
            mad = pl.col("error").abs().cum_sum().over(ID) / pl.int_range(1, pl.len() + 1).over(ID)
        data = data.with_columns(rsfe.alias("rsfe"), mad.alias("mad")).with_columns(
            pl.when(pl.col("mad") > 0)
            .then(pl.col("rsfe") / pl.col("mad"))
            .otherwise(0.0)
            .alias("tracking_signal")
        )
        extra = ["rsfe", "mad"]
    else:
        raise ValueError("method must be 'trigg' or 'rsfe'")

    data = data.with_columns(
        pl.when(pl.col("tracking_signal") > threshold)
        .then(pl.lit("under-forecast"))
        .when(pl.col("tracking_signal") < -threshold)
        .then(pl.lit("over-forecast"))
        .otherwise(pl.lit("ok"))
        .alias("flag"),
        pl.lit(threshold).alias("threshold"),
    )
    if full:
        return data.select(ID, DS, Y, model, "error", *extra, "tracking_signal", "flag")
    return (
        data.group_by(ID, maintain_order=True)
        .last()
        .select(ID, *extra, "tracking_signal", "threshold", "flag")
    )


__all__ = [
    "METRICS",
    "error_table",
    "mase_scale",
    "melt_forecasts",
    "series_errors",
    "tracking_signal",
    "trigg_limit",
]
