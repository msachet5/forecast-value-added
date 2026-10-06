"""Safety stock from forecast error, the way the forecast is actually used.

Textbook safety stock uses the standard deviation of demand. When a
forecast drives replenishment, the uncertainty that matters is the
forecast error, which is why FVA and safety stock belong in one library:
a step that destroys forecast value also inflates inventory.

    SS  = z(service level) * sigma_error * sqrt(lead time + review period)
    ROP = mean demand * lead time + SS

`method="demand"` adds lead-time variability:

    SS = z * sqrt((L + R) * sigma_d^2 + d^2 * sigma_L^2)

Both use the normal approximation, which understates risk for
intermittent and lumpy items. Those rows get a `caution` note.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Literal

import polars as pl

from ._frame import ID, Y, to_polars


def z_value(service_level: float) -> float:
    """Inverse standard normal for a cycle service level such as 0.95."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1, for example 0.95")
    return NormalDist().inv_cdf(service_level)


def safety_stock(
    evaluation: Any,
    step: str,
    lead_time: float | Any = 1.0,
    service_level: float = 0.95,
    review_period: float = 0.0,
    method: Literal["forecast_error", "demand"] = "forecast_error",
    lead_time_std: float | Any = 0.0,
    classes: Any | None = None,
) -> pl.DataFrame:
    """Safety stock and reorder point per series.

    Parameters
    ----------
    evaluation:
        Backtest or forecast history with unique_id, y and the `step` column
        (a `horizon` column, if present, restricts error to one-step-ahead).
    step:
        The forecast step whose error drives safety stock (usually the final one).
    lead_time, lead_time_std:
        In periods. A number, or a DataFrame with unique_id and lead_time
        (and optionally lead_time_std) per series.
    classes:
        Optional demand_profile output to flag intermittent and lumpy series.
    """
    data = to_polars(evaluation)
    if "horizon" in data.columns:
        data = data.filter(pl.col("horizon") == 1)
    stats = data.group_by(ID).agg(
        pl.col(Y).mean().alias("mean_demand"),
        pl.col(Y).std(ddof=1).alias("sigma_demand"),
        ((pl.col(step) - pl.col(Y)) ** 2).mean().sqrt().alias("sigma_error"),
        pl.len().alias("n_obs"),
    )
    if isinstance(lead_time, (int, float)):
        stats = stats.with_columns(pl.lit(float(lead_time)).alias("lead_time"))
    else:
        lt = to_polars(lead_time).select(ID, pl.col("lead_time").cast(pl.Float64))
        stats = stats.join(lt, on=ID, how="left")
    if isinstance(lead_time_std, (int, float)):
        stats = stats.with_columns(pl.lit(float(lead_time_std)).alias("lead_time_std"))
    else:
        lts = to_polars(lead_time_std).select(ID, pl.col("lead_time_std").cast(pl.Float64))
        stats = stats.join(lts, on=ID, how="left")

    z = z_value(service_level)
    exposure = pl.col("lead_time") + review_period
    if method == "forecast_error":
        ss = z * pl.col("sigma_error") * exposure.sqrt()
    elif method == "demand":
        ss = (
            z
            * (
                exposure * pl.col("sigma_demand") ** 2
                + pl.col("mean_demand") ** 2 * pl.col("lead_time_std") ** 2
            ).sqrt()
        )
    else:
        raise ValueError("method must be 'forecast_error' or 'demand'")

    out = stats.with_columns(
        pl.lit(service_level).alias("service_level"),
        pl.lit(z).alias("z"),
        ss.alias("safety_stock"),
    ).with_columns(
        (pl.col("mean_demand") * pl.col("lead_time") + pl.col("safety_stock")).alias(
            "reorder_point"
        ),
        pl.when(pl.col("mean_demand") > 0)
        .then(pl.col("safety_stock") / pl.col("mean_demand"))
        .otherwise(None)
        .alias("cover_periods"),
    )
    if classes is not None:
        cls = to_polars(classes).select(ID, "demand_class")
        out = out.join(cls, on=ID, how="left").with_columns(
            pl.when(pl.col("demand_class").is_in(["intermittent", "lumpy"]))
            .then(
                pl.lit(
                    "normal approximation understates risk; consider a simulation or Poisson/negative binomial model"
                )
            )
            .otherwise(pl.lit(""))
            .alias("caution")
        )
    return out.sort("safety_stock", descending=True)


__all__ = ["safety_stock", "z_value"]
