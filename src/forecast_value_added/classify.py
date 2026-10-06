"""Demand classification: Syntetos-Boylan (ADI / CV squared) and ABC-XYZ.

Why it matters to a planner: the right forecasting method and the right
accuracy expectation both depend on the demand pattern. Judging a lumpy
spare part by the same WAPE target as a smooth fast mover is how forecast
accuracy reviews go wrong.

Syntetos-Boylan quadrants (Syntetos, Boylan and Croston, 2005):

    ADI  = average inter-demand interval (periods per non-zero demand)
    CV2  = squared coefficient of variation of the non-zero demand sizes

                      CV2 < 0.49      CV2 >= 0.49
    ADI <  1.32       smooth          erratic
    ADI >= 1.32       intermittent    lumpy
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl

from ._frame import ID, Y, validate

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49

DEMAND_CLASSES = ("smooth", "erratic", "intermittent", "lumpy")

#: Method most planners and the literature recommend per class.
RECOMMENDED_METHOD = {
    "smooth": "exponential smoothing (ETS) or seasonal naive",
    "erratic": "exponential smoothing with wider safety stock",
    "intermittent": "Croston SBA or TSB",
    "lumpy": "Croston SBA or TSB; consider make-to-order",
    "insufficient history": "judgment or a category-level forecast",
    "no demand": "candidate for discontinuation review",
}


def _adi_intervals(values: np.ndarray) -> float:
    """Mean gap between non-zero demands, counting the first gap from the series start."""
    idx = np.flatnonzero(values > 0)
    if idx.size == 0:
        return float("nan")
    gaps = np.diff(np.concatenate(([-1], idx)))
    return float(gaps.mean())


def demand_profile(
    df: Any,
    adi_method: Literal["ratio", "intervals"] = "ratio",
    adi_cutoff: float = ADI_CUTOFF,
    cv2_cutoff: float = CV2_CUTOFF,
    min_nonzero: int = 2,
) -> pl.DataFrame:
    """Classify every series into smooth, erratic, intermittent or lumpy.

    Parameters
    ----------
    df:
        Long-format actuals (unique_id, ds, y) with a complete calendar.
        Run `fill_gaps` first if your extract skips zero periods.
    adi_method:
        "ratio" = periods / non-zero periods (the common practitioner form).
        "intervals" = mean gap between non-zero demands (the paper's form).
    min_nonzero:
        Series with fewer non-zero periods are labelled "insufficient history".

    Returns
    -------
    One row per series with n_periods, n_nonzero, zero_share, mean_demand,
    adi, cv2, demand_class and recommended_method.
    """
    data = validate(df)
    stats = data.group_by(ID, maintain_order=True).agg(
        pl.len().alias("n_periods"),
        (pl.col(Y) > 0).sum().alias("n_nonzero"),
        pl.col(Y).mean().alias("mean_demand"),
        pl.col(Y).filter(pl.col(Y) > 0).mean().alias("mean_nonzero"),
        pl.col(Y).filter(pl.col(Y) > 0).std(ddof=0).alias("std_nonzero"),
        pl.col(Y).alias("_values"),
    )
    if adi_method == "ratio":
        stats = stats.with_columns(
            (pl.col("n_periods") / pl.col("n_nonzero").cast(pl.Float64)).alias("adi")
        )
    elif adi_method == "intervals":
        stats = stats.with_columns(
            pl.col("_values")
            .map_elements(lambda s: _adi_intervals(np.asarray(s)), return_dtype=pl.Float64)
            .alias("adi")
        )
    else:
        raise ValueError("adi_method must be 'ratio' or 'intervals'")

    stats = stats.with_columns(
        pl.when(pl.col("n_nonzero") > 0)
        .then((pl.col("std_nonzero") / pl.col("mean_nonzero")) ** 2)
        .otherwise(None)
        .alias("cv2"),
        (1 - pl.col("n_nonzero") / pl.col("n_periods")).alias("zero_share"),
    )
    stats = stats.with_columns(
        pl.when(pl.col("n_nonzero") == 0)
        .then(pl.lit("no demand"))
        .when(pl.col("n_nonzero") < min_nonzero)
        .then(pl.lit("insufficient history"))
        .when((pl.col("adi") < adi_cutoff) & (pl.col("cv2") < cv2_cutoff))
        .then(pl.lit("smooth"))
        .when(pl.col("adi") < adi_cutoff)
        .then(pl.lit("erratic"))
        .when(pl.col("cv2") < cv2_cutoff)
        .then(pl.lit("intermittent"))
        .otherwise(pl.lit("lumpy"))
        .alias("demand_class")
    )
    stats = stats.with_columns(
        pl.col("demand_class").replace_strict(RECOMMENDED_METHOD).alias("recommended_method")
    )
    return stats.select(
        ID,
        "n_periods",
        "n_nonzero",
        "zero_share",
        "mean_demand",
        "adi",
        "cv2",
        "demand_class",
        "recommended_method",
    )


def abc_xyz(
    df: Any,
    value_col: str | None = None,
    price: Any | None = None,
    a_share: float = 0.80,
    b_share: float = 0.95,
    x_cv: float = 0.5,
    y_cv: float = 1.0,
) -> pl.DataFrame:
    """ABC by cumulative value share and XYZ by demand variability.

    ABC: series are ranked by value (sum of `value_col`, or y times a price,
    or plain units). The series that crosses the `a_share` cumulative line is
    still an A, so A always holds at least one series.

    XYZ: coefficient of variation of per-period demand, zeros included.
    X <= x_cv (stable), Y <= y_cv (variable), Z above (unpredictable).

    `price` may be a DataFrame with unique_id and price columns.
    """
    data = validate(df, numeric=[c for c in (Y, value_col) if c])
    if price is not None:
        p = validate(price, required=(ID, "price"), numeric=("price",))
        data = data.join(p.select(ID, "price"), on=ID, how="left").with_columns(
            (pl.col(Y) * pl.col("price").fill_null(1.0)).alias("_value")
        )
    elif value_col:
        data = data.with_columns(pl.col(value_col).alias("_value"))
    else:
        data = data.with_columns(pl.col(Y).alias("_value"))

    agg = data.group_by(ID).agg(
        pl.col("_value").sum().alias("value"),
        pl.col(Y).mean().alias("_mean"),
        pl.col(Y).std(ddof=0).alias("_std"),
    )
    total = agg["value"].sum()
    agg = agg.sort("value", descending=True).with_columns(
        (pl.col("value") / total if total else pl.lit(0.0)).alias("value_share"),
    )
    agg = agg.with_columns(
        pl.col("value_share").cum_sum().alias("cum_share"),
        (pl.col("value_share").cum_sum() - pl.col("value_share")).alias("_cum_before"),
    )
    agg = agg.with_columns(
        pl.when(pl.col("_cum_before") < a_share)
        .then(pl.lit("A"))
        .when(pl.col("_cum_before") < b_share)
        .then(pl.lit("B"))
        .otherwise(pl.lit("C"))
        .alias("abc"),
        pl.when(pl.col("_mean") > 0)
        .then(pl.col("_std") / pl.col("_mean"))
        .otherwise(None)
        .alias("cv"),
    )
    agg = agg.with_columns(
        pl.when(pl.col("cv").is_null())
        .then(pl.lit("Z"))
        .when(pl.col("cv") <= x_cv)
        .then(pl.lit("X"))
        .when(pl.col("cv") <= y_cv)
        .then(pl.lit("Y"))
        .otherwise(pl.lit("Z"))
        .alias("xyz")
    ).with_columns((pl.col("abc") + pl.col("xyz")).alias("abc_xyz"))
    return agg.select(ID, "value", "value_share", "cum_share", "abc", "cv", "xyz", "abc_xyz")


def abc_xyz_matrix(classes: pl.DataFrame) -> pl.DataFrame:
    """Count of series in each ABC x XYZ cell, as a 3x3 table (rows A/B/C)."""
    counts = classes.group_by("abc", "xyz").len()
    rows = []
    for a in "ABC":
        row = {"abc": a}
        for x in "XYZ":
            hit = counts.filter((pl.col("abc") == a) & (pl.col("xyz") == x))
            row[x] = int(hit["len"][0]) if hit.height else 0
        rows.append(row)
    return pl.DataFrame(rows)


def classify(df: Any, **kwargs: Any) -> pl.DataFrame:
    """Demand profile and ABC-XYZ joined into one table per series."""
    profile_keys = {"adi_method", "adi_cutoff", "cv2_cutoff", "min_nonzero"}
    profile_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in profile_keys}
    profile = demand_profile(df, **profile_kwargs)
    abc = abc_xyz(df, **kwargs)
    return profile.join(abc, on=ID, how="left")


__all__ = [
    "ADI_CUTOFF",
    "CV2_CUTOFF",
    "DEMAND_CLASSES",
    "RECOMMENDED_METHOD",
    "abc_xyz",
    "abc_xyz_matrix",
    "classify",
    "demand_profile",
]
