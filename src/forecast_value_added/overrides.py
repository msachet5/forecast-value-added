"""Override analysis: which judgmental adjustments add value and which destroy it.

Published field studies (Fildes, Goodwin, Lawrence and Nikolopoulos, 2009,
across four supply chain companies) found that large adjustments tend to
improve accuracy, small ones mostly add noise, and upward adjustments are
wrong far more often than downward ones, because optimism leaks into the
number. This module tests those patterns on your own data.

An override is any row where the final forecast differs from the baseline
(statistical) forecast by more than `tolerance` (relative).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from ._frame import ID, Y, to_polars

DEFAULT_SIZE_BINS = (0.10, 0.25, 0.50)


def _size_labels(bins: Sequence[float]) -> list[str]:
    labels = [f"<{bins[0]:.0%}"]
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        labels.append(f"{lo:.0%}-{hi:.0%}")
    labels.append(f">={bins[-1]:.0%}")
    return labels


def tag_overrides(
    df: Any,
    baseline: str = "statistical",
    final: str = "final",
    tolerance: float = 0.005,
    size_bins: Sequence[float] = DEFAULT_SIZE_BINS,
) -> pl.DataFrame:
    """Add override direction, size bucket and outcome to every forecast row.

    New columns:
      adjustment      final - baseline (units)
      adjustment_pct  adjustment / baseline (inf-safe; baseline 0 counts as >= top bin)
      direction       "up", "down" or "none"
      size_bucket     relative size band, e.g. "10%-25%"
      abs_err_base    |actual - baseline|
      abs_err_final   |actual - final|
      error_change    abs_err_final - abs_err_base (negative = override helped)
      outcome         "improved", "worsened" or "no change"
    """
    data = to_polars(df)
    for c in (baseline, final, Y):
        if c not in data.columns:
            raise KeyError(f"Column '{c}' not found. Columns: {data.columns}")
    labels = _size_labels(size_bins)
    adj = pl.col(final) - pl.col(baseline)
    pct = (
        pl.when(pl.col(baseline).abs() > 0)
        .then(adj / pl.col(baseline).abs())
        .when(adj.abs() > 0)
        .then(pl.lit(float("inf")))
        .otherwise(0.0)
    )
    out = data.with_columns(adj.alias("adjustment"), pct.alias("adjustment_pct"))
    out = out.with_columns(
        pl.when(pl.col("adjustment_pct").abs() <= tolerance)
        .then(pl.lit("none"))
        .when(pl.col("adjustment") > 0)
        .then(pl.lit("up"))
        .otherwise(pl.lit("down"))
        .alias("direction"),
    )
    size_expr = pl.when(pl.col("direction") == "none").then(pl.lit("none"))
    abs_pct = pl.col("adjustment_pct").abs()
    size_expr = size_expr.when(abs_pct < size_bins[0]).then(pl.lit(labels[0]))
    for i in range(1, len(size_bins)):
        size_expr = size_expr.when(abs_pct < size_bins[i]).then(pl.lit(labels[i]))
    size_expr = size_expr.otherwise(pl.lit(labels[-1]))
    out = out.with_columns(
        size_expr.alias("size_bucket"),
        (pl.col(Y) - pl.col(baseline)).abs().alias("abs_err_base"),
        (pl.col(Y) - pl.col(final)).abs().alias("abs_err_final"),
    ).with_columns((pl.col("abs_err_final") - pl.col("abs_err_base")).alias("error_change"))
    return out.with_columns(
        pl.when(pl.col("error_change") < 0)
        .then(pl.lit("improved"))
        .when(pl.col("error_change") > 0)
        .then(pl.lit("worsened"))
        .otherwise(pl.lit("no change"))
        .alias("outcome")
    )


def override_summary(
    df: Any,
    baseline: str = "statistical",
    final: str = "final",
    by: Sequence[str] = ("direction", "size_bucket"),
    tolerance: float = 0.005,
    size_bins: Sequence[float] = DEFAULT_SIZE_BINS,
) -> pl.DataFrame:
    """Win rate and value added by override type.

    Returns one row per group (default: direction x size bucket) with the
    number of overrides, win rate (share improved), WAPE before and after,
    FVA in percentage points and total units of absolute error added or
    removed. Sorted with the most value-destroying groups first.
    """
    tagged = tag_overrides(df, baseline, final, tolerance, size_bins)
    tagged = tagged.filter(pl.col("direction") != "none")
    if tagged.is_empty():
        return pl.DataFrame(
            schema={
                **{b: pl.Utf8 for b in by},
                "overrides": pl.UInt32,
                "win_rate": pl.Float64,
                "wape_before": pl.Float64,
                "wape_after": pl.Float64,
                "fva_pp": pl.Float64,
                "error_units_change": pl.Float64,
            }
        )
    out = tagged.group_by(list(by)).agg(
        pl.len().alias("overrides"),
        (pl.col("outcome") == "improved").mean().alias("win_rate"),
        (pl.col("abs_err_base").sum() / pl.col(Y).sum()).alias("wape_before"),
        (pl.col("abs_err_final").sum() / pl.col(Y).sum()).alias("wape_after"),
        pl.col("error_change").sum().alias("error_units_change"),
        pl.col("adjustment").mean().alias("mean_adjustment"),
    )
    out = out.with_columns(((pl.col("wape_before") - pl.col("wape_after")) * 100).alias("fva_pp"))
    return out.sort("error_units_change", descending=True)


def override_overview(
    df: Any,
    baseline: str = "statistical",
    final: str = "final",
    tolerance: float = 0.005,
) -> dict[str, float]:
    """Single-number view: how often planners override and whether it pays."""
    tagged = tag_overrides(df, baseline, final, tolerance)
    total = tagged.height
    over = tagged.filter(pl.col("direction") != "none")
    n = over.height
    actual = tagged[Y].sum()
    return {
        "rows": float(total),
        "override_rate": n / total if total else 0.0,
        "win_rate": float((over["outcome"] == "improved").mean()) if n else 0.0,
        "up_share": float((over["direction"] == "up").mean()) if n else 0.0,
        "wape_baseline": float(tagged["abs_err_base"].sum() / actual) if actual else float("nan"),
        "wape_final": float(tagged["abs_err_final"].sum() / actual) if actual else float("nan"),
        "error_units_added": float(over.filter(pl.col("error_change") > 0)["error_change"].sum()),
        "error_units_removed": float(
            -over.filter(pl.col("error_change") < 0)["error_change"].sum()
        ),
    }


def worst_overriders(
    df: Any,
    who: str,
    baseline: str = "statistical",
    final: str = "final",
    min_overrides: int = 5,
) -> pl.DataFrame:
    """Rank planners, customers or categories (`who`) by value destroyed through overrides.

    Use carefully: the point is coaching and process design, not blame.
    """
    tagged = tag_overrides(df, baseline, final).filter(pl.col("direction") != "none")
    return (
        tagged.group_by(who)
        .agg(
            pl.len().alias("overrides"),
            (pl.col("outcome") == "improved").mean().alias("win_rate"),
            pl.col("error_change").sum().alias("error_units_change"),
            pl.col(ID).n_unique().alias("series"),
        )
        .filter(pl.col("overrides") >= min_overrides)
        .sort("error_units_change", descending=True)
    )


__all__ = ["override_overview", "override_summary", "tag_overrides", "worst_overriders"]
