"""Forecast value added: the stairstep report.

FVA (Michael Gilliland, SAS) asks of every step in the forecasting process:
did it make the forecast better than the step before it? The usual steps:

    naive  ->  statistical model  ->  planner override  ->  consensus / final

FVA of a step = error of the previous step - error of this step, in
percentage points. Positive means the step added value. Negative means the
organisation spent effort to make the forecast worse.

The stairstep table puts every step in one column so the meeting can see,
in one glance, where value is added and where it is destroyed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from ._frame import ID, to_polars
from .metrics import error_table, series_errors

LOWER_IS_BETTER = {"wape", "mae", "rmse", "mase", "abs_bias_pct"}
HIGHER_IS_BETTER = {"accuracy"}


def _metric_col(metric: str) -> pl.Expr:
    if metric == "abs_bias_pct":
        return pl.col("bias_pct").abs()
    return pl.col(metric)


def _check_metric(metric: str) -> int:
    if metric in LOWER_IS_BETTER:
        return 1
    if metric in HIGHER_IS_BETTER:
        return -1
    raise ValueError(
        f"Unsupported metric '{metric}'. Use one of {sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)}"
    )


def stairstep(
    df: Any,
    steps: Sequence[str],
    metric: str = "wape",
    by: Sequence[str] | str | None = None,
    scale: Any | None = None,
) -> pl.DataFrame:
    """The FVA stairstep table.

    Parameters
    ----------
    df:
        Rows of (unique_id, ds, y, <one column per step>), for example the
        output of `backtest` or your own forecast history joined to actuals.
    steps:
        Forecast columns in process order, the benchmark first. For example
        ["naive", "statistical", "planner", "final"].
    metric:
        "wape" (default), "mae", "rmse", "mase" (needs `scale`),
        "abs_bias_pct" or "accuracy".
    by:
        Optional grouping, such as "demand_class" or "abc".

    Returns
    -------
    One row per (group, step): the metric, bias_pct, fva_vs_previous and
    fva_vs_first, both in the metric's units (fractions for WAPE, so 0.031
    is 3.1 percentage points), signed so positive always means "this step
    added value".
    """
    if len(steps) < 2:
        raise ValueError("FVA needs at least two steps: a benchmark and one process step.")
    direction = _check_metric(metric)
    by_cols = [by] if isinstance(by, str) else list(by or [])
    if metric == "mase" and scale is None:
        raise ValueError("metric='mase' needs scale=mase_scale(history, season_length)")
    table = error_table(df, steps, by=by_cols or None, scale=scale)
    table = table.with_columns(_metric_col(metric).alias("value"))
    prev = direction * (pl.col("value").shift(1) - pl.col("value"))
    first = direction * (pl.col("value").first() - pl.col("value"))
    if by_cols:
        prev, first = prev.over(by_cols), first.over(by_cols)
    table = table.with_columns(prev.alias("fva_vs_previous"), first.alias("fva_vs_first"))
    keep = [*by_cols, "model", "n", "actual", "forecast_total", "value", "bias_pct"]
    if metric != "wape":
        keep.append("wape")
    out = table.select(*keep, "fva_vs_previous", "fva_vs_first")
    return out.rename({"model": "step", "value": metric})


def item_fva(
    df: Any,
    steps: Sequence[str],
    metric: str = "wape",
    scale: Any | None = None,
) -> pl.DataFrame:
    """FVA per series: which items does each step help, and which does it hurt?

    Returns one row per series with the metric for each step
    (`<metric>_<step>`) and the FVA of each step against the previous one and
    against the first step (`fva_<step>_vs_prev`, `fva_<step>_vs_first`).
    """
    direction = _check_metric(metric)
    per = series_errors(df, steps, scale=scale).with_columns(_metric_col(metric).alias("_v"))
    wide = per.pivot(on="model", index=ID, values="_v")
    wide = wide.rename({s: f"{metric}_{s}" for s in steps})
    first = f"{metric}_{steps[0]}"
    exprs = []
    for prev, cur in zip(steps[:-1], steps[1:], strict=True):
        exprs.append(
            (direction * (pl.col(f"{metric}_{prev}") - pl.col(f"{metric}_{cur}"))).alias(
                f"fva_{cur}_vs_prev"
            )
        )
        exprs.append(
            (direction * (pl.col(first) - pl.col(f"{metric}_{cur}"))).alias(f"fva_{cur}_vs_first")
        )
    actual = per.group_by(ID).agg(pl.col("actual").first())
    return wide.with_columns(exprs).join(actual, on=ID, how="left").sort("actual", descending=True)


def _score_row(items: pl.DataFrame, col: str, step: str, compared_to: str) -> dict[str, Any]:
    valid = items.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
    n = valid.height
    better = valid.filter(pl.col(col) > 0)
    worse = valid.filter(pl.col(col) < 0)
    return {
        "step": step,
        "compared_to": compared_to,
        "series": n,
        "share_better": better.height / n if n else None,
        "share_worse": worse.height / n if n else None,
        "share_equal": (n - better.height - worse.height) / n if n else None,
        "actual_in_better": better["actual"].sum(),
        "actual_in_worse": worse["actual"].sum(),
    }


def fva_scorecard(items: pl.DataFrame, steps: Sequence[str]) -> pl.DataFrame:
    """Share of series where each step adds or destroys value.

    This is the headline a planning director remembers: "the statistical
    model beats seasonal naive on 58% of items, and overrides make 41% of
    the items they touch worse".
    """
    rows = [
        _score_row(items, f"fva_{cur}_vs_prev", cur, prev)
        for prev, cur in zip(steps[:-1], steps[1:], strict=True)
    ]
    rows += [_score_row(items, f"fva_{cur}_vs_first", cur, steps[0]) for cur in steps[2:]]
    return pl.DataFrame(rows)


def headline(stairs: pl.DataFrame, scorecard: pl.DataFrame, metric: str = "wape") -> list[str]:
    """Plain-language findings for the top of the report."""
    lines: list[str] = []
    data = to_polars(stairs)
    if "step" not in data.columns or data.is_empty():
        return lines
    steps = data["step"].to_list()
    values = data[metric].to_list()
    is_pct = metric in {"wape", "abs_bias_pct", "accuracy"}
    unit = " pp" if is_pct else ""
    scale = 100 if is_pct else 1
    pick = max if metric in HIGHER_IS_BETTER else min
    best_i = pick(range(len(values)), key=lambda i: values[i])
    pct = "%" if is_pct else ""
    lines.append(
        f"Best step overall: {steps[best_i]} ({metric.upper()} {values[best_i] * scale:.1f}{pct})."
    )
    for row in data.iter_rows(named=True):
        fva = row["fva_vs_previous"]
        if fva is None:
            continue
        verb = "added" if fva > 0 else "destroyed" if fva < 0 else "changed"
        lines.append(
            f"{row['step']} {verb} {abs(fva) * scale:.1f}{unit} of {metric.upper()} "
            "versus the step before it."
        )
    last = steps[-1]
    for row in scorecard.iter_rows(named=True):
        if row["share_better"] is None:
            continue
        adjacent = steps.index(row["step"]) - steps.index(row["compared_to"]) == 1
        if adjacent or (row["step"] == last and row["compared_to"] == steps[0]):
            lines.append(
                f"{row['step']} beats {row['compared_to']} on {row['share_better']:.0%} of series "
                f"and is worse on {row['share_worse']:.0%}."
            )
    return lines


__all__ = ["fva_scorecard", "headline", "item_fva", "stairstep"]
