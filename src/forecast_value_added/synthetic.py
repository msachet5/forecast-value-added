"""Synthetic multi-location demand and forecast-process data.

Real demand history is confidential, so every example, test and demo in
this project runs on data generated here (or on the public M5 dataset).
The generator produces the four Syntetos-Boylan demand patterns across
items and locations, with seasonality, trend and promotions, and can
simulate a realistic forecasting process on top of it: a statistical
forecast, planner overrides with an optimism bias, and a consensus step.

Nothing here is derived from any company's data.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta

import numpy as np
import polars as pl

from ._frame import DS, ID, Y
from .backtest import backtest
from .models import class_aware, naive, seasonal_naive

DEFAULT_MIX = {"smooth": 0.40, "erratic": 0.20, "intermittent": 0.25, "lumpy": 0.15}
CATEGORIES = ("Beverages", "Snacks", "Household", "Personal Care", "Hardware", "Auto Parts")
PLANNERS = ("Planner A", "Planner B", "Planner C", "Planner D")


def _pattern(
    rng: np.random.Generator,
    kind: str,
    n: int,
    level: float,
    season: np.ndarray,
    trend: np.ndarray,
) -> np.ndarray:
    mean = np.maximum(level * season * trend, 0.01)
    if kind == "smooth":
        noise = rng.gamma(shape=25.0, scale=1 / 25.0, size=n)  # CV about 0.2
        return rng.poisson(mean * noise).astype(float)
    if kind == "erratic":
        noise = rng.gamma(shape=1.2, scale=1 / 1.2, size=n)  # CV about 0.9
        return rng.poisson(mean * noise).astype(float)
    p = rng.uniform(0.15, 0.6)
    occur = rng.random(n) < p
    size_mean = np.maximum(mean / p, 1.0)
    if kind == "intermittent":
        sizes = 1 + rng.poisson(np.maximum(size_mean - 1, 0.1))
    elif kind == "lumpy":
        sizes = 1 + rng.poisson(size_mean * rng.gamma(shape=0.8, scale=1 / 0.8, size=n))
    else:
        raise ValueError(f"Unknown pattern '{kind}'")
    return np.where(occur, sizes, 0).astype(float)


def generate_sales(
    n_items: int = 120,
    n_locations: int = 4,
    n_periods: int = 156,
    start: str | date = "2023-01-02",
    freq: str = "1w",
    season_length: int = 52,
    class_mix: Mapping[str, float] = DEFAULT_MIX,
    promo_rate: float = 0.03,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate long-format sales for items x locations.

    Returns unique_id ("SKU0001@LOC01"), item_id, location_id, category,
    pattern (the pattern the series was generated from), price, ds, y.
    """
    rng = np.random.default_rng(seed)
    start_d = date.fromisoformat(start) if isinstance(start, str) else start
    dates = pl.Series(
        DS,
        pl.date_range(start_d, _end_date(start_d, freq, n_periods), interval=freq, eager=True)[
            :n_periods
        ],
    )
    kinds = list(class_mix)
    probs = np.array([class_mix[k] for k in kinds], dtype=float)
    probs /= probs.sum()
    t = np.arange(n_periods)
    loc_mult = rng.uniform(0.5, 1.6, size=n_locations)

    frames = []
    for i in range(n_items):
        kind = str(rng.choice(kinds, p=probs))
        category = str(rng.choice(CATEGORIES))
        price = float(np.round(rng.lognormal(mean=2.5, sigma=0.8), 2))
        base = (
            float(rng.lognormal(mean=2.3, sigma=0.9))
            if kind in {"smooth", "erratic"}
            else float(rng.lognormal(mean=0.6, sigma=0.6))
        )
        amp = rng.uniform(0.0, 0.45) if kind in {"smooth", "erratic"} else rng.uniform(0, 0.2)
        phase = rng.uniform(0, 2 * np.pi)
        season = 1 + amp * np.sin(2 * np.pi * t / max(season_length, 1) + phase)
        slope = rng.normal(0, 0.003)
        trend = np.maximum(1 + slope * t, 0.2)
        for j in range(n_locations):
            y = _pattern(rng, kind, n_periods, base * loc_mult[j], season, trend)
            if kind == "smooth" and promo_rate > 0:
                promo = rng.random(n_periods) < promo_rate
                y = np.where(promo, np.round(y * rng.uniform(1.5, 2.5)), y)
            frames.append(
                pl.DataFrame(
                    {
                        ID: f"SKU{i + 1:04d}@LOC{j + 1:02d}",
                        "item_id": f"SKU{i + 1:04d}",
                        "location_id": f"LOC{j + 1:02d}",
                        "category": category,
                        "pattern": kind,
                        "price": price,
                        DS: dates,
                        Y: y,
                    }
                )
            )
    return pl.concat(frames).sort(ID, DS)


def _end_date(start: date, freq: str, n: int) -> date:
    """Generous end date so date_range yields at least n periods."""
    unit = freq.lstrip("0123456789")
    step = int(freq[: len(freq) - len(unit)] or 1)
    days = {"d": 1, "w": 7, "mo": 31, "q": 92, "y": 366}.get(unit, 7)
    return start + timedelta(days=days * step * (n + 1))


def simulate_process(
    sales: pl.DataFrame,
    h: int = 4,
    n_windows: int = 8,
    season_length: int = 52,
    override_rate: float = 0.30,
    optimism: float = 0.12,
    insight: float = 0.35,
    consensus_uplift: float = 0.05,
    seed: int = 7,
) -> pl.DataFrame:
    """Simulate a forecasting process over a rolling backtest.

    Steps produced, in process order:
      naive, seasonal_naive  benchmarks
      statistical            class-aware model (SES / seasonal naive / Croston SBA)
      planner                statistical plus judgmental overrides on `override_rate` of rows.
                             Overrides mix real insight (a share `insight` of the true
                             error, more often on big adjustments) with an upward
                             `optimism` bias, as field studies report.
      final                  consensus: sales adds `consensus_uplift` on A-category items.

    Also adds a `planner_name` column for override coaching analysis.
    """
    rng = np.random.default_rng(seed)
    base = sales.select(ID, DS, Y)
    bt = backtest(
        base,
        models={"naive": naive, "seasonal_naive": seasonal_naive, "statistical": class_aware},
        h=h,
        n_windows=n_windows,
        season_length=season_length,
    )
    n = bt.height
    stat = bt["statistical"].to_numpy()
    actual = bt[Y].to_numpy()
    touched = rng.random(n) < override_rate
    big = rng.random(n) < 0.35
    informed = np.where(big, insight, insight * 0.15) * (actual - stat)
    bias = optimism * np.maximum(stat, 1.0) * rng.uniform(0.0, 2.0, size=n)
    noise = rng.normal(0, 0.25, size=n) * np.maximum(stat, 1.0)
    adj = np.where(touched, informed + bias + noise, 0.0)
    planner = np.maximum(stat + adj, 0.0)

    items = sales.select(ID, "category").unique(ID)
    planners = {uid: PLANNERS[k % len(PLANNERS)] for k, uid in enumerate(items[ID].to_list())}
    out = bt.with_columns(pl.Series("planner", planner))
    value = (
        sales.group_by(ID)
        .agg((pl.col(Y) * pl.col("price")).sum().alias("_v"))
        .sort("_v", descending=True)
        .with_columns((pl.col("_v").cum_sum() / pl.col("_v").sum()).alias("_cum"))
        .with_columns((pl.col("_cum") - pl.col("_v") / pl.col("_v").sum() < 0.8).alias("_is_a"))
        .select(ID, "_is_a")
    )
    out = (
        out.join(value, on=ID, how="left")
        .with_columns(
            pl.when(pl.col("_is_a"))
            .then(pl.col("planner") * (1 + consensus_uplift))
            .otherwise(pl.col("planner"))
            .alias("final"),
            pl.col(ID).replace_strict(planners, default="Planner A").alias("planner_name"),
        )
        .drop("_is_a")
    )
    return out


__all__ = ["generate_sales", "simulate_process"]
