"""Benchmark forecasting methods, implemented in plain numpy.

These are the yardsticks FVA is measured against, plus the classic
intermittent-demand methods. They are deliberately simple and dependency
free so every backtest is reproducible on any laptop. For production-grade
ETS, ARIMA and the rest, use the statsforecast adapter in `backtest.py`.

Every model has the signature

    model(y: np.ndarray, h: int, season_length: int) -> np.ndarray

where `y` is the history (oldest first) and the return value has length h.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Model = Callable[[np.ndarray, int, int], np.ndarray]


def naive(y: np.ndarray, h: int, season_length: int = 1) -> np.ndarray:
    """Last observed value, repeated. The simplest benchmark in FVA practice."""
    if y.size == 0:
        return np.zeros(h)
    return np.full(h, y[-1], dtype=float)


def seasonal_naive(y: np.ndarray, h: int, season_length: int = 1) -> np.ndarray:
    """Value from the same period one season ago. Falls back to naive on short history."""
    m = max(int(season_length), 1)
    if y.size < m or m == 1:
        return naive(y, h)
    last_season = y[-m:]
    reps = int(np.ceil(h / m))
    return np.tile(last_season, reps)[:h].astype(float)


def historic_average(y: np.ndarray, h: int, season_length: int = 1) -> np.ndarray:
    """Mean of the full history."""
    return np.full(h, float(y.mean()) if y.size else 0.0)


def moving_average(window: int) -> Model:
    """Factory: mean of the last `window` periods."""

    def _ma(y: np.ndarray, h: int, season_length: int = 1) -> np.ndarray:
        if y.size == 0:
            return np.zeros(h)
        return np.full(h, float(y[-window:].mean()))

    _ma.__name__ = f"moving_average_{window}"
    return _ma


def _ses_level(y: np.ndarray, alpha: float) -> tuple[float, float]:
    """Run simple exponential smoothing; return final level and in-sample SSE."""
    level = float(y[0])
    sse = 0.0
    for value in y[1:]:
        err = value - level
        sse += err * err
        level += alpha * err
    return level, sse


def _ses_grid(y: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run SES for many alphas at once. Returns final levels and in-sample SSE per alpha."""
    level = np.full(alphas.shape, float(y[0]))
    sse = np.zeros(alphas.shape)
    for value in y[1:]:
        err = value - level
        sse += err * err
        level = level + alphas * err
    return level, sse


ALPHA_GRID = np.linspace(0.05, 0.95, 19)


def ses(y: np.ndarray, h: int, season_length: int = 1, alpha: float | None = None) -> np.ndarray:
    """Simple exponential smoothing. Alpha is grid-searched on one-step SSE when not given."""
    if y.size == 0:
        return np.zeros(h)
    if y.size < 3:
        return naive(y, h)
    if alpha is None:
        levels, sse = _ses_grid(y, ALPHA_GRID)
        return np.full(h, float(levels[int(np.argmin(sse))]))
    level, _ = _ses_level(y, alpha)
    return np.full(h, level)


def croston(
    y: np.ndarray,
    h: int,
    season_length: int = 1,
    alpha: float = 0.1,
    variant: str = "classic",
) -> np.ndarray:
    """Croston's method for intermittent demand.

    Smooths non-zero demand size (z) and the interval between demands (p)
    separately and forecasts z / p per period. `variant="sba"` applies the
    Syntetos-Boylan bias correction (1 - alpha / 2).
    """
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return np.zeros(h)
    sizes = y[nz].astype(float)
    intervals = np.diff(np.concatenate(([-1], nz))).astype(float)
    z, p = sizes[0], intervals[0]
    for size, interval in zip(sizes[1:], intervals[1:], strict=True):
        z += alpha * (size - z)
        p += alpha * (interval - p)
    rate = z / p
    if variant == "sba":
        rate *= 1 - alpha / 2
    elif variant != "classic":
        raise ValueError("variant must be 'classic' or 'sba'")
    return np.full(h, rate)


def croston_sba(y: np.ndarray, h: int, season_length: int = 1) -> np.ndarray:
    """Croston with the Syntetos-Boylan Approximation bias correction."""
    return croston(y, h, season_length, variant="sba")


def tsb(
    y: np.ndarray,
    h: int,
    season_length: int = 1,
    alpha_d: float = 0.1,
    alpha_p: float = 0.1,
) -> np.ndarray:
    """Teunter-Syntetos-Babai: smooths demand probability every period.

    Unlike Croston, the forecast decays when demand stops, which suits
    items heading toward obsolescence.
    """
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return np.zeros(h)
    prob = float((y > 0).mean())
    size = float(y[nz[0]])
    for value in y:
        occurred = 1.0 if value > 0 else 0.0
        prob += alpha_p * (occurred - prob)
        if occurred:
            size += alpha_d * (value - size)
    return np.full(h, prob * size)


def drift(y: np.ndarray, h: int, season_length: int = 1) -> np.ndarray:
    """Random walk with drift: naive plus the average historical change."""
    if y.size < 2:
        return naive(y, h)
    slope = (y[-1] - y[0]) / (y.size - 1)
    return y[-1] + slope * np.arange(1, h + 1)


def class_aware(y: np.ndarray, h: int, season_length: int = 1) -> np.ndarray:
    """Pick a method from the series' own demand pattern.

    Smooth and erratic series get the better of SES and seasonal naive on
    the last season of history; intermittent and lumpy series get Croston SBA.
    A planner's rule of thumb, encoded.
    """
    if y.size == 0:
        return np.zeros(h)
    n_nonzero = int((y > 0).sum())
    adi = y.size / n_nonzero if n_nonzero else np.inf
    if adi >= 1.32:
        return croston_sba(y, h, season_length)
    m = max(int(season_length), 1)
    if m > 1 and y.size >= 3 * m:
        train, test = y[:-m], y[-m:]
        ses_err = np.abs(ses(train, m, m) - test).sum()
        snaive_err = np.abs(seasonal_naive(train, m, m) - test).sum()
        if snaive_err < ses_err:
            return seasonal_naive(y, h, m)
    return ses(y, h, m)


BUILTIN_MODELS: dict[str, Model] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "historic_average": historic_average,
    "moving_average_4": moving_average(4),
    "moving_average_13": moving_average(13),
    "ses": ses,
    "croston": croston,
    "croston_sba": croston_sba,
    "tsb": tsb,
    "drift": drift,
    "class_aware": class_aware,
}


def get_model(name: str) -> Model:
    """Look up a built-in model by name, including moving_average_<k>."""
    if name in BUILTIN_MODELS:
        return BUILTIN_MODELS[name]
    if name.startswith("moving_average_") and name.rsplit("_", 1)[-1].isdigit():
        return moving_average(int(name.rsplit("_", 1)[-1]))
    raise KeyError(f"Unknown model '{name}'. Built-ins: {sorted(BUILTIN_MODELS)}")


__all__ = [
    "BUILTIN_MODELS",
    "Model",
    "class_aware",
    "croston",
    "croston_sba",
    "drift",
    "get_model",
    "historic_average",
    "moving_average",
    "naive",
    "seasonal_naive",
    "ses",
    "tsb",
]
