"""One call from raw files to a full FVA review.

from forecast_value_added import analyze
result = analyze(sales, forecasts, steps=["naive", "statistical", "final"])
result.to_html("fva_report.html")
result.to_parquet("fva_tables/")   # for Power BI
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ._frame import DS, ID, Y, to_polars, validate
from .backtest import backtest
from .classify import abc_xyz, abc_xyz_matrix, demand_profile
from .fva import fva_scorecard, headline, item_fva, stairstep
from .metrics import error_table, mase_scale, tracking_signal
from .overrides import override_overview, override_summary

DEFAULT_MODELS = ("naive", "seasonal_naive", "class_aware")


@dataclass
class FVAResult:
    """Everything an FVA review needs, as polars tables."""

    steps: list[str]
    metric: str
    evaluation: pl.DataFrame
    classes: pl.DataFrame
    stairstep: pl.DataFrame
    stairstep_by_class: pl.DataFrame
    stairstep_by_abc: pl.DataFrame
    errors_by_horizon: pl.DataFrame
    items: pl.DataFrame
    scorecard: pl.DataFrame
    tracking: pl.DataFrame
    abc_xyz_matrix: pl.DataFrame
    override_summary: pl.DataFrame | None = None
    override_overview: dict[str, float] | None = None
    findings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def tables(self) -> dict[str, pl.DataFrame]:
        """All tables by name, ready to export."""
        out = {
            "evaluation": self.evaluation,
            "classes": self.classes,
            "stairstep": self.stairstep,
            "stairstep_by_class": self.stairstep_by_class,
            "stairstep_by_abc": self.stairstep_by_abc,
            "errors_by_horizon": self.errors_by_horizon,
            "items": self.items,
            "scorecard": self.scorecard,
            "tracking": self.tracking,
            "abc_xyz_matrix": self.abc_xyz_matrix,
        }
        if self.override_summary is not None:
            out["override_summary"] = self.override_summary
        return out

    def to_parquet(self, directory: str | Path) -> list[Path]:
        """Write every table as Parquet (Power BI: Get Data > Parquet, or a folder source)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, table in self.tables().items():
            path = directory / f"{name}.parquet"
            table.write_parquet(path)
            paths.append(path)
        return paths

    def to_csv(self, directory: str | Path) -> list[Path]:
        """Write every table as CSV, for Excel users."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, table in self.tables().items():
            path = directory / f"{name}.csv"
            table.write_csv(path)
            paths.append(path)
        return paths

    def to_html(self, path: str | Path, title: str = "Forecast Value Added review") -> Path:
        """Render the one-page HTML report."""
        from .report import render_html  # noqa: PLC0415

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(self, title=title), encoding="utf-8")
        return path

    def summary(self) -> str:
        """Findings as plain text, for a terminal or an email."""
        lines = [
            f"FVA review: {self.meta.get('n_series', '?')} series, steps {' > '.join(self.steps)}"
        ]
        lines += [f"- {f}" for f in self.findings]
        return "\n".join(lines)


def analyze(
    actuals: Any,
    forecasts: Any | None = None,
    steps: Sequence[str] | None = None,
    metric: str = "wape",
    season_length: int = 52,
    h: int = 4,
    n_windows: int = 6,
    models: Sequence[str] = DEFAULT_MODELS,
    overrides: tuple[str, str] | None = None,
    price: Any | None = None,
    value_col: str | None = None,
    ts_method: str = "trigg",
    ts_threshold: float | None = None,
    extra_dims: Sequence[str] = (),
) -> FVAResult:
    """Run a complete FVA review.

    Parameters
    ----------
    actuals:
        Long-format demand history (unique_id, ds, y), complete calendar.
    forecasts:
        Optional forecast history: unique_id, ds and one column per process
        step (y optional; joined from actuals). Without it, a rolling-origin
        backtest of `models` is run and the models become the steps.
    steps:
        Forecast columns in process order, benchmark first. Defaults to every
        numeric forecast column in `forecasts`, or to `models`.
    overrides:
        (baseline, final) column names to analyse judgmental overrides, for
        example ("statistical", "planner").
    extra_dims:
        Extra columns in `forecasts` to keep for slicing (category, planner).
    """
    hist = validate(actuals, numeric=[c for c in (Y, value_col) if c])

    if forecasts is None:
        steps = list(steps or models)
        evaluation = backtest(
            hist.select(ID, DS, Y),
            models=steps,
            h=h,
            n_windows=n_windows,
            season_length=season_length,
        )
        source = "backtest"
    else:
        fc = to_polars(forecasts)
        fc = validate(fc, required=(ID, DS), numeric=())
        if steps is None:
            reserved = {ID, DS, Y, "cutoff", "horizon", *extra_dims}
            steps = [c for c, t in fc.schema.items() if c not in reserved and t.is_numeric()]
        steps = list(steps)
        fc = fc.with_columns([pl.col(s).cast(pl.Float64) for s in steps])
        if Y not in fc.columns:
            fc = fc.join(hist.select(ID, DS, Y), on=[ID, DS], how="inner")
        evaluation = fc
        source = "forecasts"

    if len(steps) < 2:
        raise ValueError("Need at least two steps (a benchmark and one process step).")

    first_fc = evaluation[DS].min()
    history_before = hist.filter(pl.col(DS) < first_fc)
    if history_before.is_empty():
        history_before = hist
    scale = mase_scale(history_before.select(ID, DS, Y), season_length=season_length)

    profile = demand_profile(history_before.select(ID, DS, Y))
    abc = abc_xyz(history_before, price=price, value_col=value_col)
    classes = profile.join(abc, on=ID, how="left")
    evaluation = evaluation.join(
        classes.select(ID, "demand_class", "abc", "xyz", "abc_xyz"), on=ID, how="left"
    )

    stairs = stairstep(evaluation, steps, metric=metric, scale=scale)
    by_class = stairstep(evaluation, steps, metric=metric, by="demand_class", scale=scale)
    by_abc = stairstep(evaluation, steps, metric=metric, by="abc", scale=scale)
    by_h = (
        error_table(evaluation, steps, by="horizon", scale=scale)
        if "horizon" in evaluation.columns
        else pl.DataFrame()
    )
    items = item_fva(evaluation, steps, metric=metric, scale=scale).join(
        classes.select(ID, "demand_class", "abc_xyz"), on=ID, how="left"
    )
    card = fva_scorecard(items, steps)
    ts = tracking_signal(
        evaluation.unique([ID, DS], keep="last"),
        steps[-1],
        method=ts_method,
        threshold=ts_threshold,
    )
    ts = ts.join(classes.select(ID, "demand_class", "abc"), on=ID, how="left").sort(
        pl.col("tracking_signal").abs(), descending=True
    )

    ov_summary = ov_overview = None
    if overrides is not None:
        base_col, final_col = overrides
        ov_summary = override_summary(evaluation, baseline=base_col, final=final_col)
        ov_overview = override_overview(evaluation, baseline=base_col, final=final_col)

    findings = headline(stairs, card, metric=metric)
    flagged = ts.filter(pl.col("flag") != "ok").height
    findings.append(
        f"{flagged} of {ts.height} series have a {ts_method} tracking signal beyond "
        f"+/-{ts['threshold'][0]:.2f} on '{steps[-1]}': biased, not just noisy."
    )
    if ov_overview:
        findings.append(
            f"Overrides touch {ov_overview['override_rate']:.0%} of forecasts and improve "
            f"{ov_overview['win_rate']:.0%} of the ones they touch; "
            f"{ov_overview['up_share']:.0%} of overrides are upward."
        )

    meta = {
        "source": source,
        "n_series": int(evaluation[ID].n_unique()),
        "n_rows": evaluation.height,
        "first_forecast": str(first_fc),
        "last_forecast": str(evaluation[DS].max()),
        "history_start": str(hist[DS].min()),
        "season_length": season_length,
        "h": h if source == "backtest" else None,
        "n_windows": n_windows if source == "backtest" else None,
        "class_mix": dict(
            profile.group_by("demand_class").len().sort("len", descending=True).iter_rows()
        ),
    }
    return FVAResult(
        steps=steps,
        metric=metric,
        evaluation=evaluation,
        classes=classes,
        stairstep=stairs,
        stairstep_by_class=by_class,
        stairstep_by_abc=by_abc,
        errors_by_horizon=by_h,
        items=items,
        scorecard=card,
        tracking=ts,
        abc_xyz_matrix=abc_xyz_matrix(abc),
        override_summary=ov_summary,
        override_overview=ov_overview,
        findings=findings,
        meta=meta,
    )


__all__ = ["DEFAULT_MODELS", "FVAResult", "analyze"]
