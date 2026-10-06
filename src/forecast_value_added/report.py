"""One-page HTML report. Self-contained: inline CSS and SVG, no JavaScript, prints cleanly."""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from importlib import resources
from typing import TYPE_CHECKING, Any

import polars as pl
from jinja2 import Environment, select_autoescape

if TYPE_CHECKING:
    from .analyze import FVAResult

PCT_METRICS = {"wape", "abs_bias_pct", "accuracy", "bias_pct"}


def _fmt(value: Any, kind: str = "num") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "pp":
        sign = "+" if value > 0 else ""
        return f"{sign}{value * 100:.1f} pp"
    if kind == "signed":
        sign = "+" if value > 0 else ""
        return f"{sign}{value:,.2f}"
    if kind == "int":
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _fva_class(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "neutral"
    if value > 0.005:
        return "good"
    if value < -0.005:
        return "bad"
    return "neutral"


def bar_chart(labels: list[str], values: list[float], pct: bool = True, width: int = 940) -> str:
    """Horizontal bar chart as inline SVG. The lowest (best) bar is highlighted."""
    clean = [v if v is not None and not math.isnan(v) else 0.0 for v in values]
    vmax = max(clean) if clean and max(clean) > 0 else 1.0
    bar_h, gap, label_w, pad = 22, 8, 150, 80
    height = len(labels) * (bar_h + gap) + gap
    best = min(range(len(clean)), key=lambda i: clean[i]) if clean else -1
    parts = [
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px" role="img" '
        f'aria-label="Error by forecast step" xmlns="http://www.w3.org/2000/svg">'
    ]
    for i, (label, v) in enumerate(zip(labels, clean, strict=True)):
        y = gap + i * (bar_h + gap)
        w = (width - label_w - pad) * v / vmax
        cls = "bar best" if i == best else "bar"
        text = f"{v * 100:.1f}%" if pct else f"{v:,.2f}"
        parts.append(
            f'<text x="{label_w - 8}" y="{y + bar_h * 0.68}" class="lbl" text-anchor="end">'
            f"{html.escape(label)}</text>"
            f'<rect x="{label_w}" y="{y}" width="{max(w, 1):.1f}" height="{bar_h}" rx="3" class="{cls}"/>'
            f'<text x="{label_w + w + 6:.1f}" y="{y + bar_h * 0.68}" class="val">{text}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _rows(df: pl.DataFrame | None, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.is_empty():
        return []
    data = df.head(limit) if limit else df
    return list(data.iter_rows(named=True))


def render_html(result: FVAResult, title: str = "Forecast Value Added review") -> str:
    from . import __version__  # noqa: PLC0415

    template_text = (
        resources.files("forecast_value_added")
        .joinpath("templates/report.html.j2")
        .read_text("utf-8")
    )
    env = Environment(
        autoescape=select_autoescape(["html", "j2"]), trim_blocks=True, lstrip_blocks=True
    )
    env.filters["fmt"] = _fmt
    env.filters["fva_class"] = _fva_class
    template = env.from_string(template_text)

    metric = result.metric
    pct = metric in PCT_METRICS
    stairs = result.stairstep
    chart = bar_chart(stairs["step"].to_list(), stairs[metric].to_list(), pct=pct)

    by_class = result.stairstep_by_class
    class_rows = []
    if not by_class.is_empty():
        for cls in by_class["demand_class"].unique(maintain_order=True).to_list():
            sub = by_class.filter(pl.col("demand_class") == cls)
            class_rows.append(
                {
                    "group": cls,
                    "n": int(sub["n"][0]),
                    "cells": [
                        {"value": v, "fva": f}
                        for v, f in zip(
                            sub[metric].to_list(), sub["fva_vs_previous"].to_list(), strict=True
                        )
                    ],
                }
            )
    abc_rows = []
    if not result.stairstep_by_abc.is_empty():
        for grp in sorted(result.stairstep_by_abc["abc"].drop_nulls().unique().to_list()):
            sub = result.stairstep_by_abc.filter(pl.col("abc") == grp)
            abc_rows.append(
                {
                    "group": grp,
                    "n": int(sub["n"][0]),
                    "cells": [
                        {"value": v, "fva": f}
                        for v, f in zip(
                            sub[metric].to_list(), sub["fva_vs_previous"].to_list(), strict=True
                        )
                    ],
                }
            )

    flagged = result.tracking.filter(pl.col("flag") != "ok")
    worst_items = result.items.sort(f"fva_{result.steps[-1]}_vs_prev").head(10)

    return template.render(
        title=title,
        version=__version__,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        meta=result.meta,
        steps=result.steps,
        metric=metric,
        metric_label=metric.upper().replace("_", " "),
        pct=pct,
        findings=result.findings,
        stairs=_rows(stairs),
        chart=chart,
        scorecard=_rows(result.scorecard),
        class_rows=class_rows,
        abc_rows=abc_rows,
        matrix=_rows(result.abc_xyz_matrix),
        tracking=_rows(flagged, 15),
        n_flagged=flagged.height,
        worst_items=_rows(worst_items),
        last_step=result.steps[-1],
        overview=result.override_overview,
        overrides=_rows(result.override_summary),
    )


__all__ = ["bar_chart", "render_html"]
