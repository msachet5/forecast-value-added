"""forecast-value-added: did your forecasting process beat a naive forecast?

Forecast value added (FVA) analysis, demand classification and override
analysis for demand planners, on top of any forecasting tool.

Quick start
-----------
>>> from forecast_value_added import analyze, generate_sales
>>> sales = generate_sales(n_items=50, n_locations=3)
>>> result = analyze(sales.select("unique_id", "ds", "y"))
>>> print(result.summary())
>>> result.to_html("fva_report.html")
"""

from __future__ import annotations

from ._frame import fill_gaps, read_table
from .analyze import FVAResult, analyze
from .backtest import backtest, backtest_statsforecast
from .classify import abc_xyz, abc_xyz_matrix, classify, demand_profile
from .fva import fva_scorecard, item_fva, stairstep
from .metrics import error_table, mase_scale, series_errors, tracking_signal
from .models import BUILTIN_MODELS
from .overrides import override_overview, override_summary, tag_overrides, worst_overriders
from .safety_stock import safety_stock
from .synthetic import generate_sales, simulate_process

__version__ = "0.3.0"

__all__ = [
    "BUILTIN_MODELS",
    "FVAResult",
    "__version__",
    "abc_xyz",
    "abc_xyz_matrix",
    "analyze",
    "backtest",
    "backtest_statsforecast",
    "classify",
    "demand_profile",
    "error_table",
    "fill_gaps",
    "fva_scorecard",
    "generate_sales",
    "item_fva",
    "mase_scale",
    "override_overview",
    "override_summary",
    "read_table",
    "safety_stock",
    "series_errors",
    "simulate_process",
    "stairstep",
    "tag_overrides",
    "tracking_signal",
    "worst_overriders",
]
