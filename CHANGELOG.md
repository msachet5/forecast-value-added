# Changelog

All notable changes are listed here. The project follows semantic versioning; until 1.0, minor versions may change the API.

## 0.3.0 (planned 2027-01-31)

- Override analysis: `tag_overrides`, `override_summary`, `override_overview`, `worst_overriders`, and the `fva overrides` command.
- Override section in the HTML report.
- `safety_stock` from forecast error or demand variability, and `fva safety-stock`.

## 0.2.0 (planned 2026-12-13)

- `fva` command line: `demo`, `report`, `classify`, `backtest`, `generate`.
- One-page HTML report (inline CSS and SVG, dark mode, prints cleanly).
- Parquet and CSV export of every table for Power BI and Excel.
- `backtest_statsforecast` adapter.

## 0.1.0 (planned 2026-11-08)

- Syntetos-Boylan demand classification (ADI, CV²) and ABC-XYZ.
- Rolling-origin backtest with naive, seasonal naive, moving average, SES, Croston, Croston SBA, TSB, drift and a class-aware selector.
- Error metrics: WAPE, MAE, RMSE, bias, MASE, accuracy; Trigg and RSFE tracking signals.
- FVA stairstep, per-item FVA and scorecard.
- Synthetic multi-location demand generator and forecasting-process simulator.
