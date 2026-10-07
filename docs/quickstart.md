# Quick start

## 1. Shape your data

One row per series per period. A series is an item, or an item at a location.

| unique_id | ds | y |
|---|---|---|
| SKU1@TOR | 2026-01-05 | 12 |
| SKU1@TOR | 2026-01-12 | 0 |

Different column names are fine: pass `--id-col`, `--date-col`, `--target-col`. If your extract only has rows for periods with sales, add `--fill-gaps 1w` (or `1d`, `1mo`).

## 2. No forecast history yet? Benchmark first

```bash
fva report sales.csv --h 4 --windows 8 --season-length 52
```

This runs a rolling-origin backtest of naive, seasonal naive and a class-aware method (SES, seasonal naive or Croston SBA by demand pattern) and reports what a simple, automatic process would have achieved. That number is the bar your current process must clear.

## 3. With forecast history

Export the forecast as each step produced it, one column per step:

| unique_id | ds | statistical | planner | final |
|---|---|---|---|---|

```bash
fva report sales.csv --forecasts forecasts.csv \
    --steps naive,statistical,planner,final \
    --overrides statistical,planner
```

If your file has no `naive` column, compute one first (`fva backtest sales.csv --models naive,seasonal_naive --out bench.csv`) and join it on `unique_id` and `ds`.

!!! warning "Lag matters"
    Compare forecasts at the lag the business acts on. If purchase orders are placed four weeks out, evaluate the forecast that existed four weeks before each period, not the last revision.

## 4. In Python

```python
from forecast_value_added import analyze, read_table

result = analyze(
    read_table("sales.csv"),
    forecasts=read_table("forecasts.csv"),
    steps=["naive", "statistical", "planner", "final"],
    overrides=("statistical", "planner"),
)
result.findings  # plain-language headlines
result.stairstep  # polars DataFrame
result.items  # per-item FVA
result.to_html("report.html")
result.to_parquet("tables/")
```
