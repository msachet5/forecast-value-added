"""M5 benchmark: how often does a machine learning forecast fail to beat seasonal naive?

This is the launch finding for forecast-value-added. It runs a rolling-origin
backtest on the public M5 (Walmart) dataset at weekly item-store level and
reports, by demand class, the share of series where each method fails to
beat the seasonal naive benchmark.

Data
----
Download `sales_train_evaluation.csv` and `calendar.csv` from the M5
Forecasting Accuracy competition on Kaggle:
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
and pass the folder with --m5-dir. Or let `datasetsforecast` fetch it:
pip install datasetsforecast, then --m5-dir data/m5 --download.

Models
------
naive, seasonal_naive (52 weeks), ses, croston_sba, class_aware
(built in), AutoETS (statsforecast, optional) and a global LightGBM
model (mlforecast, optional). Missing optional packages are skipped.

Usage
-----
    python benchmarks/m5_fva_benchmark.py --m5-dir data/m5 --store CA_1 --dept FOODS_3
    python benchmarks/m5_fva_benchmark.py --synthetic         # smoke test, no data needed

Outputs benchmarks/results/<run>/: backtest.parquet, findings.md, report.html
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from forecast_value_added import analyze, backtest, generate_sales, item_fva
from forecast_value_added.classify import demand_profile


def load_m5_weekly(m5_dir: Path, store: str | None, dept: str | None, max_series: int | None) -> pl.DataFrame:
    """Daily M5 sales to weekly (Saturday-start Walmart weeks) item-store series."""
    sales_path = m5_dir / "sales_train_evaluation.csv"
    cal_path = m5_dir / "calendar.csv"
    if not sales_path.exists():
        raise FileNotFoundError(
            f"{sales_path} not found. Download the M5 files from Kaggle "
            "(m5-forecasting-accuracy) or run with --download."
        )
    sales = pl.scan_csv(sales_path)
    if store:
        sales = sales.filter(pl.col("store_id") == store)
    if dept:
        sales = sales.filter(pl.col("dept_id") == dept)
    sales = sales.collect()
    if max_series:
        sales = sales.head(max_series)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.select("id", *day_cols).unpivot(index="id", variable_name="d", value_name="y")
    cal = pl.read_csv(cal_path).select("d", pl.col("date").str.to_date(), "wm_yr_wk")
    long = long.join(cal, on="d", how="left")
    weekly = (
        long.group_by("id", "wm_yr_wk")
        .agg(pl.col("y").sum().cast(pl.Float64), pl.col("date").min().alias("ds"), pl.len().alias("days"))
        .filter(pl.col("days") == 7)  # drop partial weeks at the edges
        .rename({"id": "unique_id"})
        .select("unique_id", "ds", "y")
        .sort("unique_id", "ds")
    )
    # drop leading zeros before an item's first sale (product not yet listed)
    weekly = weekly.with_columns(pl.col("y").cum_sum().over("unique_id").alias("_c")).filter(pl.col("_c") > 0).drop("_c")
    return weekly


def download_m5(m5_dir: Path) -> None:
    try:
        from datasetsforecast.m5 import M5  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("pip install datasetsforecast to use --download") from exc
    M5.download(str(m5_dir))
    print(f"downloaded M5 into {m5_dir}; point --m5-dir at the folder holding sales_train_evaluation.csv")


def optional_models(df: pl.DataFrame, h: int, n_windows: int, season: int) -> pl.DataFrame | None:
    """AutoETS and LightGBM backtests when their packages are installed."""
    frames = []
    try:
        from statsforecast.models import AutoETS  # noqa: PLC0415

        from forecast_value_added import backtest_statsforecast  # noqa: PLC0415

        t = time.time()
        sf = backtest_statsforecast(df, [AutoETS(season_length=season)], h=h, n_windows=n_windows, freq="W-SAT")
        frames.append(sf.select("unique_id", "ds", "cutoff", pl.col("AutoETS").alias("auto_ets")))
        print(f"  AutoETS done in {time.time() - t:.0f}s")
    except ImportError:
        print("  statsforecast not installed: skipping AutoETS")
    try:
        import lightgbm as lgb  # noqa: PLC0415
        from mlforecast import MLForecast  # noqa: PLC0415
        from mlforecast.lag_transforms import RollingMean  # noqa: PLC0415

        t = time.time()
        mlf = MLForecast(
            models={"lightgbm": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1)},
            freq="W-SAT",
            lags=[1, 2, 4, 8, 13, 26, 52],
            lag_transforms={1: [RollingMean(window_size=4), RollingMean(window_size=13)]},
            date_features=["month", "week"],
        )
        cv = mlf.cross_validation(df.to_pandas(), h=h, n_windows=n_windows, step_size=h)
        ml = pl.from_pandas(cv).with_columns(pl.col("ds").cast(pl.Date), pl.col("cutoff").cast(pl.Date))
        frames.append(ml.select("unique_id", "ds", "cutoff", pl.col("lightgbm").clip(lower_bound=0)))
        print(f"  LightGBM done in {time.time() - t:.0f}s")
    except ImportError:
        print("  mlforecast/lightgbm not installed: skipping the ML model")
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on=["unique_id", "ds", "cutoff"], how="inner")
    return out


def findings_markdown(items: pl.DataFrame, classes: pl.DataFrame, steps: list[str], bench: str) -> str:
    """Share of series where each method fails to beat the benchmark, by demand class."""
    data = items.join(classes.select("unique_id", "demand_class"), on="unique_id", how="left")
    lines = [
        f"| Method | Series | Fails to beat {bench} | Smooth | Erratic | Intermittent | Lumpy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for step in steps:
        if step == bench:
            continue
        col = f"wape_{step}"
        ref = f"wape_{bench}"
        sub = data.filter(pl.col(col).is_not_null() & pl.col(ref).is_not_null() & pl.col(col).is_finite() & pl.col(ref).is_finite())
        fails = (sub[col] >= sub[ref]).mean() if sub.height else float("nan")
        cells = []
        for cls in ("smooth", "erratic", "intermittent", "lumpy"):
            c = sub.filter(pl.col("demand_class") == cls)
            cells.append(f"{(c[col] >= c[ref]).mean():.0%}" if c.height else "n/a")
        lines.append(f"| {step} | {sub.height:,} | {fails:.0%} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m5-dir", type=Path, default=Path("data/m5"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--store", default="CA_1")
    ap.add_argument("--dept", default="FOODS_3")
    ap.add_argument("--max-series", type=int)
    ap.add_argument("--h", type=int, default=4)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--synthetic", action="store_true", help="smoke test on generated data")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()

    if args.download:
        download_m5(args.m5_dir)
        return
    if args.synthetic:
        df = generate_sales(n_items=60, n_locations=3, seed=11).select("unique_id", "ds", "y")
        run = "synthetic"
    else:
        df = load_m5_weekly(args.m5_dir, args.store, args.dept, args.max_series)
        run = f"m5_{args.store or 'all'}_{args.dept or 'all'}".lower()
    print(f"{df['unique_id'].n_unique():,} weekly series, {df.height:,} rows")

    season = 52
    t = time.time()
    bt = backtest(
        df,
        models=["naive", "seasonal_naive", "ses", "croston_sba", "class_aware"],
        h=args.h,
        n_windows=args.windows,
        season_length=season,
    )
    print(f"  built-in models done in {time.time() - t:.0f}s")
    extra = optional_models(df, args.h, args.windows, season) if not args.synthetic else None
    if extra is not None:
        bt = bt.join(extra, on=["unique_id", "ds", "cutoff"], how="inner")
    steps = [c for c in ["seasonal_naive", "naive", "ses", "croston_sba", "class_aware", "auto_ets", "lightgbm"] if c in bt.columns]

    out = args.out / run
    out.mkdir(parents=True, exist_ok=True)
    bt.write_parquet(out / "backtest.parquet")
    history = df.filter(pl.col("ds") < bt["ds"].min())
    classes = demand_profile(history)
    items = item_fva(bt, steps)
    table = findings_markdown(items, classes, steps, bench="seasonal_naive")
    result = analyze(df, forecasts=bt.drop("y"), steps=steps, season_length=season)
    result.to_html(out / "report.html", title=f"FVA benchmark: {run}")
    mix = classes.group_by("demand_class").len().sort("len", descending=True)
    md = [
        f"# FVA benchmark: {run}",
        "",
        f"{df['unique_id'].n_unique():,} weekly item-store series, rolling origin with {args.windows} windows of {args.h} weeks.",
        "Benchmark: seasonal naive (same week last year). A method 'fails' on a series when its WAPE is equal to or worse than the benchmark's.",
        "",
        table,
        "",
        "Demand class mix (history before the first cutoff):",
        "",
        *[f"- {r[0]}: {r[1]:,}" for r in mix.iter_rows()],
        "",
        "Stairstep (WAPE, all series pooled):",
        "",
        "| Step | WAPE | Bias | FVA vs seasonal naive |",
        "|---|---:|---:|---:|",
        *[
            f"| {r['step']} | {r['wape']:.1%} | {r['bias_pct']:+.1%} | {(r['fva_vs_first'] or 0) * 100:+.1f} pp |"
            for r in result.stairstep.iter_rows(named=True)
        ],
    ]
    (out / "findings.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
