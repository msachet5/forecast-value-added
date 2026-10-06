"""Command line interface.

    fva demo                                  # synthetic data, full report, no setup
    fva report sales.csv                      # backtest benchmarks on your history
    fva report sales.csv --forecasts fc.csv --steps naive,statistical,final \
        --overrides statistical,final
    fva classify sales.csv --out classes.csv
    fva backtest sales.csv --h 4 --windows 6 --season-length 52 --out backtest.csv
    fva overrides fc.csv --baseline statistical --final final --who planner
    fva safety-stock backtest.csv --step class_aware --lead-time 2 --service-level 0.95
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from . import __version__
from ._frame import DS, ID, Y, fill_gaps, read_table, rename_columns
from .analyze import DEFAULT_MODELS, analyze
from .backtest import backtest
from .classify import classify
from .overrides import override_overview, override_summary, worst_overriders
from .safety_stock import safety_stock
from .synthetic import generate_sales, simulate_process


def _csv_list(text: str | None) -> list[str] | None:
    return [t.strip() for t in text.split(",") if t.strip()] if text else None


def _write(df: pl.DataFrame, path: str | None) -> None:
    if not path:
        with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=160):
            print(df)
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".parquet", ".pq"}:
        df.write_parquet(out)
    else:
        df.write_csv(out)
    print(f"wrote {out} ({df.height:,} rows)")


def _load_sales(args: argparse.Namespace) -> pl.DataFrame:
    df = read_table(args.sales)
    df = rename_columns(df, args.id_col, args.date_col, args.target_col)
    if getattr(args, "fill_gaps", None):
        df = fill_gaps(df.select(ID, DS, Y), freq=args.fill_gaps)
    return df


def _add_columns(p: argparse.ArgumentParser) -> None:
    p.add_argument("--id-col", default=ID, help="series key column (default unique_id)")
    p.add_argument("--date-col", default=DS, help="period date column (default ds)")
    p.add_argument("--target-col", default=Y, help="actual demand column (default y)")
    p.add_argument(
        "--fill-gaps",
        metavar="FREQ",
        help="insert zero rows for missing periods, e.g. 1w, 1d, 1mo",
    )


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"generating {args.items} items x {args.locations} locations of weekly demand ...")
    sales = generate_sales(n_items=args.items, n_locations=args.locations, seed=args.seed)
    process = simulate_process(sales, seed=args.seed)
    sales.write_csv(out / "sales.csv")
    process.write_csv(out / "forecasts.csv")
    result = analyze(
        sales.select(ID, DS, Y),
        forecasts=process.drop(Y),
        steps=["naive", "seasonal_naive", "statistical", "planner", "final"],
        overrides=("statistical", "planner"),
        value_col=None,
        price=sales.select(ID, "price").unique(ID),
        extra_dims=("planner_name",),
    )
    result.to_html(out / "report.html", title="Forecast Value Added review (synthetic demo)")
    try:
        result.to_parquet(out / "tables")
    except ImportError:
        result.to_csv(out / "tables")
    print(result.summary())
    print(f"\nreport: {out / 'report.html'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    sales = _load_sales(args)
    forecasts = None
    if args.forecasts:
        forecasts = rename_columns(read_table(args.forecasts), args.id_col, args.date_col, None)
    overrides = tuple(_csv_list(args.overrides) or []) or None
    if overrides is not None and len(overrides) != 2:
        print("--overrides needs exactly two columns: baseline,final", file=sys.stderr)
        return 2
    result = analyze(
        sales,
        forecasts=forecasts,
        steps=_csv_list(args.steps),
        metric=args.metric,
        season_length=args.season_length,
        h=args.h,
        n_windows=args.windows,
        models=_csv_list(args.models) or DEFAULT_MODELS,
        overrides=overrides,  # type: ignore[arg-type]
        value_col=args.value_col,
    )
    out = Path(args.out)
    formats = set(_csv_list(args.format) or ["html"])
    if "html" in formats:
        print(f"report: {result.to_html(out / 'report.html', title=args.title)}")
    if "parquet" in formats:
        result.to_parquet(out / "tables")
        print(f"parquet tables: {out / 'tables'}")
    if "csv" in formats:
        result.to_csv(out / "tables_csv")
        print(f"csv tables: {out / 'tables_csv'}")
    print(result.summary())
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    sales = _load_sales(args)
    table = classify(sales, value_col=args.value_col)
    _write(table, args.out)
    if not args.out:
        print(table.group_by("demand_class").len().sort("len", descending=True))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    sales = _load_sales(args)
    bt = backtest(
        sales.select(ID, DS, Y),
        models=_csv_list(args.models) or list(DEFAULT_MODELS),
        h=args.h,
        n_windows=args.windows,
        season_length=args.season_length,
    )
    _write(bt, args.out)
    return 0


def cmd_overrides(args: argparse.Namespace) -> int:
    fc = rename_columns(read_table(args.forecasts), args.id_col, args.date_col, args.target_col)
    overview = override_overview(fc, args.baseline, args.final)
    for k, v in overview.items():
        print(f"{k:>22}: {v:,.3f}")
    _write(override_summary(fc, args.baseline, args.final), args.out)
    if args.who:
        print(f"\nby {args.who}:")
        _write(worst_overriders(fc, args.who, args.baseline, args.final), None)
    return 0


def cmd_safety(args: argparse.Namespace) -> int:
    ev = rename_columns(read_table(args.evaluation), args.id_col, args.date_col, args.target_col)
    table = safety_stock(
        ev,
        step=args.step,
        lead_time=args.lead_time,
        service_level=args.service_level,
        review_period=args.review_period,
    )
    _write(table, args.out)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    sales = generate_sales(n_items=args.items, n_locations=args.locations, seed=args.seed)
    _write(sales, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fva",
        description="Forecast value added: did your forecasting process beat a naive forecast?",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run the full analysis on synthetic data")
    p.add_argument("--out", default="fva_demo")
    p.add_argument("--items", type=int, default=120)
    p.add_argument("--locations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("report", help="one-page FVA report from your files")
    p.add_argument("sales", help="actuals: CSV or Parquet with unique_id, ds, y")
    p.add_argument("--forecasts", help="forecast history with one column per process step")
    p.add_argument("--steps", help="forecast columns in process order, benchmark first")
    p.add_argument("--overrides", help="baseline,final columns for override analysis")
    p.add_argument(
        "--metric",
        default="wape",
        choices=["wape", "mae", "rmse", "mase", "abs_bias_pct", "accuracy"],
    )
    p.add_argument(
        "--models",
        help=f"backtest models when no forecasts given (default {','.join(DEFAULT_MODELS)})",
    )
    p.add_argument("--h", type=int, default=4, help="forecast horizon in periods")
    p.add_argument("--windows", type=int, default=6, help="backtest cutoffs")
    p.add_argument("--season-length", type=int, default=52)
    p.add_argument("--value-col", help="column for ABC value (e.g. revenue); default units")
    p.add_argument("--format", default="html,parquet", help="html, parquet, csv (comma separated)")
    p.add_argument("--title", default="Forecast Value Added review")
    p.add_argument("--out", default="fva_report")
    _add_columns(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("classify", help="demand class (ADI/CV2) and ABC-XYZ per series")
    p.add_argument("sales")
    p.add_argument("--value-col")
    p.add_argument("--out")
    _add_columns(p)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("backtest", help="rolling-origin backtest of benchmark models")
    p.add_argument("sales")
    p.add_argument("--models")
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--season-length", type=int, default=52)
    p.add_argument("--out")
    _add_columns(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("overrides", help="which judgmental overrides add or destroy value")
    p.add_argument("forecasts", help="rows with y, baseline and final columns")
    p.add_argument("--baseline", default="statistical")
    p.add_argument("--final", default="final")
    p.add_argument("--who", help="column to rank by, e.g. planner or customer")
    p.add_argument("--out")
    _add_columns(p)
    p.set_defaults(func=cmd_overrides)

    p = sub.add_parser("safety-stock", help="safety stock and reorder point from forecast error")
    p.add_argument("evaluation", help="backtest or forecast history with y and the step column")
    p.add_argument("--step", required=True)
    p.add_argument("--lead-time", type=float, default=1.0, help="periods")
    p.add_argument("--review-period", type=float, default=0.0, help="periods")
    p.add_argument("--service-level", type=float, default=0.95)
    p.add_argument("--out")
    _add_columns(p)
    p.set_defaults(func=cmd_safety)

    p = sub.add_parser("generate", help="write a synthetic multi-location sales file")
    p.add_argument("--items", type=int, default=120)
    p.add_argument("--locations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="sales.csv")
    p.set_defaults(func=cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"fva: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
