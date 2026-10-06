"""Input handling shared by every module.

The library speaks one data shape, the long format used by Nixtla's
statsforecast and most forecasting tools:

    unique_id | ds         | y
    SKU1@TOR  | 2026-01-05 | 12.0

`unique_id` is a series key (an item, or an item at a location), `ds` is the
period start date, `y` is the actual demand. Forecast tables add one column
per forecast step (for example `naive`, `statistical`, `final`).

Every public function accepts a polars or pandas DataFrame and returns polars.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import polars as pl

ID, DS, Y = "unique_id", "ds", "y"


def to_polars(df: Any) -> pl.DataFrame:
    """Coerce a pandas or polars DataFrame (or a LazyFrame) to an eager polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    try:  # pandas, without importing it eagerly
        import pandas as pd  # noqa: PLC0415

        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df.reset_index(drop=True))
    except ImportError:  # pragma: no cover - pandas is optional
        pass
    raise TypeError(f"Expected a polars or pandas DataFrame, got {type(df).__name__}")


def rename_columns(
    df: pl.DataFrame,
    id_col: str = ID,
    date_col: str = DS,
    target_col: str | None = Y,
) -> pl.DataFrame:
    """Rename user columns to the canonical unique_id / ds / y names."""
    mapping = {}
    if id_col != ID:
        mapping[id_col] = ID
    if date_col != DS:
        mapping[date_col] = DS
    if target_col is not None and target_col != Y:
        mapping[target_col] = Y
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Available: {df.columns}")
    return df.rename(mapping) if mapping else df


def validate(
    df: Any,
    required: Sequence[str] = (ID, DS, Y),
    numeric: Iterable[str] = (Y,),
) -> pl.DataFrame:
    """Check required columns, cast types and sort. Returns a clean polars frame."""
    out = to_polars(df)
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(
            f"Missing required columns {missing}. Expected long format with "
            f"'{ID}', '{DS}' and '{Y}'. Use id_col/date_col/target_col to map yours."
        )
    casts = [pl.col(ID).cast(pl.Utf8)]
    if DS in out.columns and out.schema[DS] not in (pl.Date, pl.Datetime):
        casts.append(pl.col(DS).cast(pl.Utf8).str.to_date(strict=False))
    for col in numeric:
        if col in out.columns:
            casts.append(pl.col(col).cast(pl.Float64))
    out = out.with_columns(casts)
    if DS in out.columns and out.schema[DS] == pl.Datetime:
        out = out.with_columns(pl.col(DS).dt.date())
    if DS in out.columns:
        if out[DS].null_count():
            raise ValueError(f"Could not parse {out[DS].null_count()} dates in '{DS}'.")
        dupes = out.select(ID, DS).is_duplicated().sum()
        if dupes:
            raise ValueError(f"{dupes} duplicated ({ID}, {DS}) rows. Aggregate before analysis.")
        out = out.sort(ID, DS)
    return out


def read_table(path: str | Path, **kwargs: Any) -> pl.DataFrame:
    """Read CSV or Parquet by extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, try_parse_dates=True, infer_schema_length=10_000, **kwargs)
    if suffix == ".tsv":
        return pl.read_csv(path, separator="\t", try_parse_dates=True, **kwargs)
    raise ValueError(f"Unsupported file type '{suffix}'. Use .csv or .parquet.")


def fill_gaps(df: pl.DataFrame, freq: str = "1w") -> pl.DataFrame:
    """Insert zero-demand rows for missing periods inside each series.

    Many ERP extracts only store periods with sales. Demand classification and
    every error metric assume a complete calendar, so fill gaps first.
    `freq` is a polars interval string: "1d", "1w", "1mo".
    """
    df = validate(df)
    bounds = df.group_by(ID).agg(pl.col(DS).min().alias("start"), pl.col(DS).max().alias("end"))
    grid = (
        bounds.with_columns(pl.date_ranges("start", "end", interval=freq).alias(DS))
        .explode(DS)
        .select(ID, DS)
    )
    return (
        grid.join(df, on=[ID, DS], how="left").with_columns(pl.col(Y).fill_null(0.0)).sort(ID, DS)
    )
