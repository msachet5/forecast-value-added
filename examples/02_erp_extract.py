"""From a typical ERP extract to an FVA report.

ERP sales extracts usually have their own column names, daily dates and no
rows for days without sales. This example cleans that up first.
"""

import polars as pl

from forecast_value_added import analyze, fill_gaps, generate_sales

# Stand-in for an ERP export: columns named the ERP way, zero rows removed.
erp = (
    generate_sales(n_items=20, n_locations=2, seed=5)
    .filter(pl.col("y") > 0)
    .select(
        pl.col("item_id").alias("MATERIAL"),
        pl.col("location_id").alias("PLANT"),
        pl.col("ds").alias("WEEK_START"),
        pl.col("y").alias("QTY"),
    )
)

sales = (
    erp.with_columns((pl.col("MATERIAL") + "@" + pl.col("PLANT")).alias("unique_id"))
    .rename({"WEEK_START": "ds", "QTY": "y"})
    .group_by("unique_id", "ds")
    .agg(pl.col("y").sum())
)
sales = fill_gaps(sales, freq="1w")  # put back the zero weeks

result = analyze(sales, h=4, n_windows=6, models=["naive", "seasonal_naive", "class_aware"])
print(result.summary())
