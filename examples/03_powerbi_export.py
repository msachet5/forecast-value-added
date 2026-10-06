"""Write every FVA table as Parquet for a Power BI model."""

from pathlib import Path

from forecast_value_added import analyze, generate_sales, simulate_process

sales = generate_sales(n_items=60, n_locations=4, seed=2)
process = simulate_process(sales)
result = analyze(
    sales.select("unique_id", "ds", "y"),
    forecasts=process.drop("y"),
    steps=["naive", "statistical", "planner", "final"],
    overrides=("statistical", "planner"),
    extra_dims=("planner_name",),
)
paths = result.to_parquet(Path("powerbi_tables"))
for p in paths:
    print(p)
print("Power BI Desktop: Get Data > Folder > powerbi_tables")
