"""Five-minute tour on synthetic data. Run: python examples/01_quickstart.py"""

from forecast_value_added import analyze, generate_sales, simulate_process

# 1. Two years of weekly demand for 40 items in 3 stores, with every demand pattern.
sales = generate_sales(n_items=40, n_locations=3, seed=1)

# 2. A simulated planning process: statistical forecast, planner overrides, consensus.
process = simulate_process(sales, n_windows=6)

# 3. The FVA review.
result = analyze(
    sales.select("unique_id", "ds", "y"),
    forecasts=process.drop("y"),
    steps=["naive", "seasonal_naive", "statistical", "planner", "final"],
    overrides=("statistical", "planner"),
    extra_dims=("planner_name",),
)
print(result.summary())
print(result.stairstep)
print(result.stairstep_by_class)
result.to_html("quickstart_report.html")
print("open quickstart_report.html")
