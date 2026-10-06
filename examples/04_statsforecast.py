"""Feed statsforecast models into the FVA stairstep.

pip install "forecast-value-added[statsforecast]"
"""

from statsforecast.models import AutoETS, CrostonSBA, Naive, SeasonalNaive

from forecast_value_added import backtest_statsforecast, generate_sales, stairstep

sales = generate_sales(n_items=30, n_locations=2, seed=3).select("unique_id", "ds", "y")
cv = backtest_statsforecast(
    sales,
    [Naive(), SeasonalNaive(season_length=52), AutoETS(season_length=52), CrostonSBA()],
    h=4,
    n_windows=6,
    freq="W-MON",
)
print(stairstep(cv, ["Naive", "SeasonalNaive", "AutoETS", "CrostonSBA"]))
