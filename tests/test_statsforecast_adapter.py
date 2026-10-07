import pytest

pytest.importorskip("statsforecast")

from statsforecast.models import CrostonSBA, Naive, SeasonalNaive  # noqa: E402

from forecast_value_added import backtest_statsforecast, generate_sales, stairstep  # noqa: E402


def test_adapter_matches_backtest_shape():
    sales = generate_sales(n_items=4, n_locations=1, n_periods=80, seed=3).select(
        "unique_id", "ds", "y"
    )
    cv = backtest_statsforecast(
        sales,
        [Naive(), SeasonalNaive(season_length=52), CrostonSBA()],
        h=4,
        n_windows=2,
        freq="W-MON",
        n_jobs=1,
    )
    assert {
        "unique_id",
        "ds",
        "cutoff",
        "horizon",
        "y",
        "Naive",
        "SeasonalNaive",
        "CrostonSBA",
    } <= set(cv.columns)
    assert cv["horizon"].max() == 4
    assert stairstep(cv, ["Naive", "SeasonalNaive", "CrostonSBA"]).height == 3
