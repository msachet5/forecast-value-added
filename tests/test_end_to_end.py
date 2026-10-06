import polars as pl
import pytest

from forecast_value_added import analyze, fill_gaps, generate_sales, safety_stock, simulate_process
from forecast_value_added.classify import demand_profile
from forecast_value_added.cli import main
from forecast_value_added.safety_stock import z_value


@pytest.fixture(scope="module")
def sales():
    return generate_sales(n_items=12, n_locations=2, n_periods=120, seed=1)


def test_generator_covers_all_patterns(sales):
    prof = demand_profile(sales.select("unique_id", "ds", "y"))
    assert prof["demand_class"].n_unique() >= 3
    assert sales["y"].min() >= 0


def test_analyze_backtest_mode(sales, tmp_path):
    res = analyze(sales.select("unique_id", "ds", "y"), h=4, n_windows=3)
    assert res.steps == ["naive", "seasonal_naive", "class_aware"]
    assert res.stairstep.height == 3
    html = res.to_html(tmp_path / "r.html").read_text()
    assert "Forecast Value Added" in html and "<svg" in html
    paths = res.to_parquet(tmp_path / "t")
    assert any(p.name == "stairstep.parquet" for p in paths)
    assert "FVA review" in res.summary()


def test_analyze_forecast_mode_with_overrides(sales, tmp_path):
    proc = simulate_process(sales, n_windows=3)
    res = analyze(
        sales.select("unique_id", "ds", "y"),
        forecasts=proc.drop("y"),
        steps=["naive", "statistical", "planner", "final"],
        overrides=("statistical", "planner"),
        extra_dims=("planner_name",),
    )
    assert res.override_overview is not None
    assert 0.2 < res.override_overview["override_rate"] < 0.4
    assert res.to_html(tmp_path / "r.html").exists()


def test_fill_gaps():
    df = pl.DataFrame(
        {"unique_id": ["a", "a"], "ds": ["2024-01-01", "2024-01-15"], "y": [1.0, 2.0]}
    )
    out = fill_gaps(df, "1w")
    assert out["y"].to_list() == [1.0, 0.0, 2.0]


def test_safety_stock():
    assert z_value(0.95) == pytest.approx(1.6449, abs=1e-4)
    ev = pl.DataFrame(
        {"unique_id": ["a"] * 4, "y": [10.0, 10.0, 10.0, 10.0], "f": [12.0, 8.0, 12.0, 8.0]}
    )
    ss = safety_stock(ev, "f", lead_time=4, service_level=0.95)
    # sigma_error = 2, sqrt(L) = 2
    assert ss["safety_stock"][0] == pytest.approx(1.6449 * 2 * 2, abs=1e-3)
    assert ss["reorder_point"][0] == pytest.approx(40 + 1.6449 * 4, abs=1e-3)


def test_cli_demo(tmp_path):
    assert main(["demo", "--out", str(tmp_path / "d"), "--items", "10", "--locations", "2"]) == 0
    assert (tmp_path / "d" / "report.html").exists()
    assert (
        main(["classify", str(tmp_path / "d" / "sales.csv"), "--out", str(tmp_path / "c.csv")]) == 0
    )
    assert (
        main(
            [
                "report",
                str(tmp_path / "d" / "sales.csv"),
                "--h",
                "4",
                "--windows",
                "2",
                "--out",
                str(tmp_path / "r"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "overrides",
                str(tmp_path / "d" / "forecasts.csv"),
                "--baseline",
                "statistical",
                "--final",
                "planner",
                "--who",
                "planner_name",
            ]
        )
        == 0
    )


def test_cli_error_is_clean(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("sku,date,qty\na,2024-01-01,1\n")
    assert main(["classify", str(bad)]) == 1
    assert "Missing required columns" in capsys.readouterr().err
    assert (
        main(["classify", str(bad), "--id-col", "sku", "--date-col", "date", "--target-col", "qty"])
        == 0
    )
