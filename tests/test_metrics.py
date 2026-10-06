import polars as pl
import pytest

from forecast_value_added import error_table, mase_scale, tracking_signal
from forecast_value_added.metrics import trigg_limit


def frame(y, **models):
    n = len(y)
    return pl.DataFrame(
        {
            "unique_id": ["a"] * n,
            "ds": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, n), eager=True),
            "y": [float(v) for v in y],
            **{k: [float(x) for x in v] for k, v in models.items()},
        }
    )


def test_wape_bias_accuracy():
    df = frame([10, 10], f=[8, 13])
    t = error_table(df, ["f"])
    row = t.row(0, named=True)
    assert row["wape"] == pytest.approx(0.25)
    assert row["bias_pct"] == pytest.approx(0.05)
    assert row["accuracy"] == pytest.approx(0.75)
    assert row["mae"] == pytest.approx(2.5)
    assert row["rmse"] == pytest.approx(((4 + 9) / 2) ** 0.5)


def test_accuracy_floored_at_zero():
    t = error_table(frame([1, 1], f=[5, 5]), ["f"])
    assert t["accuracy"][0] == 0.0


def test_mase_scale_and_mase():
    hist = frame([1, 2, 3, 4])
    scale = mase_scale(hist, season_length=1)
    assert scale["scale"][0] == pytest.approx(1.0)
    t = error_table(frame([5, 6], f=[4, 8]), ["f"], scale=scale)
    assert t["mase"][0] == pytest.approx(1.5)


def test_constant_history_scale_is_null():
    assert mase_scale(frame([3, 3, 3]))["scale"][0] is None


def test_model_order_preserved():
    df = frame([1, 2], z=[1, 2], a=[2, 3])
    assert error_table(df, ["z", "a"])["model"].to_list() == ["z", "a"]


def test_grouped():
    df = pl.concat(
        [
            frame([10, 10], f=[10, 10]),
            frame([10, 10], f=[0, 0]).with_columns(pl.lit("b").alias("unique_id")),
        ]
    )
    t = error_table(df, ["f"], by="unique_id").sort("unique_id")
    assert t["wape"].to_list() == [0.0, 1.0]


def test_trigg_limit_value():
    # 1.96 * sqrt(0.1 / 1.9) / sqrt(2 / pi)
    assert trigg_limit(0.1) == pytest.approx(0.5643, abs=1e-3)


def test_tracking_signal_flags_persistent_under_forecast():
    df = frame([10] * 20, f=[8] * 20)
    ts = tracking_signal(df, "f")
    assert ts["tracking_signal"][0] == pytest.approx(1.0)
    assert ts["flag"][0] == "under-forecast"
    ts2 = tracking_signal(df, "f", method="rsfe")
    assert ts2["tracking_signal"][0] == pytest.approx(20.0)
    ts3 = tracking_signal(df, "f", method="rsfe", window=5)
    assert ts3["tracking_signal"][0] == pytest.approx(5.0)


def test_tracking_signal_unbiased_is_ok():
    df = frame([10, 12] * 10, f=[11] * 20)
    assert tracking_signal(df, "f")["flag"][0] == "ok"
