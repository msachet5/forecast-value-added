import polars as pl
import pytest

from forecast_value_added import fva_scorecard, item_fva, stairstep


def panel():
    return pl.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pl.Series(["2024-01-01", "2024-01-08"] * 2).str.to_date(),
            "y": [10.0, 10.0, 10.0, 10.0],
            "naive": [5.0, 5.0, 10.0, 10.0],  # a: 50% error, b: 0%
            "stat": [8.0, 8.0, 9.0, 9.0],  # a: 20%, b: 10%
            "final": [9.0, 9.0, 7.0, 7.0],  # a: 10%, b: 30%
        }
    )


def test_stairstep_signs():
    s = stairstep(panel(), ["naive", "stat", "final"])
    w = dict(zip(s["step"], s["wape"], strict=True))
    assert w["naive"] == pytest.approx(0.25)
    assert w["stat"] == pytest.approx(0.15)
    assert w["final"] == pytest.approx(0.20)
    fva = dict(zip(s["step"], s["fva_vs_previous"], strict=True))
    assert fva["naive"] is None
    assert fva["stat"] == pytest.approx(0.10)  # added value
    assert fva["final"] == pytest.approx(-0.05)  # destroyed value
    assert s.filter(pl.col("step") == "final")["fva_vs_first"][0] == pytest.approx(0.05)


def test_accuracy_metric_direction():
    s = stairstep(panel(), ["naive", "stat"], metric="accuracy")
    assert s["fva_vs_previous"][1] == pytest.approx(0.10)


def test_grouped_stairstep():
    df = panel().with_columns(pl.col("unique_id").alias("grp"))
    s = stairstep(df, ["naive", "stat"], by="grp")
    b = s.filter((pl.col("grp") == "b") & (pl.col("step") == "stat"))
    assert b["fva_vs_previous"][0] == pytest.approx(-0.10)


def test_item_fva_and_scorecard():
    items = item_fva(panel(), ["naive", "stat", "final"])
    a = items.filter(pl.col("unique_id") == "a").row(0, named=True)
    assert a["fva_stat_vs_prev"] == pytest.approx(0.30)
    assert a["fva_final_vs_first"] == pytest.approx(0.40)
    card = fva_scorecard(items, ["naive", "stat", "final"])
    stat = card.filter((pl.col("step") == "stat") & (pl.col("compared_to") == "naive")).row(
        0, named=True
    )
    assert stat["share_better"] == pytest.approx(0.5)
    assert stat["share_worse"] == pytest.approx(0.5)


def test_needs_two_steps():
    with pytest.raises(ValueError):
        stairstep(panel(), ["naive"])


def test_mase_requires_scale():
    with pytest.raises(ValueError, match="scale"):
        stairstep(panel(), ["naive", "stat"], metric="mase")
