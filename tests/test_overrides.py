import polars as pl
import pytest

from forecast_value_added import (
    override_overview,
    override_summary,
    tag_overrides,
    worst_overriders,
)


def frame():
    return pl.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b", "c"],
            "planner": ["P1", "P1", "P2", "P2", "P2"],
            "y": [10.0, 10.0, 10.0, 10.0, 10.0],
            "statistical": [8.0, 10.0, 10.0, 10.0, 0.0],
            "final": [10.0, 10.0, 14.0, 9.5, 2.0],
        }
    )


def test_tagging():
    t = tag_overrides(frame())
    assert t["direction"].to_list() == ["up", "none", "up", "down", "up"]
    assert t["size_bucket"].to_list() == ["25%-50%", "none", "25%-50%", "<10%", ">=50%"]
    assert t["outcome"].to_list() == ["improved", "no change", "worsened", "worsened", "improved"]


def test_summary_and_overview():
    s = override_summary(frame())
    assert s["overrides"].sum() == 4
    ov = override_overview(frame())
    assert ov["override_rate"] == pytest.approx(0.8)
    assert ov["win_rate"] == pytest.approx(0.5)
    assert ov["up_share"] == pytest.approx(0.75)


def test_worst_overriders():
    w = worst_overriders(frame(), "planner", min_overrides=1)
    assert w["planner"][0] == "P2"


def test_empty_when_no_overrides():
    df = frame().with_columns(pl.col("statistical").alias("final"))
    assert override_summary(df).is_empty()
