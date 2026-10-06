import math

import polars as pl
import pytest

from forecast_value_added import abc_xyz, abc_xyz_matrix, classify, demand_profile


def test_syntetos_boylan_quadrants(small_panel):
    prof = demand_profile(small_panel)
    got = dict(zip(prof["unique_id"], prof["demand_class"], strict=True))
    assert got == {
        "smooth": "smooth",
        "intermittent": "intermittent",
        "lumpy": "lumpy",
        "erratic": "erratic",
    }


def test_adi_and_cv2_values(series):
    df = series("a", [0, 0, 5, 0, 0, 5])
    prof = demand_profile(df)
    assert prof["adi"][0] == pytest.approx(3.0)
    assert prof["cv2"][0] == pytest.approx(0.0)
    assert prof["zero_share"][0] == pytest.approx(4 / 6)


def test_erratic_cv2(series):
    prof = demand_profile(series("e", [1, 10] * 6))
    # population std of {1, 10} is 4.5, mean 5.5
    assert prof["cv2"][0] == pytest.approx((4.5 / 5.5) ** 2)
    assert prof["demand_class"][0] == "erratic"


def test_adi_interval_method(series):
    # non-zero at index 1 and 4 -> gaps from start: 2 and 3 -> mean 2.5
    prof = demand_profile(series("a", [0, 3, 0, 0, 3, 0]), adi_method="intervals")
    assert prof["adi"][0] == pytest.approx(2.5)


def test_no_demand_and_insufficient(series):
    df = pl.concat([series("zero", [0, 0, 0, 0]), series("one", [0, 0, 7, 0])])
    prof = demand_profile(df)
    got = dict(zip(prof["unique_id"], prof["demand_class"], strict=True))
    assert got == {"zero": "no demand", "one": "insufficient history"}


def test_abc_threshold_crossing_item_is_a(series):
    df = pl.concat([series("i1", [70]), series("i2", [20]), series("i3", [6]), series("i4", [4])])
    abc = abc_xyz(df).sort("unique_id")
    assert abc["abc"].to_list() == ["A", "A", "B", "C"]
    assert abc["value_share"].sum() == pytest.approx(1.0)


def test_abc_with_price(series):
    df = pl.concat([series("cheap", [100]), series("dear", [10])])
    price = pl.DataFrame({"unique_id": ["cheap", "dear"], "price": [1.0, 50.0]})
    abc = abc_xyz(df, price=price)
    top = abc.sort("value", descending=True)["unique_id"][0]
    assert top == "dear"


def test_xyz_bands(series):
    df = pl.concat(
        [
            series("x", [10, 10, 10, 10]),
            series("y", [5, 15, 5, 15]),  # cv = 0.5 -> X boundary inclusive
            series("z", [0, 0, 0, 40]),  # cv = sqrt(3) -> Z
        ]
    )
    abc = abc_xyz(df).sort("unique_id")
    assert abc["xyz"].to_list() == ["X", "X", "Z"]
    assert math.isclose(abc.filter(pl.col("unique_id") == "z")["cv"][0], 3**0.5)


def test_matrix_and_classify(small_panel):
    cls = classify(small_panel)
    assert {"demand_class", "abc", "xyz", "abc_xyz"} <= set(cls.columns)
    m = abc_xyz_matrix(abc_xyz(small_panel))
    assert m.height == 3
    assert m.select(["X", "Y", "Z"]).sum_horizontal().sum() == 4


def test_pandas_input(small_panel):
    pd = pytest.importorskip("pandas")
    prof = demand_profile(small_panel.to_pandas())
    assert isinstance(small_panel.to_pandas(), pd.DataFrame)
    assert prof.height == 4


def test_missing_columns_message():
    with pytest.raises(KeyError, match="unique_id"):
        demand_profile(pl.DataFrame({"sku": ["a"], "ds": ["2024-01-01"], "y": [1.0]}))
