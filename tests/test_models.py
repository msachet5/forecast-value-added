import numpy as np
import pytest

from forecast_value_added import models as m


def test_naive_and_seasonal_naive():
    y = np.array([1.0, 2, 3, 4, 5, 6])
    assert m.naive(y, 3).tolist() == [6, 6, 6]
    assert m.seasonal_naive(y, 5, season_length=3).tolist() == [4, 5, 6, 4, 5]
    # short history falls back to naive
    assert m.seasonal_naive(np.array([2.0]), 2, season_length=4).tolist() == [2, 2]


def test_croston_textbook():
    y = np.array([0, 0, 4, 0, 0, 4], dtype=float)
    # sizes 4, 4 and intervals 3, 3: z = 4, p = 3
    assert m.croston(y, 2)[0] == pytest.approx(4 / 3)
    assert m.croston_sba(y, 2)[0] == pytest.approx(4 / 3 * (1 - 0.1 / 2))


def test_croston_all_zero():
    assert m.croston(np.zeros(5), 3).tolist() == [0, 0, 0]
    assert m.tsb(np.zeros(5), 3).tolist() == [0, 0, 0]


def test_tsb_decays_after_demand_stops():
    active = np.array([5, 5, 5, 5, 0, 0, 0, 0, 0, 0], dtype=float)
    early = m.tsb(active[:4], 1)[0]
    late = m.tsb(active, 1)[0]
    assert late < early


def test_ses_constant_series():
    assert m.ses(np.full(10, 7.0), 3).tolist() == pytest.approx([7, 7, 7])


def test_ses_fixed_alpha():
    y = np.array([10.0, 20.0])
    # level 10, error 10, alpha .5 -> 15
    assert m.ses(np.array([10.0, 20.0, 20.0]), 1, alpha=0.5)[0] == pytest.approx(17.5)
    assert y.size == 2


def test_moving_average_and_lookup():
    ma = m.get_model("moving_average_3")
    assert ma(np.array([1.0, 2, 3, 6]), 2).tolist() == pytest.approx([11 / 3, 11 / 3])
    with pytest.raises(KeyError):
        m.get_model("prophet")


def test_drift():
    assert m.drift(np.array([1.0, 2, 3]), 2).tolist() == [4, 5]


def test_class_aware_routes_intermittent_to_croston():
    y = np.array([0, 0, 4, 0, 0, 4], dtype=float)
    assert m.class_aware(y, 1)[0] == pytest.approx(m.croston_sba(y, 1)[0])
