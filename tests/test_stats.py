"""Unit tests for the pure statistical/financial functions - no database or
generated data required, so these always run (including in CI, which has
no copy of the dataset)."""
import numpy as np
import pytest

from assumptions import expected_revenue
from build_npv_excel import annuity_factor
from hypothesis import cramers_v


def test_expected_revenue_floors_negative_balance():
    assert expected_revenue(-500) == 0.0


def test_expected_revenue_matches_hand_calculation():
    # $1,000 balance, 2% margin, 3 years, 8% discount - assumptions.py defaults
    assert expected_revenue(1000) == pytest.approx(51.54, abs=0.01)


def test_annuity_factor_matches_hand_calculation():
    assert annuity_factor(years=3, discount_rate=0.08) == pytest.approx(2.5771, abs=0.001)


def test_cramers_v_is_zero_for_independent_table():
    independent = np.array([[10, 10], [10, 10]])
    assert cramers_v(independent) == pytest.approx(0.0, abs=1e-9)


def test_cramers_v_ranks_stronger_association_higher():
    weak = np.array([[12, 8], [8, 12]])
    strong = np.array([[20, 0], [0, 20]])
    assert cramers_v(strong) > cramers_v(weak)
