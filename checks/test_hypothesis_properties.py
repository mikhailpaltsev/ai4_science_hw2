"""Пять рандомизированных тестов свойств для практики недели 7.

[порождено: ChatGPT (GPT-6)], сверить с METHOD.md перед сдачей.
"""

import math
import numpy as np
from hypothesis import given, settings, strategies as st
from src.exponential import BACKGROUND, dose_rate, half_life

positive_a = st.floats(min_value=1.0, max_value=30.0, allow_nan=False, allow_infinity=False)
nonnegative_a = st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False)
valid_lam = st.floats(min_value=0.005, max_value=0.05, allow_nan=False, allow_infinity=False)
valid_t = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


@given(t=valid_t, A=nonnegative_a, lam=valid_lam)
@settings(max_examples=40)
def test_hypothesis_background_bound(t, A, lam):
    assert dose_rate(t, A, lam) >= BACKGROUND - 1e-12


@given(t=valid_t, dt=st.floats(min_value=0.5, max_value=30.0), A=positive_a, lam=valid_lam)
@settings(max_examples=40)
def test_hypothesis_monotone(t, dt, A, lam):
    assert dose_rate(t + dt, A, lam) < dose_rate(t, A, lam)


@given(A=nonnegative_a, lam=valid_lam)
@settings(max_examples=40)
def test_hypothesis_initial_value(A, lam):
    assert np.isclose(dose_rate(0.0, A, lam), BACKGROUND + A, rtol=1e-12, atol=1e-12)


@given(t=valid_t, A=nonnegative_a, lam=valid_lam)
@settings(max_examples=40)
def test_hypothesis_time_units(t, A, lam):
    hourly = dose_rate(t, A, lam)
    daily = dose_rate(t / 24.0, A, lam * 24.0)
    assert np.isclose(hourly, daily, rtol=1e-12, atol=1e-12)


@given(A=positive_a, lam=valid_lam)
@settings(max_examples=40)
def test_hypothesis_half_life(A, lam):
    value = dose_rate(half_life(lam), A, lam)
    assert np.isclose(value, BACKGROUND + A / 2.0, rtol=1e-10, atol=1e-10)
