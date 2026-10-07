"""Unit tests for the deterministic tenor-basis overlay.

``valax.pricing.multicurve`` is three small pure functions that the MC and
PDE rates recipes lean on for their discount/projection split, so they are
pinned down here independently of any solver:

- :func:`with_discount_curve` must be a *no-op* for ``None`` (that is the
  single-curve non-regression guarantee) and re-fit an exact-fit model
  otherwise.
- :func:`tenor_basis` must return ``None`` — not an array of ones — when the
  two curves coincide, and otherwise the ratio of the curves'
  simply-compounded growth factors.
- :func:`apply_swap_basis` must leave the single-curve cash-flow vector
  untouched and, for a flat basis, reproduce the value a swap would have
  under a shifted projection curve.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.models.hull_white import HullWhiteModel
from valax.pricing.multicurve import (
    apply_swap_basis,
    require_single_curve,
    swap_leg_basis,
    tenor_basis,
    with_discount_curve,
)

_DAY_COUNT = "act_365"


def _flat_curve(ref: int, rate: float, n_years: int = 11) -> DiscountCurve:
    """Flat continuously-compounded curve anchored at ``df(ref) = 1``."""
    pillars = jnp.array(
        [ref + int(round(y * 365)) for y in range(n_years)], dtype=jnp.int32
    )
    times = (pillars - ref).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(rate) * times),
        reference_date=jnp.int32(ref),
        day_count=_DAY_COUNT,
    )


@pytest.fixture(scope="module")
def ref_date() -> int:
    return int(ymd_to_ordinal(2026, 1, 1))


@pytest.fixture(scope="module")
def ois(ref_date) -> DiscountCurve:
    return _flat_curve(ref_date, 0.03)


@pytest.fixture(scope="module")
def projection(ref_date) -> DiscountCurve:
    return _flat_curve(ref_date, 0.035)


# ── with_discount_curve ───────────────────────────────────────────────


def test_with_discount_curve_is_identity_for_none(ois):
    """``None`` must return the *same object*, not an equal copy.

    The whole single-curve non-regression argument rests on this: a recipe
    that is handed no discount curve must run the code it ran before.
    """
    model = HullWhiteModel(
        mean_reversion=jnp.asarray(0.1),
        volatility=jnp.asarray(0.01),
        initial_curve=ois,
    )
    assert with_discount_curve(model, None) is model


def test_with_discount_curve_rebinds(ois, projection):
    model = HullWhiteModel(
        mean_reversion=jnp.asarray(0.1),
        volatility=jnp.asarray(0.01),
        initial_curve=ois,
    )
    rebound = with_discount_curve(model, projection)
    assert jnp.array_equal(
        rebound.initial_curve.discount_factors, projection.discount_factors
    )
    # Intrinsic parameters are untouched.
    assert rebound.mean_reversion == model.mean_reversion
    assert rebound.volatility == model.volatility
    # Re-binding a model to the curve it already carries is a no-op, which is
    # what keeps the `is`-checks in `tenor_basis` meaningful downstream.
    assert with_discount_curve(model, ois) is model


def test_with_discount_curve_rejects_non_exact_fit_model(ois):
    class NotAShortRateModel(eqx.Module):
        vol: jnp.ndarray

    with pytest.raises(AttributeError, match="initial_curve"):
        with_discount_curve(NotAShortRateModel(vol=jnp.asarray(0.2)), ois)


# ── tenor_basis ───────────────────────────────────────────────────────


def test_tenor_basis_is_none_when_single_curve(ois, ref_date):
    starts = jnp.array([ref_date], dtype=jnp.int32)
    ends = jnp.array([ref_date + 365], dtype=jnp.int32)
    assert tenor_basis(ois, None, starts, ends) is None
    assert tenor_basis(ois, ois, starts, ends) is None


def test_tenor_basis_matches_growth_factor_ratio(ois, projection, ref_date):
    """b = (1 + tau F_fwd) / (1 + tau F_disc), computed from the curves."""
    starts = jnp.array([ref_date, ref_date + 365], dtype=jnp.int32)
    ends = jnp.array([ref_date + 365, ref_date + 730], dtype=jnp.int32)

    basis = tenor_basis(ois, projection, starts, ends)
    expected = (projection(starts) / projection(ends)) / (
        ois(starts) / ois(ends)
    )
    assert jnp.allclose(basis, expected, atol=1e-14)
    # Projection curve is above the discount curve, so forwards are higher.
    assert jnp.all(basis > 1.0)


def test_tenor_basis_flat_curves_is_the_rate_spread(ois, projection, ref_date):
    """For flat curves, ``b = exp((r_fwd - r_disc) * tau)`` exactly."""
    starts = jnp.array([ref_date + 365], dtype=jnp.int32)
    ends = jnp.array([ref_date + 730], dtype=jnp.int32)
    basis = tenor_basis(ois, projection, starts, ends)
    assert jnp.allclose(basis, jnp.exp((0.035 - 0.03) * 1.0), atol=1e-12)


def test_swap_leg_basis_uses_start_date_for_the_first_period(
    ois, projection, ref_date
):
    payments = jnp.array(
        [ref_date + 365, ref_date + 730, ref_date + 1095], dtype=jnp.int32
    )
    start = jnp.int32(ref_date)

    basis = swap_leg_basis(ois, projection, start, payments)
    starts = jnp.array([ref_date, ref_date + 365, ref_date + 730], jnp.int32)
    expected = tenor_basis(ois, projection, starts, payments)
    assert jnp.allclose(basis, expected, atol=1e-14)
    assert swap_leg_basis(ois, None, start, payments) is None


# ── apply_swap_basis ──────────────────────────────────────────────────


def test_apply_swap_basis_is_identity_for_none():
    cashflows = jnp.array([0.01, 0.01, 1.01])
    float_start, adjusted = apply_swap_basis(cashflows, None)
    assert float_start == 1.0
    assert adjusted is cashflows


def test_apply_swap_basis_reproduces_the_dual_curve_swap_pv(
    ois, projection, ref_date
):
    r"""``float_start - sum c_i P(0,T_i)`` must equal the explicit legs.

    The overlay's whole claim is that a *linear* swap is priced exactly, so
    check the folded form against the textbook dual-curve NPV computed
    directly from the two curves: floating leg
    :math:`\sum_j (DF_f(T_{j-1})/DF_f(T_j) - 1) DF_d(T_j)` minus fixed leg
    :math:`K \sum_j \tau_j DF_d(T_j)`.
    """
    payments = jnp.array(
        [ref_date + 365 * k for k in (1, 2, 3, 4, 5)], dtype=jnp.int32
    )
    starts = jnp.concatenate([jnp.array([ref_date], jnp.int32), payments[:-1]])
    taus = jnp.full((5,), 1.0)
    strike = 0.03

    cashflows = (strike * taus).at[-1].add(1.0)
    basis = swap_leg_basis(ois, projection, jnp.int32(ref_date), payments)
    float_start, adjusted = apply_swap_basis(cashflows, basis)

    df = ois(payments)
    folded = float_start - jnp.sum(adjusted * df)

    floating = jnp.sum(
        (projection(starts) / projection(payments) - 1.0) * df
    )
    fixed = strike * jnp.sum(taus * df)
    assert jnp.allclose(folded, floating - fixed, atol=1e-12)


# ── require_single_curve ──────────────────────────────────────────────


def test_require_single_curve_allows_the_degenerate_cases(ois):
    require_single_curve(ois, None, "a payoff")
    require_single_curve(ois, ois, "a payoff")


def test_require_single_curve_rejects_a_distinct_curve(ois, projection):
    with pytest.raises(ValueError, match="projection curve"):
        require_single_curve(ois, projection, "The widget recipe")
