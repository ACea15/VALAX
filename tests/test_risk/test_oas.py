"""Tests for option-adjusted spread / Z-spread (:mod:`valax.risk.oas`).

The suite is a ladder, each test acting as an oracle for the next (the
modelling background is ``docs/theory/risk-measures.md`` §7.4 and the recipe is
``docs/guide/callable-bonds.md``):

1. **``hw_alpha`` shift identity** — a parallel curve shift ``s`` moves the
   exact-fit shift ``alpha(t)`` by exactly ``s``; free, exact, and it proves the
   spread is threaded into the model correctly.
2. **Round-trip to zero** — feeding a model price back as the market price
   recovers ``OAS = 0`` to solver tolerance.
3. **Option-free bond: OAS ≡ Z-spread** — the PDE-based OAS and the closed-form
   Z-spread (a completely separate code path) agree to numerical tolerance,
   because there is genuinely no option component to adjust for.
4. **Z-spread > OAS for a callable** — the call is valuable to the issuer, so
   ignoring it overstates the credit/liquidity spread. A model-free directional
   invariant.
5. **Effective duration(callable) < duration(bullet)** — the call truncates the
   price upside, compressing duration.
6. **Negative effective convexity near the call boundary** — the money shot: the
   defining economic signature of the embedded call.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import CallableBond, FixedRateBond
from valax.models.hull_white import HullWhiteModel, hw_alpha_average
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.pde import PDEConfig, pde_price_dispatch
from valax.risk.oas import (
    bond_z_spread,
    callable_bond_oas,
    effective_convexity,
    effective_duration,
    price_under_spread,
    z_spread_convexity,
    z_spread_dv01,
    z_spread_duration,
)
from valax.risk.shocks import bump_curve_zero_rates, parallel_shift


CONFIG = PDEConfig(n_spot=201, n_time=200, spot_range=6.0)
CALL_YEARS = (2027, 2028, 2029)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(21)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.05 * times),
        reference_date=ref_date,
    )


@pytest.fixture
def model(flat_curve):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.01),
        initial_curve=flat_curve,
    )


@pytest.fixture
def schedule():
    return generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2).payment_dates


@pytest.fixture
def call_dates():
    return jnp.array(
        [int(ymd_to_ordinal(y, 1, 1)) for y in CALL_YEARS], dtype=jnp.int32
    )


def _bullet(ref_date, schedule, coupon):
    """Option-free fixed-rate bond."""
    return FixedRateBond(
        payment_dates=schedule,
        settlement_date=ref_date,
        coupon_rate=jnp.array(coupon),
        face_value=jnp.array(100.0),
        frequency=2,
    )


def _callable(ref_date, schedule, dates, coupon, call_price=1.0):
    """Fixed-rate bond callable at ``call_price`` (fraction of face)."""
    return CallableBond(
        payment_dates=schedule,
        settlement_date=ref_date,
        coupon_rate=jnp.array(coupon),
        face_value=jnp.array(100.0),
        call_dates=dates,
        call_prices=jnp.full(dates.shape[0], call_price),
        frequency=2,
    )


# A longer bond callable at *every* coupon date (near-continuous American
# call) plus a low model vol is what makes the call pin the price at par when
# it is deep in-the-money: effective convexity is then negative, most negative
# at the call boundary and shrinking toward zero from below deep in the money.
CONV_CONFIG = PDEConfig(n_spot=251, n_time=400, spot_range=6.0)


@pytest.fixture
def conv_model(flat_curve):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.03),
        volatility=jnp.array(0.01),
        initial_curve=flat_curve,
    )


@pytest.fixture
def long_schedule():
    return generate_schedule(2025, 1, 1, 2035, 1, 1, frequency=2).payment_dates


# ── 1. hw_alpha shift identity ───────────────────────────────────────


@pytest.mark.parametrize("s", [0.0025, 0.0075, -0.0050])
def test_alpha_shift_identity(model, s):
    """A parallel shift ``s`` moves ``alpha`` by exactly ``s`` everywhere."""
    shifted = HullWhiteModel(
        mean_reversion=model.mean_reversion,
        volatility=model.volatility,
        initial_curve=parallel_shift(model.initial_curve, jnp.asarray(s)),
    )
    for t0, t1 in [(0.0, 1.0), (1.5, 2.5), (3.0, 4.75)]:
        diff = hw_alpha_average(shifted, t0, t1) - hw_alpha_average(model, t0, t1)
        assert float(diff) == pytest.approx(s, abs=1e-12)


# ── 2. Round-trip to zero ────────────────────────────────────────────


def test_oas_round_trip_zero(ref_date, schedule, model):
    """Pricing the model then inverting it recovers OAS = 0."""
    bond = _bullet(ref_date, schedule, coupon=0.04)
    model_price = price_under_spread(bond, model, CONFIG, jnp.array(0.0))
    oas = callable_bond_oas(bond, model, model_price, CONFIG)
    assert float(oas) == pytest.approx(0.0, abs=1e-9)


def test_oas_recovers_known_spread(ref_date, schedule, call_dates, model):
    """A callable priced at a known spread has that spread as its OAS."""
    bond = _callable(ref_date, schedule, call_dates, coupon=0.08)
    s_true = 0.0075
    price = price_under_spread(bond, model, CONFIG, jnp.array(s_true))
    oas = callable_bond_oas(bond, model, price, CONFIG)
    assert float(oas) == pytest.approx(s_true, abs=1e-8)


# ── 3. Option-free bond: OAS ≡ Z-spread ──────────────────────────────


def test_option_free_oas_equals_z_spread(ref_date, schedule, flat_curve, model):
    """With no optionality the PDE OAS matches the closed-form Z-spread.

    The two are computed through entirely independent code paths (Hull-White
    finite differences versus the analytic curve pricer), so agreement to
    numerical tolerance is a strong cross-check.
    """
    bond = _bullet(ref_date, schedule, coupon=0.04)
    # A market price below the model's fair value forces a positive spread.
    market_price = jnp.array(93.0)
    oas = callable_bond_oas(bond, model, market_price, CONFIG)
    z = bond_z_spread(bond, flat_curve, market_price)
    assert float(oas) == pytest.approx(float(z), abs=1e-5)
    assert float(oas) > 0.0  # cheap vs fair value => positive spread


# ── 4. Z-spread > OAS for a callable ─────────────────────────────────


def test_z_spread_exceeds_oas_for_callable(
    ref_date, schedule, call_dates, flat_curve, model
):
    """Ignoring a valuable call overstates the spread (Z-spread > OAS)."""
    coupon = 0.08  # well above the 5% curve => the call is valuable
    callable_bond = _callable(ref_date, schedule, call_dates, coupon=coupon)
    bullet = _bullet(ref_date, schedule, coupon=coupon)

    # Market price = the callable's model price at a known spread.
    s_true = 0.0050
    market_price = price_under_spread(callable_bond, model, CONFIG, jnp.array(s_true))

    oas = callable_bond_oas(callable_bond, model, market_price, CONFIG)
    z = bond_z_spread(bullet, flat_curve, market_price)

    assert float(oas) == pytest.approx(s_true, abs=1e-7)
    assert float(z) > float(oas) + 1e-4  # meaningfully wider, not just noise


# ── 5. Effective duration compression ────────────────────────────────


def test_callable_duration_below_bullet(
    ref_date, schedule, call_dates, model
):
    """The embedded call compresses effective duration vs the bullet."""
    coupon = 0.08
    callable_bond = _callable(ref_date, schedule, call_dates, coupon=coupon)
    bullet = _bullet(ref_date, schedule, coupon=coupon)

    dur_callable = effective_duration(callable_bond, model, CONFIG)
    dur_bullet = effective_duration(bullet, model, CONFIG)

    assert float(dur_bullet) > 0.0
    assert float(dur_callable) < float(dur_bullet)


# ── 6. Convexity compression near the call boundary (the money shot) ──


def test_callable_convexity_negative_at_call_boundary(
    ref_date, long_schedule, conv_model
):
    """The embedded call turns effective convexity negative as rates fall.

    Measured as the second derivative of the full Hull-White PDE price with
    respect to a parallel discount-curve spread, a deep-in-the-money callable
    shows the textbook **negative effective convexity**: the call caps the
    price, so the bond is short the call's gamma.

    History: this test used to assert the opposite (convexity "floors at a
    small positive value", +0.33 here). That number was an artifact. With the
    call projection sampled pointwise, ``max`` is piecewise linear in the
    spread, so autodiff's second derivative misses the curvature concentrated
    at the node crossings — a wide-step finite difference on the same
    pointwise grid gave -4.6 / -2.4 / -0.6 / +0.3 at h = 10 / 5 / 2 / 1 bp,
    i.e. it was dominated by kinks. With the cell-averaged projection
    (``PDEConfig.smooth_payoff``, the default) the price is C¹, autodiff gives
    -4.19 and the finite difference converges to it (-4.19 at 2 bp and 1 bp).
    At the money (6% coupon on a 5% curve) the convexity is about -229: the
    duration collapses from ~7 to ~0 across the call boundary.
    """
    # Callable at every coupon date => near-continuous American call.
    callable_bond = _callable(
        ref_date, long_schedule, long_schedule, coupon=0.06
    )
    bullet = _bullet(ref_date, long_schedule, coupon=0.06)

    conv_bullet = effective_convexity(bullet, conv_model, CONV_CONFIG)
    conv_atm = effective_convexity(callable_bond, conv_model, CONV_CONFIG, 0.0)
    # Deep in-the-money: rates well below the coupon, call almost certain.
    conv_itm = effective_convexity(
        callable_bond, conv_model, CONV_CONFIG, -0.02
    )

    # Bullet is strongly positively convex; deep in the money the call makes
    # the callable *negatively* convex, by a small fraction of the bullet's.
    assert float(conv_bullet) > 10.0
    assert float(conv_itm) < 0.0
    assert abs(float(conv_itm)) < 0.25 * float(conv_bullet)
    assert float(conv_atm) < float(conv_bullet)

    # Negative convexity is deepest where the call boundary sits (at the
    # money here: 6% coupon on a 5% curve) and shrinks toward zero deep in the
    # money, where the price is pinned near the call price.
    assert float(conv_atm) < float(conv_itm) < 0.0

    # The autodiff convexity is the derivative of a C¹ price: a central
    # second difference converges to it, at both points.
    p = lambda s: price_under_spread(
        callable_bond, conv_model, CONV_CONFIG, jnp.asarray(s)
    )
    h = 1e-4
    for s0, conv in ((-0.02, conv_itm), (0.0, conv_atm)):
        fd = (p(s0 + h) - 2 * p(s0) + p(s0 - h)) / h**2 / p(s0)
        np.testing.assert_allclose(float(conv), float(fd), rtol=5e-3)


def test_effective_convexity_matches_finite_difference(
    ref_date, long_schedule, conv_model
):
    """Validate the autodiff second derivative on the smooth bullet bond.

    The effective-convexity autodiff path is exercised on an option-free bond,
    whose price in the spread is smooth (no exercise obstacle), so a central
    difference agrees tightly.  The callable is deliberately *not* used here: at
    its exercise boundary the numerical price carries fine-scale kinks, so a
    finite difference and the exact pointwise derivative legitimately measure
    curvature at different scales.
    """
    bullet = _bullet(ref_date, long_schedule, coupon=0.06)
    conv_ad = effective_convexity(bullet, conv_model, CONV_CONFIG, 0.0)

    h = 1e-3
    p_up = price_under_spread(bullet, conv_model, CONV_CONFIG, jnp.array(h))
    p_0 = price_under_spread(bullet, conv_model, CONV_CONFIG, jnp.array(0.0))
    p_dn = price_under_spread(bullet, conv_model, CONV_CONFIG, jnp.array(-h))
    conv_fd = (p_up - 2.0 * p_0 + p_dn) / (h * h) / p_0
    assert float(conv_ad) == pytest.approx(float(conv_fd), rel=1e-3)


# ── JIT smoke test ───────────────────────────────────────────────────


def test_oas_filter_jit_smoke(ref_date, schedule, call_dates, model):
    """`callable_bond_oas` compiles and runs under ``eqx.filter_jit``."""

    @eqx.filter_jit
    def solve(bond, model, price):
        return callable_bond_oas(bond, model, price, CONFIG)

    bond = _callable(ref_date, schedule, call_dates, coupon=0.06)
    price = price_under_spread(bond, model, CONFIG, jnp.array(0.003))
    oas = solve(bond, model, price)
    assert jnp.isfinite(oas)
    assert float(oas) == pytest.approx(0.003, abs=1e-7)


# ── Z-spread risk (option-free bonds) ────────────────────────────────


def test_z_spread_dv01_positive_and_matches_grad(ref_date, schedule, flat_curve):
    """DV01 is positive for a long bond and equals -dP/dz * 1bp."""
    bond = _bullet(ref_date, schedule, coupon=0.04)
    z = bond_z_spread(bond, flat_curve, jnp.array(96.0))  # its market spread

    dv01 = float(z_spread_dv01(bond, flat_curve, z))
    price_fn = lambda x: fixed_rate_bond_price(bond, parallel_shift(flat_curve, x))
    dprice = float(jax.grad(price_fn)(z))

    assert dv01 > 0.0
    assert dv01 == pytest.approx(-dprice * 1e-4, rel=1e-10)


def test_z_spread_dv01_duration_relation(ref_date, schedule, flat_curve):
    """DV01 = duration * price * 1bp."""
    bond = _bullet(ref_date, schedule, coupon=0.04)
    z = jnp.array(0.0075)
    dv01 = float(z_spread_dv01(bond, flat_curve, z))
    dur = float(z_spread_duration(bond, flat_curve, z))
    price = float(fixed_rate_bond_price(bond, parallel_shift(flat_curve, z)))
    assert dv01 == pytest.approx(dur * price * 1e-4, rel=1e-10)


def test_z_spread_convexity_positive(ref_date, schedule, flat_curve):
    """An option-free bond has strictly positive spread convexity."""
    bond = _bullet(ref_date, schedule, coupon=0.04)
    assert float(z_spread_convexity(bond, flat_curve, 0.0)) > 0.0


def test_z_spread_dv01_equals_parallel_curve_dv01(ref_date, schedule, flat_curve):
    """For a single-curve bullet, spread DV01 equals parallel-curve DV01.

    A Z-spread shift and a parallel zero-rate shift are the same curve move, so
    the two DV01s must coincide.  Cross-checks the spread helper against an
    independent central-difference bump through ``bump_curve_zero_rates``.
    """
    bond = _bullet(ref_date, schedule, coupon=0.04)
    n = flat_curve.pillar_dates.shape[0]
    h = 1e-6
    p_up = float(fixed_rate_bond_price(bond, bump_curve_zero_rates(flat_curve, jnp.full(n, h))))
    p_dn = float(fixed_rate_bond_price(bond, bump_curve_zero_rates(flat_curve, jnp.full(n, -h))))
    curve_dv01 = -(p_up - p_dn) / (2.0 * h) * 1e-4

    dv01 = float(z_spread_dv01(bond, flat_curve, 0.0))
    assert dv01 == pytest.approx(curve_dv01, rel=1e-6)
