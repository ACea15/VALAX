"""Multi-curve (OIS discounting + tenor projection) in the short-rate MC.

Companion to ``tests/test_pde/test_multicurve.py``; the acceptance criteria are
``docs/architecture/pricing-ontology.md`` §12.6.  The floating-rate note is the
load-bearing instrument here: its PV is *linear*, so it does not depend on the
model's dynamics at all, and the Hull-White / G2++ MC price must therefore
reproduce the analytic dual-curve
:func:`~valax.pricing.analytic.floating.floating_rate_bond_price` exactly (up
to sampling error).  That isolates the discount/projection split from the
model.

Two tolerances are used deliberately:

* the **differential** test (dual-curve minus single-curve, same PRNG key)
  cancels both the sampling noise and the event-time snapping bias, so it is
  asserted tightly — this is what actually pins down the basis arithmetic;
* the **absolute** test against the analytic price is asserted within a few
  standard errors, which is all Monte Carlo can promise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import FloatingRateBond
from valax.instruments.rates import CMSSpreadSwap
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.floating import floating_rate_bond_price
from valax.pricing.mc.dispatch import MCConfig, mc_price_dispatch

_CONFIG = MCConfig(n_paths=40_000, n_steps=100)
_KEY = jax.random.PRNGKey(20260923)

_OIS_ID = "USD.SOFR.OIS"
_PROJ_ID = "USD.LIBOR.3M"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


def _flat_curve(ref, rate: float) -> DiscountCurve:
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(21)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=ref,
    )


@pytest.fixture(scope="module")
def ois(ref_date) -> DiscountCurve:
    return _flat_curve(ref_date, 0.03)


@pytest.fixture(scope="module")
def projection(ref_date) -> DiscountCurve:
    """Tenor curve 50 bp above OIS — a realistic IBOR/OIS basis."""
    return _flat_curve(ref_date, 0.035)


@pytest.fixture(scope="module")
def hw(ois) -> HullWhiteModel:
    return HullWhiteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.01),
        initial_curve=ois,
        curve_id=_OIS_ID,
    )


@pytest.fixture(scope="module")
def g2pp(ois) -> G2PPModel:
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        volatility_x=jnp.array(0.010),
        mean_reversion_y=jnp.array(0.10),
        volatility_y=jnp.array(0.006),
        correlation=jnp.array(-0.70),
        initial_curve=ois,
        curve_id=_OIS_ID,
    )


@pytest.fixture(scope="module")
def frn(ref_date) -> FloatingRateBond:
    """5y annual FRN resetting in arrears-free fashion at each period start.

    ``fixing_dates`` are the period starts, which is the schedule the MC
    recipe's floating-leg replication assumes, so the analytic pricer and the
    MC recipe are describing the *same* contract.  ``act_365`` matches the
    curve's day count, so the model's time axis and the curve agree (see the
    note in the PDE companion test).
    """
    payments = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)
    fixings = jnp.concatenate(
        [jnp.asarray(ref_date, dtype=payments.dtype)[None], payments[:-1]]
    )
    return FloatingRateBond(
        payment_dates=payments,
        fixing_dates=fixings,
        settlement_date=jnp.asarray(ref_date, dtype=payments.dtype),
        spread=jnp.array(0.002),
        face_value=jnp.array(100.0),
        frequency=1,
        day_count="act_365",
        discount_id=_OIS_ID,
        forward_id=_PROJ_ID,
    )


# ── 1. Single-curve non-regression ────────────────────────────────────


@pytest.mark.parametrize("model_name", ["hw", "g2pp"])
def test_frn_model_curve_changes_nothing(request, model_name, frn):
    """Passing the model's own curve must change literally nothing."""
    model = request.getfixturevalue(model_name)
    plain = mc_price_dispatch(frn, model, _CONFIG, _KEY)
    explicit = mc_price_dispatch(
        frn, model, _CONFIG, _KEY,
        discount_curve=model.initial_curve, forward_curve=None,
    )
    assert float(explicit.price) == float(plain.price)


# ── 2. Linear product: the dual-curve PV is the analytic one ──────────


@pytest.mark.parametrize("model_name", ["hw", "g2pp"])
def test_frn_dual_curve_matches_analytic(
    request, model_name, frn, ois, projection
):
    """§12.6.2 — a linear product prices exactly under the overlay.

    An FRN's PV depends only on the two curves, never on the short-rate
    dynamics, so agreement with the analytic dual-curve price is a direct
    check of the discount/projection split.
    """
    model = request.getfixturevalue(model_name)
    result = mc_price_dispatch(
        frn, model, _CONFIG, _KEY,
        discount_curve=ois, forward_curve=projection,
    )
    expected = float(floating_rate_bond_price(frn, ois, projection))
    assert float(result.price) == pytest.approx(
        expected, abs=4.0 * float(result.stderr) + 5e-3
    )


@pytest.mark.parametrize("model_name", ["hw", "g2pp"])
def test_frn_basis_increment_matches_analytic(
    request, model_name, frn, ois, projection
):
    """The *effect* of the basis, with the sampling noise cancelled.

    Both runs use the same PRNG key and therefore the same paths, so the
    difference is essentially noise-free and the event-time snapping bias
    cancels too.  This is the sharp test of the basis arithmetic itself.
    """
    model = request.getfixturevalue(model_name)
    single = mc_price_dispatch(frn, model, _CONFIG, _KEY, discount_curve=ois)
    dual = mc_price_dispatch(
        frn, model, _CONFIG, _KEY,
        discount_curve=ois, forward_curve=projection,
    )
    mc_increment = float(dual.price) - float(single.price)
    analytic_increment = float(
        floating_rate_bond_price(frn, ois, projection)
        - floating_rate_bond_price(frn, ois, ois)
    )
    assert mc_increment == pytest.approx(analytic_increment, rel=2e-3)
    # A projection curve above OIS means higher coupons, hence a richer note.
    assert mc_increment > 0.0


# ── 3. End-to-end through the unified front door ──────────────────────


def test_price_forwards_the_resolved_curves(frn, hw, ois, projection):
    """``price(...)`` must bind ``discount_id`` / ``forward_id`` for MC.

    This is the P5.1a plumbing: the instrument names its curves, ``resolve``
    looks them up on the :class:`MarketState`, and the dispatch bridge hands
    them to the recipe because the recipe declares them.
    """
    market = MarketState(
        curves=CurveGraph(curves={_OIS_ID: ois, _PROJ_ID: projection})
    )
    unified = price(frn, hw, market, method="mc", config=_CONFIG, key=_KEY)
    direct = mc_price_dispatch(
        frn, hw, _CONFIG, _KEY,
        discount_curve=ois, forward_curve=projection,
    )
    assert float(unified.price) == float(direct.price)
    assert unified.method == "mc"


def test_price_single_curve_market_is_unchanged(frn, hw, ois):
    """An instrument naming only a discount curve still prices single-curve."""
    plain_frn = FloatingRateBond(
        payment_dates=frn.payment_dates,
        fixing_dates=frn.fixing_dates,
        settlement_date=frn.settlement_date,
        spread=frn.spread,
        face_value=frn.face_value,
        frequency=frn.frequency,
        day_count=frn.day_count,
    )
    market = MarketState.single_curve(ois)
    unified = price(plain_frn, hw, market, method="mc", config=_CONFIG, key=_KEY)
    direct = mc_price_dispatch(plain_frn, hw, _CONFIG, _KEY)
    assert float(unified.price) == float(direct.price)


# ── 4. Payoffs that cannot project must refuse, not ignore ────────────


def test_cms_spread_rejects_a_distinct_projection_curve(
    g2pp, ois, projection, ref_date
):
    swap = CMSSpreadSwap(
        start_date=jnp.asarray(ref_date, dtype=jnp.int32),
        payment_dates=generate_schedule(2026, 1, 1, 2028, 1, 1, frequency=1),
        fixed_rate=jnp.array(0.0),
        notional=jnp.array(1_000_000.0),
        cms_tenor_long=10,
        cms_tenor_short=2,
        pay_fixed=True,
    )
    with pytest.raises(ValueError, match="projection curve"):
        mc_price_dispatch(
            swap, g2pp, MCConfig(n_paths=256, n_steps=8), _KEY,
            discount_curve=ois, forward_curve=projection,
        )
