"""Multi-curve (OIS discounting + tenor projection) in the short-rate PDEs.

Hull-White and G2++ are exact-fit-to-*one*-curve models, so "multi-curve" here
means the deterministic tenor-basis overlay of :mod:`valax.pricing.multicurve`:
discount with the model re-fitted to the trade's discount curve, project
floating legs off a separate curve whose basis to the discount curve is frozen
at its time-zero value.  The tests below are the acceptance criteria of
``PRICING_ONTOLOGY.md`` §12.6:

1. **Single-curve non-regression (the hard gate).**  Passing no curves, or
   passing the model's own curve, must reproduce the pre-existing number
   *exactly* — not to a tolerance.
2. **Discounting correctness.**  Payer-minus-receiver swaption parity is the
   forward swap value, which is model-independent: comparing it against the
   textbook dual-curve swap NPV computed straight off the two curves isolates
   the discount/projection split from anything the model does.
3. **Optionality.**  A payer swaption must rise monotonically with the tenor
   basis and collapse onto the single-curve price as the basis vanishes.

Also checks that a payoff which cannot yet project (the G2++ CMS spread)
*rejects* a distinct projection curve rather than silently ignoring it.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import FixedRateBond
from valax.instruments.rates import CMSSpreadSwap, Swaption
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.pde import PDEConfig, pde_price_dispatch
from valax.pricing.pde.config import PDEConfig2D

_CONFIG = PDEConfig(n_spot=301, n_time=300, spot_range=6.0)
_CONFIG_2D = PDEConfig2D(n_x=61, n_y=61, n_time=60, x_range=5.0)


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
    )


@pytest.fixture(scope="module")
def payer(ref_date) -> Swaption:
    """2y into 5y annual payer, so the first accrual runs from expiry.

    ``day_count="act_365"`` (rather than the ``act_360`` default) is
    deliberate: the short-rate recipes build their *time axis* from the
    instrument's day count, so an ``act_360`` trade is modelled on times that
    differ from the ``act_365`` curve's own by ~1.4 %.  That pre-existing
    convention mismatch is orthogonal to multi-curve, and matching the
    conventions here lets the curve-level oracle below be exact rather than
    approximate.
    """
    return Swaption(
        expiry_date=jnp.int32(int(ymd_to_ordinal(2027, 1, 1))),
        fixed_dates=generate_schedule(2027, 1, 1, 2032, 1, 1, frequency=1).payment_dates,
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        is_payer=True,
        day_count="act_365",
    )


def _receiver(payer: Swaption) -> Swaption:
    return Swaption(
        expiry_date=payer.expiry_date,
        fixed_dates=payer.fixed_dates,
        strike=payer.strike,
        notional=payer.notional,
        is_payer=False,
        day_count=payer.day_count,
    )


def _forward_swap_npv(
    payer: Swaption,
    discount: DiscountCurve,
    forward: DiscountCurve,
) -> float:
    """Textbook dual-curve payer-swap NPV, straight off the two curves.

    Floating leg ``sum_j (DF_f(T_{j-1}) / DF_f(T_j) - 1) DF_d(T_j)`` minus
    fixed leg ``K sum_j tau_j DF_d(T_j)``, with the floating schedule equal to
    the fixed schedule (the convention the short-rate recipes replicate).
    """
    payments = payer.fixed_dates
    starts = jnp.concatenate([payer.expiry_date[None], payments[:-1]])
    taus = year_fraction(starts, payments, payer.day_count)

    df = discount(payments)
    floating = jnp.sum((forward(starts) / forward(payments) - 1.0) * df)
    fixed = payer.strike * jnp.sum(taus * df)
    return float(payer.notional * (floating - fixed))


# ── 1. Single-curve non-regression ────────────────────────────────────


def test_hw_swaption_no_curves_equals_model_curve(hw, payer):
    """Passing the model's own curve must change literally nothing."""
    plain = pde_price_dispatch(payer, hw, _CONFIG)
    explicit = pde_price_dispatch(
        payer, hw, _CONFIG, discount_curve=hw.initial_curve, forward_curve=None
    )
    assert float(explicit.price) == float(plain.price)


def test_hw_cap_single_curve_non_regression(hw, ref_date):
    from valax.instruments.rates import Cap

    fixings = generate_schedule(2025, 7, 1, 2027, 7, 1, frequency=2).payment_dates
    cap = Cap(
        fixing_dates=fixings[:-1],
        start_dates=fixings[:-1],
        end_dates=fixings[1:],
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        is_cap=True,
    )
    plain = pde_price_dispatch(cap, hw, _CONFIG)
    explicit = pde_price_dispatch(
        cap, hw, _CONFIG, discount_curve=hw.initial_curve
    )
    assert float(explicit.price) == float(plain.price)


def test_g2pp_swaption_no_curves_equals_model_curve(g2pp, payer):
    plain = pde_price_dispatch(payer, g2pp, _CONFIG_2D)
    explicit = pde_price_dispatch(
        payer, g2pp, _CONFIG_2D, discount_curve=g2pp.initial_curve
    )
    assert float(explicit.price) == float(plain.price)


# ── 2. Discounting: the model is re-fitted to the discount curve ──────


def test_hw_bond_honours_a_distinct_discount_curve(hw, projection, ref_date):
    """An option-free bond must reprice off whatever curve it discounts on.

    The bond carries no model risk, so this isolates the re-fitting step: the
    PDE price on ``projection`` has to match the analytic curve price on
    ``projection``, not the one on the model's embedded OIS curve.
    """
    bond = FixedRateBond(
        payment_dates=generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2).payment_dates,
        settlement_date=ref_date,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        frequency=2,
    )
    result = pde_price_dispatch(bond, hw, _CONFIG, discount_curve=projection)
    expected = float(fixed_rate_bond_price(bond, projection))
    assert float(result.price) == pytest.approx(expected, abs=1e-4)

    # And it is genuinely different from the model's own-curve price.
    own = float(pde_price_dispatch(bond, hw, _CONFIG).price)
    assert abs(own - expected) > 1e-2


# ── 3. Projection: swaption parity is the dual-curve forward swap ─────


@pytest.mark.parametrize("model_name", ["hw", "g2pp"])
def test_swaption_parity_reproduces_the_dual_curve_swap(
    request, model_name, payer, ois, projection
):
    """Payer - receiver = forward swap, model-independently.

    This is acceptance criterion §12.6.2: a linear swap's NPV under the PDE
    with OIS discounting and a distinct projection curve must equal the
    analytic dual-curve NPV, to grid tolerance.  Parity is used because the
    solvers register no standalone swap recipe — and because it cancels the
    model entirely, so any residual is the discount/projection split alone.
    """
    model = request.getfixturevalue(model_name)
    config = _CONFIG if model_name == "hw" else _CONFIG_2D

    kwargs = dict(discount_curve=ois, forward_curve=projection)
    payer_price = float(pde_price_dispatch(payer, model, config, **kwargs).price)
    receiver_price = float(
        pde_price_dispatch(_receiver(payer), model, config, **kwargs).price
    )

    expected = _forward_swap_npv(payer, ois, projection)
    assert payer_price - receiver_price == pytest.approx(expected, rel=1e-4)


def test_swaption_parity_single_curve_still_holds(payer, hw, ois):
    """The same parity with no projection curve — the degenerate case."""
    payer_price = float(pde_price_dispatch(payer, hw, _CONFIG).price)
    receiver_price = float(
        pde_price_dispatch(_receiver(payer), hw, _CONFIG).price
    )
    expected = _forward_swap_npv(payer, ois, ois)
    assert payer_price - receiver_price == pytest.approx(expected, rel=1e-4)


# ── 4. Optionality: monotone in, and continuous at, zero basis ────────


def test_payer_swaption_increases_with_the_tenor_basis(
    hw, payer, ois, ref_date
):
    """Higher projected forwards make a payer swaption worth more.

    Acceptance criterion §12.6.3.  Note the dual-curve price is *above* both
    single-curve prices here rather than between them: raising the projection
    curve raises the forwards (good for a payer) while keeping the higher OIS
    discount factors, so both effects push the same way.
    """
    prices = [
        float(
            pde_price_dispatch(
                payer,
                hw,
                _CONFIG,
                discount_curve=ois,
                forward_curve=_flat_curve(ref_date, rate),
            ).price
        )
        for rate in (0.030, 0.035, 0.040)
    ]
    assert prices[0] < prices[1] < prices[2]

    single_curve = float(pde_price_dispatch(payer, hw, _CONFIG).price)
    # A zero basis must reproduce the single-curve price.
    assert prices[0] == pytest.approx(single_curve, rel=1e-10)
    # A positive basis is worth strictly more than either single-curve leg.
    on_projection = float(
        pde_price_dispatch(
            payer,
            HullWhiteModel(
                mean_reversion=hw.mean_reversion,
                volatility=hw.volatility,
                initial_curve=_flat_curve(ref_date, 0.035),
            ),
            _CONFIG,
        ).price
    )
    assert prices[1] > max(single_curve, on_projection)


# ── 5. Payoffs that cannot project must refuse, not ignore ────────────


def test_cms_spread_rejects_a_distinct_projection_curve(
    g2pp, ois, projection, ref_date
):
    swap = CMSSpreadSwap(
        start_date=jnp.int32(int(ymd_to_ordinal(2025, 1, 1))),
        payment_dates=generate_schedule(2026, 1, 1, 2028, 1, 1, frequency=1).payment_dates,
        fixed_rate=jnp.array(0.0),
        notional=jnp.array(1_000_000.0),
        cms_tenor_long=10,
        cms_tenor_short=2,
        pay_fixed=True,
    )
    with pytest.raises(ValueError, match="projection curve"):
        pde_price_dispatch(
            swap,
            g2pp,
            _CONFIG_2D,
            discount_curve=ois,
            forward_curve=projection,
        )
