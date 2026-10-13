r"""Bit-for-bit regression gate for the shared short-rate PDE payoff layer.

The migration of ``hull_white.py`` / ``g2pp.py`` / ``cheyette.py`` onto the
model-agnostic :mod:`valax.pricing.pde.rates_payoffs` backbone must be
**price-preserving**: every recipe has to return the *same float* it did before
the refactor. These golden numbers were captured on the pre-refactor code (see
the module history / ``_capture_golden.py``) and are asserted here to machine
precision, so any accidental change in the arithmetic — a reordered sum, a
dropped ``alpha`` term, a different bond-price call — trips immediately.

This is deliberately a stored-number test (unlike the rest of ``test_pde``,
which checks against independent oracles): its whole job is to guard the
refactor, not to re-validate the models.

**Baseline moved once, deliberately.** The bond goldens were re-captured when
coupon sizing changed from ``coupon_rate / frequency`` to a day-count year
fraction over each accrual period, and when callable/puttable exercise began
comparing a clean call price against a clean continuation value. Both were
corrections, not refactors, so price preservation could not hold: the bullet
prices moved by 0.87 cents per 100 of face and the callable/puttable by
between 0.13 and 0.68 cents. Every bond here now carries an explicit
``accrual_start_dates``, so the goldens are anchored on a fully specified
schedule rather than on one inferred from the payment dates.

The swaption and Bermudan goldens are **unchanged to the last bit**, which is
the useful signal: the correction touches bond coupon accrual and the exercise
obstacle, and nothing else.

**Cell-averaged payoffs.** The recipes now cell-average kinked payoffs and
exercise projections by default (``smooth_payoff=True``, see
:mod:`valax.pricing.pde.payoff_smoothing`). The goldens stay the pointwise
numbers and every config here sets ``smooth_payoff=False``, so the switch is
proven to restore the original arithmetic exactly; the smoothed prices are
validated against oracles in ``test_payoff_smoothing.py`` and the per-model
files instead.
"""

import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import CallableBond, FixedRateBond, PuttableBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde import PDEConfig, PDEConfig2D, Scheme, pde_price_dispatch
from valax.pricing.pde.cheyette import cheyette_exact_price

# Golden prices captured on the pre-refactor recipes, re-captured once when
# day-count coupon accrual and the clean/dirty exercise comparison landed
# (see the module docstring).
GOLDEN = {
    "hw_bullet": 104.50702170207438,
    "hw_callable": 101.4338759620375,
    "hw_puttable": 105.00039307221081,
    "hw_swaption": 588.8695461364847,
    "hw_bermudan": 2887.7488420481063,
    "g2_bullet": 104.50642169585382,
    "g2_swaption": 9.942659392497438,
    "g2_bermudan": 341.4774890058857,
    "ch1_bullet": 104.50708284611957,
    "ch1_swaption": 1349.0715337959873,
    "ch1_bermudan": 4636.66777250074,
    "ch1_callable": 101.31731866746532,
    "ch1_puttable": 105.12588429287793,
    "ch2_bullet": 104.50642222588172,
    "ch2_swaption": 10.074270539247307,
    "ch2_bermudan": 342.99139729908006,
    "chex_bullet": 104.506231027728,
    "chex_swaption": 1366.3178400013967,
}

REF = int(ymd_to_ordinal(2025, 1, 1))


def _flat_curve(rate=0.03, n=26):
    pillars = jnp.array(
        [REF + int(round(y * 365)) for y in range(n)], dtype=jnp.int32
    )
    times = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=jnp.int32(REF),
        day_count="act_365",
    )


CURVE = _flat_curve()
BOND_ACCRUAL_START, _, BOND_SCHED = generate_schedule(
    2025, 1, 1, 2030, 1, 1, frequency=2
)
SWAP_SCHED = generate_schedule(2027, 1, 1, 2032, 1, 1, frequency=1).payment_dates
CALL_DATES = jnp.array(
    [int(ymd_to_ordinal(y, 1, 1)) for y in (2027, 2028, 2029)], dtype=jnp.int32
)
EX_DATES = jnp.array(
    [int(ymd_to_ordinal(y, 1, 1)) for y in (2027, 2028, 2029, 2030, 2031)],
    dtype=jnp.int32,
)

BULLET = FixedRateBond(
    payment_dates=BOND_SCHED, accrual_start_dates=BOND_ACCRUAL_START,
    settlement_date=jnp.int32(REF),
    coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0), frequency=2,
)
CALLABLE = CallableBond(
    payment_dates=BOND_SCHED, accrual_start_dates=BOND_ACCRUAL_START,
    settlement_date=jnp.int32(REF),
    coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
    call_dates=CALL_DATES, call_prices=jnp.full(CALL_DATES.shape[0], 1.0),
    frequency=2,
)
PUTTABLE = PuttableBond(
    payment_dates=BOND_SCHED, accrual_start_dates=BOND_ACCRUAL_START,
    settlement_date=jnp.int32(REF),
    coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
    put_dates=CALL_DATES, put_prices=jnp.full(CALL_DATES.shape[0], 1.0),
    frequency=2,
)
SWAPTION = Swaption(
    expiry_date=jnp.array(int(ymd_to_ordinal(2027, 1, 1)), dtype=jnp.int32),
    fixed_dates=SWAP_SCHED, strike=jnp.array(0.05), notional=jnp.array(1.0e6),
    is_payer=True, day_count="act_365",
)
BERMUDAN = BermudanSwaption(
    exercise_dates=EX_DATES, fixed_dates=SWAP_SCHED, strike=jnp.array(0.05),
    notional=jnp.array(1.0e6), is_payer=True, day_count="act_365",
)

HW = HullWhiteModel(
    mean_reversion=jnp.array(0.10), volatility=jnp.array(0.01),
    initial_curve=CURVE,
)
G2 = G2PPModel(
    mean_reversion_x=jnp.array(0.5), mean_reversion_y=jnp.array(0.1),
    volatility_x=jnp.array(0.01), volatility_y=jnp.array(0.008),
    correlation=jnp.array(-0.7), initial_curve=CURVE,
)
CH1 = CheyetteModel(
    mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01),
    skew=jnp.array(0.1), initial_curve=CURVE,
)
CH2 = CheyetteModel2F(
    mean_reversion_1=jnp.array(0.5), mean_reversion_2=jnp.array(0.1),
    volatility_1=jnp.array(0.01), volatility_2=jnp.array(0.008),
    skew_1=jnp.array(0.1), skew_2=jnp.array(0.05), correlation=jnp.array(-0.7),
    initial_curve=CURVE,
)

# ``smooth_payoff=False``: the goldens are the pointwise-payoff arithmetic, so
# this gate also proves that switching the cell-averaged payoff off restores
# the pre-smoothing prices to the last bit.
C1 = PDEConfig(n_spot=121, n_time=60, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON,
               smooth_payoff=False)
C2 = PDEConfig2D(n_x=71, n_y=71, n_time=50, x_range=6.0, scheme=Scheme.CRAIG_SNEYD,
                 smooth_payoff=False)
CE = PDEConfig2D(n_x=81, n_y=61, n_time=60, x_range=6.0, y_max=3.0,
                 smooth_payoff=False)

DISPATCH_CASES = {
    "hw_bullet": (BULLET, HW, C1),
    "hw_callable": (CALLABLE, HW, C1),
    "hw_puttable": (PUTTABLE, HW, C1),
    "hw_swaption": (SWAPTION, HW, C1),
    "hw_bermudan": (BERMUDAN, HW, C1),
    "g2_bullet": (BULLET, G2, C2),
    "g2_swaption": (SWAPTION, G2, C2),
    "g2_bermudan": (BERMUDAN, G2, C2),
    "ch1_bullet": (BULLET, CH1, C1),
    "ch1_swaption": (SWAPTION, CH1, C1),
    "ch1_bermudan": (BERMUDAN, CH1, C1),
    "ch1_callable": (CALLABLE, CH1, C1),
    "ch1_puttable": (PUTTABLE, CH1, C1),
    "ch2_bullet": (BULLET, CH2, C2),
    "ch2_swaption": (SWAPTION, CH2, C2),
    "ch2_bermudan": (BERMUDAN, CH2, C2),
}


@pytest.mark.parametrize("name", list(DISPATCH_CASES))
def test_dispatch_recipe_price_preserved(name):
    instrument, model, config = DISPATCH_CASES[name]
    price = float(pde_price_dispatch(instrument, model, config).price)
    assert price == pytest.approx(GOLDEN[name], abs=0.0, rel=1e-12)


@pytest.mark.parametrize(
    "name,instrument", [("chex_bullet", BULLET), ("chex_swaption", SWAPTION)]
)
def test_exact_y_price_preserved(name, instrument):
    price = float(cheyette_exact_price(instrument, CH1, CE).price)
    assert price == pytest.approx(GOLDEN[name], abs=0.0, rel=1e-12)
