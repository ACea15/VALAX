"""Two-factor Cheyette frozen-y 2-D ADI PDE with piecewise σ₁(t) and ratio r(t).

Each backward step diffuses at the exact step averages of σ₁², σ₂², ρσ₁σ₂
(:func:`~valax.pricing.pde.cheyette.cheyette_operator_rows_2d`, assembled
inside the ADI loop) times the skew factors, and discounts by the exact step
integral of the G2++ twin's φ. Oracles:

* β₁ = β₂ = 0: the G2++ PDE with the same σ(t), r(t), κ₁ = a, κ₂ = b, ρ on
  the same grid — the same operators, so agreement is at round-off (measured
  on the 3Y×5Y ATM payer: 4.4e-16 relative at 61²×50, 2.2e-16 at 111²×90);
* the two-bucket-equal path against the constant single-operator path;
* β ≠ 0: the Bermudan dominates its co-terminal Europeans.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.cheyette import CheyetteModel2F
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.pde import PDEConfig2D, Scheme, pde_price_dispatch
from valax.pricing.pde.cheyette import cheyette_operator_2d, cheyette_operator_rows_2d
from valax.pricing.pde.grids import Grid2D, centred_state_grid

_ACT365 = "act_365"
REF = int(ymd_to_ordinal(2026, 1, 1))
VOLS = (0.012, 0.008, 0.015)
RATIOS = (0.8, 0.6, 0.9)
TIMES = (1.0, 2.5)

COARSE = PDEConfig2D(n_x=61, n_y=61, n_time=50, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)


def _curve():
    years = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0)
    zeros = (0.020, 0.021, 0.022, 0.025, 0.027, 0.031, 0.034, 0.038, 0.040, 0.041)
    pillars = jnp.array([REF + int(round(y * 365)) for y in years], dtype=jnp.int32)
    times = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(zeros) * times),
        reference_date=jnp.int32(REF),
        day_count=_ACT365,
    )


CURVE = _curve()


def _cheyette(skew_1=0.0, skew_2=0.0, vols=VOLS, ratios=RATIOS, times=TIMES):
    return CheyetteModel2F(
        mean_reversion_1=jnp.asarray(0.5),
        mean_reversion_2=jnp.asarray(0.1),
        volatility_1=jnp.asarray(vols),
        volatility_ratio=jnp.asarray(ratios),
        skew_1=jnp.asarray(skew_1),
        skew_2=jnp.asarray(skew_2),
        correlation=jnp.asarray(-0.7),
        initial_curve=CURVE,
        volatility_times=jnp.asarray(times),
    )


def _g2pp():
    return G2PPModel(
        mean_reversion_x=jnp.asarray(0.5),
        mean_reversion_y=jnp.asarray(0.1),
        volatility_x=jnp.asarray(VOLS),
        volatility_ratio=jnp.asarray(RATIOS),
        correlation=jnp.asarray(-0.7),
        initial_curve=CURVE,
        volatility_times=jnp.asarray(TIMES),
    )


def _fixed(start_y, n):
    return jnp.array([REF + 365 * (start_y + k) for k in range(1, n + 1)], dtype=jnp.int32)


def _atm(expiry_ord, fixed):
    ann = _annuity(jnp.int32(expiry_ord), fixed, CURVE, _ACT365)
    return float((CURVE(jnp.int32(expiry_ord)) - CURVE(fixed[-1])) / ann)


def _swaption(expiry_y, tenor_y, moneyness=0.0, is_payer=True):
    expiry = REF + 365 * expiry_y
    fixed = _fixed(expiry_y, tenor_y)
    return Swaption(
        expiry_date=jnp.int32(expiry), fixed_dates=fixed,
        strike=jnp.asarray(_atm(expiry, fixed) + moneyness),
        notional=jnp.asarray(1.0), is_payer=is_payer, day_count=_ACT365,
    )


def _pde(inst, model, config=COARSE):
    return float(pde_price_dispatch(inst, model, config).price)


class TestGaussianLimitIsG2PP:
    @pytest.mark.parametrize("expiry_y,tenor_y,moneyness,is_payer", [
        (3, 5, 0.0, True), (2, 4, 0.01, False),
    ])
    def test_european_matches_g2pp_pde(self, expiry_y, tenor_y, moneyness, is_payer):
        s = _swaption(expiry_y, tenor_y, moneyness, is_payer)
        assert _pde(s, _cheyette()) == pytest.approx(_pde(s, _g2pp()), rel=1e-13)

    def test_bermudan_matches_g2pp_pde(self):
        berm, _ = _coterminal()
        assert _pde(berm, _cheyette()) == pytest.approx(_pde(berm, _g2pp()), rel=1e-13)


class TestOperators:
    def test_constant_operator_refuses_buckets(self):
        grid = Grid2D(x=centred_state_grid(jnp.asarray(0.01), n=11, half_width=5.0),
                      y=centred_state_grid(jnp.asarray(0.01), n=11, half_width=5.0))
        with pytest.raises(ValueError, match="cheyette_operator_rows_2d"):
            cheyette_operator_2d(_cheyette(), grid, ref_time=jnp.asarray(3.0))

    def test_rows_equal_constant_operator_for_equal_buckets(self):
        grid = Grid2D(x=centred_state_grid(jnp.asarray(0.01), n=21, half_width=5.0),
                      y=centred_state_grid(jnp.asarray(0.02), n=17, half_width=5.0))
        const = _cheyette(0.4, -0.2, vols=(0.011,), ratios=(0.7,), times=())
        split = _cheyette(0.4, -0.2, vols=(0.011,) * 3, ratios=(0.7,) * 3)
        ref = cheyette_operator_2d(const, grid, ref_time=jnp.asarray(3.0))
        rows = cheyette_operator_rows_2d(
            split, grid, ref_time=jnp.asarray(3.0), expiry=jnp.asarray(3.0), n_time=10
        )
        for m in (0, 4, 9):
            for a, b in zip(jax.tree.leaves(rows(jnp.asarray(m))), jax.tree.leaves(ref)):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-18)

    def test_equal_buckets_price_matches_constant(self):
        s = _swaption(3, 5)
        const = _cheyette(0.4, -0.2, vols=(0.011,), ratios=(0.7,), times=())
        split = _cheyette(0.4, -0.2, vols=(0.011,) * 3, ratios=(0.7,) * 3)
        assert _pde(s, split) == pytest.approx(_pde(s, const), rel=1e-11)


def _coterminal():
    exercise_y = (1, 2, 3, 4)
    fixed = _fixed(1, 5)  # years 2..6
    strike = _atm(REF + 365, fixed)
    berm = BermudanSwaption(
        exercise_dates=jnp.array([REF + 365 * y for y in exercise_y], dtype=jnp.int32),
        fixed_dates=fixed, strike=jnp.asarray(strike), notional=jnp.asarray(1.0),
        is_payer=True, day_count=_ACT365,
    )
    europeans = [
        Swaption(
            expiry_date=jnp.int32(REF + 365 * y), fixed_dates=fixed[i:],
            strike=jnp.asarray(strike), notional=jnp.asarray(1.0),
            is_payer=True, day_count=_ACT365,
        )
        for i, y in enumerate(exercise_y)
    ]
    return berm, europeans


class TestSkewedBermudan:
    def test_dominates_coterminal_europeans(self):
        model = _cheyette(0.5, -0.3)
        berm, europeans = _coterminal()
        b = _pde(berm, model)
        euro = [_pde(e, model) for e in europeans]
        assert b >= max(euro) - 1e-9
        assert b <= sum(euro) + 1e-9

    def test_skew_moves_price(self):
        s = _swaption(3, 5)
        assert _pde(s, _cheyette(0.5, -0.3)) != pytest.approx(_pde(s, _cheyette()), rel=1e-4)


class TestDifferentiationAndJIT:
    def test_filter_jit_smoke(self):
        s = _swaption(3, 5)

        @eqx.filter_jit
        def f(m):
            return pde_price_dispatch(s, m, COARSE).price

        model = _cheyette(0.5, -0.3)
        assert float(f(model)) == pytest.approx(_pde(s, model), rel=1e-12)

    @pytest.mark.parametrize("field", ["volatility_1", "volatility_ratio"])
    def test_bucket_vega_matches_fd(self, field):
        """Autodiff through the per-step operators vs FD of the same grid."""
        s = _swaption(3, 5)
        model = _cheyette(0.5, -0.3)
        cfg = PDEConfig2D(n_x=41, n_y=41, n_time=30, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)

        def f(v):
            return pde_price_dispatch(
                s, eqx.tree_at(lambda m: getattr(m, field), model, v), cfg
            ).price

        v0 = getattr(model, field)
        grad = np.asarray(jax.grad(f)(v0))
        for i in range(v0.shape[0]):
            h = 1e-6 * float(v0[i])
            e = jnp.zeros(v0.shape[0]).at[i].set(h)
            fd = (float(f(v0 + e)) - float(f(v0 - e))) / (2 * h)
            assert grad[i] == pytest.approx(fd, rel=1e-4)
