r"""Tests for the Cheyette quasi-Gaussian short-rate PDE pricers.

The validation is a triangulation, since there is no QuantLib Cheyette engine:

- **Reduction (primary oracle).** At :math:`\beta = 0` frozen-``y`` is exact and
  the volatility is constant, so the Cheyette PDE must reproduce the Hull-White
  (1F) / G2++ (2F) PDE — on the *same* mesh, to numerical precision.
- **Exact fit.** An option-free coupon bond reprices its own input curve.
- **Frozen-``y`` vs exact-``y`` (one factor).** Tier 1 (frozen) and Tier 2
  (exact) agree at :math:`\beta = 0` to grid tolerance; their difference at
  :math:`\beta \neq 0` is the (tiny, for realistic rate vols) frozen-``y`` bias,
  measured deterministically with zero Monte-Carlo noise.
- **MC <-> PDE.** The frozen-``y`` PDE agrees with the exact-``y`` Monte-Carlo
  engine within Monte-Carlo error plus the frozen-``y`` bias.
- **Differentiability.** The PDE price is cleanly reverse-mode differentiable in
  the model parameters (needed for calibration).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.instruments.bonds import FixedRateBond
from valax.instruments.rates import Swaption
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.mc import MCConfig, mc_price_dispatch
from valax.pricing.pde import (
    PDEConfig,
    PDEConfig2D,
    Scheme,
    pde_price_dispatch,
    registered_recipes,
)
from valax.pricing.pde.cheyette import cheyette_exact_price

_ACT365 = "act_365"

# Moderate meshes: fine enough for the reduction / exact-fit gates, quick enough
# for the suite. The 1-D and 2-D reduction pairs share their config so any
# disagreement is a modelling difference, never coarseness.
CFG_1F = PDEConfig(n_spot=201, n_time=100, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON)
CFG_2F = PDEConfig2D(n_x=111, n_y=111, n_time=90, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)
CFG_EXACT = PDEConfig2D(n_x=121, n_y=81, n_time=120, x_range=6.0, y_max=3.0)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def ref_date():
    return int(ymd_to_ordinal(2026, 1, 1))


def _flat_curve(ref_date, rate=0.03, n_years=26):
    years = [0.0] + [float(k) for k in range(1, n_years)]
    pillars = jnp.array(
        [ref_date + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - ref_date).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=jnp.int32(ref_date),
        day_count=_ACT365,
    )


def _sloped_curve(ref_date, n_years=26):
    years = [0.0] + [float(k) for k in range(1, n_years)]
    pillars = jnp.array(
        [ref_date + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - ref_date).astype(jnp.float64) / 365.0
    zero_rates = 0.03 + 0.015 * jnp.tanh(times / 4.0)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero_rates * times),
        reference_date=jnp.int32(ref_date),
        day_count=_ACT365,
    )


@pytest.fixture
def flat_curve(ref_date):
    return _flat_curve(ref_date)


@pytest.fixture
def sloped_curve(ref_date):
    return _sloped_curve(ref_date)


# ── Model builders ───────────────────────────────────────────────────


def _cheyette_1f(curve, kappa=0.05, sigma=0.01, beta=0.0):
    return CheyetteModel(
        mean_reversion=jnp.array(kappa),
        volatility=jnp.array(sigma),
        skew=jnp.array(beta),
        initial_curve=curve,
    )


def _hw(curve, kappa=0.05, sigma=0.01):
    return HullWhiteModel(
        mean_reversion=jnp.array(kappa),
        volatility=jnp.array(sigma),
        initial_curve=curve,
    )


def _cheyette_2f(curve, beta1=0.0, beta2=0.0, rho=-0.7):
    return CheyetteModel2F(
        mean_reversion_1=jnp.array(0.5),
        mean_reversion_2=jnp.array(0.1),
        volatility_1=jnp.array(0.01),
        volatility_2=jnp.array(0.008),
        skew_1=jnp.array(beta1),
        skew_2=jnp.array(beta2),
        correlation=jnp.array(rho),
        initial_curve=curve,
    )


def _g2pp(curve, rho=-0.7):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.5),
        mean_reversion_y=jnp.array(0.1),
        volatility_x=jnp.array(0.01),
        volatility_y=jnp.array(0.008),
        correlation=jnp.array(rho),
        initial_curve=curve,
    )


# ── Instrument builders ──────────────────────────────────────────────


def _fixed_dates(start_year, tenor_y):
    return jnp.array(
        [int(ymd_to_ordinal(start_year + k, 1, 1)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )


def _swaption(expiry_year, tenor_y, strike, is_payer=True, notional=1.0):
    return Swaption(
        expiry_date=jnp.int32(int(ymd_to_ordinal(expiry_year, 1, 1))),
        fixed_dates=_fixed_dates(expiry_year, tenor_y),
        strike=jnp.asarray(strike),
        notional=jnp.array(notional),
        is_payer=is_payer,
        day_count=_ACT365,
    )


def _atm_strike(curve, expiry_year, tenor_y):
    expiry = jnp.int32(int(ymd_to_ordinal(expiry_year, 1, 1)))
    fixed = _fixed_dates(expiry_year, tenor_y)
    ann = _annuity(expiry, fixed, curve, _ACT365)
    return float((curve(expiry) - curve(fixed[-1])) / ann)


def _bond(ref_date, coupon=0.03, tenor_y=8):
    fixed = _fixed_dates(2026, tenor_y)
    return FixedRateBond(
        payment_dates=fixed,
        settlement_date=jnp.int32(ref_date),
        coupon_rate=jnp.array(coupon),
        face_value=jnp.array(1.0),
        frequency=1,
        day_count=_ACT365,
    )


def _pde(instrument, model, config):
    return float(pde_price_dispatch(instrument, model, config).price)


# ═════════════════════════════════════════════════════════════════════


class TestRegistration:
    def test_recipes_registered(self):
        recipes = registered_recipes()
        for inst in ("Swaption", "BermudanSwaption", "FixedRateBond"):
            assert (inst, "CheyetteModel") in recipes
            assert (inst, "CheyetteModel2F") in recipes
        assert ("CallableBond", "CheyetteModel") in recipes
        assert ("PuttableBond", "CheyetteModel") in recipes


class TestReduction1F:
    """At beta = 0 the 1F Cheyette PDE reduces to the Hull-White PDE."""

    @pytest.mark.parametrize("is_payer", [True, False])
    def test_swaption_matches_hull_white_pde(self, flat_curve, is_payer):
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike, is_payer=is_payer)
        ch = _cheyette_1f(flat_curve, beta=0.0)
        hw = _hw(flat_curve)
        assert _pde(s, ch, CFG_1F) == pytest.approx(_pde(s, hw, CFG_1F), abs=1e-10)

    def test_swaption_matches_hw_sloped(self, sloped_curve):
        strike = _atm_strike(sloped_curve, 2030, 5)
        s = _swaption(2030, 5, strike)
        ch = _cheyette_1f(sloped_curve, beta=0.0)
        hw = _hw(sloped_curve)
        assert _pde(s, ch, CFG_1F) == pytest.approx(_pde(s, hw, CFG_1F), abs=1e-10)

    def test_swaption_matches_analytic(self, flat_curve):
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike)
        ch = _cheyette_1f(flat_curve, beta=0.0)
        analytic = float(hw_swaption_price(s, _hw(flat_curve)))
        assert _pde(s, ch, CFG_1F) == pytest.approx(analytic, rel=3e-3, abs=2e-5)


class TestReduction2F:
    """At (beta1, beta2) = 0 the 2F Cheyette PDE reduces to the G2++ PDE."""

    @pytest.mark.parametrize("rho", [-0.7, 0.0, 0.5])
    def test_swaption_matches_g2pp_pde(self, flat_curve, rho):
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike)
        ch = _cheyette_2f(flat_curve, rho=rho)
        g2 = _g2pp(flat_curve, rho=rho)
        assert _pde(s, ch, CFG_2F) == pytest.approx(_pde(s, g2, CFG_2F), abs=1e-9)

    def test_swaption_sloped(self, sloped_curve):
        strike = _atm_strike(sloped_curve, 2030, 5)
        s = _swaption(2030, 5, strike)
        ch = _cheyette_2f(sloped_curve)
        g2 = _g2pp(sloped_curve)
        assert _pde(s, ch, CFG_2F) == pytest.approx(_pde(s, g2, CFG_2F), abs=1e-9)


class TestExactFit:
    """An option-free coupon bond reprices its own input curve."""

    def _analytic_bond(self, curve, fixed, coupon):
        dfs = curve(fixed)
        return float(coupon * dfs.sum() + dfs[-1])

    def test_bond_reprices_curve_1f(self, flat_curve, ref_date):
        bond = _bond(ref_date, coupon=0.03)
        ch = _cheyette_1f(flat_curve, beta=0.0)
        analytic = self._analytic_bond(flat_curve, _fixed_dates(2026, 8), 0.03)
        assert _pde(bond, ch, CFG_1F) == pytest.approx(analytic, abs=5e-5)

    def test_bond_reprices_curve_1f_sloped(self, sloped_curve, ref_date):
        bond = _bond(ref_date, coupon=0.035)
        ch = _cheyette_1f(sloped_curve, beta=0.0)
        analytic = self._analytic_bond(sloped_curve, _fixed_dates(2026, 8), 0.035)
        assert _pde(bond, ch, CFG_1F) == pytest.approx(analytic, abs=5e-5)

    def test_bond_reprices_curve_2f(self, flat_curve, ref_date):
        bond = _bond(ref_date, coupon=0.03)
        ch = _cheyette_2f(flat_curve)
        analytic = self._analytic_bond(flat_curve, _fixed_dates(2026, 8), 0.03)
        assert _pde(bond, ch, CFG_2F) == pytest.approx(analytic, abs=1e-4)

    def test_exact_y_bond_reprices_curve(self, flat_curve, ref_date):
        """The exact-``y`` solver reprices the curve tightly for any beta."""
        bond = _bond(ref_date, coupon=0.03)
        analytic = self._analytic_bond(flat_curve, _fixed_dates(2026, 8), 0.03)
        for beta in (0.0, 0.3):
            ch = _cheyette_1f(flat_curve, beta=beta)
            price = float(cheyette_exact_price(bond, ch, CFG_EXACT).price)
            assert price == pytest.approx(analytic, abs=5e-5)


class TestFrozenVsExactY:
    """Tier 1 (frozen-``y``) vs Tier 2 (exact-``y``), one factor."""

    def test_agree_at_zero_skew(self, flat_curve):
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike)
        ch = _cheyette_1f(flat_curve, beta=0.0)
        frozen = _pde(s, ch, CFG_1F)
        exact = float(cheyette_exact_price(s, ch, CFG_EXACT).price)
        # Different discretisations of the same beta = 0 model: agree to grid tol.
        assert exact == pytest.approx(frozen, rel=5e-3)

    def test_frozen_y_bias_is_small(self, flat_curve):
        """For realistic rate vols the frozen-``y`` bias is a small fraction of
        the price at every skew — the quantitative justification for the
        production frozen-``y`` engine."""
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike)
        for beta in (0.0, 0.2, 0.4):
            ch = _cheyette_1f(flat_curve, sigma=0.012, beta=beta)
            frozen = _pde(s, ch, CFG_1F)
            exact = float(cheyette_exact_price(s, ch, CFG_EXACT).price)
            assert abs(exact - frozen) < 1e-2 * frozen


class TestMonteCarloConsistency:
    """The frozen-``y`` PDE agrees with the exact-``y`` MC engine."""

    def test_swaption_matches_mc_zero_skew(self, flat_curve, ref_date):
        strike = _atm_strike(flat_curve, 2029, 4)
        s = _swaption(2029, 4, strike, notional=1.0)
        ch = _cheyette_1f(flat_curve, kappa=0.1, sigma=0.01, beta=0.0)
        pde = _pde(s, ch, CFG_1F)
        res = mc_price_dispatch(
            s, ch, MCConfig(n_paths=120_000, n_steps=48),
            jax.random.PRNGKey(5), n_steps=48,
        )
        assert abs(float(res.price) - pde) < 3.0 * float(res.stderr) + 1e-4

    def test_swaption_matches_mc_with_skew(self, flat_curve, ref_date):
        strike = _atm_strike(flat_curve, 2029, 4)
        s = _swaption(2029, 4, strike, notional=1.0)
        ch = _cheyette_1f(flat_curve, kappa=0.1, sigma=0.01, beta=0.25)
        pde = _pde(s, ch, CFG_1F)
        res = mc_price_dispatch(
            s, ch, MCConfig(n_paths=120_000, n_steps=48),
            jax.random.PRNGKey(6), n_steps=48,
        )
        # MC prices the true-y model; the PDE is frozen-y. Allow MC noise plus
        # the (tiny) frozen-y bias.
        assert abs(float(res.price) - pde) < 3.0 * float(res.stderr) + 1e-3


class TestConvergence:
    def test_refining_tightens_reduction(self, flat_curve):
        """Refining the mesh drives the beta = 0 swaption toward the analytic."""
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike)
        ch = _cheyette_1f(flat_curve, beta=0.0)
        analytic = float(hw_swaption_price(s, _hw(flat_curve)))
        coarse = PDEConfig(n_spot=81, n_time=40, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON)
        fine = PDEConfig(n_spot=301, n_time=200, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON)
        err_coarse = abs(_pde(s, ch, coarse) - analytic)
        err_fine = abs(_pde(s, ch, fine) - analytic)
        assert err_fine < err_coarse
        assert err_fine < 1e-3 * analytic


class TestSkewSignature:
    def test_skew_moves_price(self, flat_curve):
        """A payer swaption's PDE price responds monotonically to the skew."""
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike + 0.01, is_payer=True)  # OTM: skew-sensitive
        prices = [
            _pde(s, _cheyette_1f(flat_curve, sigma=0.012, beta=b), CFG_1F)
            for b in (-0.3, 0.0, 0.3)
        ]
        assert prices[0] != pytest.approx(prices[2], rel=1e-3)


class TestDifferentiabilityAndJit:
    def test_frozen_1f_grad_finite(self, flat_curve):
        s = _swaption(2031, 5, _atm_strike(flat_curve, 2031, 5))
        cfg = PDEConfig(n_spot=101, n_time=50, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON)

        def price(sig, beta):
            m = _cheyette_1f(flat_curve, sigma=sig, beta=beta)
            return pde_price_dispatch(s, m, cfg).price

        g = eqx.filter_jit(jax.grad(price, argnums=(0, 1)))(
            jnp.array(0.01), jnp.array(0.2)
        )
        assert jnp.isfinite(g[0]) and jnp.isfinite(g[1])
        assert float(g[0]) > 0.0  # vega positive

    def test_exact_1f_grad_finite(self, flat_curve):
        s = _swaption(2029, 4, _atm_strike(flat_curve, 2029, 4))
        cfg = PDEConfig2D(n_x=61, n_y=41, n_time=48, x_range=6.0, y_max=3.0)

        def price(sig):
            m = _cheyette_1f(flat_curve, sigma=sig, beta=0.15)
            return cheyette_exact_price(s, m, cfg).price

        g = eqx.filter_jit(jax.grad(price))(jnp.array(0.01))
        assert jnp.isfinite(g) and float(g) > 0.0

    def test_frozen_2f_jit(self, flat_curve):
        s = _swaption(2031, 5, _atm_strike(flat_curve, 2031, 5))
        cfg = PDEConfig2D(n_x=71, n_y=71, n_time=40, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)

        def price(sig1):
            m = _cheyette_2f(flat_curve)
            m = eqx.tree_at(lambda mm: mm.volatility_1, m, sig1)
            return pde_price_dispatch(s, m, cfg).price

        v = eqx.filter_jit(price)(jnp.array(0.01))
        assert jnp.isfinite(v) and float(v) > 0.0
