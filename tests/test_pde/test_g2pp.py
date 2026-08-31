"""Tests for the G2++ two-factor Gaussian short-rate PDE pricers.

Every instrument is checked against an **independent** oracle rather than a
stored number:

- ``FixedRateBond``     -> the analytic curve price. No optionality, so the
                           answer cannot depend on the model parameters — any
                           error is pure numerics (mesh, ADI stepping, or the
                           exact-fit ``phi`` treatment).
- ``Swaption``          -> the exact Gauss-Hermite ``g2pp_swaption_price``,
                           itself validated to ~1e-10 against QuantLib. This is
                           the tight gate that sizes the grid.
- ``BermudanSwaption``  -> a single-exercise Bermudan must reproduce the
                           European exactly, and the multi-exercise price must
                           sit between its co-terminal European bounds (and
                           carry a non-negative early-exercise premium).

The cross term ``rho sigma eta V_xy`` is the crux of two-factor decorrelation;
:class:`TestDecorrelation` checks the PDE tracks the analytic price's sign of
response to ``rho`` — evidence the mixed operator is wired correctly. Models use
well-separated ``a != b`` to avoid the degenerate ``a ~ b`` case and keep
``|rho| < 1`` (``rho -> +/-1`` is singular).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import FixedRateBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.pde import (
    PDEConfig2D,
    Scheme,
    pde_price_dispatch,
    registered_recipes,
)


_ACT365 = "act_365"

# Moderate mesh: the tight European gate lands well inside a few e-3 here, and
# every test in the file uses it (or coarser) to keep the suite quick.
MODERATE = PDEConfig2D(
    n_x=111, n_y=111, n_time=90, x_range=6.0, scheme=Scheme.CRAIG_SNEYD
)
COARSE = PDEConfig2D(
    n_x=61, n_y=61, n_time=50, x_range=6.0, scheme=Scheme.CRAIG_SNEYD
)
FINE = PDEConfig2D(
    n_x=161, n_y=161, n_time=140, x_range=6.0, scheme=Scheme.CRAIG_SNEYD
)


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
    """Upward-sloping curve — exercises the non-constant ``phi(t)`` path."""
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


def _model(curve, rho=-0.70):
    # Well-separated a != b avoids the a ~ b degeneracy; |rho| < 1.
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.010),
        volatility_y=jnp.array(0.008),
        correlation=jnp.array(rho),
        initial_curve=curve,
    )


@pytest.fixture
def model(flat_curve):
    return _model(flat_curve)


# ── Instrument builders ──────────────────────────────────────────────


def _fixed_dates(start_year, tenor_y):
    return jnp.array(
        [int(ymd_to_ordinal(start_year + k, 1, 1)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )


def _swaption(expiry_year, tenor_y, strike, is_payer=True, notional=1.0):
    expiry = jnp.int32(int(ymd_to_ordinal(expiry_year, 1, 1)))
    return Swaption(
        expiry_date=expiry,
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
    fwd = (curve(expiry) - curve(fixed[-1])) / ann
    return float(fwd)


def _pde(instrument, model, config=MODERATE):
    return float(pde_price_dispatch(instrument, model, config).price)


# ── Registration ─────────────────────────────────────────────────────


class TestRegistration:
    def test_recipes_registered(self):
        recipes = registered_recipes()
        assert ("Swaption", "G2PPModel") in recipes
        assert ("BermudanSwaption", "G2PPModel") in recipes
        assert ("FixedRateBond", "G2PPModel") in recipes


# ── European swaption vs the analytic gold reference (the tight gate) ─


class TestEuropeanVsAnalytic:
    @pytest.mark.parametrize("is_payer", [True, False])
    @pytest.mark.parametrize("moneyness", [0.0, 0.01, -0.01])
    def test_atm_and_off_atm_flat(self, flat_curve, model, is_payer, moneyness):
        strike = _atm_strike(flat_curve, 2031, 5) + moneyness
        s = _swaption(2031, 5, strike, is_payer=is_payer)
        analytic = float(g2pp_swaption_price(s, model))
        pde = _pde(s, model)
        assert pde == pytest.approx(analytic, rel=3e-3, abs=2e-5)

    def test_sloped_curve(self, sloped_curve):
        m = _model(sloped_curve)
        strike = _atm_strike(sloped_curve, 2030, 5)
        s = _swaption(2030, 5, strike, is_payer=True)
        analytic = float(g2pp_swaption_price(s, m))
        assert _pde(s, m) == pytest.approx(analytic, rel=3e-3, abs=2e-5)

    def test_convergence_to_analytic(self, flat_curve, model):
        """Refining the mesh drives the error toward the analytic price."""
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike, is_payer=True)
        analytic = float(g2pp_swaption_price(s, model))
        err_coarse = abs(_pde(s, model, COARSE) - analytic)
        err_fine = abs(_pde(s, model, FINE) - analytic)
        assert err_fine < err_coarse
        assert err_fine < 1e-3 * analytic


# ── Fixed-rate bond: exact-fit calibration of the scheme ─────────────


class TestFixedRateBondExactFit:
    def _analytic_bond(self, curve, fixed, coupon):
        dfs = curve(fixed)
        return float(coupon * dfs.sum() + dfs[-1])

    def test_reprices_curve_flat(self, flat_curve, model):
        fixed = _fixed_dates(2026, 8)
        bond = FixedRateBond(
            payment_dates=fixed,
            settlement_date=model.initial_curve.reference_date,
            coupon_rate=jnp.array(0.03),
            face_value=jnp.array(1.0),
            frequency=1,
            day_count=_ACT365,
        )
        analytic = self._analytic_bond(flat_curve, fixed, 0.03)
        assert _pde(bond, model) == pytest.approx(analytic, abs=1e-4)

    def test_reprices_curve_sloped(self, sloped_curve):
        m = _model(sloped_curve)
        fixed = _fixed_dates(2026, 8)
        bond = FixedRateBond(
            payment_dates=fixed,
            settlement_date=m.initial_curve.reference_date,
            coupon_rate=jnp.array(0.035),
            face_value=jnp.array(1.0),
            frequency=1,
            day_count=_ACT365,
        )
        analytic = self._analytic_bond(sloped_curve, fixed, 0.035)
        assert _pde(bond, m) == pytest.approx(analytic, abs=1e-4)

    def test_independent_of_model_params(self, flat_curve):
        """No optionality => the price cannot depend on a, b, sigma, eta, rho."""
        fixed = _fixed_dates(2026, 8)
        bond = FixedRateBond(
            payment_dates=fixed,
            settlement_date=jnp.int32(flat_curve.reference_date),
            coupon_rate=jnp.array(0.03),
            face_value=jnp.array(1.0),
            frequency=1,
            day_count=_ACT365,
        )
        m1 = G2PPModel(
            mean_reversion_x=jnp.array(0.30),
            mean_reversion_y=jnp.array(0.05),
            volatility_x=jnp.array(0.015),
            volatility_y=jnp.array(0.004),
            correlation=jnp.array(0.40),
            initial_curve=flat_curve,
        )
        m2 = _model(flat_curve)
        assert _pde(bond, m1) == pytest.approx(_pde(bond, m2), abs=2e-4)


# ── Bermudan swaption ────────────────────────────────────────────────


class TestBermudan:
    def test_single_exercise_reproduces_european(self, flat_curve, model):
        strike = _atm_strike(flat_curve, 2031, 5)
        euro = _swaption(2031, 5, strike, is_payer=True)
        berm = BermudanSwaption(
            exercise_dates=jnp.array(
                [int(ymd_to_ordinal(2031, 1, 1))], dtype=jnp.int32
            ),
            fixed_dates=_fixed_dates(2031, 5),
            strike=jnp.asarray(strike),
            notional=jnp.array(1.0),
            is_payer=True,
            day_count=_ACT365,
        )
        assert _pde(berm, model) == pytest.approx(_pde(euro, model), rel=3e-3)

    def _coterminal(self, curve, model, exercise_years, final_year, strike):
        """The Bermudan and its co-terminal European constituents."""
        exercise_dates = jnp.array(
            [int(ymd_to_ordinal(y, 1, 1)) for y in exercise_years],
            dtype=jnp.int32,
        )
        fixed_dates = jnp.array(
            [
                int(ymd_to_ordinal(y, 1, 1))
                for y in range(exercise_years[0] + 1, final_year + 1)
            ],
            dtype=jnp.int32,
        )
        berm = BermudanSwaption(
            exercise_dates=exercise_dates,
            fixed_dates=fixed_dates,
            strike=jnp.asarray(strike),
            notional=jnp.array(1.0),
            is_payer=True,
            day_count=_ACT365,
        )
        europeans = []
        for i, ey in enumerate(exercise_years):
            e = jnp.int32(int(ymd_to_ordinal(ey, 1, 1)))
            tail = fixed_dates[i:]
            s = Swaption(
                expiry_date=e,
                fixed_dates=tail,
                strike=jnp.asarray(strike),
                notional=jnp.array(1.0),
                is_payer=True,
                day_count=_ACT365,
            )
            europeans.append(_pde(s, model))
        return _pde(berm, model), europeans

    def test_within_coterminal_bounds(self, flat_curve, model):
        strike = _atm_strike(flat_curve, 2031, 5)
        berm, europeans = self._coterminal(
            flat_curve, model, (2029, 2030, 2031, 2032, 2033), 2034, strike
        )
        # The Bermudan dominates each European (early-exercise premium >= 0)
        # and cannot exceed the sum of its co-terminal constituents.
        assert berm >= max(europeans) - 1e-9
        assert berm <= sum(europeans) + 1e-9

    def test_monotone_in_exercise_count(self, flat_curve, model):
        """More exercise opportunities cannot reduce the Bermudan's value."""
        strike = _atm_strike(flat_curve, 2031, 5)
        few, _ = self._coterminal(flat_curve, model, (2031, 2032), 2034, strike)
        many, _ = self._coterminal(
            flat_curve, model, (2029, 2030, 2031, 2032, 2033), 2034, strike
        )
        assert many >= few - 1e-9


# ── Decorrelation: the mixed operator is wired correctly ─────────────


class TestDecorrelation:
    def test_rho_sensitivity_tracks_analytic(self, flat_curve):
        """The PDE and the analytic price respond to rho with the same sign."""
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike, is_payer=True)

        m_lo = _model(flat_curve, rho=-0.70)
        m_hi = _model(flat_curve, rho=0.50)

        pde_lo, pde_hi = _pde(s, m_lo), _pde(s, m_hi)
        an_lo = float(g2pp_swaption_price(s, m_lo))
        an_hi = float(g2pp_swaption_price(s, m_hi))

        # Same ordering as the analytic reference...
        assert (pde_hi - pde_lo) * (an_hi - an_lo) > 0.0
        # ...and each matches its analytic counterpart tightly.
        assert pde_lo == pytest.approx(an_lo, rel=3e-3, abs=2e-5)
        assert pde_hi == pytest.approx(an_hi, rel=3e-3, abs=2e-5)


# ── JIT ──────────────────────────────────────────────────────────────


class TestJIT:
    def test_filter_jit_smoke(self, flat_curve, model):
        strike = _atm_strike(flat_curve, 2031, 5)
        s = _swaption(2031, 5, strike, is_payer=True)

        @eqx.filter_jit
        def price(m):
            return pde_price_dispatch(s, m, COARSE).price

        eager = float(pde_price_dispatch(s, model, COARSE).price)
        assert float(price(model)) == pytest.approx(eager, rel=1e-6)
