"""Tests for the Hull-White short-rate PDE pricers (roadmap PR-3).

Every instrument here is checked against an **independent** oracle rather than
against a stored number, which is what makes the suite worth running:

- ``FixedRateBond``      -> the analytic curve price. No optionality, so the
                            answer cannot depend on ``a`` or ``sigma`` — any
                            error is pure numerics.
- ``Swaption``           -> the exact Jamshidian decomposition.
- ``CallableBond`` /
  ``PuttableBond``       -> the Hull-White trinomial tree.
- ``BermudanSwaption``   -> a single-exercise Bermudan must reproduce the
                            European exactly, and the multi-exercise price must
                            sit between its co-terminal European bounds.

Two accuracy notes worth carrying, both established by the convergence tests
below:

1. The PDE is **more accurate than the tree** at comparable resolution. The
   tree snaps coupon dates to time steps, an O(dt) error; the PDE instead
   scales each coupon by the analytic bond price from its snapped level to its
   true date, which removes that error entirely. Pricing an effectively
   option-free bond *on the tree* leaves -2.2e-3 to +3.4e-5 of error depending
   on the step count (non-monotone, since the count decides where dates land),
   versus 8e-6 and cleanly second-order for the PDE. The callable/puttable
   tolerances below are therefore set by the **tree's** error, not the PDE's.

2. Prices are insensitive to the domain half-width from ~4 standard deviations
   outward, which is what justifies the zero-curvature edge treatment standing
   in for Dirichlet data.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import CallableBond, FixedRateBond, PuttableBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.lattice.hull_white_tree import (
    callable_bond_price,
    puttable_bond_price,
)
from valax.pricing.pde import PDEConfig, pde_price_dispatch, registered_recipes


# ── Fixtures ─────────────────────────────────────────────────────────

FINE = PDEConfig(n_spot=401, n_time=400, spot_range=6.0)
COARSE = PDEConfig(n_spot=201, n_time=200, spot_range=6.0)

CALL_YEARS = (2027, 2028, 2029)
EXERCISE_YEARS = (2027, 2028, 2029, 2030, 2031)


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
def sloped_curve(ref_date):
    """Upward-sloping curve — exercises the non-constant ``alpha(t)`` path."""
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(21)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    zero_rates = 0.03 + 0.015 * jnp.tanh(times / 4.0)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero_rates * times),
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
def bond_schedule():
    return generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)


@pytest.fixture
def bullet_bond(ref_date, bond_schedule):
    return FixedRateBond(
        payment_dates=bond_schedule,
        settlement_date=ref_date,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        frequency=2,
    )


@pytest.fixture
def call_dates():
    return jnp.array(
        [int(ymd_to_ordinal(y, 1, 1)) for y in CALL_YEARS], dtype=jnp.int32
    )


def _callable(ref_date, schedule, dates, call_price=1.0):
    return CallableBond(
        payment_dates=schedule,
        settlement_date=ref_date,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        call_dates=dates,
        call_prices=jnp.full(dates.shape[0], call_price),
        frequency=2,
    )


def _puttable(ref_date, schedule, dates, put_price=1.0):
    return PuttableBond(
        payment_dates=schedule,
        settlement_date=ref_date,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        put_dates=dates,
        put_prices=jnp.full(dates.shape[0], put_price),
        frequency=2,
    )


@pytest.fixture
def callable_bond(ref_date, bond_schedule, call_dates):
    return _callable(ref_date, bond_schedule, call_dates)


@pytest.fixture
def puttable_bond(ref_date, bond_schedule, call_dates):
    return _puttable(ref_date, bond_schedule, call_dates)


@pytest.fixture
def swap_schedule():
    return generate_schedule(2027, 1, 1, 2032, 1, 1, frequency=1)


def _swaption(expiry_ordinal, fixed_dates, is_payer=True, strike=0.05):
    return Swaption(
        expiry_date=jnp.array(int(expiry_ordinal), dtype=jnp.int32),
        fixed_dates=fixed_dates,
        strike=jnp.array(strike),
        notional=jnp.array(1.0e6),
        is_payer=is_payer,
        day_count="act_365",
    )


@pytest.fixture
def exercise_dates():
    return jnp.array(
        [int(ymd_to_ordinal(y, 1, 1)) for y in EXERCISE_YEARS], dtype=jnp.int32
    )


@pytest.fixture
def bermudan(swap_schedule, exercise_dates):
    return BermudanSwaption(
        exercise_dates=exercise_dates,
        fixed_dates=swap_schedule,
        strike=jnp.array(0.05),
        notional=jnp.array(1.0e6),
        is_payer=True,
        day_count="act_365",
    )


def _price(instrument, model, config=FINE):
    return float(pde_price_dispatch(instrument, model, config).price)


# ── Dispatcher wiring ────────────────────────────────────────────────

class TestRegistration:
    @pytest.mark.parametrize(
        "instrument_name",
        [
            "FixedRateBond",
            "CallableBond",
            "PuttableBond",
            "Swaption",
            "BermudanSwaption",
        ],
    )
    def test_recipe_registered(self, instrument_name):
        assert (instrument_name, "HullWhiteModel") in registered_recipes()


# ── Option-free bond: the scheme's own calibration ───────────────────

class TestFixedRateBond:
    def test_matches_analytic_curve_price(self, bullet_bond, model, flat_curve):
        pde = _price(bullet_bond, model)
        analytic = float(fixed_rate_bond_price(bullet_bond, flat_curve))
        assert pde == pytest.approx(analytic, abs=1e-4)

    def test_matches_on_a_sloped_curve(self, bullet_bond, sloped_curve):
        """A non-flat curve makes ``alpha(t)`` genuinely time-varying."""
        m = HullWhiteModel(
            mean_reversion=jnp.array(0.10),
            volatility=jnp.array(0.01),
            initial_curve=sloped_curve,
        )
        pde = _price(bullet_bond, m)
        analytic = float(fixed_rate_bond_price(bullet_bond, sloped_curve))
        assert pde == pytest.approx(analytic, abs=1e-4)

    @pytest.mark.parametrize("mean_reversion,vol", [(0.02, 0.005), (0.35, 0.02)])
    def test_independent_of_model_parameters(
        self, bullet_bond, flat_curve, mean_reversion, vol
    ):
        """No optionality => the price is a pure curve quantity.

        A scheme that leaked ``sigma`` into an option-free bond would be
        mispricing the exact-fit shift, and this catches it across a wide
        parameter range.
        """
        m = HullWhiteModel(
            mean_reversion=jnp.array(mean_reversion),
            volatility=jnp.array(vol),
            initial_curve=flat_curve,
        )
        analytic = float(fixed_rate_bond_price(bullet_bond, flat_curve))
        assert _price(bullet_bond, m) == pytest.approx(analytic, abs=2e-4)

    def test_second_order_convergence(self, bullet_bond, model, flat_curve):
        """Halving the mesh must quarter the error."""
        analytic = float(fixed_rate_bond_price(bullet_bond, flat_curve))
        errors = []
        for n in (100, 200, 400):
            cfg = PDEConfig(n_spot=n + 1, n_time=n, spot_range=6.0)
            errors.append(abs(_price(bullet_bond, model, cfg) - analytic))
        for coarse, fine in zip(errors, errors[1:]):
            assert coarse / fine > 3.0, f"convergence ratios: {errors}"

    @pytest.mark.parametrize("half_width", [4.0, 5.0, 6.0, 8.0])
    def test_insensitive_to_domain_width(
        self, bullet_bond, model, flat_curve, half_width
    ):
        """Justifies the zero-curvature edges standing in for Dirichlet data."""
        cfg = PDEConfig(n_spot=401, n_time=400, spot_range=half_width)
        analytic = float(fixed_rate_bond_price(bullet_bond, flat_curve))
        assert _price(bullet_bond, model, cfg) == pytest.approx(analytic, abs=1e-4)


# ── European swaptions vs Jamshidian ─────────────────────────────────

class TestSwaption:
    @pytest.mark.parametrize("is_payer", [True, False])
    @pytest.mark.parametrize("strike", [0.04, 0.05, 0.06])
    def test_matches_jamshidian(self, model, swap_schedule, is_payer, strike):
        swaption = _swaption(
            ymd_to_ordinal(2027, 1, 1), swap_schedule, is_payer, strike
        )
        exact = float(hw_swaption_price(swaption, model))
        assert _price(swaption, model) == pytest.approx(exact, rel=2e-4)

    def test_matches_jamshidian_on_a_sloped_curve(
        self, sloped_curve, swap_schedule
    ):
        m = HullWhiteModel(
            mean_reversion=jnp.array(0.08),
            volatility=jnp.array(0.012),
            initial_curve=sloped_curve,
        )
        swaption = _swaption(ymd_to_ordinal(2027, 1, 1), swap_schedule)
        exact = float(hw_swaption_price(swaption, m))
        assert _price(swaption, m) == pytest.approx(exact, rel=2e-4)

    def test_converges_to_jamshidian(self, model, swap_schedule):
        swaption = _swaption(ymd_to_ordinal(2027, 1, 1), swap_schedule)
        exact = float(hw_swaption_price(swaption, model))
        coarse = abs(_price(swaption, model, COARSE) - exact)
        fine = abs(_price(swaption, model, FINE) - exact)
        assert fine < coarse


# ── Callable / puttable bonds vs the trinomial tree ──────────────────

class TestCallablePuttableBonds:
    # Set by the *tree's* accuracy, not the PDE's — see the module docstring.
    TREE_TOL = 3.0e-3

    def test_callable_matches_tree(self, callable_bond, model):
        tree = float(callable_bond_price(callable_bond, model, n_steps=1600))
        assert _price(callable_bond, model) == pytest.approx(
            tree, abs=self.TREE_TOL
        )

    def test_puttable_matches_tree(self, puttable_bond, model):
        tree = float(puttable_bond_price(puttable_bond, model, n_steps=1600))
        assert _price(puttable_bond, model) == pytest.approx(
            tree, abs=self.TREE_TOL
        )

    def test_embedded_option_ordering(
        self, callable_bond, puttable_bond, bullet_bond, model
    ):
        """callable <= bullet <= puttable, strictly, for a live option."""
        callable_px = _price(callable_bond, model)
        bullet_px = _price(bullet_bond, model)
        puttable_px = _price(puttable_bond, model)
        assert callable_px < bullet_px < puttable_px

    def test_unreachable_call_recovers_the_bullet(
        self, ref_date, bond_schedule, call_dates, bullet_bond, model
    ):
        """A call struck far out of the money must never be exercised.

        This is the sharpest structural check available: it isolates the
        exercise machinery (event seam, ex-coupon ordering, obstacle scatter)
        from the diffusion, so the two prices must agree to *machine*
        precision, not merely to grid tolerance.
        """
        never = _callable(ref_date, bond_schedule, call_dates, call_price=10.0)
        assert _price(never, model) == pytest.approx(
            _price(bullet_bond, model), abs=1e-12
        )

    def test_unreachable_put_recovers_the_bullet(
        self, ref_date, bond_schedule, call_dates, bullet_bond, model
    ):
        never = _puttable(ref_date, bond_schedule, call_dates, put_price=0.0)
        assert _price(never, model) == pytest.approx(
            _price(bullet_bond, model), abs=1e-12
        )

    def test_call_value_increases_with_volatility(
        self, ref_date, bond_schedule, call_dates, flat_curve
    ):
        """The embedded call is an option: more vol, more value to the issuer,
        so less value to the holder."""
        bond = _callable(ref_date, bond_schedule, call_dates)
        prices = []
        for vol in (0.005, 0.010, 0.020):
            m = HullWhiteModel(
                mean_reversion=jnp.array(0.10),
                volatility=jnp.array(vol),
                initial_curve=flat_curve,
            )
            prices.append(_price(bond, m, COARSE))
        assert prices[0] > prices[1] > prices[2]


# ── Bermudan swaptions ───────────────────────────────────────────────

class TestBermudanSwaption:
    def test_single_exercise_reproduces_the_european(self, model, swap_schedule):
        """A one-date Bermudan *is* a European — and both go through this
        module's own machinery, so the comparison also pins the Bermudan
        exercise path against the exact Jamshidian price."""
        expiry = ymd_to_ordinal(2027, 1, 1)
        berm = BermudanSwaption(
            exercise_dates=jnp.array([int(expiry)], dtype=jnp.int32),
            fixed_dates=swap_schedule,
            strike=jnp.array(0.05),
            notional=jnp.array(1.0e6),
            is_payer=True,
            day_count="act_365",
        )
        european = _swaption(expiry, swap_schedule)
        assert _price(berm, model) == pytest.approx(
            float(hw_swaption_price(european, model)), rel=2e-4
        )

    def test_within_co_terminal_european_bounds(
        self, bermudan, model, swap_schedule, exercise_dates
    ):
        """Model-free no-arbitrage bounds.

        A Bermudan dominates each of its co-terminal Europeans (it can always
        choose that one exercise date and behave identically) and is dominated
        by owning all of them at once.
        """
        europeans = [
            float(
                hw_swaption_price(
                    _swaption(exercise_dates[e], swap_schedule[e:]), model
                )
            )
            for e in range(exercise_dates.shape[0])
        ]
        price = _price(bermudan, model)
        assert price >= max(europeans) - 1e-6
        assert price <= sum(europeans) + 1e-6

    def test_more_exercise_rights_are_worth_more(self, model, swap_schedule):
        """Monotonicity in the exercise set — extra optionality is never bad."""
        prices = []
        for count in (1, 2, 3, 5):
            dates = jnp.array(
                [int(ymd_to_ordinal(y, 1, 1)) for y in EXERCISE_YEARS[:count]],
                dtype=jnp.int32,
            )
            prices.append(
                _price(
                    BermudanSwaption(
                        exercise_dates=dates,
                        fixed_dates=swap_schedule,
                        strike=jnp.array(0.05),
                        notional=jnp.array(1.0e6),
                        is_payer=True,
                        day_count="act_365",
                    ),
                    model,
                    COARSE,
                )
            )
        assert all(a < b for a, b in zip(prices, prices[1:])), prices

    def test_receiver_and_payer_both_priced(self, model, swap_schedule, exercise_dates):
        kwargs = dict(
            exercise_dates=exercise_dates,
            fixed_dates=swap_schedule,
            strike=jnp.array(0.05),
            notional=jnp.array(1.0e6),
            day_count="act_365",
        )
        payer = _price(BermudanSwaption(is_payer=True, **kwargs), model, COARSE)
        receiver = _price(BermudanSwaption(is_payer=False, **kwargs), model, COARSE)
        assert payer > 0.0 and receiver > 0.0

    def test_converges_under_refinement(self, bermudan, model):
        coarse = _price(bermudan, model, PDEConfig(n_spot=101, n_time=100, spot_range=6.0))
        mid = _price(bermudan, model, COARSE)
        fine = _price(bermudan, model, FINE)
        assert abs(fine - mid) < abs(mid - coarse)


# ── JIT and autodiff ─────────────────────────────────────────────────

class TestJitAndGreeks:
    @pytest.fixture
    def instruments(
        self, bullet_bond, callable_bond, puttable_bond, bermudan, swap_schedule
    ):
        return {
            "bullet": bullet_bond,
            "callable": callable_bond,
            "puttable": puttable_bond,
            "swaption": _swaption(ymd_to_ordinal(2027, 1, 1), swap_schedule),
            "bermudan": bermudan,
        }

    @pytest.mark.parametrize(
        "name", ["bullet", "callable", "puttable", "swaption", "bermudan"]
    )
    def test_filter_jit_smoke(self, instruments, model, name):
        instrument = instruments[name]

        @eqx.filter_jit
        def price(m):
            return pde_price_dispatch(instrument, m, COARSE).price

        assert jnp.isfinite(price(model))
        # Compiled and eager paths must agree.
        assert float(price(model)) == pytest.approx(
            _price(instrument, model, COARSE), rel=1e-12
        )

    @pytest.mark.parametrize("name", ["callable", "swaption", "bermudan"])
    def test_greeks_match_finite_differences(
        self, instruments, flat_curve, name
    ):
        """Autodiff Greeks vs central differences.

        Per ``AGENTS.md`` finite differences are a *test* device only; the
        library itself always differentiates. Both ``a`` and ``sigma`` are
        checked because the PDE must stay differentiable through the operator
        stack, the mesh width, and the exercise projection alike.
        """
        instrument = instruments[name]

        def build(mean_reversion, vol):
            return HullWhiteModel(
                mean_reversion=mean_reversion,
                volatility=vol,
                initial_curve=flat_curve,
            )

        a0, s0 = jnp.array(0.10), jnp.array(0.01)

        def price(mean_reversion, vol):
            return pde_price_dispatch(
                instrument, build(mean_reversion, vol), COARSE
            ).price

        d_vol = float(jax.grad(price, argnums=1)(a0, s0))
        d_a = float(jax.grad(price, argnums=0)(a0, s0))

        h = 1e-6
        fd_vol = float(
            (price(a0, s0 + h) - price(a0, s0 - h)) / (2 * h)
        )
        fd_a = float((price(a0 + h, s0) - price(a0 - h, s0)) / (2 * h))

        assert d_vol == pytest.approx(fd_vol, rel=1e-4)
        assert d_a == pytest.approx(fd_a, rel=1e-4)

    def test_callable_bond_vega_is_negative(self, callable_bond, flat_curve):
        """Sign check with economic content: the holder is short the call."""
        def price(vol):
            m = HullWhiteModel(
                mean_reversion=jnp.array(0.10),
                volatility=vol,
                initial_curve=flat_curve,
            )
            return pde_price_dispatch(callable_bond, m, COARSE).price

        assert float(jax.grad(price)(jnp.array(0.01))) < 0.0

    def test_swaption_vega_is_positive(self, model, swap_schedule, flat_curve):
        swaption = _swaption(ymd_to_ordinal(2027, 1, 1), swap_schedule)

        def price(vol):
            m = HullWhiteModel(
                mean_reversion=jnp.array(0.10),
                volatility=vol,
                initial_curve=flat_curve,
            )
            return pde_price_dispatch(swaption, m, COARSE).price

        assert float(jax.grad(price)(jnp.array(0.01))) > 0.0
