"""Tests for Bermudan swaption pricing via Longstaff-Schwartz MC.

Validates the backward induction algorithm against known properties:
- Bermudan >= European (more exercise dates = more value)
- Single exercise date reduces to European swaption
- Price is finite and positive for ITM swaptions
- JIT compatibility
- Payer and receiver both produce positive prices

Companion example: examples/comparisons/08_bermudan_swaption.py (planned)
"""

import jax
import jax.numpy as jnp
import pytest

from valax.models.lmm import (
    PiecewiseConstantVol,
    ExponentialCorrelation,
    build_lmm_model,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption, BermudanSwaption
from valax.pricing.mc.lmm_paths import generate_lmm_paths
from valax.pricing.mc.rate_payoffs import swaption_mc_payoff
from valax.pricing.mc.bermudan import bermudan_swaption_lsm, LSMConfig, _tail_swap_value


# ── Helpers ───────────────────────────────────────────────────────────

def _flat_curve(ref_date, rate, pillar_years):
    """Build a flat continuously-compounded discount curve."""
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + y, 1, 1)) for y in pillar_years],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-rate * times)
    return DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs, reference_date=ref_date,
    )


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    return _flat_curve(ref_date, 0.05, list(range(13)))


@pytest.fixture
def model(flat_curve):
    """LMM with 4 annual forward rates (5y tenor structure)."""
    tenor_dates = jnp.array([
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2027, 1, 1)),
        int(ymd_to_ordinal(2028, 1, 1)),
        int(ymd_to_ordinal(2029, 1, 1)),
        int(ymd_to_ordinal(2030, 1, 1)),
    ], dtype=jnp.int32)
    N = 4
    vol = jnp.array(0.20)
    vol_structure = PiecewiseConstantVol(vols=vol * jnp.ones((N, N)))
    corr_structure = ExponentialCorrelation(beta=jnp.array(0.05))
    return build_lmm_model(flat_curve, tenor_dates, vol_structure, corr_structure)


@pytest.fixture
def paths(model):
    """Generate LMM paths for tests (50k paths, cached per session)."""
    return generate_lmm_paths(model, 10, 50_000, jax.random.PRNGKey(42))


# ── forwards_at_tenors shape validation ──────────────────────────────

class TestForwardsAtTenors:
    def test_shape(self, paths):
        """forwards_at_tenors should be (n_paths, N, N)."""
        assert paths.forwards_at_tenors.shape == (50_000, 4, 4)

    def test_dead_forwards_zero(self, paths):
        """Dead forwards (j < i) should be zero."""
        # Row i: forwards j=0..i-1 should be zero
        for i in range(1, 4):
            dead = paths.forwards_at_tenors[:, i, :i]
            assert jnp.allclose(dead, 0.0), f"Dead forwards at row {i} not zero"

    def test_alive_forwards_positive(self, paths):
        """Alive forwards (j >= i) should be positive."""
        for i in range(4):
            alive = paths.forwards_at_tenors[:, i, i:]
            assert jnp.all(alive > 0.0), f"Non-positive alive forward at row {i}"

    def test_diagonal_matches_fixing(self, paths):
        """Diagonal of forwards_at_tenors should match forwards_at_fixing."""
        diag = jnp.array([paths.forwards_at_tenors[:, i, i] for i in range(4)]).T
        assert jnp.allclose(diag, paths.forwards_at_fixing, atol=1e-10)


# ── Bermudan pricing smoke tests ─────────────────────────────────────

class TestBermudanSmoke:
    def test_price_finite_and_positive(self, model, paths):
        """Basic smoke: price should be finite and positive for ATM swaption."""
        swaption = BermudanSwaption(
            exercise_dates=model.tenor_dates[:-1],
            fixed_dates=model.tenor_dates[1:],
            strike=jnp.array(0.05),  # near ATM for flat 5% curve
            notional=jnp.array(1_000_000.0),
            is_payer=True,
        )
        exercise_indices = jnp.arange(4, dtype=jnp.int32)
        price = bermudan_swaption_lsm(
            paths, swaption, exercise_indices, model.accrual_fractions,
        )
        assert jnp.isfinite(price), f"Price is not finite: {price}"
        assert float(price) > 0.0, f"Price should be positive: {price}"

    def test_receiver_price_positive(self, model, paths):
        """Receiver Bermudan should also have positive price."""
        swaption = BermudanSwaption(
            exercise_dates=model.tenor_dates[:-1],
            fixed_dates=model.tenor_dates[1:],
            strike=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            is_payer=False,
        )
        exercise_indices = jnp.arange(4, dtype=jnp.int32)
        price = bermudan_swaption_lsm(
            paths, swaption, exercise_indices, model.accrual_fractions,
        )
        assert jnp.isfinite(price)
        assert float(price) > 0.0


# ── Bermudan >= European ─────────────────────────────────────────────

class TestBermudanVsEuropean:
    def test_bermudan_geq_european(self, model, paths):
        """Bermudan price >= European price (more exercise = more value).

        Uses the same paths and same payoff computation (_tail_swap_value)
        to ensure an apples-to-apples comparison.
        """
        K = jnp.array(0.05)
        N = model.initial_forwards.shape[0]
        indices = jnp.arange(N, dtype=jnp.int32)
        taus = model.accrual_fractions

        # European: exercise only at T_0, using _tail_swap_value for consistency
        ex_val_0, _ = _tail_swap_value(
            paths.forwards_at_tenors[:, 0, :], taus,
            jnp.array(0, dtype=jnp.int32), K,
            jnp.array(1_000_000.0), True,
        )
        df_0 = paths.discount_factors[:, 0]
        euro_price = float(jnp.mean(ex_val_0 * df_0))

        # Bermudan: exercise at all tenor dates
        berm_swaption = BermudanSwaption(
            exercise_dates=model.tenor_dates[:-1],
            fixed_dates=model.tenor_dates[1:],
            strike=K,
            notional=jnp.array(1_000_000.0),
            is_payer=True,
        )
        berm_price = float(bermudan_swaption_lsm(
            paths, berm_swaption, indices, taus,
        ))

        # Allow small numerical tolerance
        assert berm_price >= euro_price - 50.0, (
            f"Bermudan ({berm_price:.2f}) should be >= European ({euro_price:.2f})"
        )


# ── Single exercise reduces to European ──────────────────────────────

class TestSingleExercise:
    def test_single_exercise_close_to_european(self, model, paths):
        """Bermudan with 1 exercise date should match European MC price.

        Both use _tail_swap_value (forward curve at exercise date) to ensure
        identical payoff computation.
        """
        K = jnp.array(0.05)
        taus = model.accrual_fractions

        # European: _tail_swap_value at T_0, discounted to time 0
        ex_val_0, _ = _tail_swap_value(
            paths.forwards_at_tenors[:, 0, :], taus,
            jnp.array(0, dtype=jnp.int32), K,
            jnp.array(1_000_000.0), True,
        )
        df_0 = paths.discount_factors[:, 0]
        euro_price = float(jnp.mean(ex_val_0 * df_0))

        # Bermudan with single exercise date (= European)
        berm = BermudanSwaption(
            exercise_dates=model.tenor_dates[:1],
            fixed_dates=model.tenor_dates[1:],
            strike=K,
            notional=jnp.array(1_000_000.0),
            is_payer=True,
        )
        berm_price = float(bermudan_swaption_lsm(
            paths, berm, jnp.array([0], dtype=jnp.int32), taus,
        ))

        # Should be very close — same exercise date, same paths, same payoff
        rel_err = abs(berm_price - euro_price) / max(abs(euro_price), 1.0)
        assert rel_err < 0.01, (
            f"Single-exercise Bermudan ({berm_price:.2f}) should ≈ "
            f"European ({euro_price:.2f}), rel_err={rel_err:.2%}"
        )


# ── JIT compatibility ─────────────────────────────────────────────────

class TestJIT:
    def test_bermudan_jit(self, model, paths):
        """bermudan_swaption_lsm should be JIT-compilable."""
        swaption = BermudanSwaption(
            exercise_dates=model.tenor_dates[:-1],
            fixed_dates=model.tenor_dates[1:],
            strike=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            is_payer=True,
        )
        indices = jnp.arange(4, dtype=jnp.int32)
        taus = model.accrual_fractions

        price_eager = bermudan_swaption_lsm(paths, swaption, indices, taus)
        price_jit = jax.jit(
            lambda r: bermudan_swaption_lsm(r, swaption, indices, taus)
        )(paths)

        assert jnp.allclose(price_eager, price_jit, atol=1e-4)


# ── Convergence ───────────────────────────────────────────────────────

class TestConvergence:
    def test_price_stabilizes_with_paths(self, model):
        """Price should stabilize as n_paths increases."""
        swaption = BermudanSwaption(
            exercise_dates=model.tenor_dates[:-1],
            fixed_dates=model.tenor_dates[1:],
            strike=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            is_payer=True,
        )
        indices = jnp.arange(4, dtype=jnp.int32)
        taus = model.accrual_fractions

        prices = []
        for n in [5_000, 20_000, 50_000]:
            result = generate_lmm_paths(model, 10, n, jax.random.PRNGKey(42))
            p = float(bermudan_swaption_lsm(result, swaption, indices, taus))
            prices.append(p)

        # The spread between 20k and 50k should be tighter than 5k and 50k
        spread_small = abs(prices[1] - prices[2])
        spread_large = abs(prices[0] - prices[2])
        assert spread_small <= spread_large + 100.0, (
            f"Prices not converging: 5k={prices[0]:.0f}, "
            f"20k={prices[1]:.0f}, 50k={prices[2]:.0f}"
        )
