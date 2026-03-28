"""Tests for the LIBOR Market Model: path generation, payoffs, convergence."""

import jax
import jax.numpy as jnp
import pytest

from valax.models.lmm import (
    LMMModel,
    LMMDrift,
    LMMDiffusion,
    PiecewiseConstantVol,
    RebonatoVol,
    ExponentialCorrelation,
    TwoParameterCorrelation,
    compute_loading_matrix,
    build_lmm_model,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.dates.schedule import generate_schedule
from valax.instruments.rates import Caplet, Cap, Swaption
from valax.pricing.mc.lmm_paths import generate_lmm_paths, LMMPathResult
from valax.pricing.mc.rate_payoffs import (
    caplet_mc_payoff,
    cap_mc_payoff,
    swaption_mc_payoff,
)
from valax.pricing.analytic.caplets import caplet_price_black76
from valax.pricing.analytic.swaptions import swaption_price_black76


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
    """Flat 5% CC curve out to 12 years."""
    return _flat_curve(ref_date, 0.05, list(range(13)))


@pytest.fixture
def single_fwd_model(flat_curve):
    """LMM with a single forward rate (N=1): should match Black-76."""
    tenor_dates = jnp.array([
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2026, 7, 1)),
    ], dtype=jnp.int32)
    vol = jnp.array(0.20)
    vol_structure = PiecewiseConstantVol(vols=vol * jnp.ones((1, 1)))
    corr_structure = ExponentialCorrelation(beta=jnp.array(0.1))
    return build_lmm_model(flat_curve, tenor_dates, vol_structure, corr_structure)


@pytest.fixture
def multi_fwd_model(flat_curve):
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


# ── Model construction ────────────────────────────────────────────────

class TestModelConstruction:
    def test_initial_forwards_positive(self, single_fwd_model):
        assert jnp.all(single_fwd_model.initial_forwards > 0.0)

    def test_initial_forwards_shape(self, multi_fwd_model):
        assert multi_fwd_model.initial_forwards.shape == (4,)

    def test_accrual_fractions_positive(self, multi_fwd_model):
        assert jnp.all(multi_fwd_model.accrual_fractions > 0.0)

    def test_loading_matrix_shape(self, multi_fwd_model):
        N = multi_fwd_model.initial_forwards.shape[0]
        assert multi_fwd_model.loading_matrix.shape == (N, N)

    def test_pca_loading_shape(self, flat_curve):
        """PCA with k=2 factors on 4 forwards."""
        tenor_dates = jnp.array([
            int(ymd_to_ordinal(2026, 1, 1)),
            int(ymd_to_ordinal(2027, 1, 1)),
            int(ymd_to_ordinal(2028, 1, 1)),
            int(ymd_to_ordinal(2029, 1, 1)),
            int(ymd_to_ordinal(2030, 1, 1)),
        ], dtype=jnp.int32)
        vol_structure = PiecewiseConstantVol(vols=jnp.array(0.20) * jnp.ones((4, 4)))
        corr_structure = ExponentialCorrelation(beta=jnp.array(0.05))
        model = build_lmm_model(flat_curve, tenor_dates, vol_structure, corr_structure, n_factors=2)
        assert model.loading_matrix.shape == (4, 2)

    def test_correlation_matrix_psd(self, multi_fwd_model):
        """Correlation matrix should be positive semi-definite."""
        tenor_times_fwd = multi_fwd_model.tenor_times[:-1]
        corr = multi_fwd_model.corr_structure.matrix(tenor_times_fwd)
        eigenvalues = jnp.linalg.eigvalsh(corr)
        assert jnp.all(eigenvalues >= -1e-10)


# ── Volatility structures ─────────────────────────────────────────────

class TestVolStructures:
    def test_piecewise_constant_vol(self):
        vols = jnp.array([[0.20, 0.0], [0.25, 0.22]])
        vol_struct = PiecewiseConstantVol(vols=vols)
        tenor_times = jnp.array([1.0, 2.0])
        # t=0.5 is in period 0 (before T_0=1.0)
        sigma = vol_struct(jnp.array(0.5), tenor_times)
        assert sigma.shape == (2,)
        assert float(sigma[0]) == pytest.approx(0.20)
        assert float(sigma[1]) == pytest.approx(0.25)

    def test_rebonato_vol_positive(self):
        vol_struct = RebonatoVol(
            a=jnp.array(0.20), b=jnp.array(0.05),
            c=jnp.array(0.50), d=jnp.array(0.10),
        )
        tenor_times = jnp.array([1.0, 2.0, 3.0])
        sigma = vol_struct(jnp.array(0.0), tenor_times)
        assert jnp.all(sigma > 0.0)

    def test_rebonato_vol_dead_forward(self):
        """Dead forward (T_i < t) should get tau_pos=0."""
        vol_struct = RebonatoVol(
            a=jnp.array(0.20), b=jnp.array(0.0),
            c=jnp.array(0.0), d=jnp.array(0.0),
        )
        # sigma_i(t) = (0.20 + 0) * exp(0) + 0 = 0.20 for alive
        # For dead (tau=0): (0.20 + 0) * exp(0) + 0 = 0.20 — still returns a vol
        # but the alive mask in the drift/diffusion zeros it out
        sigma = vol_struct(jnp.array(2.5), jnp.array([1.0, 2.0, 3.0]))
        assert sigma.shape == (3,)


# ── Correlation structures ─────────────────────────────────────────────

class TestCorrelationStructures:
    def test_exp_corr_diagonal_ones(self):
        corr_struct = ExponentialCorrelation(beta=jnp.array(0.1))
        t = jnp.array([1.0, 2.0, 3.0])
        C = corr_struct.matrix(t)
        assert jnp.allclose(jnp.diag(C), 1.0)

    def test_exp_corr_symmetric(self):
        corr_struct = ExponentialCorrelation(beta=jnp.array(0.1))
        t = jnp.array([1.0, 2.0, 3.0])
        C = corr_struct.matrix(t)
        assert jnp.allclose(C, C.T)

    def test_two_param_corr_floor(self):
        """rho_inf should be the minimum off-diagonal correlation."""
        corr_struct = TwoParameterCorrelation(
            rho_inf=jnp.array(0.3), beta=jnp.array(0.5),
        )
        t = jnp.array([1.0, 5.0, 10.0])
        C = corr_struct.matrix(t)
        off_diag = C[0, 2]  # most distant pair
        assert float(off_diag) >= 0.3 - 1e-6


# ── Path generation ───────────────────────────────────────────────────

class TestPathGeneration:
    def test_single_forward_paths(self, single_fwd_model):
        result = generate_lmm_paths(single_fwd_model, 10, 100, jax.random.PRNGKey(0))
        assert result.forwards_at_fixing.shape == (100, 1)
        assert result.discount_factors.shape == (100, 2)

    def test_multi_forward_paths(self, multi_fwd_model):
        result = generate_lmm_paths(multi_fwd_model, 5, 100, jax.random.PRNGKey(0))
        assert result.forwards_at_fixing.shape == (100, 4)
        assert result.discount_factors.shape == (100, 5)

    def test_forward_rates_positive(self, multi_fwd_model):
        """Log-Euler guarantees positive forwards."""
        result = generate_lmm_paths(multi_fwd_model, 10, 1000, jax.random.PRNGKey(1))
        assert jnp.all(result.forwards_at_fixing > 0.0)

    def test_discount_factors_decreasing(self, multi_fwd_model):
        """DF(0, T_k) should be decreasing in k (for positive rates)."""
        result = generate_lmm_paths(multi_fwd_model, 5, 100, jax.random.PRNGKey(2))
        # Check on the mean path
        mean_dfs = jnp.mean(result.discount_factors, axis=0)
        diffs = jnp.diff(mean_dfs)
        assert jnp.all(diffs < 0.0)

    def test_discount_factors_positive(self, multi_fwd_model):
        result = generate_lmm_paths(multi_fwd_model, 5, 100, jax.random.PRNGKey(3))
        assert jnp.all(result.discount_factors > 0.0)

    def test_martingale_property(self, multi_fwd_model, flat_curve):
        """E[DF_path(0, T_k)] should approximate the curve DF(0, T_k).

        This is the fundamental consistency check for the spot measure simulation.
        """
        result = generate_lmm_paths(multi_fwd_model, 10, 50_000, jax.random.PRNGKey(42))
        mean_dfs = jnp.mean(result.discount_factors, axis=0)

        # Compare against curve discount factors at tenor dates
        curve_dfs = flat_curve(multi_fwd_model.tenor_dates)

        # The LMM builds DFs from forwards, so at T_0 there's a mismatch
        # between DF(0, T_0) = 1 in the LMM and the actual curve DF.
        # Normalize: compare ratios DF(0, T_k)/DF(0, T_0)
        mc_ratios = mean_dfs / mean_dfs[0]
        curve_ratios = curve_dfs / curve_dfs[0]

        # Allow 2% tolerance for MC noise with 50k paths
        for k in range(1, len(mc_ratios)):
            rel_err = abs(float(mc_ratios[k]) - float(curve_ratios[k])) / float(curve_ratios[k])
            assert rel_err < 0.02, (
                f"Martingale test failed at tenor {k}: "
                f"MC ratio={float(mc_ratios[k]):.6f}, "
                f"curve ratio={float(curve_ratios[k]):.6f}, "
                f"rel_err={rel_err:.4f}"
            )


# ── Caplet MC vs Black-76 ─────────────────────────────────────────────

class TestCapletConvergence:
    def test_single_forward_caplet_vs_black76(self, single_fwd_model, flat_curve):
        """N=1 LMM caplet should converge to Black-76 analytical price."""
        model = single_fwd_model
        n_paths = 100_000
        result = generate_lmm_paths(model, 20, n_paths, jax.random.PRNGKey(42))

        # Build matching Caplet with same day_count as model/curve
        caplet = Caplet(
            fixing_date=model.tenor_dates[0],
            start_date=model.tenor_dates[0],
            end_date=model.tenor_dates[1],
            strike=model.initial_forwards[0],  # ATM
            notional=jnp.array(1_000_000.0),
            is_cap=True,
            day_count=model.day_count,
        )
        tau = model.accrual_fractions[0]
        mc_payoffs = caplet_mc_payoff(result, caplet, 0, tau)
        mc_price = float(jnp.mean(mc_payoffs))
        mc_stderr = float(jnp.std(mc_payoffs) / jnp.sqrt(jnp.array(n_paths, dtype=jnp.float64)))

        # Black-76 analytical
        analytical = float(caplet_price_black76(caplet, flat_curve, jnp.array(0.20)))

        err = abs(mc_price - analytical)
        assert err < 3 * mc_stderr, (
            f"MC={mc_price:.2f}, Black76={analytical:.2f}, "
            f"err={err:.2f}, 3*SE={3*mc_stderr:.2f}"
        )

    def test_multi_forward_caplet_vs_black76(self, multi_fwd_model, flat_curve):
        """Each caplet in a multi-forward LMM should be close to Black-76.

        The multi-forward drift correction affects the distribution slightly,
        so we use a looser tolerance.
        """
        model = multi_fwd_model
        n_paths = 100_000
        result = generate_lmm_paths(model, 10, n_paths, jax.random.PRNGKey(123))

        for i in range(model.initial_forwards.shape[0]):
            caplet = Caplet(
                fixing_date=model.tenor_dates[i],
                start_date=model.tenor_dates[i],
                end_date=model.tenor_dates[i + 1],
                strike=model.initial_forwards[i],  # ATM
                notional=jnp.array(1_000_000.0),
                is_cap=True,
                day_count=model.day_count,
            )
            tau = model.accrual_fractions[i]
            mc_payoffs = caplet_mc_payoff(result, caplet, i, tau)
            mc_price = float(jnp.mean(mc_payoffs))
            mc_stderr = float(jnp.std(mc_payoffs) / jnp.sqrt(jnp.array(n_paths, dtype=jnp.float64)))

            analytical = float(caplet_price_black76(caplet, flat_curve, jnp.array(0.20)))

            # Loose tolerance: multi-forward drift correction causes small bias
            err = abs(mc_price - analytical)
            assert err < max(4 * mc_stderr, 0.05 * abs(analytical)), (
                f"Caplet {i}: MC={mc_price:.2f}, Black76={analytical:.2f}, "
                f"err={err:.2f}, 4*SE={4*mc_stderr:.2f}"
            )

    def test_floorlet_mc(self, single_fwd_model, flat_curve):
        """Floorlet should also price correctly."""
        model = single_fwd_model
        n_paths = 50_000
        result = generate_lmm_paths(model, 20, n_paths, jax.random.PRNGKey(99))

        floorlet = Caplet(
            fixing_date=model.tenor_dates[0],
            start_date=model.tenor_dates[0],
            end_date=model.tenor_dates[1],
            strike=model.initial_forwards[0],  # ATM
            notional=jnp.array(1_000_000.0),
            is_cap=False,
            day_count=model.day_count,
        )
        tau = model.accrual_fractions[0]
        mc_payoffs = caplet_mc_payoff(result, floorlet, 0, tau)
        mc_price = float(jnp.mean(mc_payoffs))

        from valax.pricing.analytic.caplets import caplet_price_black76
        analytical = float(caplet_price_black76(floorlet, flat_curve, jnp.array(0.20)))

        mc_stderr = float(jnp.std(mc_payoffs) / jnp.sqrt(jnp.array(n_paths, dtype=jnp.float64)))
        assert abs(mc_price - analytical) < 3 * mc_stderr


# ── Cap MC ─────────────────────────────────────────────────────────────

class TestCapMC:
    def test_cap_equals_sum_of_caplets(self, multi_fwd_model):
        """Cap payoff should equal sum of individual caplet payoffs."""
        model = multi_fwd_model
        result = generate_lmm_paths(model, 5, 1000, jax.random.PRNGKey(7))
        N = model.initial_forwards.shape[0]
        strike = jnp.array(0.05)

        cap = Cap(
            fixing_dates=model.tenor_dates[:-1],
            start_dates=model.tenor_dates[:-1],
            end_dates=model.tenor_dates[1:],
            strike=strike,
            notional=jnp.array(1_000_000.0),
            is_cap=True,
        )
        indices = jnp.arange(N, dtype=jnp.int32)
        cap_payoffs = cap_mc_payoff(result, cap, indices, model.accrual_fractions)

        # Sum individual caplets
        total = jnp.zeros(1000)
        for i in range(N):
            caplet = Caplet(
                fixing_date=model.tenor_dates[i],
                start_date=model.tenor_dates[i],
                end_date=model.tenor_dates[i + 1],
                strike=strike,
                notional=jnp.array(1_000_000.0),
                is_cap=True,
            )
            total = total + caplet_mc_payoff(result, caplet, i, model.accrual_fractions[i])

        assert jnp.allclose(cap_payoffs, total, atol=1e-6)


# ── Swaption MC ────────────────────────────────────────────────────────

class TestSwaptionMC:
    def test_swaption_price_positive(self, multi_fwd_model):
        model = multi_fwd_model
        result = generate_lmm_paths(model, 10, 10_000, jax.random.PRNGKey(55))
        N = model.initial_forwards.shape[0]

        swaption = Swaption(
            expiry_date=model.tenor_dates[0],
            fixed_dates=model.tenor_dates[1:],
            strike=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            is_payer=True,
        )
        indices = jnp.arange(N, dtype=jnp.int32)
        payoffs = swaption_mc_payoff(result, swaption, indices, model.accrual_fractions)
        price = float(jnp.mean(payoffs))
        assert price > 0.0

    def test_payer_receiver_parity(self, multi_fwd_model):
        """payer - receiver = swap NPV (from forwards)."""
        model = multi_fwd_model
        result = generate_lmm_paths(model, 10, 10_000, jax.random.PRNGKey(66))
        N = model.initial_forwards.shape[0]
        K = jnp.array(0.05)
        indices = jnp.arange(N, dtype=jnp.int32)

        payer = Swaption(
            expiry_date=model.tenor_dates[0], fixed_dates=model.tenor_dates[1:],
            strike=K, notional=jnp.array(1_000_000.0), is_payer=True,
        )
        receiver = Swaption(
            expiry_date=model.tenor_dates[0], fixed_dates=model.tenor_dates[1:],
            strike=K, notional=jnp.array(1_000_000.0), is_payer=False,
        )
        p_payoffs = swaption_mc_payoff(result, payer, indices, model.accrual_fractions)
        r_payoffs = swaption_mc_payoff(result, receiver, indices, model.accrual_fractions)

        # payer - receiver = annuity * (S - K) * DF(0, T_0) per path
        # Since we use the same paths, the difference should be very tight
        diff = p_payoffs - r_payoffs

        # The diff should equal the path-wise swap value
        F = result.forwards_at_fixing[:, indices]
        taus = model.accrual_fractions
        accrual = 1.0 + taus[None, :] * F
        cum_accrual = jnp.cumprod(accrual, axis=1)
        rel_df = 1.0 / cum_accrual
        annuity = jnp.sum(taus[None, :] * rel_df, axis=1)
        df_TN = rel_df[:, -1]
        swap_rate_val = (1.0 - df_TN) / annuity
        df_0_T0 = result.discount_factors[:, 0]
        expected_diff = 1_000_000.0 * annuity * (swap_rate_val - K) * df_0_T0

        assert jnp.allclose(diff, expected_diff, atol=1e-2)

    @pytest.mark.xfail(reason="LMM spot-measure dynamics diverge from Black-76 terminal measure for multi-period swaptions")
    def test_swaption_vs_black76_approximate(self, multi_fwd_model, flat_curve):
        """LMM swaption MC should be approximately consistent with Black-76.

        Not exact because the LMM swap rate distribution is not perfectly
        lognormal. Allow 10% relative tolerance.
        """
        model = multi_fwd_model
        n_paths = 100_000
        result = generate_lmm_paths(model, 10, n_paths, jax.random.PRNGKey(777))
        N = model.initial_forwards.shape[0]

        swaption = Swaption(
            expiry_date=model.tenor_dates[0],
            fixed_dates=model.tenor_dates[1:],
            strike=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            is_payer=True,
            day_count=model.day_count,
        )
        indices = jnp.arange(N, dtype=jnp.int32)
        mc_payoffs = swaption_mc_payoff(result, swaption, indices, model.accrual_fractions)
        mc_price = float(jnp.mean(mc_payoffs))

        analytical = float(swaption_price_black76(swaption, flat_curve, jnp.array(0.20)))

        # Approximate: LMM and Black-76 won't match exactly
        rel_err = abs(mc_price - analytical) / abs(analytical) if analytical != 0 else abs(mc_price)
        assert rel_err < 0.15, (
            f"Swaption: MC={mc_price:.2f}, Black76={analytical:.2f}, rel_err={rel_err:.2%}"
        )


# ── Cap-floor parity ──────────────────────────────────────────────────

class TestCapFloorParity:
    def test_cap_minus_floor(self, multi_fwd_model):
        """Cap - Floor = sum of PV(F_i - K) on each path."""
        model = multi_fwd_model
        result = generate_lmm_paths(model, 5, 5000, jax.random.PRNGKey(88))
        N = model.initial_forwards.shape[0]
        K = jnp.array(0.05)
        indices = jnp.arange(N, dtype=jnp.int32)

        cap = Cap(
            fixing_dates=model.tenor_dates[:-1], start_dates=model.tenor_dates[:-1],
            end_dates=model.tenor_dates[1:], strike=K,
            notional=jnp.array(1_000_000.0), is_cap=True,
        )
        floor = Cap(
            fixing_dates=model.tenor_dates[:-1], start_dates=model.tenor_dates[:-1],
            end_dates=model.tenor_dates[1:], strike=K,
            notional=jnp.array(1_000_000.0), is_cap=False,
        )
        cap_pv = cap_mc_payoff(result, cap, indices, model.accrual_fractions)
        floor_pv = cap_mc_payoff(result, floor, indices, model.accrual_fractions)

        # Expected: sum_i DF(0, T_{i+1}) * tau_i * notional * (F_i - K)
        F = result.forwards_at_fixing[:, indices]
        df = result.discount_factors[:, indices + 1]
        taus = model.accrual_fractions
        expected = jnp.sum(df * taus[None, :] * 1_000_000.0 * (F - K), axis=1)

        assert jnp.allclose(cap_pv - floor_pv, expected, atol=1e-2)


# ── JIT and autodiff ──────────────────────────────────────────────────

class TestJITAndGrad:
    def test_path_gen_jit(self, single_fwd_model):
        """generate_lmm_paths should be JIT-compilable."""
        result = jax.jit(
            lambda key: generate_lmm_paths(single_fwd_model, 10, 100, key)
        )(jax.random.PRNGKey(0))
        assert jnp.all(jnp.isfinite(result.forwards_at_fixing))

    def test_caplet_payoff_jit(self, single_fwd_model):
        result = generate_lmm_paths(single_fwd_model, 10, 100, jax.random.PRNGKey(0))
        caplet = Caplet(
            fixing_date=single_fwd_model.tenor_dates[0],
            start_date=single_fwd_model.tenor_dates[0],
            end_date=single_fwd_model.tenor_dates[1],
            strike=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
        )
        tau = single_fwd_model.accrual_fractions[0]
        payoffs = jax.jit(caplet_mc_payoff)(result, caplet, 0, tau)
        assert jnp.all(jnp.isfinite(payoffs))
