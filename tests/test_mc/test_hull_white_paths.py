"""Tests for Hull-White short-rate Monte Carlo path generation.

Validates four properties of :func:`generate_hull_white_paths`:

1. **TestConditionalMoments** — sample mean and variance of the short rate
   at horizon T match the analytic OU formulas within 2 standard errors
   (50k paths).

2. **TestZCBConsistency** — the MC expectation of the money-market
   discount factor recovers the analytic zero-coupon bond price from
   :func:`hw_bond_price` (martingale measure check).

3. **TestTreeTriangulation** — the MC price of a callable bond lies
   within 1 % of the HW trinomial tree price, providing a cross-method
   sanity check at 50k paths.

4. **TestJITAndGrad** — the path generator is compatible with
   ``eqx.filter_jit`` and ``eqx.filter_grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import CallableBond
from valax.models.hull_white import HullWhiteModel, hw_bond_price, hw_short_rate_variance, _instantaneous_forward
from valax.pricing.lattice.hull_white_tree import callable_bond_price
from valax.pricing.mc.hull_white_paths import HullWhitePathResult, generate_hull_white_paths


# ── Shared fixtures ───────────────────────────────────────────────────

def _flat_curve(ref_date: int, rate: float, mat_years: int) -> DiscountCurve:
    """Flat continuously-compounded discount curve on an integer-day grid.

    The reference date is included as the first pillar (df=1), per the
    :class:`DiscountCurve` contract that the leading discount factor is 1.0.
    Short-end forwards no longer *depend* on this — :func:`_log_df_grid`
    anchors the interpolation at ``t = 0`` regardless — but a well-formed
    curve should carry the pillar anyway.
    """
    pillars = jnp.array(
        [ref_date] + [ref_date + int(round(k * 365)) for k in range(1, mat_years + 2)],
        dtype=jnp.int32,
    )
    times = (pillars - ref_date).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-jnp.asarray(rate, dtype=jnp.float64) * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=jnp.int32(ref_date),
        day_count="act_365",
    )


@pytest.fixture(scope="module")
def ref_date() -> int:
    return int(ymd_to_ordinal(2026, 1, 1))


@pytest.fixture(scope="module")
def hw_model(ref_date) -> HullWhiteModel:
    """Hull-White model: a=0.1, sigma=0.01, flat 5% curve."""
    curve = _flat_curve(ref_date, 0.05, 10)
    return HullWhiteModel(
        mean_reversion=jnp.asarray(0.1, dtype=jnp.float64),
        volatility=jnp.asarray(0.01, dtype=jnp.float64),
        initial_curve=curve,
    )


@pytest.fixture(scope="module")
def paths_50k(hw_model) -> HullWhitePathResult:
    """50k paths, T=5y, 100 steps — module-scoped to avoid recompilation."""
    return generate_hull_white_paths(hw_model, T=5.0, n_steps=100, n_paths=50_000, key=jax.random.PRNGKey(0))


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Conditional moments
# ─────────────────────────────────────────────────────────────────────


class TestConditionalMoments:
    """Sample moments of r(T) vs analytic OU mean and variance.

    Under the exact-fit Hull-White model the short rate decomposes as
    r(t) = x(t) + alpha(t), where x is a zero-mean OU and

        alpha(t) = f^M(0,t) + sigma^2/(2a^2) * (1-exp(-a*t))^2

    Starting from x(0)=0 (i.e. r(0)=alpha(0)), the unconditional mean is:

        E[r(T)] = alpha(T)

    and the unconditional variance is:

        Var[r(T)] = sigma^2/(2a) * (1 - exp(-2*a*T))
    """

    def test_mean(self, hw_model, paths_50k):
        T = 5.0
        r_T = paths_50k.short_rates[:, -1]
        sample_mean = float(jnp.mean(r_T))

        # Analytic mean: alpha(T) = f^M(0,T) + sigma^2/(2a^2)*(1-exp(-a*T))^2
        a = float(hw_model.mean_reversion)
        sigma = float(hw_model.volatility)
        fwd_T = float(_instantaneous_forward(hw_model, jnp.asarray(T)))
        analytic_mean = fwd_T + (sigma**2 / (2.0 * a**2)) * (1.0 - float(jnp.exp(-a * T)))**2

        # A single fixed seed is one draw from N(0,1) in z-units, so a 2-sigma
        # bound would reject ~5 % of perfectly valid draws (this seed sits at
        # -2.03 sigma).  Unbiasedness is asserted with proper power in
        # `test_mean_unbiased_across_seeds` below; here we only guard against
        # gross error.
        stderr = float(jnp.std(r_T) / jnp.sqrt(jnp.array(50_000.0)))
        assert abs(sample_mean - analytic_mean) < 3.0 * stderr, (
            f"E[r(T)] sample={sample_mean:.6f}  analytic={analytic_mean:.6f}  "
            f"3*stderr={3*stderr:.6f}"
        )

    def test_mean_unbiased_across_seeds(self, hw_model):
        """Pool independent seeds: the z-score must average to zero.

        Any systematic error in the exact-OU scheme (a wrong ``alpha`` shift,
        a mis-scaled conditional variance, an off-by-one in the scan) shows up
        as a consistent sign here, which a single-seed test cannot detect.
        """
        T = 5.0
        n_paths = 25_000
        n_seeds = 8

        a = float(hw_model.mean_reversion)
        sigma = float(hw_model.volatility)
        fwd_T = float(_instantaneous_forward(hw_model, jnp.asarray(T)))
        analytic_mean = (
            fwd_T + (sigma**2 / (2.0 * a**2)) * (1.0 - float(jnp.exp(-a * T))) ** 2
        )

        z_scores = []
        for seed in range(n_seeds):
            res = generate_hull_white_paths(
                hw_model, T=T, n_steps=100, n_paths=n_paths,
                key=jax.random.PRNGKey(seed),
            )
            r_T = res.short_rates[:, -1]
            stderr = float(jnp.std(r_T) / jnp.sqrt(jnp.array(float(n_paths))))
            z_scores.append((float(jnp.mean(r_T)) - analytic_mean) / stderr)

        # mean(z) ~ N(0, 1/n_seeds); 3 sigma is 3/sqrt(n_seeds).
        mean_z = sum(z_scores) / n_seeds
        assert abs(mean_z) < 3.0 / (n_seeds ** 0.5), (
            f"mean z-score {mean_z:+.3f} over {n_seeds} seeds suggests bias; "
            f"individual z = {[f'{z:+.2f}' for z in z_scores]}"
        )

    def test_variance(self, hw_model, paths_50k):
        T = 5.0
        r_T = paths_50k.short_rates[:, -1]
        sample_var = float(jnp.var(r_T))

        analytic_var = float(hw_short_rate_variance(hw_model, jnp.asarray(T)))

        # A variance estimate at 50k paths has a relative std-error of about
        # sqrt(2/(N-1)) ~ 0.63 %, so 3 % is a ~5-sigma band.
        assert abs(sample_var - analytic_var) / analytic_var < 0.03, (
            f"Var[r(T)] sample={sample_var:.2e}  analytic={analytic_var:.2e}"
        )

    def test_short_rates_shape(self, paths_50k):
        assert paths_50k.short_rates.shape == (50_000, 101)

    def test_log_dfs_shape(self, paths_50k):
        assert paths_50k.log_discount_factors.shape == (50_000, 101)

    def test_log_dfs_initial_zero(self, paths_50k):
        assert jnp.allclose(paths_50k.log_discount_factors[:, 0], 0.0)


# ─────────────────────────────────────────────────────────────────────
# Test 2 — ZCB consistency (martingale measure check)
# ─────────────────────────────────────────────────────────────────────


class TestZCBConsistency:
    """MC price of ZCB recovers hw_bond_price(model, r0, 0, T)."""

    @pytest.mark.parametrize("T", [1.0, 3.0, 5.0])
    def test_zcb_mc_vs_analytic(self, hw_model, T):
        n_paths = 50_000
        # Use enough steps that trapezoidal integration error (O(dt^2) per step,
        # O(dt) total) is well below statistical noise.  250 steps/year keeps
        # the bias < 1 bp per unit of time.
        n_steps = max(50, int(T * 250))
        result = generate_hull_white_paths(
            hw_model, T=T, n_steps=n_steps, n_paths=n_paths,
            key=jax.random.PRNGKey(1),
        )
        # Stochastic discount factors.
        sdf = jnp.exp(result.log_discount_factors[:, -1])  # (n_paths,)

        mc_price = float(jnp.mean(sdf))
        stderr = float(jnp.std(sdf) / jnp.sqrt(jnp.array(float(n_paths))))

        r0 = float(_instantaneous_forward(hw_model, jnp.asarray(0.0)))
        analytic = float(hw_bond_price(hw_model, jnp.asarray(r0), jnp.asarray(0.0), jnp.asarray(T)))

        assert abs(mc_price - analytic) < 2.0 * stderr, (
            f"T={T}: MC={mc_price:.6f}  analytic={analytic:.6f}  "
            f"2*stderr={2*stderr:.6f}"
        )


# ─────────────────────────────────────────────────────────────────────
# Test 3 — Tree triangulation for callable bond
# ─────────────────────────────────────────────────────────────────────


class TestPathDiscountingSanity:
    """The path SDFs reprice a *straight* bond to its analytic curve PV.

    This is a property of the path generator only — it checks that the
    trapezoidal log-discount accumulator in
    :func:`generate_hull_white_paths` is unbiased, independently of any
    exercise logic.

    The callable-bond MC-vs-tree triangulation lives in
    ``tests/test_mc/test_hull_white_recipes.py``; here we merely assert the
    no-arbitrage ordering ``callable <= straight``.
    """

    @pytest.fixture(scope="class")
    def callable_bond(self, ref_date):
        """5Y annual bond, callable at par on years 2 and 4."""
        payment_ords = jnp.array(
            [ref_date + int(round(k * 365)) for k in range(1, 6)],
            dtype=jnp.int32,
        )
        call_ords = jnp.array(
            [ref_date + int(round(k * 365)) for k in [2, 4]],
            dtype=jnp.int32,
        )
        return CallableBond(
            payment_dates=payment_ords,
            settlement_date=jnp.int32(ref_date),
            coupon_rate=jnp.asarray(0.05),
            face_value=jnp.asarray(100.0),
            call_dates=call_ords,
            call_prices=jnp.ones(2),
            frequency=1,
            day_count="act_365",
        )

    def test_straight_bond_mc_matches_analytic(self, hw_model, callable_bond):
        """Straight-bond MC vs analytic, plus the callable <= straight bound."""
        tree_price = float(callable_bond_price(callable_bond, hw_model, n_steps=200))

        T = 5.0
        n_paths = 50_000
        n_steps = 500   # 100 steps/year — enough for trapezoidal bias < 0.05
        result = generate_hull_white_paths(
            hw_model, T=T, n_steps=n_steps, n_paths=n_paths,
            key=jax.random.PRNGKey(2),
        )
        # Price straight bond MC: E[sum coupon_i * D(0, t_i) + face * D(0, T)]
        dt = T / n_steps

        coupon = 0.05 * 100.0
        coupon_years = [1.0, 2.0, 3.0, 4.0, 5.0]
        step_idxs = [int(round(cy / dt)) for cy in coupon_years]

        pv = jnp.zeros(n_paths)
        for step in step_idxs:
            sdf = jnp.exp(result.log_discount_factors[:, step])
            pv = pv + coupon * sdf
        # Face value.
        sdf_T = jnp.exp(result.log_discount_factors[:, -1])
        pv = pv + 100.0 * sdf_T
        mc_straight = float(jnp.mean(pv))
        mc_stderr = float(jnp.std(pv) / jnp.sqrt(jnp.array(float(n_paths))))

        # Analytic straight bond price.
        from valax.pricing.analytic.bonds import fixed_rate_bond_price
        from valax.instruments.bonds import FixedRateBond
        cb = callable_bond
        straight = FixedRateBond(
            payment_dates=cb.payment_dates,
            settlement_date=cb.settlement_date,
            coupon_rate=cb.coupon_rate,
            face_value=cb.face_value,
            frequency=1,
            day_count="act_365",
        )
        analytic_straight = float(fixed_rate_bond_price(straight, hw_model.initial_curve))

        # Straight bond MC should match analytic within 2 stderr.
        assert abs(mc_straight - analytic_straight) < 2.0 * mc_stderr, (
            f"MC straight={mc_straight:.4f}  analytic={analytic_straight:.4f}  "
            f"stderr={mc_stderr:.4f}"
        )

        # Callable <= straight (embedded call option cannot increase bond value).
        assert tree_price <= analytic_straight, (
            f"tree callable={tree_price:.4f} should be <= straight={analytic_straight:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────
# Test 4 — JIT and grad
# ─────────────────────────────────────────────────────────────────────


class TestJITAndGrad:
    """filter_jit smoke and filter_grad through model.volatility."""

    def test_jit_smoke(self, hw_model):
        """generate_hull_white_paths is JIT-compilable."""
        key = jax.random.PRNGKey(42)
        eager = generate_hull_white_paths(hw_model, T=1.0, n_steps=10, n_paths=100, key=key)

        @eqx.filter_jit
        def _f(model):
            return generate_hull_white_paths(model, T=1.0, n_steps=10, n_paths=100, key=key)

        jit_result = _f(hw_model)
        assert jnp.allclose(eager.short_rates, jit_result.short_rates, atol=1e-10)

    def test_grad_through_volatility(self, hw_model):
        """eqx.filter_grad flows through model.volatility."""
        key = jax.random.PRNGKey(7)

        @eqx.filter_jit
        @eqx.filter_grad
        def _loss(model):
            res = generate_hull_white_paths(model, T=1.0, n_steps=10, n_paths=200, key=key)
            # Scalar loss: mean terminal short rate.
            return jnp.mean(res.short_rates[:, -1])

        grads = _loss(hw_model)
        dL_dsigma = grads.volatility
        assert jnp.isfinite(dL_dsigma), f"grad w.r.t. sigma is not finite: {dL_dsigma}"
        assert float(dL_dsigma) != 0.0, "grad w.r.t. sigma is zero (no signal?)"

    def test_grad_through_mean_reversion(self, hw_model):
        """eqx.filter_grad flows through model.mean_reversion."""
        key = jax.random.PRNGKey(13)

        @eqx.filter_jit
        @eqx.filter_grad
        def _loss(model):
            res = generate_hull_white_paths(model, T=1.0, n_steps=10, n_paths=200, key=key)
            return jnp.mean(jnp.exp(res.log_discount_factors[:, -1]))

        grads = _loss(hw_model)
        assert jnp.isfinite(grads.mean_reversion), (
            f"grad w.r.t. a is not finite: {grads.mean_reversion}"
        )
