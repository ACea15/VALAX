"""Tests for VaR backtesting and FRTB PLA statistics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.risk.backtesting import (
    basel_traffic_light,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    ks_statistic,
    kupiec_pof,
    pla_ks,
    pla_spearman,
    pla_traffic_light,
    var_breaches,
)


# ── var_breaches ─────────────────────────────────────────────────────


class TestVarBreaches:
    def test_breach_when_loss_exceeds_var(self):
        actual = jnp.array([-1.0, -5.0, 0.5, -3.0])
        var = jnp.array([2.0, 2.0, 2.0, 2.0])
        breaches = var_breaches(actual, var)
        # Only -5.0 and -3.0 exceed the 2.0 threshold
        expected = jnp.array([False, True, False, True])
        assert jnp.all(breaches == expected)

    def test_no_breaches_when_var_is_high(self):
        actual = -jax.random.normal(jax.random.PRNGKey(0), (100,))
        var = jnp.full((100,), 100.0)
        breaches = var_breaches(actual, var)
        assert int(jnp.sum(breaches)) == 0


# ── Kupiec POF ───────────────────────────────────────────────────────


class TestKupiecPOF:
    def test_perfect_coverage_gives_zero_lr(self):
        """When the empirical breach rate equals 1 - confidence exactly,
        the LR statistic is zero."""
        # 250 days, 0.01 expected rate, 1% breach rate ⇒ 2.5 expected.
        # Choose n=200, x=2 (exactly 1% breach rate) ⇒ LR = 0.
        n, x = 200, 2
        breaches = jnp.array([True] * x + [False] * (n - x))
        result = kupiec_pof(breaches, confidence=0.99)
        assert int(result["x"]) == x
        assert int(result["n"]) == n
        assert float(result["lr_uc"]) < 1e-6

    def test_large_lr_for_far_off_rate(self):
        """50 breaches in 200 days at 99% VaR ⇒ overwhelming rejection."""
        n, x = 200, 50
        breaches = jnp.array([True] * x + [False] * (n - x))
        result = kupiec_pof(breaches, confidence=0.99)
        assert float(result["lr_uc"]) > 50.0
        assert float(result["p_value"]) < 1e-6

    def test_zero_breaches_handled(self):
        breaches = jnp.zeros(250, dtype=bool)
        result = kupiec_pof(breaches, confidence=0.99)
        # With p=0.01, n=250, the LR for x=0 is -2 * (n*log(0.99) - 0) ≈ 5.03
        assert float(result["lr_uc"]) > 0.0
        assert jnp.isfinite(result["lr_uc"])

    def test_p_value_in_unit_interval(self):
        breaches = jnp.array([True, False, False, True, False] * 50)
        result = kupiec_pof(breaches, confidence=0.99)
        assert 0.0 <= float(result["p_value"]) <= 1.0


# ── Christoffersen independence ─────────────────────────────────────


class TestChristoffersenIndependence:
    def test_clustered_breaches_high_lr(self):
        """All breaches at the start ⇒ strong dependence ⇒ large LR_ind."""
        n = 250
        breaches = jnp.array([True] * 10 + [False] * (n - 10))
        result = christoffersen_independence(breaches)
        # n01=1, n11=9, n10=1, n00=238 ⇒ clear clustering
        assert float(result["lr_ind"]) > 10.0

    def test_evenly_spaced_breaches_low_lr(self):
        """Evenly spaced breaches (one every 25 days, 10 total) should
        give LR_ind close to zero."""
        n = 250
        idx = np.arange(0, n, 25)  # ten indices, 0,25,50,...,225
        b = np.zeros(n, dtype=bool)
        b[idx] = True
        result = christoffersen_independence(jnp.asarray(b))
        # n11 should be 0 (no consecutive breaches) which gives pi11=0;
        # the LR with pi11=0 reflects independence so should be small
        assert float(result["lr_ind"]) < 5.0

    def test_no_breaches_lr_zero(self):
        breaches = jnp.zeros(100, dtype=bool)
        result = christoffersen_independence(breaches)
        assert float(result["lr_ind"]) == 0.0


# ── Christoffersen conditional coverage ─────────────────────────────


class TestChristoffersenCC:
    def test_lr_cc_equals_sum(self):
        breaches = jnp.array([True] * 5 + [False] * 245)
        cc = christoffersen_conditional_coverage(breaches, confidence=0.99)
        assert jnp.isclose(
            cc["lr_cc"], cc["lr_uc"] + cc["lr_ind"], atol=1e-6,
        )


# ── Basel traffic light ─────────────────────────────────────────────


class TestBaselTrafficLight:
    def test_zones_at_canonical_thresholds(self):
        # 250 days, 99% confidence: 0-4 green, 5-9 yellow, 10+ red
        assert basel_traffic_light(0) == "green"
        assert basel_traffic_light(4) == "green"
        assert basel_traffic_light(5) == "yellow"
        assert basel_traffic_light(9) == "yellow"
        assert basel_traffic_light(10) == "red"
        assert basel_traffic_light(50) == "red"

    def test_non_standard_window_recomputes_thresholds(self):
        # 500 days, 99% confidence: expected 5 breaches, green up to ~8
        zone_low = basel_traffic_light(2, n_obs=500, confidence=0.99)
        zone_high = basel_traffic_light(40, n_obs=500, confidence=0.99)
        assert zone_low == "green"
        assert zone_high == "red"


# ── Spearman correlation ────────────────────────────────────────────


class TestPlaSpearman:
    def test_monotonic_data_correlation_one(self):
        x = jnp.linspace(0.0, 1.0, 100)
        y = 3.0 * x + 1.0
        rho = pla_spearman(x, y)
        assert float(rho) > 0.999

    def test_reversed_data_correlation_minus_one(self):
        x = jnp.linspace(0.0, 1.0, 100)
        y = -x
        rho = pla_spearman(x, y)
        assert float(rho) < -0.999

    def test_nonlinear_monotonic_correlation_one(self):
        """Spearman captures monotonic, not just linear, agreement."""
        x = jnp.linspace(0.1, 1.0, 100)
        y = jnp.exp(x)
        rho = pla_spearman(x, y)
        assert float(rho) > 0.999

    def test_independent_data_correlation_near_zero(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(key1, (1000,))
        y = jax.random.normal(key2, (1000,))
        rho = pla_spearman(x, y)
        assert abs(float(rho)) < 0.1


# ── KS statistic ────────────────────────────────────────────────────


class TestKSStatistic:
    def test_identical_distributions_low_d(self):
        x = jnp.arange(100, dtype=jnp.float64)
        y = jnp.arange(100, dtype=jnp.float64)
        d = ks_statistic(x, y)
        assert float(d) < 1e-6

    def test_shifted_distributions_nonzero_d(self):
        x = jnp.arange(100, dtype=jnp.float64)
        y = jnp.arange(100, dtype=jnp.float64) + 50.0
        d = ks_statistic(x, y)
        assert float(d) > 0.4

    def test_d_in_unit_interval(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (200,))
        y = jax.random.normal(k2, (200,)) + 1.0
        d = ks_statistic(x, y)
        assert 0.0 <= float(d) <= 1.0

    def test_pla_ks_alias(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        assert jnp.isclose(pla_ks(x, y), ks_statistic(x, y))


# ── PLA traffic light ───────────────────────────────────────────────


class TestPlaTrafficLight:
    def test_high_spearman_low_ks_green(self):
        # Identical distributions, perfect rank correlation
        assert pla_traffic_light(spearman=0.95, ks_stat=0.02, n_obs=250) == "green"

    def test_low_spearman_red(self):
        assert pla_traffic_light(spearman=0.5, ks_stat=0.02, n_obs=250) == "red"

    def test_high_ks_red(self):
        assert pla_traffic_light(spearman=0.95, ks_stat=0.3, n_obs=250) == "red"

    def test_mid_spearman_amber(self):
        # Spearman in amber zone (0.70-0.80), KS in green ⇒ amber overall
        zone = pla_traffic_light(spearman=0.75, ks_stat=0.02, n_obs=250)
        assert zone == "amber"

    def test_zone_is_worst_of_two(self):
        # Spearman green, KS amber ⇒ amber.
        # For n=250 the KS green/amber boundary is at D≈0.089 (p≈0.264),
        # the amber/red boundary at D≈0.119 (p≈0.055), so 0.10 sits in amber.
        zone = pla_traffic_light(spearman=0.95, ks_stat=0.10, n_obs=250)
        assert zone == "amber"


# ── End-to-end PLA pipeline ─────────────────────────────────────────


class TestPlaEndToEnd:
    def test_identical_series_green(self):
        """When RTPL == HPL the PLA test must pass with green zone."""
        key = jax.random.PRNGKey(0)
        pnl = jax.random.normal(key, (250,))
        rho = pla_spearman(pnl, pnl)
        d = pla_ks(pnl, pnl)
        assert float(rho) > 0.999
        assert float(d) < 1e-6
        assert pla_traffic_light(float(rho), float(d), 250) == "green"

    def test_noisy_rtpl_still_green(self):
        """Small RTPL noise should still pass PLA (FRTB designed for this)."""
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        hpl = jax.random.normal(k1, (250,))
        # RTPL = HPL + small relative noise
        rtpl = hpl + 0.05 * jax.random.normal(k2, (250,))
        rho = pla_spearman(rtpl, hpl)
        d = pla_ks(rtpl, hpl)
        zone = pla_traffic_light(float(rho), float(d), 250)
        # Should be at least amber, ideally green
        assert zone in ("green", "amber")
