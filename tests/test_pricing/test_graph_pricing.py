"""Tests for the dual-curve pricing seam and the CurveGraph wrappers.

Covers, per AGENTS.md:

- **Parity**: ``forward_curve=None`` reproduces the single-curve result,
  and passing the discount curve as the forward curve agrees with it.
- **Delegation**: every ``*_from_graph`` wrapper equals the underlying
  ``DiscountCurve`` pricer called on ``graph[...]`` lookups.
- **Dual-curve correctness**: hand-computed dual-curve PVs, par-rate
  consistency, and basis-direction monotonicity.
- **JIT/grad smoke tests** for the new code paths.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.instruments.bonds import FloatingRateBond
from valax.instruments.rates import (
    Cap,
    Caplet,
    CMSCapFloor,
    CMSSwap,
    InterestRateSwap,
    OISSwap,
    RangeAccrual,
    Swaption,
    TotalReturnSwap,
)
from valax.pricing.analytic import (
    cap_price_bachelier,
    cap_price_bachelier_from_graph,
    cap_price_black76,
    cap_price_black76_from_graph,
    caplet_price_bachelier,
    caplet_price_bachelier_from_graph,
    caplet_price_black76,
    caplet_price_black76_from_graph,
    cms_cap_floor_price_black76,
    cms_cap_floor_price_black76_from_graph,
    cms_swap_price,
    cms_swap_price_from_graph,
    floating_rate_bond_price,
    floating_rate_bond_price_from_graph,
    ois_swap_price,
    ois_swap_price_from_graph,
    ois_swap_rate,
    ois_swap_rate_from_graph,
    range_accrual_price_black76,
    range_accrual_price_black76_from_graph,
    swap_price,
    swap_price_from_graph,
    swap_rate,
    swap_rate_from_graph,
    swaption_price_bachelier,
    swaption_price_bachelier_from_graph,
    swaption_price_black76,
    swaption_price_black76_from_graph,
    total_return_swap_price,
    total_return_swap_price_from_graph,
)

REF = ymd_to_ordinal(2026, 1, 1)

DISC_ID = "USD.SOFR.OIS"
FWD_ID = "USD.LIBOR.3M"


def _flat_curve(rate: float) -> DiscountCurve:
    """Flat continuously-compounded curve with quarterly pillars to 4Y."""
    pillars = REF + jnp.arange(1, 17, dtype=jnp.int32) * 91
    times = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=REF,
    )


@pytest.fixture
def disc_curve():
    return _flat_curve(0.030)


@pytest.fixture
def fwd_curve():
    # Forward (projection) curve above the discount curve: positive basis.
    return _flat_curve(0.035)


@pytest.fixture
def graph(disc_curve, fwd_curve):
    return CurveGraph(curves={DISC_ID: disc_curve, FWD_ID: fwd_curve})


# ── Instruments ───────────────────────────────────────────────────────


@pytest.fixture
def swap():
    return InterestRateSwap(
        start_date=REF,
        fixed_dates=REF + jnp.array([365, 730], dtype=jnp.int32),
        fixed_rate=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        day_count="act_365",
    )


@pytest.fixture
def ois(swap):
    return OISSwap(
        start_date=REF,
        fixed_dates=swap.fixed_dates,
        float_dates=REF + jnp.arange(1, 9, dtype=jnp.int32) * 91,
        fixed_rate=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        day_count="act_365",
    )


@pytest.fixture
def swaption():
    return Swaption(
        expiry_date=REF + jnp.array(365, dtype=jnp.int32),
        fixed_dates=REF + jnp.array([730, 1095], dtype=jnp.int32),
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        day_count="act_365",
    )


@pytest.fixture
def caplet():
    return Caplet(
        fixing_date=REF + jnp.array(365, dtype=jnp.int32),
        start_date=REF + jnp.array(365, dtype=jnp.int32),
        end_date=REF + jnp.array(456, dtype=jnp.int32),
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        day_count="act_365",
    )


@pytest.fixture
def cap():
    fixings = REF + jnp.array([91, 182, 273], dtype=jnp.int32)
    return Cap(
        fixing_dates=fixings,
        start_dates=fixings,
        end_dates=fixings + 91,
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        day_count="act_365",
    )


@pytest.fixture
def frn():
    payments = REF + jnp.arange(1, 9, dtype=jnp.int32) * 91
    return FloatingRateBond(
        payment_dates=payments,
        fixing_dates=payments - 91,
        settlement_date=REF,
        spread=jnp.array(0.001),
        face_value=jnp.array(100.0),
        day_count="act_365",
    )


@pytest.fixture
def trs():
    return TotalReturnSwap(
        start_date=REF,
        payment_dates=REF + jnp.array([91, 182, 273, 365], dtype=jnp.int32),
        notional=jnp.array(1_000_000.0),
        funding_spread=jnp.array(0.002),
        day_count="act_365",
    )


@pytest.fixture
def cms_swap():
    return CMSSwap(
        start_date=REF,
        payment_dates=REF + jnp.array([365, 730], dtype=jnp.int32),
        fixed_rate=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        cms_tenor=2,
        day_count="act_365",
    )


@pytest.fixture
def cms_cap():
    return CMSCapFloor(
        payment_dates=REF + jnp.array([365, 730], dtype=jnp.int32),
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        cms_tenor=2,
        day_count="act_365",
    )


@pytest.fixture
def range_accrual():
    return RangeAccrual(
        payment_dates=REF + jnp.array([91, 182, 273, 365], dtype=jnp.int32),
        coupon_rate=jnp.array(0.05),
        lower_barrier=jnp.array(0.01),
        upper_barrier=jnp.array(0.06),
        notional=jnp.array(1_000_000.0),
        day_count="act_365",
    )


VOL = jnp.array(0.20)


# ── Parity: forward_curve=None ≡ single-curve behaviour ──────────────


class TestSingleCurveParity:
    """Passing the discount curve as forward_curve must reproduce the
    default (``forward_curve=None``) result to numerical noise."""

    def test_swap_price(self, swap, disc_curve):
        base = swap_price(swap, disc_curve)
        dual = swap_price(swap, disc_curve, forward_curve=disc_curve)
        assert jnp.allclose(base, dual, atol=1e-6)

    def test_swap_rate(self, swap, disc_curve):
        base = swap_rate(swap, disc_curve)
        dual = swap_rate(swap, disc_curve, forward_curve=disc_curve)
        assert jnp.allclose(base, dual, atol=1e-12)

    def test_ois_swap_price(self, ois, disc_curve):
        base = ois_swap_price(ois, disc_curve)
        dual = ois_swap_price(ois, disc_curve, forward_curve=disc_curve)
        assert jnp.allclose(base, dual, atol=1e-6)

    def test_ois_swap_rate(self, ois, disc_curve):
        base = ois_swap_rate(ois, disc_curve)
        dual = ois_swap_rate(ois, disc_curve, forward_curve=disc_curve)
        assert jnp.allclose(base, dual, atol=1e-12)

    def test_swaptions(self, swaption, disc_curve):
        for fn in (swaption_price_black76, swaption_price_bachelier):
            base = fn(swaption, disc_curve, VOL)
            dual = fn(swaption, disc_curve, VOL, forward_curve=disc_curve)
            assert jnp.allclose(base, dual, atol=1e-6)

    def test_caplets_and_caps(self, caplet, cap, disc_curve):
        pairs = [
            (caplet_price_black76, caplet),
            (caplet_price_bachelier, caplet),
            (cap_price_black76, cap),
            (cap_price_bachelier, cap),
        ]
        for fn, inst in pairs:
            base = fn(inst, disc_curve, VOL)
            dual = fn(inst, disc_curve, VOL, forward_curve=disc_curve)
            assert jnp.allclose(base, dual, atol=1e-12)

    def test_frn(self, frn, disc_curve):
        base = floating_rate_bond_price(frn, disc_curve)
        dual = floating_rate_bond_price(frn, disc_curve, forward_curve=disc_curve)
        assert jnp.allclose(base, dual, atol=1e-12)

    def test_trs(self, trs, disc_curve):
        base = total_return_swap_price(trs, disc_curve)
        dual = total_return_swap_price(
            trs, disc_curve, forward_curve=disc_curve
        )
        assert jnp.allclose(base, dual, atol=1e-6)

    def test_cms(self, cms_swap, cms_cap, disc_curve):
        base = cms_swap_price(cms_swap, disc_curve)
        dual = cms_swap_price(cms_swap, disc_curve, forward_curve=disc_curve)
        assert jnp.allclose(base, dual, atol=1e-6)

        base = cms_cap_floor_price_black76(cms_cap, disc_curve, VOL)
        dual = cms_cap_floor_price_black76(
            cms_cap, disc_curve, VOL, forward_curve=disc_curve
        )
        assert jnp.allclose(base, dual, atol=1e-6)

    def test_range_accrual(self, range_accrual, disc_curve):
        base = range_accrual_price_black76(range_accrual, disc_curve, VOL)
        dual = range_accrual_price_black76(
            range_accrual, disc_curve, VOL, forward_curve=disc_curve
        )
        assert jnp.allclose(base, dual, atol=1e-12)


# ── Wrapper delegation ────────────────────────────────────────────────


class TestGraphWrapperDelegation:
    """Every ``*_from_graph`` call equals the underlying pricer on
    explicit ``graph[...]`` lookups."""

    def test_swap(self, swap, graph, disc_curve, fwd_curve):
        assert jnp.allclose(
            swap_price_from_graph(swap, graph, DISC_ID, FWD_ID),
            swap_price(swap, disc_curve, fwd_curve),
            atol=1e-12,
        )
        assert jnp.allclose(
            swap_rate_from_graph(swap, graph, DISC_ID, FWD_ID),
            swap_rate(swap, disc_curve, fwd_curve),
            atol=1e-12,
        )

    def test_swap_single_curve_default(self, swap, graph, disc_curve):
        assert jnp.allclose(
            swap_price_from_graph(swap, graph, DISC_ID),
            swap_price(swap, disc_curve),
            atol=1e-12,
        )

    def test_ois(self, ois, graph, disc_curve, fwd_curve):
        assert jnp.allclose(
            ois_swap_price_from_graph(ois, graph, DISC_ID, FWD_ID),
            ois_swap_price(ois, disc_curve, fwd_curve),
            atol=1e-12,
        )
        assert jnp.allclose(
            ois_swap_rate_from_graph(ois, graph, DISC_ID, FWD_ID),
            ois_swap_rate(ois, disc_curve, fwd_curve),
            atol=1e-12,
        )

    def test_swaptions(self, swaption, graph, disc_curve, fwd_curve):
        assert jnp.allclose(
            swaption_price_black76_from_graph(
                swaption, graph, DISC_ID, VOL, FWD_ID
            ),
            swaption_price_black76(swaption, disc_curve, VOL, fwd_curve),
            atol=1e-12,
        )
        assert jnp.allclose(
            swaption_price_bachelier_from_graph(
                swaption, graph, DISC_ID, VOL, FWD_ID
            ),
            swaption_price_bachelier(swaption, disc_curve, VOL, fwd_curve),
            atol=1e-12,
        )

    def test_caplets_caps(self, caplet, cap, graph, disc_curve, fwd_curve):
        pairs = [
            (caplet_price_black76_from_graph, caplet_price_black76, caplet),
            (caplet_price_bachelier_from_graph, caplet_price_bachelier, caplet),
            (cap_price_black76_from_graph, cap_price_black76, cap),
            (cap_price_bachelier_from_graph, cap_price_bachelier, cap),
        ]
        for wrapper, fn, inst in pairs:
            assert jnp.allclose(
                wrapper(inst, graph, DISC_ID, VOL, FWD_ID),
                fn(inst, disc_curve, VOL, fwd_curve),
                atol=1e-12,
            )

    def test_frn(self, frn, graph, disc_curve, fwd_curve):
        assert jnp.allclose(
            floating_rate_bond_price_from_graph(frn, graph, DISC_ID, FWD_ID),
            floating_rate_bond_price(frn, disc_curve, fwd_curve),
            atol=1e-12,
        )

    def test_trs(self, trs, graph, disc_curve, fwd_curve):
        assert jnp.allclose(
            total_return_swap_price_from_graph(
                trs, graph, DISC_ID, forward_id=FWD_ID
            ),
            total_return_swap_price(trs, disc_curve, forward_curve=fwd_curve),
            atol=1e-12,
        )

    def test_cms_and_range_accrual(
        self, cms_swap, cms_cap, range_accrual, graph, disc_curve, fwd_curve
    ):
        assert jnp.allclose(
            cms_swap_price_from_graph(cms_swap, graph, DISC_ID, FWD_ID),
            cms_swap_price(cms_swap, disc_curve, fwd_curve),
            atol=1e-12,
        )
        assert jnp.allclose(
            cms_cap_floor_price_black76_from_graph(
                cms_cap, graph, DISC_ID, VOL, FWD_ID
            ),
            cms_cap_floor_price_black76(cms_cap, disc_curve, VOL, fwd_curve),
            atol=1e-12,
        )
        assert jnp.allclose(
            range_accrual_price_black76_from_graph(
                range_accrual, graph, DISC_ID, VOL, FWD_ID
            ),
            range_accrual_price_black76(
                range_accrual, disc_curve, VOL, fwd_curve
            ),
            atol=1e-12,
        )

    def test_unknown_curve_id_raises(self, swap, graph):
        with pytest.raises(KeyError):
            swap_price_from_graph(swap, graph, "NO.SUCH.CURVE")


# ── Dual-curve correctness ────────────────────────────────────────────


class TestDualCurveCorrectness:
    def test_ois_float_leg_matches_hand_computation(
        self, ois, disc_curve, fwd_curve
    ):
        """Dual-curve float leg = sum of projected forwards discounted
        on the OIS curve — computed by hand on the float schedule."""
        starts = jnp.concatenate([ois.start_date[None], ois.float_dates[:-1]])
        tau = year_fraction(starts, ois.float_dates, ois.day_count)
        fwd_rates = (fwd_curve(starts) / fwd_curve(ois.float_dates) - 1.0) / tau
        float_pv = jnp.sum(fwd_rates * tau * disc_curve(ois.float_dates))

        starts_f = jnp.concatenate([ois.start_date[None], ois.fixed_dates[:-1]])
        tau_f = year_fraction(starts_f, ois.fixed_dates, ois.day_count)
        fixed_pv = ois.fixed_rate * jnp.sum(tau_f * disc_curve(ois.fixed_dates))

        expected = ois.notional * (float_pv - fixed_pv)
        actual = ois_swap_price(ois, disc_curve, forward_curve=fwd_curve)
        assert jnp.allclose(actual, expected, rtol=1e-10)

    def test_positive_basis_raises_payer_value(self, swap, ois, disc_curve, fwd_curve):
        """Forward curve above the discount curve ⇒ higher projected
        forwards ⇒ the payer (receive-float) swap gains value."""
        assert swap_price(swap, disc_curve, fwd_curve) > swap_price(
            swap, disc_curve
        )
        assert ois_swap_price(ois, disc_curve, fwd_curve) > ois_swap_price(
            ois, disc_curve
        )

    def test_par_rate_zeroes_the_swap(self, ois, disc_curve, fwd_curve):
        """Repricing at the dual-curve par rate gives NPV ≈ 0."""
        par = ois_swap_rate(ois, disc_curve, forward_curve=fwd_curve)
        par_swap = eqx.tree_at(lambda s: s.fixed_rate, ois, par)
        pv = ois_swap_price(par_swap, disc_curve, forward_curve=fwd_curve)
        assert jnp.allclose(pv, 0.0, atol=1e-6)

    def test_dual_curve_par_rate_above_single_curve(
        self, swap, disc_curve, fwd_curve
    ):
        assert swap_rate(swap, disc_curve, fwd_curve) > swap_rate(
            swap, disc_curve
        )

    def test_caplet_forward_projection(self, caplet, disc_curve, fwd_curve):
        """Dual-curve caplet uses the forward curve's (higher) rate but
        the discount curve's (higher) DF — price must exceed the
        single-curve price for an ATM caplet."""
        single = caplet_price_black76(caplet, disc_curve, VOL)
        dual = caplet_price_black76(caplet, disc_curve, VOL, fwd_curve)
        assert dual > single


# ── JIT / grad smoke tests ────────────────────────────────────────────


class TestJitAndGrad:
    def test_filter_jit_dual_curve_swap(self, swap, disc_curve, fwd_curve):
        jitted = eqx.filter_jit(swap_price)
        out = jitted(swap, disc_curve, fwd_curve)
        assert jnp.allclose(out, swap_price(swap, disc_curve, fwd_curve))

    def test_filter_jit_graph_wrapper(self, swap, graph):
        @eqx.filter_jit
        def price(s, g):
            return swap_price_from_graph(s, g, DISC_ID, FWD_ID)

        out = price(swap, graph)
        assert jnp.allclose(out, swap_price_from_graph(swap, graph, DISC_ID, FWD_ID))

    def test_grad_flows_through_forward_curve(self, swap, disc_curve, fwd_curve):
        def price_of_fwd_dfs(dfs):
            fwd = DiscountCurve(
                pillar_dates=fwd_curve.pillar_dates,
                discount_factors=dfs,
                reference_date=fwd_curve.reference_date,
            )
            return swap_price(swap, disc_curve, fwd)

        g = jax.grad(price_of_fwd_dfs)(fwd_curve.discount_factors)
        assert jnp.any(g != 0.0)
        assert jnp.all(jnp.isfinite(g))
