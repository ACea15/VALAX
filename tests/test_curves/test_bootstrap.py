"""Tests for curve bootstrapping: sequential, simultaneous, and multi-curve."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve, forward_rate, zero_rate
from valax.curves.instruments import DepositRate, FRA, SwapRate
from valax.curves.bootstrap import bootstrap_sequential, bootstrap_simultaneous
from valax.curves.multi_curve import MultiCurveSet, bootstrap_multi_curve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.instruments.rates import InterestRateSwap
from valax.pricing.analytic.swaptions import swap_price


# ── Helpers ───────────────────────────────────────────────────────────

REF = ymd_to_ordinal(2025, 1, 1)


def _make_date(y, m, d):
    return ymd_to_ordinal(y, m, d)


def _make_deposit(end_y, end_m, end_d, rate, start=None, day_count="act_365"):
    """Create a deposit starting at REF (or given start)."""
    if start is None:
        start = REF
    return DepositRate(
        start_date=start,
        end_date=_make_date(end_y, end_m, end_d),
        rate=jnp.array(rate),
        day_count=day_count,
    )


def _make_fra(start_y, start_m, start_d, end_y, end_m, end_d, rate, day_count="act_365"):
    return FRA(
        start_date=_make_date(start_y, start_m, start_d),
        end_date=_make_date(end_y, end_m, end_d),
        rate=jnp.array(rate),
        day_count=day_count,
    )


def _make_swap_rate(start, end_year, rate, frequency=1, day_count="act_365"):
    """Create a SwapRate with annual (or given frequency) fixed payments."""
    # Generate annual payment dates from start_year+1 to end_year
    start_y = 2025
    dates = []
    months_per_period = 12 // frequency
    ref_month = 1
    ref_day = 1
    current_y = start_y
    current_m = ref_month + months_per_period
    while True:
        if current_m > 12:
            current_y += 1
            current_m -= 12
        d = _make_date(current_y, current_m, ref_day)
        if int(d) > int(_make_date(end_year, 1, 1)):
            break
        dates.append(d)
        current_m += months_per_period
    # Always include the final maturity
    mat = _make_date(end_year, 1, 1)
    if not dates or int(dates[-1]) != int(mat):
        dates.append(mat)
    return SwapRate(
        start_date=start,
        fixed_dates=jnp.stack(dates).astype(jnp.int32),
        rate=jnp.array(rate),
        day_count=day_count,
    )


# ── Test: Deposit-only round-trip ────────────────────────────────────

class TestDepositBootstrap:
    def test_single_deposit(self):
        """One deposit should give correct DF."""
        dep = _make_deposit(2025, 4, 1, 0.05)  # 3M at 5%
        curve = bootstrap_sequential(REF, [dep])

        tau = year_fraction(REF, dep.end_date, dep.day_count)
        expected_df = 1.0 / (1.0 + 0.05 * float(tau))
        actual_df = float(curve(dep.end_date))
        assert abs(actual_df - expected_df) < 1e-10

    def test_multiple_deposits(self):
        """Multiple deposits should each produce the correct DF."""
        deposits = [
            _make_deposit(2025, 2, 1, 0.045),   # 1M
            _make_deposit(2025, 4, 1, 0.048),   # 3M
            _make_deposit(2025, 7, 1, 0.050),   # 6M
            _make_deposit(2026, 1, 1, 0.052),   # 1Y
        ]
        curve = bootstrap_sequential(REF, deposits)

        for dep in deposits:
            tau = year_fraction(REF, dep.end_date, dep.day_count)
            expected = 1.0 / (1.0 + float(dep.rate) * float(tau))
            actual = float(curve(dep.end_date))
            assert abs(actual - expected) < 1e-10, (
                f"Deposit maturity {int(dep.end_date)}: "
                f"expected DF={expected:.10f}, got {actual:.10f}"
            )

    def test_df_at_reference_is_one(self):
        dep = _make_deposit(2025, 4, 1, 0.05)
        curve = bootstrap_sequential(REF, [dep])
        assert abs(float(curve(REF)) - 1.0) < 1e-12


# ── Test: FRA chain round-trip ───────────────────────────────────────

class TestFRABootstrap:
    def test_deposit_then_fra(self):
        """Deposit + FRA: forward rate should match FRA rate."""
        dep = _make_deposit(2025, 4, 1, 0.05)
        fra = _make_fra(2025, 4, 1, 2025, 7, 1, 0.055)

        curve = bootstrap_sequential(REF, [dep, fra])

        # Extract forward rate from curve and compare
        fwd = forward_rate(curve, fra.start_date, fra.end_date)
        assert abs(float(fwd) - 0.055) < 1e-8

    def test_fra_chain(self):
        """Chain of FRAs: each forward should match."""
        dep = _make_deposit(2025, 4, 1, 0.05)
        fra1 = _make_fra(2025, 4, 1, 2025, 7, 1, 0.052)
        fra2 = _make_fra(2025, 7, 1, 2025, 10, 1, 0.054)
        fra3 = _make_fra(2025, 10, 1, 2026, 1, 1, 0.056)

        curve = bootstrap_sequential(REF, [dep, fra1, fra2, fra3])

        for fra in [fra1, fra2, fra3]:
            fwd = forward_rate(curve, fra.start_date, fra.end_date)
            assert abs(float(fwd) - float(fra.rate)) < 1e-8


# ── Test: Swap round-trip ────────────────────────────────────────────

class TestSwapBootstrap:
    def test_single_swap(self):
        """A swap bootstrapped at par should reprice to zero."""
        dep = _make_deposit(2026, 1, 1, 0.05)
        swap = _make_swap_rate(REF, 2027, 0.052)

        curve = bootstrap_sequential(REF, [dep, swap])

        # Build an InterestRateSwap at the par rate and check NPV ≈ 0
        irs = InterestRateSwap(
            start_date=swap.start_date,
            fixed_dates=swap.fixed_dates,
            fixed_rate=swap.rate,
            notional=jnp.array(1e6),
            day_count=swap.day_count,
        )
        npv = float(swap_price(irs, curve))
        assert abs(npv) < 1.0, f"Swap NPV should be ~0, got {npv:.4f}"

    def test_swap_strip(self):
        """Strip of swaps: each should reprice to par."""
        dep = _make_deposit(2026, 1, 1, 0.05)
        swaps = [
            _make_swap_rate(REF, 2027, 0.051),
            _make_swap_rate(REF, 2028, 0.052),
            _make_swap_rate(REF, 2029, 0.053),
            _make_swap_rate(REF, 2030, 0.054),
        ]
        curve = bootstrap_sequential(REF, [dep] + swaps)

        for sw in swaps:
            irs = InterestRateSwap(
                start_date=sw.start_date,
                fixed_dates=sw.fixed_dates,
                fixed_rate=sw.rate,
                notional=jnp.array(1e6),
                day_count=sw.day_count,
            )
            npv = float(swap_price(irs, curve))
            assert abs(npv) < 1.0, (
                f"Swap to {int(sw.fixed_dates[-1])}: NPV={npv:.4f}"
            )


# ── Test: Flat curve invariant ───────────────────────────────────────

class TestFlatCurveInvariant:
    def test_flat_rate_deposits(self):
        """If all deposits imply the same rate, zero rates should be flat."""
        r = 0.05
        deposits = [
            _make_deposit(2025, 4, 1, r),
            _make_deposit(2025, 7, 1, r),
            _make_deposit(2026, 1, 1, r),
        ]
        curve = bootstrap_sequential(REF, deposits)

        for dep in deposits:
            zr = float(zero_rate(curve, dep.end_date))
            # Simply-compounded rate r gives CC rate = ln(1 + r*tau)/tau
            tau = float(year_fraction(REF, dep.end_date, "act_365"))
            expected_cc = jnp.log(1.0 + r * tau) / tau
            assert abs(zr - float(expected_cc)) < 1e-6


# ── Test: Sequential vs Simultaneous consistency ─────────────────────

class TestSequentialVsSimultaneous:
    def test_deposits_match(self):
        """Both methods should produce the same curve for deposits."""
        deposits = [
            _make_deposit(2025, 4, 1, 0.045),
            _make_deposit(2025, 7, 1, 0.048),
            _make_deposit(2026, 1, 1, 0.050),
        ]

        seq_curve = bootstrap_sequential(REF, deposits)

        pillar_dates = jnp.stack(
            [jnp.asarray(d.end_date, dtype=jnp.int32) for d in deposits]
        )
        sim_curve = bootstrap_simultaneous(REF, pillar_dates, deposits)

        # Compare DFs at pillar dates
        seq_dfs = seq_curve(pillar_dates)
        sim_dfs = sim_curve(pillar_dates)
        assert jnp.allclose(seq_dfs, sim_dfs, atol=1e-8), (
            f"Sequential: {seq_dfs}\nSimultaneous: {sim_dfs}"
        )

    def test_mixed_instruments_match(self):
        """Both methods agree for deposits + swaps."""
        dep = _make_deposit(2026, 1, 1, 0.05)
        sw1 = _make_swap_rate(REF, 2027, 0.051)
        sw2 = _make_swap_rate(REF, 2028, 0.052)
        instruments = [dep, sw1, sw2]

        seq_curve = bootstrap_sequential(REF, instruments)

        pillar_dates = jnp.array([
            int(dep.end_date),
            int(sw1.fixed_dates[-1]),
            int(sw2.fixed_dates[-1]),
        ], dtype=jnp.int32)
        sim_curve = bootstrap_simultaneous(REF, pillar_dates, instruments)

        seq_dfs = seq_curve(pillar_dates)
        sim_dfs = sim_curve(pillar_dates)
        assert jnp.allclose(seq_dfs, sim_dfs, atol=1e-6), (
            f"Sequential: {seq_dfs}\nSimultaneous: {sim_dfs}"
        )


# ── Test: Differentiability ──────────────────────────────────────────

class TestDifferentiability:
    def test_grad_through_sequential_bootstrap(self):
        """Gradients should flow through the sequential bootstrap."""
        dep_rate = jnp.array(0.05)
        sw_rate = jnp.array(0.052)

        # Use a 2Y swap with annual payments so DF(2026) from the deposit
        # feeds into the swap's annuity calculation for DF(2027).
        date_2026 = _make_date(2026, 1, 1)
        date_2027 = _make_date(2027, 1, 1)

        def price_fn(dr, sr):
            dep = DepositRate(
                start_date=REF,
                end_date=date_2026,
                rate=dr,
                day_count="act_365",
            )
            sw = SwapRate(
                start_date=REF,
                fixed_dates=jnp.array(
                    [int(date_2026), int(date_2027)], dtype=jnp.int32
                ),
                rate=sr,
                day_count="act_365",
            )
            curve = bootstrap_sequential(REF, [dep, sw])
            return curve(date_2027)

        grads = jax.grad(price_fn, argnums=(0, 1))(dep_rate, sw_rate)
        assert all(jnp.isfinite(g) for g in grads), f"Non-finite grads: {grads}"
        # Higher deposit rate → lower DF(2026) → but with fixed swap rate,
        # DF(2027) must increase to satisfy the par condition. So d(DF_2027)/d(dep_rate) > 0.
        assert float(grads[0]) > 0, f"d(DF)/d(dep_rate) should be positive: {grads[0]}"
        # Higher swap rate → lower DF(2027) directly
        assert float(grads[1]) < 0, f"d(DF)/d(swap_rate) should be negative: {grads[1]}"

    def test_grad_through_simultaneous_bootstrap(self):
        """Gradients through the simultaneous bootstrap via implicit adjoint."""
        dep_rate = jnp.array(0.05)

        def price_fn(r):
            dep = DepositRate(
                start_date=REF,
                end_date=_make_date(2026, 1, 1),
                rate=r,
            )
            pillar_dates = jnp.array(
                [int(_make_date(2026, 1, 1))], dtype=jnp.int32
            )
            curve = bootstrap_simultaneous(REF, pillar_dates, [dep])
            return curve(_make_date(2026, 1, 1))

        grad = jax.grad(price_fn)(dep_rate)
        assert jnp.isfinite(grad)
        assert float(grad) < 0  # DF decreases with rate


# ── Test: Multi-curve bootstrap ──────────────────────────────────────

class TestMultiCurve:
    def test_discount_curve_built(self):
        """OIS discount curve should be correctly bootstrapped."""
        ois_deps = [
            _make_deposit(2025, 4, 1, 0.04),
            _make_deposit(2026, 1, 1, 0.042),
        ]
        fwd_deps = [
            _make_deposit(2025, 4, 1, 0.045),
            _make_deposit(2026, 1, 1, 0.047),
        ]

        mcs = bootstrap_multi_curve(
            reference_date=REF,
            discount_instruments=ois_deps,
            forward_instruments={"3M": fwd_deps},
        )

        assert isinstance(mcs, MultiCurveSet)
        assert isinstance(mcs.discount_curve, DiscountCurve)
        assert "3M" in mcs.forward_curves

        # OIS curve should reprice deposits
        for dep in ois_deps:
            tau = year_fraction(REF, dep.end_date, "act_365")
            expected = 1.0 / (1.0 + float(dep.rate) * float(tau))
            actual = float(mcs.discount_curve(dep.end_date))
            assert abs(actual - expected) < 1e-10

    def test_forward_curve_deposit_rates(self):
        """Forward curve deposits should produce correct DFs."""
        ois_deps = [
            _make_deposit(2025, 4, 1, 0.04),
            _make_deposit(2026, 1, 1, 0.042),
        ]
        fwd_deps = [
            _make_deposit(2025, 4, 1, 0.045),
            _make_deposit(2026, 1, 1, 0.047),
        ]

        mcs = bootstrap_multi_curve(
            reference_date=REF,
            discount_instruments=ois_deps,
            forward_instruments={"3M": fwd_deps},
        )

        fwd_curve = mcs.forward_curves["3M"]
        for dep in fwd_deps:
            tau = year_fraction(REF, dep.end_date, "act_365")
            expected = 1.0 / (1.0 + float(dep.rate) * float(tau))
            actual = float(fwd_curve(dep.end_date))
            assert abs(actual - expected) < 1e-10


# ── Test: Edge cases ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_rate_deposit(self):
        """Zero rate should give DF = 1."""
        dep = _make_deposit(2026, 1, 1, 0.0)
        curve = bootstrap_sequential(REF, [dep])
        assert abs(float(curve(dep.end_date)) - 1.0) < 1e-12

    def test_negative_rate_deposit(self):
        """Negative rates should give DF > 1."""
        dep = _make_deposit(2026, 1, 1, -0.01)
        curve = bootstrap_sequential(REF, [dep])
        assert float(curve(dep.end_date)) > 1.0

    def test_wrong_instrument_count_raises(self):
        """Simultaneous bootstrap should reject mismatched counts."""
        dep = _make_deposit(2026, 1, 1, 0.05)
        pillar_dates = jnp.array([
            int(_make_date(2026, 1, 1)),
            int(_make_date(2027, 1, 1)),
        ], dtype=jnp.int32)
        with pytest.raises(ValueError, match="one instrument per pillar"):
            bootstrap_simultaneous(REF, pillar_dates, [dep])
