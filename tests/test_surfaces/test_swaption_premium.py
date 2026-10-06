"""Swaption premium → normal (Bachelier) volatility conversion.

Covers the round trip to machine precision, that the conversion is genuinely
curve-dependent (the gradient w.r.t. the discount and projection pillars is
non-zero and matches finite differences), and that physical and cash
settlement are distinct conventions rather than aliases.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.dates.schedule import generate_schedule
from valax.instruments.rates import Swaption
from valax.pricing.analytic.cms_convexity import cash_settled_annuity
from valax.pricing.analytic.swaptions import (
    swaption_forward_and_annuity,
    swaption_price_bachelier,
)
from valax.surfaces.swaption_premium import (
    SETTLEMENTS,
    swaption_forward_and_numeraire,
    swaption_implied_normal_vol,
    swaption_implied_normal_vol_from_graph,
    swaption_premium_from_normal_vol,
    swaption_premium_residual,
)


REF = ymd_to_ordinal(2025, 1, 1)

#: Round-trip tolerance: relative on the recovered vol.
ROUNDTRIP_RTOL = 1e-10


def _curve(zero=0.03):
    pillars = jnp.array([
        int(ymd_to_ordinal(y, 1, 1))
        for y in (2025, 2026, 2027, 2028, 2030, 2032, 2035, 2040, 2045)
    ], dtype=jnp.int32)
    times = (pillars - int(REF)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero * times),
        reference_date=REF,
    )


def _swaption(strike=0.03, is_payer=True, freq=1, tenor=5):
    return Swaption(
        expiry_date=ymd_to_ordinal(2026, 1, 1),
        fixed_dates=generate_schedule(
            2026, 1, 1, 2026 + tenor, 1, 1, frequency=freq,
        ).payment_dates,
        strike=jnp.array(strike),
        notional=jnp.array(1_000_000.0),
        is_payer=is_payer,
    )


@pytest.fixture
def curve():
    return _curve()


@pytest.fixture
def proj_curve():
    return _curve(zero=0.035)


# ── The physical numeraire is the pricer's own annuity ────────────────

class TestNumeraire:
    def test_physical_numeraire_is_the_discount_annuity(self, curve):
        s = _swaption()
        F, N, T = swaption_forward_and_numeraire(s, curve)
        S, A = swaption_forward_and_annuity(s, curve)
        assert float(F) == float(S)
        assert float(N) == float(A)
        assert float(T) == float(
            year_fraction(curve.reference_date, s.expiry_date, s.day_count)
        )

    def test_cash_numeraire_is_df_times_flat_yield_annuity(self, curve):
        s = _swaption()
        F, N, _ = swaption_forward_and_numeraire(s, curve, settlement="cash")
        n_periods = int(s.fixed_dates.shape[0])
        want = curve(s.expiry_date) * cash_settled_annuity(F, n_periods, 1)
        assert float(N) == float(want)

    def test_cash_and_physical_numeraires_differ(self, curve):
        s = _swaption()
        _, phys, _ = swaption_forward_and_numeraire(s, curve)
        _, cash, _ = swaption_forward_and_numeraire(
            s, curve, settlement="cash"
        )
        assert abs(float(phys) - float(cash)) / float(phys) > 1e-4

    @pytest.mark.parametrize("freq,tenor", [(1, 5), (2, 5), (4, 10)])
    def test_frequency_is_inferred_from_the_schedule(self, curve, freq, tenor):
        s = _swaption(freq=freq, tenor=tenor)
        _, inferred, _ = swaption_forward_and_numeraire(
            s, curve, settlement="cash"
        )
        _, explicit, _ = swaption_forward_and_numeraire(
            s, curve, settlement="cash", freq=freq
        )
        assert float(inferred) == float(explicit)

    def test_unknown_settlement_is_rejected(self, curve):
        with pytest.raises(ValueError, match="settlement must be one of"):
            swaption_forward_and_numeraire(
                _swaption(), curve, settlement="swap"
            )

    def test_settlements_tuple_is_what_the_api_accepts(self, curve):
        for s in SETTLEMENTS:
            swaption_forward_and_numeraire(_swaption(), curve, settlement=s)


# ── Round trip ────────────────────────────────────────────────────────

class TestRoundTrip:
    @pytest.mark.parametrize("settlement", list(SETTLEMENTS))
    @pytest.mark.parametrize("strike", [0.02, 0.03, 0.04])
    @pytest.mark.parametrize("is_payer", [True, False])
    def test_vol_to_premium_to_vol(self, curve, settlement, strike, is_payer):
        s = _swaption(strike=strike, is_payer=is_payer)
        vol = jnp.array(0.0075)
        premium = swaption_premium_from_normal_vol(
            s, vol, curve, settlement=settlement
        )
        assert float(premium) > 0.0
        back = swaption_implied_normal_vol(
            s, premium, curve, settlement=settlement
        )
        assert float(back) == pytest.approx(float(vol), rel=ROUNDTRIP_RTOL)

    def test_residual_is_zero_at_the_solution(self, curve):
        s = _swaption()
        premium = swaption_premium_from_normal_vol(
            s, jnp.array(0.0075), curve
        )
        vol = swaption_implied_normal_vol(s, premium, curve)
        r = swaption_premium_residual(vol, s, premium, curve)
        assert abs(float(r)) < 1e-6  # currency units on a 1 mm notional

    def test_physical_route_matches_the_standalone_pricer(self, curve):
        s = _swaption()
        vol = jnp.array(0.0075)
        got = swaption_premium_from_normal_vol(s, vol, curve)
        want = swaption_price_bachelier(s, curve, vol)
        assert float(got) == float(want)

    def test_dual_curve_round_trip(self, curve, proj_curve):
        s = _swaption()
        vol = jnp.array(0.0075)
        premium = swaption_premium_from_normal_vol(s, vol, curve, proj_curve)
        back = swaption_implied_normal_vol(s, premium, curve, proj_curve)
        assert float(back) == pytest.approx(float(vol), rel=ROUNDTRIP_RTOL)

    def test_the_basis_changes_the_vol_a_premium_implies(
        self, curve, proj_curve
    ):
        """The headline claim: the same premium is a different vol."""
        s = _swaption()
        premium = swaption_premium_from_normal_vol(
            s, jnp.array(0.0075), curve, proj_curve
        )
        single = swaption_implied_normal_vol(s, premium, curve)
        dual = swaption_implied_normal_vol(s, premium, curve, proj_curve)
        assert abs(float(single) - float(dual)) > 1e-5

    def test_cash_and_physical_imply_different_vols(self, curve):
        s = _swaption()
        premium = swaption_premium_from_normal_vol(s, jnp.array(0.0075), curve)
        phys = swaption_implied_normal_vol(s, premium, curve)
        cash = swaption_implied_normal_vol(s, premium, curve, settlement="cash")
        assert abs(float(phys) - float(cash)) > 1e-5

    def test_initial_vol_override_reaches_the_same_root(self, curve):
        s = _swaption()
        premium = swaption_premium_from_normal_vol(s, jnp.array(0.0075), curve)
        a = swaption_implied_normal_vol(s, premium, curve)
        b = swaption_implied_normal_vol(s, premium, curve, initial_vol=0.05)
        assert float(b) == pytest.approx(float(a), rel=1e-9)

    def test_graph_entry_point_agrees(self, curve, proj_curve):
        s = _swaption()
        graph = CurveGraph(curves={
            "USD.SOFR.OIS": curve, "USD.TERM.3M": proj_curve,
        })
        premium = swaption_premium_from_normal_vol(s, jnp.array(0.0075), curve, proj_curve)
        a = swaption_implied_normal_vol(s, premium, curve, proj_curve)
        b = swaption_implied_normal_vol_from_graph(
            s, premium, graph, "USD.SOFR.OIS", "USD.TERM.3M"
        )
        assert float(a) == float(b)


# ── Differentiability (the reason the residual exists) ────────────────

class TestGradients:
    @staticmethod
    def _fd(fn, x, h):
        """Central difference of a scalar-of-vector function, per component."""
        out = []
        for i in range(x.shape[0]):
            e = jnp.zeros_like(x).at[i].set(h)
            out.append((float(fn(x + e)) - float(fn(x - e))) / (2.0 * h))
        return jnp.asarray(out)

    @pytest.mark.parametrize("settlement", list(SETTLEMENTS))
    def test_vol_has_nonzero_discount_curve_gradient(self, curve, settlement):
        s = _swaption()
        premium = swaption_premium_from_normal_vol(
            s, jnp.array(0.0075), curve, settlement=settlement
        )

        def vol_of_dfs(dfs):
            c = eqx.tree_at(lambda x: x.discount_factors, curve, dfs)
            return swaption_implied_normal_vol(
                s, premium, c, settlement=settlement
            )

        g = jax.grad(vol_of_dfs)(curve.discount_factors)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 1e-6

        fd = self._fd(vol_of_dfs, curve.discount_factors, 1e-7)
        scale = float(jnp.max(jnp.abs(fd))) + 1e-12
        assert float(jnp.max(jnp.abs(g - fd))) / scale < 1e-4

    def test_vol_has_nonzero_projection_curve_gradient(
        self, curve, proj_curve
    ):
        s = _swaption()
        premium = swaption_premium_from_normal_vol(
            s, jnp.array(0.0075), curve, proj_curve
        )

        def vol_of_proj(dfs):
            p = eqx.tree_at(lambda x: x.discount_factors, proj_curve, dfs)
            return swaption_implied_normal_vol(s, premium, curve, p)

        g = jax.grad(vol_of_proj)(proj_curve.discount_factors)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 1e-6

    def test_premium_gradient_is_the_reciprocal_vega(self, curve):
        s = _swaption()
        vol = jnp.array(0.0075)
        premium = swaption_premium_from_normal_vol(s, vol, curve)

        dvol_dq = jax.grad(
            lambda q: swaption_implied_normal_vol(s, q, curve)
        )(premium)
        vega = jax.grad(
            lambda v: swaption_premium_from_normal_vol(s, v, curve)
        )(vol)
        assert float(dvol_dq) == pytest.approx(1.0 / float(vega), rel=1e-6)


class TestJit:
    def test_round_trip_is_jittable(self, curve):
        s = _swaption()
        premium = swaption_premium_from_normal_vol(s, jnp.array(0.0075), curve)
        vol = eqx.filter_jit(swaption_implied_normal_vol)(s, premium, curve)
        assert jnp.isfinite(vol)
        assert float(vol) == pytest.approx(0.0075, rel=ROUNDTRIP_RTOL)

    def test_cash_branch_is_jittable_with_an_explicit_freq(self, curve):
        s = _swaption()
        premium = swaption_premium_from_normal_vol(
            s, jnp.array(0.0075), curve, settlement="cash", freq=1
        )
        vol = eqx.filter_jit(
            lambda sw, q, c: swaption_implied_normal_vol(
                sw, q, c, settlement="cash", freq=1
            )
        )(s, premium, curve)
        assert float(vol) == pytest.approx(0.0075, rel=ROUNDTRIP_RTOL)

    def test_cash_branch_asks_for_freq_when_the_schedule_is_traced(self, curve):
        """The frequency is a convention, so it is never guessed under trace."""
        s = _swaption()
        premium = swaption_premium_from_normal_vol(
            s, jnp.array(0.0075), curve, settlement="cash"
        )
        with pytest.raises(ValueError, match="Pass freq"):
            eqx.filter_jit(
                lambda sw, q, c: swaption_implied_normal_vol(
                    sw, q, c, settlement="cash"
                )
            )(s, premium, curve)
