"""Tests for curve-aware, smile-aware rates pricing (vol-source entry points).

Verifies that a flat scalar reproduces the existing Black-76/Bachelier pricers
bit-for-bit, and that a SABR smile source (SwaptionCube / OptionletVolSurface)
reads the vol at the instrument's coordinate and routes by convention.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from jaxtyping import Float
from jax import Array

from valax.instruments.rates import Caplet, Cap, Swaption
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.dates.schedule import generate_schedule
from valax.models.sabr import SABRModel

from valax.pricing.analytic.swaptions import (
    swaption_forward_and_annuity,
    swaption_price_black76,
    swaption_price_bachelier,
)
from valax.pricing.analytic.caplets import (
    caplet_forward,
    cap_forwards,
    caplet_price_black76,
    cap_price_black76,
)
from valax.pricing.analytic.rates_smile import (
    swaption_price,
    caplet_price,
    cap_price,
)
from valax.surfaces import SwaptionCube, OptionletVolSurface, ConstantVol
from valax.surfaces.shifted import ShiftedVolSurface


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    pillars = jnp.array([
        int(ymd_to_ordinal(2025, 1, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2027, 1, 1)),
        int(ymd_to_ordinal(2028, 1, 1)),
        int(ymd_to_ordinal(2029, 1, 1)),
        int(ymd_to_ordinal(2030, 1, 1)),
        int(ymd_to_ordinal(2032, 1, 1)),
        int(ymd_to_ordinal(2035, 1, 1)),
    ], dtype=jnp.int32)
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-0.03 * times)
    return DiscountCurve(pillar_dates=pillars, discount_factors=dfs, reference_date=ref_date)


@pytest.fixture
def payer_swaption(ref_date):
    """1y x 5y payer swaption."""
    expiry = ymd_to_ordinal(2026, 1, 1)
    fixed_dates = generate_schedule(2026, 1, 1, 2031, 1, 1, frequency=1).payment_dates
    return Swaption(
        expiry_date=expiry,
        fixed_dates=fixed_dates,
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        is_payer=True,
    )


@pytest.fixture
def caplet_3m6m():
    return Caplet(
        fixing_date=ymd_to_ordinal(2025, 4, 1),
        start_date=ymd_to_ordinal(2025, 4, 1),
        end_date=ymd_to_ordinal(2025, 7, 1),
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        is_cap=True,
    )


@pytest.fixture
def cap_2y():
    """2-year quarterly cap."""
    fixings, starts, ends = [], [], []
    months = [(2025, 4), (2025, 7), (2025, 10), (2026, 1),
              (2026, 4), (2026, 7), (2026, 10), (2027, 1)]
    for (y, m), (y2, m2) in zip(months[:-1], months[1:]):
        fixings.append(int(ymd_to_ordinal(y, m, 1)))
        starts.append(int(ymd_to_ordinal(y, m, 1)))
        ends.append(int(ymd_to_ordinal(y2, m2, 1)))
    return Cap(
        fixing_dates=jnp.array(fixings, dtype=jnp.int32),
        start_dates=jnp.array(starts, dtype=jnp.int32),
        end_dates=jnp.array(ends, dtype=jnp.int32),
        strike=jnp.array(0.03),
        notional=jnp.array(1_000_000.0),
        is_cap=True,
    )


def _lognormal_cube():
    expiries = jnp.array([0.5, 1.0, 2.0, 5.0])
    tenors = jnp.array([1.0, 5.0, 10.0])
    shape = (4, 3)
    return SwaptionCube(
        expiries=expiries, tenors=tenors,
        forwards=jnp.full(shape, 0.03),
        alphas=jnp.full(shape, 0.22),
        betas=jnp.full(shape, 0.5),
        rhos=jnp.full(shape, -0.25),
        nus=jnp.full(shape, 0.4),
    )


def _lognormal_optionlet_surface():
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0])
    n = expiries.shape[0]
    # Vary alpha with expiry so the strip genuinely differs per caplet.
    return OptionletVolSurface(
        expiries=expiries,
        forwards=jnp.full(n, 0.03),
        alphas=jnp.linspace(0.20, 0.28, n),
        betas=jnp.full(n, 0.5),
        rhos=jnp.full(n, -0.25),
        nus=jnp.full(n, 0.4),
    )


# ── Flat vol reproduces the existing pricers bit-for-bit ──────────────

class TestConstantVolEquivalence:
    def test_swaption_flat_scalar_equals_black76(self, payer_swaption, flat_curve):
        v = jnp.array(0.20)
        got = swaption_price(payer_swaption, flat_curve, v)
        want = swaption_price_black76(payer_swaption, flat_curve, v)
        assert float(got) == float(want)

    def test_swaption_constantvol_normal_equals_bachelier(self, payer_swaption, flat_curve):
        v = jnp.array(0.010)
        got = swaption_price(payer_swaption, flat_curve, ConstantVol(v, is_normal=True))
        want = swaption_price_bachelier(payer_swaption, flat_curve, v)
        assert float(got) == float(want)

    def test_caplet_flat_scalar_equals_black76(self, caplet_3m6m, flat_curve):
        v = jnp.array(0.25)
        got = caplet_price(caplet_3m6m, flat_curve, v)
        want = caplet_price_black76(caplet_3m6m, flat_curve, v)
        assert float(got) == float(want)

    def test_cap_flat_scalar_equals_black76(self, cap_2y, flat_curve):
        v = jnp.array(0.25)
        got = cap_price(cap_2y, flat_curve, v)
        want = cap_price_black76(cap_2y, flat_curve, v)
        assert abs(float(got) - float(want)) < 1e-9


# ── Smile sources read the vol at the instrument coordinate ───────────

class TestSmileSources:
    def test_swaption_cube_lookup(self, payer_swaption, flat_curve):
        cube = _lognormal_cube()
        T = year_fraction(flat_curve.reference_date, payer_swaption.expiry_date,
                          payer_swaption.day_count)
        tenor = year_fraction(payer_swaption.expiry_date,
                              payer_swaption.fixed_dates[-1], payer_swaption.day_count)
        v = cube(payer_swaption.strike, T, tenor)
        got = swaption_price(payer_swaption, flat_curve, cube)
        want = swaption_price_black76(payer_swaption, flat_curve, v)
        assert float(got) == float(want)

    def test_normal_cube_routes_to_bachelier(self, payer_swaption, flat_curve):
        # A normal (beta=0) cube must route the price through Bachelier.
        cube = SwaptionCube(
            expiries=jnp.array([0.5, 1.0, 2.0, 5.0]),
            tenors=jnp.array([1.0, 5.0, 10.0]),
            forwards=jnp.full((4, 3), 0.03),
            alphas=jnp.full((4, 3), 0.01),
            betas=jnp.full((4, 3), 0.0),
            rhos=jnp.full((4, 3), -0.2),
            nus=jnp.full((4, 3), 0.3),
            shift=jnp.array(0.0), is_normal=True,
        )
        T = year_fraction(flat_curve.reference_date, payer_swaption.expiry_date,
                          payer_swaption.day_count)
        tenor = year_fraction(payer_swaption.expiry_date,
                              payer_swaption.fixed_dates[-1], payer_swaption.day_count)
        v = cube(payer_swaption.strike, T, tenor)
        got = swaption_price(payer_swaption, flat_curve, cube)
        want = swaption_price_bachelier(payer_swaption, flat_curve, v)
        assert float(got) == float(want)

    def test_cap_strip_reads_per_caplet_vols(self, cap_2y, flat_curve):
        surf = _lognormal_optionlet_surface()
        T = year_fraction(flat_curve.reference_date, cap_2y.fixing_dates, cap_2y.day_count)
        vols = jax.vmap(lambda t: surf(cap_2y.strike, t))(T)
        # The per-caplet vols must genuinely differ (real strip, not flat).
        assert float(jnp.max(vols) - jnp.min(vols)) > 1e-4
        got = cap_price(cap_2y, flat_curve, surf)
        want = cap_price_black76(cap_2y, flat_curve, vols)
        assert abs(float(got) - float(want)) < 1e-9

    def test_caplet_surface_lookup(self, caplet_3m6m, flat_curve):
        surf = _lognormal_optionlet_surface()
        T = year_fraction(flat_curve.reference_date, caplet_3m6m.fixing_date,
                          caplet_3m6m.day_count)
        v = surf(caplet_3m6m.strike, T)
        got = caplet_price(caplet_3m6m, flat_curve, surf)
        want = caplet_price_black76(caplet_3m6m, flat_curve, v)
        assert float(got) == float(want)


# ── The vol-source protocol passes the live forward (spec V-1) ────────

class _ForwardProbe(eqx.Module):
    """Vol source whose quote is ``base + forward``.

    Makes the forward the pricers hand in observable *through the price*
    rather than through a Python side effect, so the assertion still holds
    under ``vmap`` (the cap strip) and inside ``jit``.
    """

    base: Float[Array, ""]
    is_normal: bool = eqx.field(static=True, default=False)

    def __call__(self, *coords, forward=None):
        if forward is None:
            raise AssertionError(
                "vol source was queried without the live forward; the "
                "curve-aware protocol requires forward=..."
            )
        return self.base + forward


@pytest.fixture
def steep_forward_curve(ref_date):
    """A projection curve distinctly above the discount curve."""
    pillars = jnp.array([
        int(ymd_to_ordinal(2025, 1, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2027, 1, 1)),
        int(ymd_to_ordinal(2028, 1, 1)),
        int(ymd_to_ordinal(2029, 1, 1)),
        int(ymd_to_ordinal(2030, 1, 1)),
        int(ymd_to_ordinal(2032, 1, 1)),
        int(ymd_to_ordinal(2035, 1, 1)),
    ], dtype=jnp.int32)
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-(0.03 + 0.004 + 0.002 * times) * times)
    return DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs, reference_date=ref_date
    )


class TestVolSourceProtocol:
    """Every pricer must query the source at the forward it prices with."""

    def test_swaption_source_sees_the_forward_par_swap_rate(
        self, payer_swaption, flat_curve
    ):
        S, ann = swaption_forward_and_annuity(payer_swaption, flat_curve)
        got = swaption_price(payer_swaption, flat_curve, _ForwardProbe(jnp.array(0.20)))
        want = swaption_price_black76(payer_swaption, flat_curve, 0.20 + S)
        assert float(got) == float(want)
        # And the probe genuinely moved the vol, so the test has teeth.
        assert float(S) > 1e-3

    def test_swaption_source_sees_the_dual_curve_forward(
        self, payer_swaption, flat_curve, steep_forward_curve
    ):
        S, _ = swaption_forward_and_annuity(
            payer_swaption, flat_curve, steep_forward_curve
        )
        S_single, _ = swaption_forward_and_annuity(payer_swaption, flat_curve)
        # The basis must actually change the forward handed to the source.
        assert abs(float(S) - float(S_single)) > 1e-4
        got = swaption_price(
            payer_swaption, flat_curve,
            _ForwardProbe(jnp.array(0.20)), steep_forward_curve,
        )
        want = swaption_price_black76(
            payer_swaption, flat_curve, 0.20 + S, steep_forward_curve
        )
        assert float(got) == float(want)

    def test_caplet_source_sees_the_period_forward(self, caplet_3m6m, flat_curve):
        F = caplet_forward(caplet_3m6m, flat_curve)
        got = caplet_price(caplet_3m6m, flat_curve, _ForwardProbe(jnp.array(0.25)))
        want = caplet_price_black76(caplet_3m6m, flat_curve, 0.25 + F)
        assert float(got) == float(want)

    def test_cap_source_sees_each_periods_own_forward(self, cap_2y, flat_curve):
        F = cap_forwards(cap_2y, flat_curve)
        got = cap_price(cap_2y, flat_curve, _ForwardProbe(jnp.array(0.25)))
        want = cap_price_black76(cap_2y, flat_curve, 0.25 + F)
        assert abs(float(got) - float(want)) < 1e-9

    def test_forward_helpers_match_the_pricers_own_projection(
        self, caplet_3m6m, cap_2y, flat_curve, steep_forward_curve
    ):
        # Supplying the helper's forward explicitly must be a no-op against
        # letting the pricer project it itself.
        F = caplet_forward(caplet_3m6m, flat_curve, steep_forward_curve)
        a = caplet_price_black76(
            caplet_3m6m, flat_curve, jnp.array(0.25), steep_forward_curve
        )
        b = caplet_price_black76(
            caplet_3m6m, flat_curve, jnp.array(0.25), steep_forward_curve,
            forward=F,
        )
        assert float(a) == float(b)

        Fs = cap_forwards(cap_2y, flat_curve, steep_forward_curve)
        c = cap_price_black76(
            cap_2y, flat_curve, jnp.array(0.25), steep_forward_curve
        )
        d = cap_price_black76(
            cap_2y, flat_curve, jnp.array(0.25), steep_forward_curve,
            forwards=Fs,
        )
        assert float(c) == float(d)

    def test_shifted_surface_passes_the_forward_through(
        self, payer_swaption, flat_curve
    ):
        probe = _ForwardProbe(jnp.array(0.20))
        shifted = ShiftedVolSurface(base=probe, shift=jnp.array(0.01))
        S, _ = swaption_forward_and_annuity(payer_swaption, flat_curve)
        got = swaption_price(payer_swaption, flat_curve, shifted)
        want = swaption_price_black76(payer_swaption, flat_curve, 0.21 + S)
        assert float(got) == float(want)

    def test_cube_and_surface_accept_but_ignore_the_forward(self, flat_curve):
        # Task-6 contract: the default sticky-absolute-strike convention reads
        # the stored forward, so passing a live one must not change the quote.
        cube = _lognormal_cube()
        a = cube(jnp.array(0.03), jnp.array(1.0), jnp.array(5.0))
        b = cube(
            jnp.array(0.03), jnp.array(1.0), jnp.array(5.0),
            forward=jnp.array(0.055),
        )
        assert float(a) == float(b)

        surf = _lognormal_optionlet_surface()
        c = surf(jnp.array(0.03), jnp.array(1.0))
        d = surf(jnp.array(0.03), jnp.array(1.0), forward=jnp.array(0.055))
        assert float(c) == float(d)

    def test_probe_price_is_jittable(self, payer_swaption, flat_curve):
        price = eqx.filter_jit(swaption_price)(
            payer_swaption, flat_curve, _ForwardProbe(jnp.array(0.20))
        )
        assert jnp.isfinite(price)


class TestJit:
    def test_swaption_jit(self, payer_swaption, flat_curve):
        cube = _lognormal_cube()
        price = eqx.filter_jit(swaption_price)(payer_swaption, flat_curve, cube)
        assert jnp.isfinite(price)

    def test_cap_jit(self, cap_2y, flat_curve):
        surf = _lognormal_optionlet_surface()
        price = eqx.filter_jit(cap_price)(cap_2y, flat_curve, surf)
        assert jnp.isfinite(price)
