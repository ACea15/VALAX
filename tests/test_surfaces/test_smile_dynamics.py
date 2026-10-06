"""Smile-dynamics conventions on the rates vol objects.

Covers the three conventions of
``valax.surfaces.dynamics`` as carried by :class:`SwaptionCube` and
:class:`OptionletVolSurface`: that the default (`sticky_strike`) preserves the
pre-existing numbers exactly, that each branch evaluates the Hagan expansion at
the documented ``(strike, forward)`` pair, that the two forward-dependent
branches collapse onto sticky strike when the curve has not moved, and that a
contradictory ``(dynamics, quote_form)`` pair is rejected at construction.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.dates.schedule import generate_schedule
from valax.instruments.rates import Caplet, Swaption
from valax.pricing.analytic.rates_smile import caplet_price, swaption_price
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity
from valax.surfaces import OptionletVolSurface, SwaptionCube
from valax.surfaces.dynamics import (
    QUOTE_FORMS,
    SMILE_DYNAMICS,
    resolve_smile_query,
)


# ── Fixtures ──────────────────────────────────────────────────────────

#: The fitted forward every node of the test cube/surface carries.
F0 = 0.030

#: A live forward distinctly away from ``F0`` so the conventions separate.
F_LIVE = 0.042

#: Query strike, deliberately away from both forwards.
STRIKE = 0.035

#: Query coordinate: on a node of the grids below, so no interpolation error.
EXPIRY = 1.0
TENOR = 5.0


def _cube(**kwargs) -> SwaptionCube:
    shape = (4, 3)
    base = dict(
        expiries=jnp.array([0.5, 1.0, 2.0, 5.0]),
        tenors=jnp.array([1.0, 5.0, 10.0]),
        forwards=jnp.full(shape, F0),
        alphas=jnp.full(shape, 0.22),
        betas=jnp.full(shape, 0.5),
        rhos=jnp.full(shape, -0.25),
        nus=jnp.full(shape, 0.4),
    )
    base.update(kwargs)
    return SwaptionCube(**base)


def _surface(**kwargs) -> OptionletVolSurface:
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0])
    n = expiries.shape[0]
    base = dict(
        expiries=expiries,
        forwards=jnp.full(n, F0),
        alphas=jnp.full(n, 0.22),
        betas=jnp.full(n, 0.5),
        rhos=jnp.full(n, -0.25),
        nus=jnp.full(n, 0.4),
    )
    base.update(kwargs)
    return OptionletVolSurface(**base)


@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    pillars = jnp.array([
        int(ymd_to_ordinal(y, 1, 1))
        for y in (2025, 2026, 2027, 2028, 2029, 2030, 2032, 2035)
    ], dtype=jnp.int32)
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-0.03 * times)
    return DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs, reference_date=ref_date
    )


@pytest.fixture
def payer_swaption():
    """1y x 5y payer swaption."""
    return Swaption(
        expiry_date=ymd_to_ordinal(2026, 1, 1),
        fixed_dates=generate_schedule(
            2026, 1, 1, 2031, 1, 1, frequency=1
        ).payment_dates,
        strike=jnp.array(STRIKE),
        notional=jnp.array(1_000_000.0),
        is_payer=True,
    )


# ── Configuration validation ──────────────────────────────────────────

class TestConfiguration:
    def test_default_is_sticky_absolute_strike(self):
        cube = _cube()
        surf = _surface()
        assert cube.dynamics == "sticky_strike"
        assert cube.quote_form == "absolute_strike"
        assert surf.dynamics == "sticky_strike"
        assert surf.quote_form == "absolute_strike"

    @pytest.mark.parametrize("quote_form,dynamics", [
        ("absolute_strike", "sticky_strike"),
        ("absolute_strike", "sabr_backbone"),
        ("atm_offset", "sticky_moneyness"),
        ("atm_offset", "sabr_backbone"),
    ])
    def test_consistent_pairs_are_accepted(self, quote_form, dynamics):
        _cube(dynamics=dynamics, quote_form=quote_form)
        _surface(dynamics=dynamics, quote_form=quote_form)

    @pytest.mark.parametrize("quote_form,dynamics", [
        ("absolute_strike", "sticky_moneyness"),
        ("atm_offset", "sticky_strike"),
    ])
    def test_contradictory_pairs_are_rejected(self, quote_form, dynamics):
        with pytest.raises(ValueError, match="contradict each other"):
            _cube(dynamics=dynamics, quote_form=quote_form)
        with pytest.raises(ValueError, match="contradict each other"):
            _surface(dynamics=dynamics, quote_form=quote_form)

    def test_unknown_dynamics_is_rejected(self):
        with pytest.raises(ValueError, match="dynamics must be one of"):
            _cube(dynamics="sticky_delta")

    def test_unknown_quote_form_is_rejected(self):
        with pytest.raises(ValueError, match="quote_form must be one of"):
            _cube(quote_form="log_moneyness")

    def test_every_dynamics_is_consistent_with_some_quote_form(self):
        # Guards the CONSISTENT_DYNAMICS table against a new convention being
        # added to SMILE_DYNAMICS without a quote form that admits it.
        from valax.surfaces.dynamics import CONSISTENT_DYNAMICS

        reachable = set()
        for form in QUOTE_FORMS:
            reachable.update(CONSISTENT_DYNAMICS[form])
        assert reachable == set(SMILE_DYNAMICS)


# ── The three conventions evaluate where the table says ───────────────

class TestConventionsOnTheCube:
    def test_sticky_strike_ignores_the_live_forward(self):
        cube = _cube()
        a = cube(jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR))
        b = cube(
            jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR),
            forward=jnp.array(F_LIVE),
        )
        assert float(a) == float(b)

    def test_sticky_strike_matches_hagan_at_the_fitted_forward(self):
        cube = _cube()
        model = cube.model_at(jnp.array(EXPIRY), jnp.array(TENOR))
        want = sabr_implied_vol(
            model, jnp.array(F0), jnp.array(STRIKE), jnp.array(EXPIRY)
        )
        got = cube(
            jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR),
            forward=jnp.array(F_LIVE),
        )
        assert float(got) == float(want)

    def test_sabr_backbone_matches_hagan_at_the_live_forward(self):
        cube = _cube(dynamics="sabr_backbone")
        model = cube.model_at(jnp.array(EXPIRY), jnp.array(TENOR))
        want = sabr_implied_vol(
            model, jnp.array(F_LIVE), jnp.array(STRIKE), jnp.array(EXPIRY)
        )
        got = cube(
            jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR),
            forward=jnp.array(F_LIVE),
        )
        assert float(got) == float(want)

    def test_sticky_moneyness_shifts_the_strike_by_the_forward_move(self):
        cube = _cube(dynamics="sticky_moneyness", quote_form="atm_offset")
        model = cube.model_at(jnp.array(EXPIRY), jnp.array(TENOR))
        want = sabr_implied_vol(
            model,
            jnp.array(F0),
            jnp.array(STRIKE - (F_LIVE - F0)),
            jnp.array(EXPIRY),
        )
        got = cube(
            jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR),
            forward=jnp.array(F_LIVE),
        )
        assert float(got) == float(want)

    def test_the_three_conventions_genuinely_disagree(self):
        args = (jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR))
        kw = dict(forward=jnp.array(F_LIVE))
        sticky = float(_cube()(*args, **kw))
        backbone = float(_cube(dynamics="sabr_backbone")(*args, **kw))
        moneyness = float(
            _cube(dynamics="sticky_moneyness", quote_form="atm_offset")(
                *args, **kw
            )
        )
        assert abs(sticky - backbone) > 1e-4
        assert abs(sticky - moneyness) > 1e-4
        assert abs(backbone - moneyness) > 1e-4

    @pytest.mark.parametrize("dynamics,quote_form", [
        ("sabr_backbone", "absolute_strike"),
        ("sticky_moneyness", "atm_offset"),
    ])
    def test_collapses_onto_sticky_strike_when_the_curve_has_not_moved(
        self, dynamics, quote_form
    ):
        # F == F0: all three conventions must agree, which is the
        # self-consistency prerequisite documented for sabr_backbone.
        args = (jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR))
        want = float(_cube()(*args))
        got = float(
            _cube(dynamics=dynamics, quote_form=quote_form)(
                *args, forward=jnp.array(F0)
            )
        )
        assert got == pytest.approx(want, abs=1e-12)

    @pytest.mark.parametrize("dynamics,quote_form", [
        ("sabr_backbone", "absolute_strike"),
        ("sticky_moneyness", "atm_offset"),
    ])
    def test_forward_dependent_dynamics_require_a_forward(
        self, dynamics, quote_form
    ):
        cube = _cube(dynamics=dynamics, quote_form=quote_form)
        with pytest.raises(ValueError, match="queried without one"):
            cube(jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR))

    def test_normal_branch_honours_the_dynamics_too(self):
        cube = _cube(
            dynamics="sabr_backbone",
            betas=jnp.full((4, 3), 0.0),
            alphas=jnp.full((4, 3), 0.01),
            is_normal=True,
        )
        model = cube.model_at(jnp.array(EXPIRY), jnp.array(TENOR))
        want = sabr_normal_implied_vol(
            model, jnp.array(F_LIVE), jnp.array(STRIKE), jnp.array(EXPIRY),
            cube.shift,
        )
        got = cube(
            jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR),
            forward=jnp.array(F_LIVE),
        )
        assert float(got) == float(want)

    @pytest.mark.parametrize("dynamics,quote_form", [
        ("sticky_strike", "absolute_strike"),
        ("sabr_backbone", "absolute_strike"),
        ("sticky_moneyness", "atm_offset"),
    ])
    def test_every_branch_is_jittable(self, dynamics, quote_form):
        cube = _cube(dynamics=dynamics, quote_form=quote_form)

        @eqx.filter_jit
        def read(c, k, t, n, f):
            return c(k, t, n, forward=f)

        v = read(
            cube, jnp.array(STRIKE), jnp.array(EXPIRY), jnp.array(TENOR),
            jnp.array(F_LIVE),
        )
        assert jnp.isfinite(v)


class TestConventionsOnTheOptionletSurface:
    def test_sticky_strike_ignores_the_live_forward(self):
        surf = _surface()
        a = surf(jnp.array(STRIKE), jnp.array(EXPIRY))
        b = surf(jnp.array(STRIKE), jnp.array(EXPIRY), forward=jnp.array(F_LIVE))
        assert float(a) == float(b)

    def test_sabr_backbone_matches_hagan_at_the_live_forward(self):
        surf = _surface(dynamics="sabr_backbone")
        model = surf.model_at(jnp.array(EXPIRY))
        want = sabr_implied_vol(
            model, jnp.array(F_LIVE), jnp.array(STRIKE), jnp.array(EXPIRY)
        )
        got = surf(jnp.array(STRIKE), jnp.array(EXPIRY), forward=jnp.array(F_LIVE))
        assert float(got) == float(want)

    def test_sticky_moneyness_shifts_the_strike_by_the_forward_move(self):
        surf = _surface(dynamics="sticky_moneyness", quote_form="atm_offset")
        model = surf.model_at(jnp.array(EXPIRY))
        want = sabr_implied_vol(
            model,
            jnp.array(F0),
            jnp.array(STRIKE - (F_LIVE - F0)),
            jnp.array(EXPIRY),
        )
        got = surf(jnp.array(STRIKE), jnp.array(EXPIRY), forward=jnp.array(F_LIVE))
        assert float(got) == float(want)

    def test_forward_dependent_dynamics_require_a_forward(self):
        surf = _surface(dynamics="sabr_backbone")
        with pytest.raises(ValueError, match="queried without one"):
            surf(jnp.array(STRIKE), jnp.array(EXPIRY))


# ── The convention reaches the price, and the curve reaches the vol ───

class TestDynamicsReachThePrice:
    def test_backbone_cube_prices_differently_from_sticky_cube(
        self, payer_swaption, flat_curve
    ):
        # A *stale* cube: fitted at F_LIVE while the curve now implies ~3%.
        # Without that gap the conventions coincide and the test is vacuous.
        stale = jnp.full((4, 3), F_LIVE)
        S, _ = swaption_forward_and_annuity(payer_swaption, flat_curve)
        assert abs(float(S) - F_LIVE) > 1e-3

        sticky = float(
            swaption_price(payer_swaption, flat_curve, _cube(forwards=stale))
        )
        backbone = float(
            swaption_price(
                payer_swaption, flat_curve,
                _cube(forwards=stale, dynamics="sabr_backbone"),
            )
        )
        assert abs(sticky - backbone) > 1.0

    def test_sticky_strike_vol_has_no_curve_sensitivity(
        self, payer_swaption, flat_curve
    ):
        def vol_of_curve(dfs, cube):
            curve = eqx.tree_at(lambda c: c.discount_factors, flat_curve, dfs)
            S, _ = swaption_forward_and_annuity(payer_swaption, curve)
            T = year_fraction(
                curve.reference_date, payer_swaption.expiry_date,
                payer_swaption.day_count,
            )
            tenor = year_fraction(
                payer_swaption.expiry_date, payer_swaption.fixed_dates[-1],
                payer_swaption.day_count,
            )
            return cube(payer_swaption.strike, T, tenor, forward=S)

        g_sticky = jax.grad(vol_of_curve)(flat_curve.discount_factors, _cube())
        assert float(jnp.max(jnp.abs(g_sticky))) == 0.0

        g_backbone = jax.grad(vol_of_curve)(
            flat_curve.discount_factors, _cube(dynamics="sabr_backbone")
        )
        assert float(jnp.max(jnp.abs(g_backbone))) > 1e-6

    def test_caplet_surface_dynamics_reach_the_price(self, flat_curve):
        caplet = Caplet(
            fixing_date=ymd_to_ordinal(2026, 1, 1),
            start_date=ymd_to_ordinal(2026, 1, 1),
            end_date=ymd_to_ordinal(2026, 4, 1),
            strike=jnp.array(STRIKE),
            notional=jnp.array(1_000_000.0),
            is_cap=True,
        )
        # Stale surface: fitted at F_LIVE, curve now implies ~3%.
        stale = jnp.full(5, F_LIVE)
        sticky = float(
            caplet_price(caplet, flat_curve, _surface(forwards=stale))
        )
        backbone = float(
            caplet_price(
                caplet, flat_curve,
                _surface(forwards=stale, dynamics="sabr_backbone"),
            )
        )
        assert abs(sticky - backbone) > 1.0


# ── The shared resolver itself ────────────────────────────────────────

class TestResolveSmileQuery:
    def test_returns_the_documented_pairs(self):
        k, f0, f = jnp.array(STRIKE), jnp.array(F0), jnp.array(F_LIVE)

        ks, a = resolve_smile_query("sticky_strike", k, f0, f, owner="X")
        assert (float(ks), float(a)) == (STRIKE, F0)

        ks, a = resolve_smile_query("sabr_backbone", k, f0, f, owner="X")
        assert (float(ks), float(a)) == (STRIKE, F_LIVE)

        ks, a = resolve_smile_query("sticky_moneyness", k, f0, f, owner="X")
        assert float(ks) == pytest.approx(STRIKE - (F_LIVE - F0))
        assert float(a) == F0

    def test_rejects_an_unknown_convention(self):
        with pytest.raises(ValueError, match="must be one of"):
            resolve_smile_query(
                "sticky_delta", jnp.array(STRIKE), jnp.array(F0),
                jnp.array(F_LIVE), owner="X",
            )
