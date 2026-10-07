"""Smile-dynamics conventions on the rates vol objects.

Covers the three conventions of
``valax.surfaces.dynamics`` as carried by :class:`SwaptionCube` and
:class:`OptionletVolSurface`: that the default (`sticky_strike`) preserves the
pre-existing numbers exactly, that each branch evaluates the Hagan expansion at
the documented ``(strike, forward)`` pair, that the two forward-dependent
branches collapse onto sticky strike when the curve has not moved, and that a
contradictory ``(dynamics, quote_form)`` pair is rejected at construction.

:class:`TestT12SmileDynamicsDelta` at the end of this file is the **T-12**
conformance gate (``pytest -m audit``): the convention is invisible in the
price and observable only in the delta, so it is checked there, against
partials taken from the Hagan expansion directly.
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
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.pricing.analytic.swaptions import (
    swaption_black76_from_forward,
    swaption_forward_and_annuity,
)
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


# ── T-12: the conformance gate ────────────────────────────────────────

#: SABR parameters giving an economically plausible smile: alpha chosen so
#: the lognormal vol at ``T12_STRIKE`` is ~29%, rather than the ~123% the
#: convention fixtures above carry (they only ever compare branches against
#: each other, so their level is irrelevant; here the *delta* is the output,
#: and a nonsense vol would give a nonsense one).
T12_ALPHA, T12_BETA, T12_RHO, T12_NU = 0.052, 0.5, -0.25, 0.4

#: Annuity held fixed. T-12 is about the smile-dynamics component of delta,
#: so the annuity is a constant here; the curve-driven part of delta is
#: gated by T-2 in ``tests/test_greeks/test_quote_adjoint.py``.
T12_ANNUITY = 4.5
T12_NOTIONAL = 1_000_000.0

#: Chain-rule and convention tolerance. Observed: the `dsigma/dF` identities
#: hold to *exactly* zero relative error, and the chain rule to ~1e-16
#: relative, so this is three orders of slack.
T12_RTOL = 1e-12

#: The one consistent ``quote_form`` for each convention -- which is itself
#: the point of spec Q-3: the dynamics is a *consequence* of what the quote
#: holds fixed, so the cube rejects a contradictory pair at construction.
T12_QUOTE_FORM = {
    "sticky_strike": "absolute_strike",
    "sabr_backbone": "absolute_strike",
    "sticky_moneyness": "atm_offset",
}


def _t12_cube(dynamics: str, **overrides) -> SwaptionCube:
    shape = (4, 3)
    params = dict(
        alpha=T12_ALPHA, beta=T12_BETA, rho=T12_RHO, nu=T12_NU
    )
    params.update(overrides)
    return SwaptionCube(
        expiries=jnp.array([0.5, 1.0, 2.0, 5.0]),
        tenors=jnp.array([1.0, 5.0, 10.0]),
        forwards=jnp.full(shape, F0),
        alphas=jnp.full(shape, params["alpha"]),
        betas=jnp.full(shape, params["beta"]),
        rhos=jnp.full(shape, params["rho"]),
        nus=jnp.full(shape, params["nu"]),
        dynamics=dynamics,
        quote_form=T12_QUOTE_FORM[dynamics],
    )


def _t12_swaption() -> Swaption:
    ref = int(ymd_to_ordinal(2025, 1, 1))
    return Swaption(
        expiry_date=jnp.int32(ref + 365),
        fixed_dates=jnp.array(
            [ref + 365 * (k + 2) for k in range(5)], jnp.int32
        ),
        strike=jnp.asarray(STRIKE),
        notional=jnp.asarray(T12_NOTIONAL),
        day_count="act_365",
    )


def _t12_price(forward, vol):
    """Black-76 payer value from an explicit forward and vol."""
    return swaption_black76_from_forward(
        _t12_swaption(),
        forward,
        jnp.asarray(T12_ANNUITY),
        vol,
        jnp.asarray(EXPIRY),
    )


def _t12_vol(cube, forward):
    return cube(
        jnp.asarray(STRIKE), jnp.asarray(EXPIRY), jnp.asarray(TENOR), forward=forward
    )


def _t12_decomposition(cube):
    """``(vol, dsigma/dF, total delta, frozen-vol delta, vega)`` at ``F == F0``.

    Evaluated *at the fitted forward*, which is the sharp point of the
    whole gate: there the three conventions agree on the vol, so they
    agree on the price, and differ only in its derivative.
    """
    forward = jnp.asarray(F0)
    vol = _t12_vol(cube, forward)
    dsigma_df = jax.grad(lambda f: _t12_vol(cube, f))(forward)
    total = jax.grad(lambda f: _t12_price(f, _t12_vol(cube, f)))(forward)
    frozen = jax.grad(lambda f: _t12_price(f, vol))(forward)
    vega = jax.grad(lambda v: _t12_price(forward, v))(vol)
    return (
        float(vol),
        float(dsigma_df),
        float(total),
        float(frozen),
        float(vega),
    )


def _t12_sabr_partials():
    """``(dsigma/dK, dsigma/dF)`` of the Hagan expansion, computed directly.

    The independent reference the conventions are checked against: taken
    from :func:`sabr_implied_vol` itself rather than from the cube, so a
    cube that applied the wrong convention cannot also define what the
    right answer is.
    """
    model = SABRModel(
        alpha=jnp.array(T12_ALPHA),
        beta=jnp.array(T12_BETA),
        rho=jnp.array(T12_RHO),
        nu=jnp.array(T12_NU),
    )
    expiry = jnp.asarray(EXPIRY)
    d_strike = jax.grad(
        lambda k: sabr_implied_vol(model, jnp.asarray(F0), k, expiry)
    )(jnp.asarray(STRIKE))
    d_forward = jax.grad(
        lambda f: sabr_implied_vol(model, f, jnp.asarray(STRIKE), expiry)
    )(jnp.asarray(F0))
    return float(d_strike), float(d_forward)


@pytest.mark.audit
class TestT12SmileDynamicsDelta:
    r"""T-12: delta of an OTM swaption, with the vol quotes held as defined.

    Verifies the three conventions of specification § 5.5 against
    requirement **V-5** ("the smile dynamics implied by how ``q_v`` is held
    fixed is documented and matches the risk policy").  The documented
    convention is owned by
    ``docs/guide/vol-surfaces.md`` § Smile dynamics; this gate checks the
    code against it in the only place the choice is observable.

    **Why delta, and only delta.**  Evaluated at the fitted forward
    :math:`F_0`, all three conventions return the *same volatility* --
    they differ in how that volatility responds to a forward move, not in
    its level.  So the three price the swaption identically and are
    indistinguishable from any repricing test.  The convention is a
    statement about :math:`d\sigma/dF`, and delta is the first quantity
    that can see it.  That is exactly why V-5 is a documentation
    requirement rather than a correctness one: there is no "right" answer
    to test against, only consistency with a stated policy.

    The identity checked, from spec § 5.5:

    .. math::

        \frac{dV}{dF} = \underbrace{\frac{\partial V}{\partial F}}_{\text{frozen vol}}
            + \underbrace{\frac{\partial V}{\partial\sigma}}_{\text{vega}}
              \cdot \frac{d\sigma}{dF},

    with, at :math:`F = F_0`,

    ========================  ==================================
    ``dynamics``              :math:`d\sigma/dF`
    ========================  ==================================
    ``sticky_strike``         :math:`0` exactly
    ``sticky_moneyness``      :math:`-\partial\sigma/\partial K`
    ``sabr_backbone``         :math:`\partial\sigma/\partial F`
    ========================  ==================================

    The right-hand column comes from :func:`sabr_implied_vol` directly,
    not from the cube, so the object under test does not get to define
    what the right answer is.

    Tolerance: ``rtol = 1e-12``; ``sticky_strike``'s zero is asserted as
    **exact**.  Observed: the three convention identities hold to exactly
    zero relative error and the chain rule to ~1e-16 relative.

    The strike is **out of the money** (3.5% against a 3.0% forward), as
    the specification requires, and that is load-bearing rather than
    incidental -- see :meth:`test_a_flat_smile_hides_the_whole_distinction`.
    """

    def test_the_chain_rule_holds_for_every_convention(self):
        """Total delta decomposes into frozen-vol delta plus vega x dsigma/dF."""
        for dynamics in SMILE_DYNAMICS:
            _, dsigma_df, total, frozen, vega = _t12_decomposition(
                _t12_cube(dynamics)
            )
            assert total == pytest.approx(
                frozen + vega * dsigma_df, rel=T12_RTOL
            ), dynamics

    def test_sticky_strike_has_no_smile_contribution(self):
        """Vol at an absolute strike is invariant, so dsigma/dF is *exactly* 0.

        Asserted as exact equality, not a tolerance: the live forward is
        not consulted at all on this branch, so its cotangent is
        structurally zero rather than numerically small.  A tolerance here
        would mask the branch starting to read the forward.
        """
        _, dsigma_df, total, frozen, _ = _t12_decomposition(
            _t12_cube("sticky_strike")
        )
        assert dsigma_df == 0.0
        assert total == frozen

    def test_sticky_moneyness_delta_carries_minus_the_skew(self):
        """The smile shifts with the forward, so dsigma/dF = -dsigma/dK."""
        d_strike, _ = _t12_sabr_partials()
        _, dsigma_df, _, _, _ = _t12_decomposition(
            _t12_cube("sticky_moneyness")
        )
        assert dsigma_df == pytest.approx(-d_strike, rel=T12_RTOL)

    def test_sabr_backbone_delta_carries_the_backbone_derivative(self):
        """Parameters frozen, smile re-evaluated at the live forward."""
        _, d_forward = _t12_sabr_partials()
        _, dsigma_df, _, _, _ = _t12_decomposition(
            _t12_cube("sabr_backbone")
        )
        assert dsigma_df == pytest.approx(d_forward, rel=T12_RTOL)

    def test_the_price_cannot_tell_the_conventions_apart(self):
        """All three agree on the vol at F0, hence on the price.

        The complement of the delta tests, and the reason this gate is
        about delta: a repricing check could never detect a wrong
        convention.
        """
        price = {
            dynamics: float(
                _t12_price(
                    jnp.asarray(F0),
                    _t12_vol(_t12_cube(dynamics), jnp.asarray(F0)),
                )
            )
            for dynamics in SMILE_DYNAMICS
        }
        values = list(price.values())
        assert values[0] == pytest.approx(values[1], rel=T12_RTOL)
        assert values[1] == pytest.approx(values[2], rel=T12_RTOL)

    def test_the_three_deltas_differ_materially(self):
        """Guards the gate against passing on a technicality.

        If the conventions produced near-identical deltas, every
        assertion above would hold while the choice carried no risk
        consequence, and V-5 would be documentation about nothing.  The
        measured spread is several percent of delta on an OTM payer.
        """
        deltas = {
            dynamics: _t12_decomposition(_t12_cube(dynamics))[2]
            for dynamics in SMILE_DYNAMICS
        }
        sticky = deltas["sticky_strike"]
        for dynamics in ("sticky_moneyness", "sabr_backbone"):
            relative = abs(deltas[dynamics] - sticky) / abs(sticky)
            assert relative > 0.02, (dynamics, relative)

    def test_the_documented_ordering_holds_for_a_negative_skew(self):
        """Which convention reports the larger delta, and why.

        With ``rho < 0`` the smile falls in the strike, so
        ``dsigma/dK < 0``.  Sticky moneyness therefore picks up
        ``-dsigma/dK > 0`` and, vega being positive, reports a delta
        *above* sticky strike; the SABR backbone derivative is negative
        at this strike and reports one *below*.  Pinned because the
        ordering is the practical consequence a desk picks a convention
        for -- it is not an accident of the fixture.
        """
        d_strike, d_forward = _t12_sabr_partials()
        assert d_strike < 0.0
        assert d_forward < 0.0

        deltas = {
            dynamics: _t12_decomposition(_t12_cube(dynamics))[2]
            for dynamics in SMILE_DYNAMICS
        }
        assert (
            deltas["sabr_backbone"]
            < deltas["sticky_strike"]
            < deltas["sticky_moneyness"]
        )

    def test_a_flat_smile_hides_the_whole_distinction(self):
        """Why the fixture must be skewed and convex, and the strike OTM.

        With ``nu = 0``, ``beta = 1`` and ``rho = 0`` the Hagan expansion
        is constant in both strike and forward, so *every* convention
        gives ``dsigma/dF == 0`` and all three deltas coincide.  Run on
        such a fixture the entire gate above would pass while testing
        nothing at all -- the recurring failure mode in this suite, where
        a degenerate configuration makes a real difference invisible.
        This test pins the degeneracy explicitly so the next reader knows
        the smile parameters are load-bearing.
        """
        flat = dict(nu=0.0, beta=1.0, rho=0.0)
        deltas = []
        for dynamics in SMILE_DYNAMICS:
            _, dsigma_df, total, frozen, _ = _t12_decomposition(
                _t12_cube(dynamics, **flat)
            )
            assert dsigma_df == 0.0, dynamics
            assert total == frozen, dynamics
            deltas.append(total)
        assert deltas[0] == pytest.approx(deltas[1], rel=T12_RTOL)
        assert deltas[1] == pytest.approx(deltas[2], rel=T12_RTOL)

    def test_the_strike_is_out_of_the_money(self):
        """The specification asks for an OTM swaption; pin that it is one."""
        assert STRIKE > F0
