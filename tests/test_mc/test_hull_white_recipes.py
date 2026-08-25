"""Hull-White Monte-Carlo dispatcher recipes.

Covers the four ``(bond, HullWhiteModel)`` recipes registered in
``valax.pricing.mc.recipes``:

- ``FixedRateBond``    — validated against the analytic curve PV.
- ``FloatingRateBond`` — validated against the par-at-reset identity.
- ``CallableBond``     — triangulated against the Hull-White trinomial tree.
- ``PuttableBond``     — triangulated against the tree, plus put-parity bounds.

The callable/puttable triangulation is the load-bearing test here: MC and the
trinomial tree are independent numerical methods sharing only the analytic
Hull-White ZCB, so agreement pins down the exercise conventions (ex-coupon
strike, coupon paid on the exercise date) on both sides at once.

Also asserts the JAX-transform contract advertised by
``valax.pricing.mc.dispatch``: every built-in recipe must survive
``eqx.filter_jit`` with the instrument *and* model passed as traced pytrees.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import (
    CallableBond,
    FixedRateBond,
    FloatingRateBond,
    PuttableBond,
)
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.lattice.hull_white_tree import (
    callable_bond_price,
    hw_tree_j_max,
    puttable_bond_price,
)
from valax.pricing.mc.dispatch import MCConfig, mc_price_dispatch

_DAY_COUNT = "act_365"
_MAT_YEARS = 5
_COUPON = 0.05
_FACE = 100.0
_RATE = 0.05
_TREE_STEPS = 400


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ref_date() -> int:
    return int(ymd_to_ordinal(2026, 1, 1))


def _flat_curve(ref: int, rate: float, n_years: int) -> DiscountCurve:
    """Flat continuously-compounded curve, anchored at t=0 with df=1."""
    years = [0.0] + [float(k) for k in range(1, n_years + 2)]
    pillars = jnp.array(
        [ref + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - ref).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(rate) * times),
        reference_date=jnp.int32(ref),
        day_count=_DAY_COUNT,
    )


@pytest.fixture(scope="module")
def curve(ref_date) -> DiscountCurve:
    return _flat_curve(ref_date, _RATE, _MAT_YEARS)


@pytest.fixture(scope="module")
def model(curve) -> HullWhiteModel:
    return HullWhiteModel(
        mean_reversion=jnp.asarray(0.10),
        volatility=jnp.asarray(0.01),
        initial_curve=curve,
    )


@pytest.fixture(scope="module")
def payment_dates(ref_date):
    return jnp.array(
        [ref_date + int(round(k * 365)) for k in range(1, _MAT_YEARS + 1)],
        dtype=jnp.int32,
    )


def _bond_kwargs(ref_date, payment_dates) -> dict:
    return dict(
        payment_dates=payment_dates,
        settlement_date=jnp.int32(ref_date),
        coupon_rate=jnp.asarray(_COUPON),
        face_value=jnp.asarray(_FACE),
        frequency=1,
        day_count=_DAY_COUNT,
    )


@pytest.fixture(scope="module")
def fixed_bond(ref_date, payment_dates) -> FixedRateBond:
    return FixedRateBond(**_bond_kwargs(ref_date, payment_dates))


@pytest.fixture(scope="module")
def floating_bond(ref_date, payment_dates) -> FloatingRateBond:
    return FloatingRateBond(
        payment_dates=payment_dates,
        fixing_dates=payment_dates,
        settlement_date=jnp.int32(ref_date),
        spread=jnp.asarray(0.0),
        face_value=jnp.asarray(_FACE),
        frequency=1,
        day_count=_DAY_COUNT,
    )


@pytest.fixture(scope="module")
def callable_bond_1cd(ref_date, payment_dates) -> CallableBond:
    """Single call date — the myopic exercise policy is *exact* here."""
    return CallableBond(
        **_bond_kwargs(ref_date, payment_dates),
        call_dates=jnp.array([ref_date + 2 * 365], dtype=jnp.int32),
        call_prices=jnp.ones(1),
    )


@pytest.fixture(scope="module")
def callable_bond_2cd(ref_date, payment_dates) -> CallableBond:
    return CallableBond(
        **_bond_kwargs(ref_date, payment_dates),
        call_dates=jnp.array(
            [ref_date + 2 * 365, ref_date + 3 * 365], dtype=jnp.int32
        ),
        call_prices=jnp.ones(2),
    )


@pytest.fixture(scope="module")
def puttable_bond_1pd(ref_date, payment_dates) -> PuttableBond:
    return PuttableBond(
        **_bond_kwargs(ref_date, payment_dates),
        put_dates=jnp.array([ref_date + 2 * 365], dtype=jnp.int32),
        put_prices=jnp.ones(1),
    )


@pytest.fixture(scope="module")
def mc_config() -> MCConfig:
    return MCConfig(n_paths=40_000, n_steps=60)


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(20260101)


# ── Straight bonds vs analytic ────────────────────────────────────────

class TestStraightBondRecipes:
    def test_fixed_bond_matches_analytic(self, fixed_bond, model, curve, mc_config, key):
        """Path-wise MC discounting must reproduce the analytic curve PV."""
        res = mc_price_dispatch(fixed_bond, model, mc_config, key)
        analytic = float(fixed_rate_bond_price(fixed_bond, curve))
        assert abs(float(res.price) - analytic) < 2.0 * float(res.stderr)

    def test_floating_bond_prices_to_par(self, floating_bond, model, mc_config, key):
        """A zero-spread FRN valued at a reset date is worth par exactly.

        The replication collapses to ``face * D(0, t_settle)`` with
        ``t_settle = 0``, so this holds path-by-path with no MC error.
        """
        res = mc_price_dispatch(floating_bond, model, mc_config, key)
        assert float(res.price) == pytest.approx(_FACE, abs=1e-9)
        assert float(res.stderr) == pytest.approx(0.0, abs=1e-9)


# ── Callable / puttable triangulated against the trinomial tree ───────

class TestTreeTriangulation:
    """MC vs trinomial tree — independent methods, same conventions."""

    def test_callable_single_call_matches_tree(
        self, callable_bond_1cd, model, mc_config, key
    ):
        """With one call date the analytic continuation value is exact.

        Any residual gap is pure MC noise, so this is the sharpest available
        check that both engines agree on the ex-coupon exercise convention.
        """
        res = mc_price_dispatch(callable_bond_1cd, model, mc_config, key)
        tree = float(callable_bond_price(callable_bond_1cd, model, n_steps=_TREE_STEPS))
        assert abs(float(res.price) - tree) < 2.0 * float(res.stderr)

    def test_puttable_single_put_matches_tree(
        self, puttable_bond_1pd, model, mc_config, key
    ):
        res = mc_price_dispatch(puttable_bond_1pd, model, mc_config, key)
        tree = float(puttable_bond_price(puttable_bond_1pd, model, n_steps=_TREE_STEPS))
        assert abs(float(res.price) - tree) < 2.0 * float(res.stderr)

    def test_multi_call_mc_is_upper_bound(
        self, callable_bond_2cd, model, mc_config, key
    ):
        """With several call dates the MC policy is myopic, hence suboptimal.

        The issuer minimises the holder's value, so any suboptimal exercise
        policy over-values the bond: MC >= tree.  The gap measures the option
        value of deferring exercise that the myopic rule discards.
        """
        res = mc_price_dispatch(callable_bond_2cd, model, mc_config, key)
        tree = float(callable_bond_price(callable_bond_2cd, model, n_steps=_TREE_STEPS))
        assert float(res.price) >= tree - 2.0 * float(res.stderr)
        # ... but still close: the myopic policy is a good approximation.
        assert float(res.price) - tree < 0.5


class TestOptionalityBounds:
    """No-arbitrage ordering against the straight bond."""

    def test_callable_below_straight(
        self, callable_bond_1cd, fixed_bond, model, curve, mc_config, key
    ):
        called = float(mc_price_dispatch(callable_bond_1cd, model, mc_config, key).price)
        straight = float(fixed_rate_bond_price(fixed_bond, curve))
        assert called < straight

    def test_puttable_above_straight(
        self, puttable_bond_1pd, fixed_bond, model, curve, mc_config, key
    ):
        put = float(mc_price_dispatch(puttable_bond_1pd, model, mc_config, key).price)
        straight = float(fixed_rate_bond_price(fixed_bond, curve))
        assert put > straight


# ── JAX transform contract ────────────────────────────────────────────

class TestJITAndGrad:
    """``dispatch.py`` advertises that all built-in recipes are jit/grad-able."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["fixed_bond", "floating_bond", "callable_bond_2cd", "puttable_bond_1pd"],
    )
    def test_filter_jit_with_traced_instrument(
        self, fixture_name, request, model, mc_config, key
    ):
        """Regression: instrument *and* model as traced pytrees.

        Previously ``int(model.initial_curve.reference_date)`` and
        ``float(instrument.face_value)`` raised ``ConcretizationTypeError``.
        """
        instrument = request.getfixturevalue(fixture_name)

        @eqx.filter_jit
        def price(inst, mdl, k):
            return mc_price_dispatch(inst, mdl, mc_config, k).price

        jitted = float(price(instrument, model, key))
        eager = float(mc_price_dispatch(instrument, model, mc_config, key).price)
        assert jitted == pytest.approx(eager, rel=1e-10)

    @pytest.mark.parametrize(
        "fixture_name", ["fixed_bond", "callable_bond_2cd", "puttable_bond_1pd"]
    )
    def test_grad_wrt_model_volatility(
        self, fixture_name, request, model, mc_config, key
    ):
        instrument = request.getfixturevalue(fixture_name)
        grads = eqx.filter_grad(
            lambda m: mc_price_dispatch(instrument, m, mc_config, key).price
        )(model)
        assert jnp.isfinite(grads.volatility)
        assert jnp.isfinite(grads.mean_reversion)

    def test_vega_signs(
        self, callable_bond_2cd, puttable_bond_1pd, model, mc_config, key
    ):
        """Optionality vega: the call hurts the holder, the put helps."""
        def dv(inst):
            return float(
                eqx.filter_grad(
                    lambda m: mc_price_dispatch(inst, m, mc_config, key).price
                )(model).volatility
            )

        assert dv(callable_bond_2cd) < 0.0
        assert dv(puttable_bond_1pd) > 0.0

    def test_grad_wrt_instrument(self, fixed_bond, model, mc_config, key):
        """Instrument leaves must stay traceable (no ``float(...)`` casts)."""
        grads = eqx.filter_grad(
            lambda inst: mc_price_dispatch(inst, model, mc_config, key).price
        )(fixed_bond)
        # d(price)/d(face_value) ~ PV of one unit of notional, just under 1.
        assert float(grads.face_value) == pytest.approx(1.0, abs=0.05)
        assert float(grads.coupon_rate) > 0.0

    def test_exercise_indicator_is_smooth(
        self, callable_bond_2cd, model, mc_config, key
    ):
        """Sigmoid smoothing gives the call strike a non-zero gradient.

        A hard Heaviside indicator is zero almost everywhere, so this
        derivative would vanish identically.
        """
        grads = eqx.filter_grad(
            lambda inst: mc_price_dispatch(inst, model, mc_config, key).price
        )(callable_bond_2cd)
        assert jnp.all(jnp.abs(grads.call_prices) > 1e-8)


# ── Tree half-width helper ────────────────────────────────────────────

class TestHWTreeJMax:
    def test_matches_hull_white_1994_bound(self):
        """j_max = max(ceil(0.1835 / (a*dt)), 1) — Hull & White (1994), §3."""
        for a, dt in [(0.10, 0.025), (0.05, 0.05), (0.01, 0.01), (5.0, 0.5)]:
            j = hw_tree_j_max(a, dt)
            assert isinstance(j, int)
            assert j == max(math.ceil(0.1835 / (a * dt)), 1)

    def test_j_max_is_at_least_one(self):
        """Degenerate (very fast mean reversion) still yields a usable tree."""
        assert hw_tree_j_max(100.0, 1.0) == 1

    def test_explicit_j_max_matches_default(self, callable_bond_1cd, model):
        j = hw_tree_j_max(0.10, _MAT_YEARS / _TREE_STEPS)
        auto = float(callable_bond_price(callable_bond_1cd, model, n_steps=_TREE_STEPS))
        explicit = float(
            callable_bond_price(
                callable_bond_1cd, model, n_steps=_TREE_STEPS, j_max=j
            )
        )
        assert explicit == pytest.approx(auto, rel=1e-12)

    def test_explicit_j_max_enables_filter_grad(self, callable_bond_1cd, model):
        """The whole point of the kwarg: hoist the shape decision out of trace.

        Without a concrete ``j_max``, ``build_hull_white_tree`` calls
        ``float(a)`` to size the lattice, which fails under tracing.
        """
        j = hw_tree_j_max(0.10, _MAT_YEARS / _TREE_STEPS)

        grads = eqx.filter_grad(
            lambda m: callable_bond_price(
                callable_bond_1cd, m, n_steps=_TREE_STEPS, j_max=j
            )
        )(model)
        assert jnp.isfinite(grads.volatility)
        assert float(grads.volatility) < 0.0  # call optionality hurts the holder

        with pytest.raises(jax.errors.ConcretizationTypeError):
            eqx.filter_grad(
                lambda m: callable_bond_price(
                    callable_bond_1cd, m, n_steps=_TREE_STEPS
                )
            )(model)
