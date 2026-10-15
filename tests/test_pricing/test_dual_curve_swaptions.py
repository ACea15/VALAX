"""Dual-curve (OIS discounting + tenor projection) analytic swaptions.

Hull-White and G2++ are exact-fit-to-*one*-curve models, so "dual-curve" here
means the frozen deterministic tenor-basis overlay of
:mod:`valax.pricing.multicurve`: discount on the model's own curve, project the
floating leg off a separate curve, and treat the basis as deterministic.  The
whole extension is that the floating leg is worth ``float_start`` rather than
par at expiry, which moves the Jamshidian critical rate (one factor) and the
``y``-boundary plus strike leg of Brigo-Mercurio eq. (4.31) (two factors).

Gates, mirroring the ones that guarded the MC/PDE overlay:

1. **Single-curve non-regression (hard gate).**  Passing no projection curve,
   or passing the discount curve itself, must reproduce the pre-existing number
   *exactly* — the branch is taken at trace time, so this is exact equality and
   not a tolerance.
2. **Payer-minus-receiver parity** equals the dual-curve forward swap value.
   Parity is model-independent, so it isolates the discount/projection split
   from anything the model does — the sharpest available oracle.
3. **Independent engine agreement**: the closed form matches the P5.1
   dual-curve PDE recipe, which implements the same overlay by a completely
   different route (finite differences on the payoff).
4. **Basis monotonicity** and collapse onto the single-curve price as the basis
   vanishes.

``day_count="act_365"`` throughout: the short-rate machinery builds its time
axis from the instrument's day count while the curve uses its own, so matching
them keeps the curve-level oracles exact (a pre-existing convention mismatch,
orthogonal to multi-curve).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import InterestRateSwap, Swaption
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.analytic.swaptions import swap_price
from valax.pricing.pde import PDEConfig
from valax.pricing.pde.config import PDEConfig2D
from valax.pricing.pde.dispatch import pde_price_dispatch


REF = ymd_to_ordinal(2025, 1, 1)
_CONFIG_1F = PDEConfig(n_spot=301, n_time=300, spot_range=6.0)
_CONFIG_2F = PDEConfig2D(n_x=61, n_y=61, n_time=60, x_range=5.0)


def _flat_curve(rate: float, n_years: int = 21) -> DiscountCurve:
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(REF)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=REF,
        day_count="act_365",
    )


@pytest.fixture(scope="module")
def ois() -> DiscountCurve:
    return _flat_curve(0.03)


@pytest.fixture(scope="module")
def projection() -> DiscountCurve:
    """Tenor curve 50 bp above OIS — a realistic IBOR/OIS basis."""
    return _flat_curve(0.035)


@pytest.fixture(scope="module")
def hw(ois) -> HullWhiteModel:
    return HullWhiteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.01),
        initial_curve=ois,
    )


@pytest.fixture(scope="module")
def g2pp(ois) -> G2PPModel:
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        volatility_x=jnp.array(0.010),
        mean_reversion_y=jnp.array(0.10),
        volatility_ratio=jnp.array(0.6),
        correlation=jnp.array(-0.70),
        initial_curve=ois,
    )


def _swaption(
    is_payer: bool = True,
    strike: float = 0.032,
    expiry_years: int = 2,
    tail_years: int = 5,
) -> Swaption:
    """A ``expiry_years`` into ``tail_years`` annual swaption."""
    return Swaption(
        expiry_date=jnp.int32(int(ymd_to_ordinal(2025 + expiry_years, 1, 1))),
        fixed_dates=jnp.array(
            [
                int(ymd_to_ordinal(2025 + expiry_years + i, 1, 1))
                for i in range(1, tail_years + 1)
            ],
            dtype=jnp.int32,
        ),
        strike=jnp.array(strike),
        notional=jnp.array(10_000_000.0),
        is_payer=is_payer,
        day_count="act_365",
    )


def _forward_swap(strike: float = 0.032) -> InterestRateSwap:
    """The swap underlying ``_swaption()`` — same schedule and strike."""
    sw = _swaption(strike=strike)
    return InterestRateSwap(
        start_date=sw.expiry_date,
        fixed_dates=sw.fixed_dates,
        fixed_rate=jnp.array(strike),
        notional=jnp.array(10_000_000.0),
        day_count="act_365",
    )


_PRICERS = (
    pytest.param(hw_swaption_price, id="hull_white"),
    pytest.param(g2pp_swaption_price, id="g2pp"),
)


def _model_for(pricer, hw, g2pp):
    return hw if pricer is hw_swaption_price else g2pp


# ── Gate 1: single-curve non-regression ───────────────────────────────

@pytest.mark.parametrize("pricer", _PRICERS)
class TestSingleCurveNonRegression:
    @pytest.mark.parametrize("is_payer", [True, False])
    @pytest.mark.parametrize("strike", [0.02, 0.032, 0.05])
    @pytest.mark.parametrize("expiry_years,tail_years", [(1, 3), (2, 5), (5, 10)])
    def test_no_projection_curve_is_exactly_single_curve(
        self, pricer, hw, g2pp, is_payer, strike, expiry_years, tail_years
    ):
        """``forward_curve=None`` must be the pre-existing arithmetic."""
        model = _model_for(pricer, hw, g2pp)
        sw = _swaption(is_payer, strike, expiry_years, tail_years)
        explicit = pricer(sw, model, forward_curve=None)
        # The basis is a trace-time branch, so this is exact, not approximate.
        assert float(pricer(sw, model)) == float(explicit)

    def test_discount_curve_as_projection_is_single_curve(
        self, pricer, hw, g2pp, ois
    ):
        """Naming the discount curve as projection changes nothing exactly."""
        model = _model_for(pricer, hw, g2pp)
        sw = _swaption()
        assert float(pricer(sw, model, forward_curve=ois)) == float(
            pricer(sw, model)
        )

    def test_zero_basis_curve_collapses_onto_single_curve(
        self, pricer, hw, g2pp
    ):
        """A *distinct object* with identical DFs gives a zero basis."""
        model = _model_for(pricer, hw, g2pp)
        sw = _swaption()
        same_level = _flat_curve(0.03)  # equal values, different object
        dual = float(pricer(sw, model, forward_curve=same_level))
        single = float(pricer(sw, model))
        assert dual == pytest.approx(single, rel=1e-12)


# ── Gate 2: parity is the dual-curve forward swap (model-independent) ──

@pytest.mark.parametrize("pricer", _PRICERS)
class TestPayerReceiverParity:
    def test_parity_equals_dual_curve_forward_swap(
        self, pricer, hw, g2pp, ois, projection
    ):
        """payer - receiver = forward swap value, computed off the two curves.

        This holds for *any* model, so it pins the discount/projection split
        without relying on the dynamics at all.
        """
        model = _model_for(pricer, hw, g2pp)
        payer = pricer(_swaption(True), model, forward_curve=projection)
        receiver = pricer(_swaption(False), model, forward_curve=projection)
        oracle = swap_price(_forward_swap(), ois, projection)
        assert float(payer - receiver) == pytest.approx(float(oracle), rel=1e-8)

    def test_single_curve_parity_still_holds(self, pricer, hw, g2pp, ois):
        model = _model_for(pricer, hw, g2pp)
        payer = pricer(_swaption(True), model)
        receiver = pricer(_swaption(False), model)
        oracle = swap_price(_forward_swap(), ois, None)
        assert float(payer - receiver) == pytest.approx(float(oracle), rel=1e-8)


# ── Gate 3: agreement with the independent dual-curve PDE recipe ───────

class TestPDEAgreement:
    @pytest.mark.parametrize("is_payer", [True, False])
    def test_hull_white_matches_pde(self, hw, ois, projection, is_payer):
        sw = _swaption(is_payer)
        analytic = hw_swaption_price(sw, hw, forward_curve=projection)
        pde = pde_price_dispatch(
            sw, hw, _CONFIG_1F, discount_curve=ois, forward_curve=projection
        ).price
        assert float(analytic) == pytest.approx(float(pde), rel=2e-4)

    @pytest.mark.parametrize("is_payer", [True, False])
    def test_g2pp_matches_pde(self, g2pp, ois, projection, is_payer):
        sw = _swaption(is_payer)
        analytic = g2pp_swaption_price(sw, g2pp, forward_curve=projection)
        pde = pde_price_dispatch(
            sw, g2pp, _CONFIG_2F, discount_curve=ois, forward_curve=projection
        ).price
        # Coarse 61x61x60 mesh; the receiver is the smaller number so its
        # relative tolerance is looser.
        assert float(analytic) == pytest.approx(float(pde), rel=2e-3)


# ── Gate 4: behaviour in the basis ────────────────────────────────────

@pytest.mark.parametrize("pricer", _PRICERS)
class TestBasisBehaviour:
    def test_payer_rises_monotonically_with_basis(self, pricer, hw, g2pp):
        """Projection above OIS lifts the forwards *and* keeps the higher OIS
        discount factors, so both effects push a payer the same way."""
        model = _model_for(pricer, hw, g2pp)
        sw = _swaption(True)
        prices = [
            float(pricer(sw, model, forward_curve=_flat_curve(0.03 + b)))
            for b in (0.0, 0.0005, 0.0025, 0.005, 0.01)
        ]
        assert all(b > a for a, b in zip(prices, prices[1:]))

    def test_receiver_falls_monotonically_with_basis(self, pricer, hw, g2pp):
        model = _model_for(pricer, hw, g2pp)
        sw = _swaption(False)
        prices = [
            float(pricer(sw, model, forward_curve=_flat_curve(0.03 + b)))
            for b in (0.0, 0.0005, 0.0025, 0.005, 0.01)
        ]
        assert all(b < a for a, b in zip(prices, prices[1:]))

    def test_adjusted_cashflows_stay_positive(self, pricer, hw, g2pp):
        """The Jamshidian/boundary validity condition, checked explicitly.

        The basis shifts each flow by ``1 - b``; the decomposition needs every
        adjusted flow to stay positive so the coupon bond stays monotone.
        """
        from valax.pricing.multicurve import apply_swap_basis, swap_leg_basis

        sw = _swaption(strike=0.02)  # low strike = smallest flows
        basis = swap_leg_basis(
            _flat_curve(0.03), _flat_curve(0.04),  # a wide 100 bp basis
            sw.expiry_date, sw.fixed_dates,
        )
        taus = jnp.diff(
            jnp.concatenate([sw.expiry_date[None], sw.fixed_dates])
        ) / 365.0
        cashflows = (sw.strike * taus).at[-1].add(1.0)
        float_start, adjusted = apply_swap_basis(cashflows, basis)
        assert float(float_start) > 0.0
        assert bool(jnp.all(adjusted > 0.0))


# ── JAX behaviour ─────────────────────────────────────────────────────

@pytest.mark.parametrize("pricer", _PRICERS)
class TestJax:
    def test_filter_jit_smoke(self, pricer, hw, g2pp, projection):
        model = _model_for(pricer, hw, g2pp)
        sw = _swaption()

        @eqx.filter_jit
        def priced(m, fc):
            return pricer(sw, m, forward_curve=fc)

        assert jnp.isfinite(priced(model, projection))

    def test_grad_flows_to_the_projection_curve(
        self, pricer, hw, g2pp, projection
    ):
        """Basis risk is differentiable: the price responds to projection DFs.

        This is what makes a dual-curve calibration's Jacobian exact autodiff
        rather than a bump.
        """
        model = _model_for(pricer, hw, g2pp)
        sw = _swaption()

        def price_of(dfs):
            fc = eqx.tree_at(lambda c: c.discount_factors, projection, dfs)
            return pricer(sw, model, forward_curve=fc)

        grad = jax.grad(price_of)(projection.discount_factors)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.sum(jnp.abs(grad))) > 0.0
