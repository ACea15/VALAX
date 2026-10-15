"""CMS-spread swap on the two-factor PDEs — agreement with Monte Carlo.

The CMS-spread swap is the decorrelation-sensitive product that justifies a
second factor, so a PDE that reproduces the Monte-Carlo price (within its
sampling error) is the strongest validation the G2++ and Cheyette-2F grids get.
Each period's coupon is injected at its (snapped) CMS-fixing level as a linear
function of the node state and discounted home in a single backward sweep; the
frozen-``y`` bias (Cheyette-2F) and the O(dt) fixing snap are both absorbed by
the three-standard-error tolerance, mirroring the caplet PDE-vs-MC discipline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import CMSSpreadSwap
from valax.models.cheyette import CheyetteModel2F
from valax.models.g2pp import G2PPModel
from valax.pricing.mc import MCConfig
from valax.pricing.pde import PDEConfig2D, Scheme, registered_recipes
from valax.pricing.reconcile import price_all_engines

_ACT365 = "act_365"
_REF = int(ymd_to_ordinal(2025, 1, 1))
_N_PATHS = 120_000
_N_STEPS = 60

C2 = PDEConfig2D(n_x=81, n_y=81, n_time=100, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def curve():
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(26)], dtype=jnp.int32
    )
    times = (pillars - _REF).astype(jnp.float64) / 365.0
    # Upward-sloping curve so the 10Y rate exceeds the 2Y rate.
    zero = 0.02 + 0.02 * (1.0 - jnp.exp(-0.2 * times))
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero * times),
        reference_date=jnp.int32(_REF),
        day_count=_ACT365,
    )


def _g2pp(curve, rho=-0.5):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.012),
        volatility_ratio=jnp.array(0.75),
        correlation=jnp.array(rho),
        initial_curve=curve,
    )


def _cheyette_2f(curve, rho=-0.5):
    # Zero skew keeps the frozen-y bias small so PDE and exact-y MC agree
    # within Monte-Carlo error; the decorrelation still comes from rho.
    return CheyetteModel2F(
        mean_reversion_1=jnp.array(0.50),
        mean_reversion_2=jnp.array(0.10),
        volatility_1=jnp.array(0.012),
        volatility_ratio=jnp.array(0.009) / jnp.array(0.012),
        skew_1=jnp.array(0.0),
        skew_2=jnp.array(0.0),
        correlation=jnp.array(rho),
        initial_curve=curve,
    )


def _swap(pay_fixed=True, fixed_rate=0.0):
    pay = jnp.array(
        [int(ymd_to_ordinal(2026 + i, 1, 1)) for i in range(5)], dtype=jnp.int32
    )
    return CMSSpreadSwap(
        start_date=jnp.int32(_REF),
        payment_dates=pay,
        fixed_rate=jnp.array(fixed_rate),
        notional=jnp.array(1e6),
        cms_tenor_long=10,
        cms_tenor_short=2,
        pay_fixed=pay_fixed,
        day_count=_ACT365,
    )


# ── Registration ─────────────────────────────────────────────────────


class TestRegistration:
    @pytest.mark.parametrize("model", ["G2PPModel", "CheyetteModel2F"])
    def test_recipe_registered(self, model):
        assert ("CMSSpreadSwap", model) in registered_recipes()


# ── PDE vs Monte Carlo ───────────────────────────────────────────────


class TestCMSSpreadVsMonteCarlo:
    def test_g2pp_cms_spread_pde_vs_mc(self, curve):
        table = price_all_engines(
            _swap(),
            _g2pp(curve),
            pde_config=C2,
            mc_config=MCConfig(n_paths=_N_PATHS, n_steps=_N_STEPS),
            mc_key=jax.random.PRNGKey(0),
        )
        diff = abs(table.price("pde") - table.price("mc"))
        assert diff < 3.0 * table.quotes["mc"].stderr

    def test_cheyette2f_cms_spread_pde_vs_mc(self, curve):
        table = price_all_engines(
            _swap(),
            _cheyette_2f(curve),
            pde_config=C2,
            mc_config=MCConfig(n_paths=_N_PATHS, n_steps=_N_STEPS),
            mc_key=jax.random.PRNGKey(0),
        )
        diff = abs(table.price("pde") - table.price("mc"))
        assert diff < 3.0 * table.quotes["mc"].stderr
