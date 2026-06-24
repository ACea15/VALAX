"""Unit tests for ``SLVModel`` (``valax/models/slv.py``).

Test layout (SLV-1 — model pytree acceptance):

* ``TestSLVModel``  — ``from_heston_and_leverage`` round-trip,
  field-order stability, pytree flatten/unflatten.
* ``TestSLVGreeks`` — ``jax.grad`` flows through ``LeverageGrid.values``
  and through Heston block (``xi``); ``eqx.tree_at`` patches ``kappa``
  without breaking the pytree shape.
* ``TestX64Guard``  — ``RuntimeError`` if ``jax_enable_x64`` is off
  (mirrors ``TestX64Guard`` in ``tests/test_pricing/test_dupire.py``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.models import HestonModel, SLVModel
from valax.surfaces import LeverageGrid, SVIVolSurface


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def heston_model():
    """Typical equity Heston parameters (Feller-violating, like real markets)."""
    return HestonModel(
        v0=jnp.array(0.04),
        kappa=jnp.array(2.0),
        theta=jnp.array(0.04),
        xi=jnp.array(0.5),
        rho=jnp.array(-0.7),
        rate=jnp.array(0.03),
        dividend=jnp.array(0.01),
    )


@pytest.fixture
def flat_surface():
    """Flat 20%-vol SVI surface (Heston-limit warm start needs *some* surface)."""
    sigma = 0.20
    expiries = jnp.array([0.05, 0.5, 1.0, 2.0])
    return SVIVolSurface(
        expiries=expiries,
        forwards=jnp.full_like(expiries, 100.0),
        a_vec=sigma ** 2 * expiries,
        b_vec=jnp.zeros_like(expiries),
        rho_vec=jnp.zeros_like(expiries),
        m_vec=jnp.zeros_like(expiries),
        sigma_vec=jnp.full_like(expiries, 0.1),
    )


@pytest.fixture
def flat_leverage():
    """L ≡ 1 on a modest 5×4 grid (pure-Heston limit)."""
    return LeverageGrid.flat(
        log_moneyness_grid=jnp.linspace(-0.3, 0.3, 5),
        time_grid=jnp.linspace(0.1, 1.0, 4),
        value=1.0,
    )


@pytest.fixture
def slv_model(heston_model, flat_surface, flat_leverage):
    return SLVModel.from_heston_and_leverage(
        heston_model, flat_surface, flat_leverage,
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Model construction + pytree
# ─────────────────────────────────────────────────────────────────────


class TestSLVModel:
    def test_from_heston_and_leverage_roundtrip(
        self, heston_model, flat_surface, flat_leverage,
    ):
        """All Heston fields copied verbatim from the input ``HestonModel``."""
        slv = SLVModel.from_heston_and_leverage(
            heston_model, flat_surface, flat_leverage,
        )
        assert float(slv.v0) == float(heston_model.v0)
        assert float(slv.kappa) == float(heston_model.kappa)
        assert float(slv.theta) == float(heston_model.theta)
        assert float(slv.xi) == float(heston_model.xi)
        assert float(slv.rho) == float(heston_model.rho)
        assert float(slv.rate) == float(heston_model.rate)
        assert float(slv.dividend) == float(heston_model.dividend)
        # Surface and leverage are stored by reference (no deep-copy).
        assert slv.surface is flat_surface
        assert slv.leverage is flat_leverage

    def test_pytree_flatten_unflatten(self, slv_model):
        """JAX pytree round-trip preserves the model exactly."""
        leaves, treedef = jax.tree_util.tree_flatten(slv_model)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        # Round-trip is bit-equal across every scalar field.
        for field in ("v0", "kappa", "theta", "xi", "rho", "rate", "dividend"):
            assert jnp.allclose(
                getattr(slv_model, field), getattr(reconstructed, field),
                atol=0.0,
            )
        # Leverage values bit-equal.
        assert jnp.allclose(
            slv_model.leverage.values,
            reconstructed.leverage.values,
            atol=0.0,
        )

    def test_eqx_tree_at_on_kappa(self, slv_model):
        """``eqx.tree_at`` patches ``kappa`` in-place pytree-wise."""
        new_kappa = jnp.array(3.5)
        slv2 = eqx.tree_at(lambda m: m.kappa, slv_model, new_kappa)
        assert float(slv2.kappa) == 3.5
        # Other fields untouched.
        assert float(slv2.v0) == float(slv_model.v0)
        assert float(slv2.xi) == float(slv_model.xi)
        # Pytree structure preserved.
        _, td1 = jax.tree_util.tree_flatten(slv_model)
        _, td2 = jax.tree_util.tree_flatten(slv2)
        assert td1 == td2


# ─────────────────────────────────────────────────────────────────────
# 2. Greeks (autodiff through SLV pytree)
# ─────────────────────────────────────────────────────────────────────


class TestSLVGreeks:
    def test_grad_through_leverage_values(self, slv_model):
        """``jax.grad`` flows through ``model.leverage.values``.

        The gradient of any function of L should be non-zero at the
        four nodes enclosing the query point (bilinear basis support).
        """
        from valax.pricing.mc.slv_paths import generate_slv_paths

        spot = jnp.array(100.0)
        T = 1.0
        key = jax.random.PRNGKey(20260101)

        def terminal_mean(lev_values):
            slv = eqx.tree_at(
                lambda m: m.leverage.values, slv_model, lev_values,
            )
            S, _V = generate_slv_paths(
                slv, spot, T, n_steps=20, n_paths=500, key=key,
            )
            return jnp.mean(S[:, -1])

        g = jax.grad(terminal_mean)(slv_model.leverage.values)
        assert g.shape == slv_model.leverage.values.shape
        assert jnp.all(jnp.isfinite(g))
        # The mean terminal spot is non-trivially sensitive to leverage
        # (variance enters the drift via the −½·σ² Itô term, so even
        # the *mean* moves under higher leverage).
        assert float(jnp.max(jnp.abs(g))) > 0.0

    def test_grad_through_heston_xi(self, slv_model):
        """``jax.grad`` flows through Heston ``xi`` to terminal variance."""
        from valax.pricing.mc.slv_paths import generate_slv_paths

        spot = jnp.array(100.0)
        T = 1.0
        key = jax.random.PRNGKey(20260102)

        def terminal_var_mean(xi_val):
            slv = eqx.tree_at(lambda m: m.xi, slv_model, xi_val)
            _S, V = generate_slv_paths(
                slv, spot, T, n_steps=20, n_paths=2000, key=key,
            )
            return jnp.mean(V[:, -1])

        g = float(jax.grad(terminal_var_mean)(slv_model.xi))
        # E[V_T] under Heston is independent of xi at the population
        # level (mean-reverting OU on V — exact mean is
        # θ + (v0 − θ)·exp(-κT)). But the *sample* mean of a finite-
        # path MC has a non-zero finite-sample dependence on xi via
        # the noise terms. We assert finiteness — not magnitude.
        assert jnp.isfinite(g)


# ─────────────────────────────────────────────────────────────────────
# 3. x64 guard (mirrors ``tests/test_pricing/test_dupire.py::TestX64Guard``)
# ─────────────────────────────────────────────────────────────────────


class TestX64Guard:
    """RuntimeError if jax_enable_x64 is disabled at construction time."""

    def test_x64_disabled_raises(self, heston_model, flat_surface, flat_leverage):
        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(RuntimeError, match="jax_enable_x64"):
                SLVModel.from_heston_and_leverage(
                    heston_model, flat_surface, flat_leverage,
                )
        finally:
            jax.config.update("jax_enable_x64", True)
