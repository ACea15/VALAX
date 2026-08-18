"""Model -> PDE operator coefficient adapters.

Maps a VALAX model onto the drift/diffusion/discount coefficients that
:func:`~valax.pricing.pde.operators.build_operator_1d` expects. PR-1 covers the
Black-Scholes model in log-spot space; local volatility (Dupire) adds a
*time-dependent* operator via :func:`lv_operator_stack`. Further models
(Heston, SLV, Hull-White) are added in later phases.
"""

import jax
import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import Float

from valax.models.black_scholes import BlackScholesModel
from valax.models.local_vol import LocalVolModel
from valax.pricing.analytic.dupire import dupire_local_vol
from valax.pricing.pde.grids import Grid1D
from valax.pricing.pde.operators import Operator1D, build_operator_1d


def bs_operator(model: BlackScholesModel, grid: Grid1D) -> Operator1D:
    """Build the log-spot Black-Scholes operator for ``model`` on ``grid``.

    In log-spot space ``x = ln S`` the drift is ``r - q - sigma^2 / 2``, the
    diffusion coefficient is ``sigma^2`` and the discount rate is ``r``.

    Args:
        model: Black-Scholes model parameters.
        grid: The log-spot grid.

    Returns:
        The assembled :class:`Operator1D`.
    """
    drift = model.rate - model.dividend - 0.5 * model.vol**2
    diffusion = model.vol**2
    return build_operator_1d(grid, drift=drift, diffusion=diffusion, discount=model.rate)


def lv_operator_stack(
    model: LocalVolModel,
    grid: Grid1D,
    spot: Float[Array, ""],
    *,
    expiry: Float[Array, ""],
    n_time: int,
) -> Operator1D:
    """Build the *time-dependent* local-volatility operator stack.

    Under Dupire's local-vol dynamics the log-spot ``x = ln S`` diffusion in
    the pricing PDE

    .. math::

        V_\\tau = \\tfrac{1}{2}\\sigma_{loc}^2(x, t)\\,V_{xx}
            + \\left(r - q - \\tfrac{1}{2}\\sigma_{loc}^2(x, t)\\right) V_x - r V

    varies with both space *and* time, so the operator must be rebuilt at every
    backward time level. This returns an :class:`Operator1D` whose three bands
    are **stacked** to shape ``(n_time, n)`` — row ``m`` is the operator used at
    backward step ``m`` — which :func:`~valax.pricing.pde.schemes.solve_backward_1d`
    consumes directly.

    The local variance is sampled at the **midpoint in time** of each step:
    backward step ``m`` spans forward time ``[m·dt, (m+1)·dt]`` and the operator
    row uses ``sigma_loc(x, (m + 1/2)·dt)``. This mirrors the midpoint-in-time
    convention of :func:`~valax.pricing.mc.local_vol_paths.generate_local_vol_paths`
    and avoids querying the Dupire formula at its singular ``t = 0`` boundary.

    Log-moneyness follows the same convention as the LV Monte-Carlo generator:
    at forward time ``t`` a node ``x`` maps to ``k = x - (ln S_0 + mu·t)`` with
    ``mu = r - q`` and ``S_0`` the current spot, i.e. ``k = ln(S / F(t))`` with
    ``F(t) = S_0 exp(mu·t)``. ``spot`` is detached from autodiff
    (:func:`jax.lax.stop_gradient`) here so the operator — a numerical scaffold
    — does not co-move with spot under differentiation; the differentiable spot
    dependence of the price lives solely in the read-off query (matching the
    Black-Scholes recipe's clean second-order spot Greek).

    Args:
        model: Local-vol model carrying the surface and rate / dividend.
        grid: The log-spot grid.
        spot: Current spot ``S_0``, used only to place the forward curve
            ``F(t) = S_0 exp((r - q) t)`` for the log-moneyness conversion.
        expiry: Time to expiry ``T`` (``dt = T / n_time``).
        n_time: Number of backward time steps (== number of stacked rows).

    Returns:
        An :class:`Operator1D` with bands of shape ``(n_time, n)``.
    """
    dt = expiry / n_time
    mu = model.rate - model.dividend
    log_spot = jnp.log(lax.stop_gradient(spot))
    surface = model.surface
    nodes = grid.nodes

    # Midpoint forward times for backward steps m = 0 .. n_time - 1:
    # (m + 1/2)·dt. Row m of the returned stack is consumed at step m.
    times = (jnp.arange(n_time) + 0.5) * dt

    def _row_operator(t: Float[Array, ""]) -> Operator1D:
        # k = ln(S / F(t)) = x - (ln S_0 + mu·t), matching the LV MC generator.
        k = nodes - (log_spot + mu * t)
        sigma = jax.vmap(lambda kk: dupire_local_vol(surface, kk, t))(k)
        sig2 = sigma * sigma
        drift = mu - 0.5 * sig2
        return build_operator_1d(
            grid, drift=drift, diffusion=sig2, discount=model.rate
        )

    # vmap over time levels -> Operator1D with (n_time, n) bands.
    return jax.vmap(_row_operator)(times)
