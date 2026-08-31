"""Model -> PDE operator coefficient adapters.

Maps a VALAX model onto the drift/diffusion/discount coefficients that
:func:`~valax.pricing.pde.operators.build_operator_1d` (1-D) and
:func:`~valax.pricing.pde.operators2d.build_operator_2d` (2-D) expect. PR-1
covers the Black-Scholes model in log-spot space; local volatility (Dupire)
adds a *time-dependent* 1-D operator via :func:`lv_operator_stack`; the Heston
stochastic-volatility model adds a 2-D (ADI) operator via
:func:`heston_operator_2d`; Hull-White adds the short-rate operator stack
:func:`hw_operator_stack`, whose *discount* coefficient is the state variable
itself; and the G2++ two-factor Gaussian short-rate model adds the 2-D operator
:func:`g2pp_operator_2d`, whose discount is the *sum* of the two state
variables ``x + y``. Further models (SLV) are added later.
"""

import jax
import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import Float

from valax.models.black_scholes import BlackScholesModel
from valax.models.g2pp import G2PPModel
from valax.models.heston import HestonModel
from valax.models.hull_white import HullWhiteModel, hw_alpha_average
from valax.models.local_vol import LocalVolModel
from valax.pricing.analytic.dupire import dupire_local_vol
from valax.pricing.pde.grids import Grid1D, Grid2D
from valax.pricing.pde.operators import Operator1D, build_operator_1d
from valax.pricing.pde.operators2d import Operator2D, build_operator_2d


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


def hw_operator_stack(
    model: HullWhiteModel,
    grid: Grid1D,
    *,
    expiry: Float[Array, ""],
    n_time: int,
) -> Operator1D:
    r"""Build the *time-dependent* Hull-White operator stack in state space.

    The solver works in the **centred state variable** ``x`` of the Hull-White
    decomposition :math:`r(t) = x(t) + \alpha(t)` (see
    :func:`~valax.models.hull_white.hw_alpha`), where

    .. math::

        dx(t) = -a\,x(t)\,dt + \sigma\,dW(t), \qquad x(0) = 0 .

    Working in ``x`` rather than ``r`` is what makes the mesh tractable: the
    drift and diffusion are then *time-independent*, the state starts exactly
    at the origin, and the entire dependence on the initial curve is pushed
    into the scalar shift :math:`\alpha(t)`. The backward pricing equation for
    a value function :math:`V(x, \tau)` with :math:`\tau = T - t` is

    .. math::

        V_\tau = \tfrac{1}{2}\sigma^2 V_{xx} - a x\,V_x
                 - \bigl(x + \alpha(t)\bigr) V ,

    so only the **discount** coefficient varies, and it varies in *both* space
    and time — it is the short rate itself. That is why this returns a stack:
    the bands have shape ``(n_time, n)``, row ``m`` being the operator used at
    backward step ``m`` (nearest expiry first), exactly the layout
    :func:`~valax.pricing.pde.schemes.solve_backward_1d` consumes.

    :math:`\alpha` is **exactly averaged** across each step rather than sampled
    at its midpoint: backward step ``m`` spans forward time
    ``[k dt, (k+1) dt]`` with ``k = n_time - m - 1``, and the row uses
    :func:`~valax.models.hull_white.hw_alpha_average` over that interval. The
    midpoint convention used by :func:`lv_operator_stack` is *not* good enough
    here, because a log-linear discount curve has a piecewise-constant
    instantaneous forward that jumps at each pillar; midpoint sampling stalls
    the scheme's time convergence at a ~4e-6 bond-repricing error, while exact
    averaging restores clean second-order behaviour and, with it, Hull-White's
    defining exact fit to the initial curve.

    Unlike the equity recipes nothing is detached from autodiff here: the grid
    is anchored at the origin rather than at a market quote (see
    :func:`~valax.pricing.pde.grids.centred_state_grid`), so ``a`` and
    ``sigma`` stay fully differentiable through both the coefficients and the
    mesh — which is what makes the PDE usable inside a calibration objective.

    Args:
        model: Hull-White model carrying ``a``, ``sigma`` and the initial curve.
        grid: The state-variable (``x``) grid.
        expiry: Horizon ``T`` in year fractions (``dt = T / n_time``).
        n_time: Number of backward time steps (== number of stacked rows).

    Returns:
        An :class:`~valax.pricing.pde.operators.Operator1D` with bands of shape
        ``(n_time, n)``.
    """
    a = model.mean_reversion
    sigma = model.volatility
    x = grid.nodes

    dt = expiry / n_time
    # Backward step m (m = 0 is nearest expiry) spans forward levels
    # k = n_time - m - 1 to k + 1.
    k = n_time - jnp.arange(n_time) - 1
    alphas = hw_alpha_average(model, k * dt, (k + 1) * dt)  # (n_time,)

    drift = -a * x
    diffusion = sigma**2

    def _row_operator(alpha: Float[Array, ""]) -> Operator1D:
        return build_operator_1d(
            grid, drift=drift, diffusion=diffusion, discount=x + alpha
        )

    return jax.vmap(_row_operator)(alphas)


def heston_operator_2d(model: HestonModel, grid: Grid2D) -> Operator2D:
    r"""Build the 2-D Heston ADI operator on a log-spot :math:`\times` variance grid.

    In log-spot ``x = ln S`` and variance ``v`` the Heston pricing PDE is

    .. math::

        V_\tau = \tfrac{1}{2} v\, V_{xx} + \rho \xi v\, V_{xv}
            + \tfrac{1}{2} \xi^2 v\, V_{vv}
            + \left(r - q - \tfrac{1}{2} v\right) V_x
            + \kappa(\theta - v) V_v - r V,

    whose coefficients are **independent of ``x``** (Heston is log-spot
    translation-invariant) and depend only on the variance ``v``. They map onto
    :func:`~valax.pricing.pde.operators2d.build_operator_2d` as

    - ``diff_x = v``               (coefficient of ``V_xx``),
    - ``drift_x = r - q - v/2``    (coefficient of ``V_x``),
    - ``diff_v = xi^2 v``          (coefficient of ``V_vv``),
    - ``drift_v = kappa(theta - v)`` (coefficient of ``V_v``),
    - ``mixed = rho xi v``         (coefficient of ``V_xv``),
    - ``discount = r``.

    Because the coefficients do not involve spot, no ``stop_gradient`` is needed
    here: the differentiable-spot dependence of the price lives entirely in the
    grid placement (the log-spot axis of ``grid`` is already spot-detached by
    :func:`~valax.pricing.pde.grids.uniform_log_spot_grid`) and in the read-off
    query, exactly as in the 1-D recipes.

    Args:
        model: Heston model parameters.
        grid: The tensor-product log-spot :math:`\times` variance grid.

    Returns:
        The assembled :class:`~valax.pricing.pde.operators2d.Operator2D`.
    """
    v = grid.y.nodes[jnp.newaxis, :]  # (1, n_y); coefficients vary in v only
    mu = model.rate - model.dividend

    diff_x = v
    drift_x = mu - 0.5 * v
    diff_v = model.xi**2 * v
    drift_v = model.kappa * (model.theta - v)
    mixed = model.rho * model.xi * v

    return build_operator_2d(
        grid,
        diff_x=diff_x,
        drift_x=drift_x,
        diff_v=diff_v,
        drift_v=drift_v,
        mixed=mixed,
        discount=model.rate,
    )


def g2pp_operator_2d(model: G2PPModel, grid: Grid2D) -> Operator2D:
    r"""Build the 2-D G2++ ADI operator on a centred ``(x, y)`` factor grid.

    The solver works in the two **centred** Gaussian factors of the G2++
    decomposition :math:`r(t) = x(t) + y(t) + \varphi(t)` (see
    :func:`~valax.models.g2pp.g2pp_phi`), where

    .. math::

        dx(t) = -a\,x(t)\,dt + \sigma\,dW_1(t), \qquad
        dy(t) = -b\,y(t)\,dt + \eta\,dW_2(t), \qquad
        dW_1\,dW_2 = \rho\,dt .

    A claim value :math:`V(t, x, y)` satisfies the backward equation

    .. math::

        V_\tau = \tfrac{1}{2}\sigma^2 V_{xx} + \tfrac{1}{2}\eta^2 V_{yy}
            + \rho\sigma\eta\, V_{xy}
            - a x\, V_x - b y\, V_y
            - \bigl(x + y + \varphi(t)\bigr) V ,

    which maps onto :func:`~valax.pricing.pde.operators2d.build_operator_2d` as

    - ``diff_x = sigma^2``          (coefficient of ``V_xx``),
    - ``drift_x = -a x``            (coefficient of ``V_x``),
    - ``diff_v = eta^2``            (coefficient of ``V_yy``),
    - ``drift_v = -b y``            (coefficient of ``V_y``),
    - ``mixed = rho sigma eta``     (coefficient of ``V_xy``; a **constant**),
    - ``discount = x + y``          (the **state** part of the short rate).

    Only the state part ``x + y`` of the discount enters the operator here. The
    deterministic, spatially-uniform shift :math:`\varphi(t)` is **factored out**
    of the operator and applied as a per-step scalar discount during the
    backward sweep (see :mod:`valax.pricing.pde.g2pp`): a spatially-constant
    discount commutes with the spatial operator, so this splitting is exact and
    keeps the operator time-independent (built once, no stacking). This is also
    what lets the scheme reproduce the initial curve exactly, provided the
    per-step :math:`\varphi` integral is done exactly.

    The mixed coefficient :math:`\rho\sigma\eta` is a genuine constant (unlike
    Heston's ``rho xi v``), and :func:`build_operator_2d` zeroes it on all four
    edges as usual — the standard in't Hout & Foulon treatment of the
    explicit-only cross term. Nothing is detached from autodiff: both factor
    axes are anchored at the origin (the read-off is always at ``(0, 0)``), so
    ``a, b, sigma, eta, rho`` stay fully differentiable through the coefficients.

    Args:
        model: G2++ model parameters.
        grid: The tensor-product ``(x, y)`` centred-factor grid.

    Returns:
        The assembled :class:`~valax.pricing.pde.operators2d.Operator2D`.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma = model.volatility_x
    eta = model.volatility_y
    rho = model.correlation

    x = grid.x.nodes[:, jnp.newaxis]  # (n_x, 1)
    y = grid.y.nodes[jnp.newaxis, :]  # (1, n_y)

    diff_x = sigma**2
    drift_x = -a * x
    diff_v = eta**2
    drift_v = -b * y
    mixed = rho * sigma * eta
    discount = x + y  # state part only; phi(t) applied per step in the recipe.

    return build_operator_2d(
        grid,
        diff_x=diff_x,
        drift_x=drift_x,
        diff_v=diff_v,
        drift_v=drift_v,
        mixed=mixed,
        discount=discount,
    )
