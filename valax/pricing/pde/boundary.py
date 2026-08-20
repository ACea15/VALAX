"""Boundary conditions for finite-difference solvers.

A :class:`Boundary1D` supplies the two exterior Dirichlet values as functions
of time-remaining ``tau``. It is a plain Python object (not a pytree): it is
closed over by the ``lax.scan`` backward loop and its callables produce traced
values from the traced ``tau`` at each step, so it composes with
``jax.grad`` / ``jax.jit`` without being a registered pytree.

Factories cover the cases needed in PR-1:

- :func:`bs_european_boundary` — Black-Scholes deep-ITM/OTM asymptotics.
- :func:`american_boundary` — intrinsic value at the far edges.
- :func:`digital_boundary` — discounted payout at the ITM edge, zero at the OTM.
- :func:`knockout_boundary` — absorbing (zero) value at the barrier edge.

For the 2-D Heston solver, :class:`Boundary2D` / :func:`heston_boundary` supply
the log-spot Dirichlet asymptotics (reusing the 1-D machinery, constant across
variance), and :func:`apply_heston_variance_bc` bakes the *variance*-axis
conditions (the degenerate ``v = 0`` transport row and the ``v = v_max``
linearity row) directly into the ADI operator ``A2``.
"""

from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.models.heston import HestonModel
from valax.pricing.pde.grids import Grid1D, Grid2D, boundary_coords
from valax.pricing.pde.operators2d import Operator2D


class Boundary1D:
    """Two Dirichlet boundary values as functions of time-remaining ``tau``.

    Attributes:
        lower_fn: Callable ``tau -> value`` at the lower boundary ``x_min``.
        upper_fn: Callable ``tau -> value`` at the upper boundary ``x_max``.
    """

    def __init__(
        self,
        lower_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
        upper_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    ) -> None:
        self.lower_fn = lower_fn
        self.upper_fn = upper_fn


def bs_european_boundary(
    grid: Grid1D,
    strike: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
    is_call: bool,
) -> Boundary1D:
    """Black-Scholes European Dirichlet boundaries in log-spot space.

    At the far edges the option value tends to its intrinsic asymptotics:
    a call is worthless as ``S -> 0`` and behaves like the discounted forward
    minus discounted strike as ``S -> infinity`` (and vice-versa for a put).

    Args:
        grid: The spatial (log-spot) grid.
        strike: Option strike.
        rate: Risk-free rate.
        dividend: Continuous dividend yield.
        is_call: True for a call, False for a put.

    Returns:
        The :class:`Boundary1D` for a European option.
    """
    x_min, x_max = boundary_coords(grid)
    s_lo = jnp.exp(x_min)
    s_hi = jnp.exp(x_max)

    if is_call:
        def lower_fn(tau):
            return jnp.zeros_like(tau)

        def upper_fn(tau):
            return s_hi * jnp.exp(-dividend * tau) - strike * jnp.exp(-rate * tau)
    else:
        def lower_fn(tau):
            return strike * jnp.exp(-rate * tau) - s_lo * jnp.exp(-dividend * tau)

        def upper_fn(tau):
            return jnp.zeros_like(tau)

    return Boundary1D(lower_fn, upper_fn)


def american_boundary(
    grid: Grid1D,
    strike: Float[Array, ""],
    is_call: bool,
) -> Boundary1D:
    """Intrinsic-value Dirichlet boundaries for an American option.

    Deep in-the-money an American option is worth (at least) its immediate
    exercise value, so the far edge is pinned to the undiscounted intrinsic
    value ``max(S - K, 0)`` / ``max(K - S, 0)``; the out-of-the-money edge is
    zero.

    Args:
        grid: The spatial (log-spot) grid.
        strike: Option strike.
        is_call: True for a call, False for a put.

    Returns:
        The :class:`Boundary1D` for an American option.
    """
    x_min, x_max = boundary_coords(grid)
    s_lo = jnp.exp(x_min)
    s_hi = jnp.exp(x_max)

    if is_call:
        def lower_fn(tau):
            return jnp.zeros_like(tau)

        def upper_fn(tau):
            return jnp.full_like(tau, s_hi - strike)
    else:
        def lower_fn(tau):
            return jnp.full_like(tau, strike - s_lo)

        def upper_fn(tau):
            return jnp.zeros_like(tau)

    return Boundary1D(lower_fn, upper_fn)


def digital_boundary(
    payout: Float[Array, ""],
    rate: Float[Array, ""],
    is_call: bool,
) -> Boundary1D:
    """Cash-or-nothing digital Dirichlet boundaries.

    Deep in-the-money the option is (almost) certain to pay, so its value is the
    discounted payout; deep out-of-the-money it is worthless. For a digital
    call the ITM edge is the upper boundary; for a put it is the lower.

    Args:
        payout: Fixed cash payout if in-the-money.
        rate: Risk-free rate.
        is_call: True for a digital call, False for a digital put.

    Returns:
        The :class:`Boundary1D` for a digital option.
    """
    if is_call:
        def lower_fn(tau):
            return jnp.zeros_like(tau)

        def upper_fn(tau):
            return payout * jnp.exp(-rate * tau)
    else:
        def lower_fn(tau):
            return payout * jnp.exp(-rate * tau)

        def upper_fn(tau):
            return jnp.zeros_like(tau)

    return Boundary1D(lower_fn, upper_fn)


def knockout_boundary(
    inner: Boundary1D,
    *,
    barrier_is_upper: bool,
) -> Boundary1D:
    """Wrap a boundary so the barrier edge is absorbing (zero value).

    For an up-and-out option the upper edge (the barrier) is set to zero and the
    lower edge keeps its vanilla asymptotic; for a down-and-out the reverse.

    Args:
        inner: The underlying (vanilla) boundary supplying the non-barrier edge.
        barrier_is_upper: True if the barrier is the upper edge.

    Returns:
        A :class:`Boundary1D` with the barrier edge pinned to zero.
    """
    zero = lambda tau: jnp.zeros_like(tau)
    if barrier_is_upper:
        return Boundary1D(inner.lower_fn, zero)
    return Boundary1D(zero, inner.upper_fn)


# ─────────────────────────────────────────────────────────────────────
# Two-dimensional (Heston) boundary conditions
# ─────────────────────────────────────────────────────────────────────


class Boundary2D:
    """Log-spot Dirichlet asymptotics for a 2-D (Heston) solver.

    Only the log-spot (``x``) axis carries Dirichlet data: the deep-ITM/OTM
    asymptotics as ``S -> 0`` and ``S -> infinity`` are (to leading order)
    independent of the instantaneous variance, so the two callables return a
    single scalar per ``tau`` which the ADI stepper broadcasts across every
    variance row. The variance-axis conditions are *not* Dirichlet; they are
    baked into the operator by :func:`apply_heston_variance_bc`.

    Attributes:
        x_lower_fn: Callable ``tau -> value`` at the lower log-spot edge
            (``S -> 0``).
        x_upper_fn: Callable ``tau -> value`` at the upper log-spot edge
            (``S -> infinity``).
    """

    def __init__(
        self,
        x_lower_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
        x_upper_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    ) -> None:
        self.x_lower_fn = x_lower_fn
        self.x_upper_fn = x_upper_fn


def heston_boundary(
    grid: Grid2D,
    strike: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
    is_call: bool,
) -> Boundary2D:
    """Log-spot Dirichlet asymptotics for a European option under Heston.

    Reuses the 1-D Black-Scholes asymptotics on the log-spot axis (``grid.x``):
    a call is worthless as ``S -> 0`` and behaves like the discounted forward
    minus discounted strike as ``S -> infinity`` (and vice-versa for a put).
    These hold per variance slice, so the same scalar boundary value is applied
    to every variance row.

    Args:
        grid: The tensor-product (log-spot :math:`\\times` variance) grid.
        strike: Option strike.
        rate: Risk-free rate.
        dividend: Continuous dividend yield.
        is_call: True for a call, False for a put.

    Returns:
        The :class:`Boundary2D` log-spot Dirichlet data.
    """
    b1 = bs_european_boundary(grid.x, strike, rate, dividend, is_call)
    return Boundary2D(b1.lower_fn, b1.upper_fn)


def apply_heston_variance_bc(
    operator: Operator2D,
    grid: Grid2D,
    model: HestonModel,
) -> Operator2D:
    r"""Bake the variance-axis boundary conditions into the ADI operator ``A2``.

    The variance axis has no natural Dirichlet data, so its boundaries are
    imposed by rewriting the first and last rows of the tridiagonal variance
    operator ``A2``:

    - **Low-variance row** ``j = 0`` (the degenerate ``v -> 0`` boundary). As
      ``v -> 0`` the variance diffusion ``1/2 xi^2 v`` vanishes and the drift
      ``kappa(theta - v) -> kappa theta > 0`` points *into* the domain (an
      inflow). The row is replaced by a **drift-only, one-sided upwind**
      (forward) difference with no coupling to a sub-zero ghost:

      .. math::

          (A_2 V)_{i,0} = \kappa(\theta - v_0)\,\frac{V_{i,1} - V_{i,0}}{v_1 - v_0}
              - \tfrac{1}{2} r V_{i,0}.

      This is the standard Feller-robust treatment: it stays well posed whether
      or not the Feller condition ``2 kappa theta >= xi^2`` holds, because it
      never differentiates across ``v = 0``.

    - **High-variance row** ``j = n_y - 1`` (the ``v = v_max`` boundary). A
      linearity condition ``V_{vv} = 0`` is imposed by linearly extrapolating
      the exterior ghost value and folding it back into the stencil, which zeros
      the super-diagonal coupling while preserving the drift/diffusion action on
      the near-boundary interior.

    The log-spot operator ``A1`` and the mixed operator ``A0`` are untouched:
    at low variance they already carry a vanishing (``propto v``) diffusion and
    mixed coefficient, so the full row correctly degenerates to the transport
    equation ``V_tau = (r - q) V_x + kappa theta V_v - r V``.

    Args:
        operator: The raw Heston :class:`Operator2D`.
        grid: The tensor-product grid.
        model: Heston model parameters (for ``kappa``, ``theta``, ``rate``).

    Returns:
        A new :class:`Operator2D` with the variance boundary rows rewritten.
    """
    v = grid.y.nodes
    half_r = 0.5 * model.rate

    # --- Low-variance row j = 0: drift-only upwind (forward) difference. ---
    v0 = v[0]
    h_up0 = v[1] - v[0]
    drift0 = model.kappa * (model.theta - v0)  # > 0 (inflow) for v0 < theta
    a2_lower = operator.a2_lower.at[:, 0].set(0.0)
    a2_diag = operator.a2_diag.at[:, 0].set(-drift0 / h_up0 - half_r)
    a2_upper = operator.a2_upper.at[:, 0].set(drift0 / h_up0)

    # --- High-variance row j = n_y - 1: linearity (V_vv = 0) ghost fold. ---
    # Linear extrapolation of the exterior ghost V_ghost = V[-1] + ratio (V[-1] - V[-2])
    # folds the super-diagonal band into the diagonal/sub-diagonal.
    _, v_hi_ghost = boundary_coords(grid.y)
    h_down = v[-1] - v[-2]
    ratio = (v_hi_ghost - v[-1]) / h_down
    au_last = operator.a2_upper[:, -1]
    a2_diag = a2_diag.at[:, -1].add(au_last * (1.0 + ratio))
    a2_lower = a2_lower.at[:, -1].add(-au_last * ratio)
    a2_upper = a2_upper.at[:, -1].set(0.0)

    return eqx.tree_at(
        lambda o: (o.a2_lower, o.a2_diag, o.a2_upper),
        operator,
        (a2_lower, a2_diag, a2_upper),
    )
