"""Boundary conditions for 1-D finite-difference solvers.

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
"""

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.pricing.pde.grids import Grid1D, boundary_coords


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
