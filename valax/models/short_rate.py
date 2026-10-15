"""Shared contract of the exact-fit short-rate models (HW, G2++, Cheyette).

Each of these models is *exact-fitted* to an initial discount curve
:math:`P^M(0, t)`: the deterministic shift (Hull-White's :math:`\\theta`,
G2++'s :math:`\\varphi`, Cheyette's initial forward) is built so the model
reprices it. The curve is therefore market data the model is *bound to*, not
one of its parameters:

* the model carries ``initial_curve: DiscountCurve | None = None`` and a static
  ``curve_id: str | None = None``;
* :func:`valax.core.pricing.price` binds ``initial_curve`` from the
  ``MarketState`` at pricing time — the curve named by ``curve_id``, else the
  instrument's discount curve (see
  :func:`valax.core.binding.curve_requirements`) — so curve risk is reported
  once, on the market, and a calibrated model is a curve-free pytree that
  stacks under ``vmap``;
* kernel-level functions (``hw_bond_price``, the Jamshidian swaption, the PDE
  and MC builders) read the bound curve and raise through
  :func:`require_initial_curve` if it is missing. Bind one explicitly with
  :func:`valax.core.binding.bind_curves` when calling them directly.

Time-dependent volatility
-------------------------

The volatilities of these models are **piecewise constant in time**: a
volatility field holds one value per bucket, ``Float[Array, " n_buckets"]``,
and ``volatility_times`` holds the ``n_buckets - 1`` interior breakpoints
:math:`0 < \\tau_1 < \\dots < \\tau_{n-1}` in year fractions from the curve's
reference date. Bucket :math:`j` covers :math:`[\\tau_j, \\tau_{j+1})` with
:math:`\\tau_0 = 0` and :math:`\\tau_n = \\infty` (the last value extends flat).
A constant volatility is one bucket with no breakpoints. Breakpoints are data,
not static structure, so models whose buckets sit on different dates (one per
trade's exercise grid) still stack under ``vmap``.

Every Gaussian quantity of these models is built from one primitive,
:func:`decayed_integral`, the exponentially weighted integral of a
piecewise-constant function and its antiderivative, in closed form.
"""

from __future__ import annotations

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve


def require_initial_curve(model: Any) -> DiscountCurve:
    """The curve an exact-fit short-rate model is bound to.

    Trace-safe: ``None`` is pytree structure, so the check is a Python
    branch resolved at trace time.

    Args:
        model: A short-rate model with an ``initial_curve`` field.

    Returns:
        The bound :class:`~valax.curves.discount.DiscountCurve`.

    Raises:
        ValueError: If the model is unbound (``initial_curve is None``).
    """
    curve = model.initial_curve
    if curve is None:
        raise ValueError(
            f"{type(model).__name__} is not bound to a curve "
            f"(initial_curve=None). Price it through valax.price, which binds "
            f"the curve from the MarketState, or bind one explicitly with "
            f"valax.core.binding.bind_curves(model, curve) before calling a "
            f"kernel directly."
        )
    return curve


# ── Piecewise-constant volatility ────────────────────────────────────


def as_buckets(volatility: Any) -> Any:
    """Promote a scalar volatility to one bucket (user-boundary convenience).

    Arrays of rank one or more, and non-array leaves equinox may pass through
    a constructor, are returned unchanged.

    Args:
        volatility: A scalar, or an array of bucket values.

    Returns:
        The volatility with at least one dimension.
    """
    if isinstance(volatility, (int, float)):
        volatility = jnp.asarray(volatility)
    if getattr(volatility, "ndim", None) == 0:
        return jnp.reshape(volatility, (1,))
    return volatility


def default_bucket_times(volatility: Any) -> Any:
    """Default breakpoints when none are given: empty, for a single bucket.

    Args:
        volatility: Bucket values (rank at least one).

    Returns:
        An empty array of ``n_buckets - 1 = 0`` breakpoints. Only valid as a
        default for a single bucket; several buckets need explicit
        breakpoints. A NumPy constant, never a traced value: a model built
        inside ``jit`` and closed over by a calibration's static ``assemble``
        (e.g. :func:`~valax.calibration.hull_white.calibrate_hull_white`'s
        default template) would otherwise leak a tracer through this leaf.

    Raises:
        ValueError: If several buckets are given without breakpoints.
    """
    shape = getattr(volatility, "shape", None)
    if shape is None:
        return volatility
    if shape[-1] != 1:
        raise ValueError(
            f"{shape[-1]} volatility buckets need {shape[-1] - 1} breakpoints "
            f"(volatility_times); none were given."
        )
    return np.zeros(shape[:-1] + (0,), dtype=getattr(volatility, "dtype", None))


def check_buckets(owner: str, volatility: Any, times: Any) -> None:
    """Validate the bucket structure (shapes only, never a traced value).

    Args:
        owner: Model name for the error message.
        volatility: Bucket values.
        times: Interior breakpoints.

    Raises:
        ValueError: If the breakpoint count is not ``n_buckets - 1``.
    """
    v, t = getattr(volatility, "shape", None), getattr(times, "shape", None)
    if v is None or t is None:
        return
    if len(v) < 1 or len(t) < 1 or v[-1] != t[-1] + 1:
        raise ValueError(
            f"{owner}: {v[-1] if v else 0} volatility buckets need "
            f"{(v[-1] - 1) if v else 0} breakpoints, got volatility_times of "
            f"shape {t}."
        )


@functools.singledispatch
def volatility_bucket_fields(model: Any) -> tuple[str, ...]:
    """Names of a model's fields that carry one value per volatility bucket.

    They all share the model's ``volatility_times``. A per-trade bootstrap
    (:func:`~valax.calibration.nodes.swaption_basket_node`) fits one of them
    bucket by bucket and broadcasts the others (held fixed) to the same
    buckets, so the model stays shape-consistent. Each piecewise model
    registers its fields; the default is none (not bucketed).

    Args:
        model: A model pytree.

    Returns:
        The bucketed field names.
    """
    return ()


def is_constant(volatility: Float[Array, " n_buckets"]) -> bool:
    """Whether a volatility is a single bucket (a static, trace-time check).

    The constant case runs the original closed forms, so a one-bucket model
    prices bit-identically to the constant-parameter model it generalises.

    Args:
        volatility: Bucket values.

    Returns:
        ``True`` for exactly one bucket.
    """
    return volatility.shape[-1] == 1


def bucket_index(
    times: Float[Array, " n_buckets_minus_1"],
    t: Float[Array, "..."],
) -> Int[Array, "..."]:
    """Bucket containing each ``t`` (right-continuous at a breakpoint).

    Args:
        times: Interior breakpoints, increasing.
        t: Times in year fractions.

    Returns:
        Index ``j`` with :math:`\\tau_j \\le t < \\tau_{j+1}`.
    """
    return jnp.searchsorted(times, t, side="right")


def _one_minus_exp(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """:math:`1 - e^{-x}`, accurate for small ``x``."""
    return -jnp.expm1(-x)


def _x_minus_one_minus_exp(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """:math:`x - (1 - e^{-x})`, accurate for small ``x`` (series below 1e-3)."""
    # Taylor series to x^7: truncation ~x^6/20160 relative, far below eps at
    # the switch point, where the closed form's cancellation error is ~2eps/x.
    small = jnp.abs(x) < 1e-3
    xs = jnp.where(small, x, 0.0)
    series = xs * xs * (
        1.0 / 2.0 - xs * (
            1.0 / 6.0 - xs * (
                1.0 / 24.0 - xs * (1.0 / 120.0 - xs * (1.0 / 720.0 - xs / 5040.0))
            )
        )
    )
    return jnp.where(small, series, x + jnp.expm1(-x))


def decayed_integral(
    values: Float[Array, " n_buckets"],
    times: Float[Array, " n_buckets_minus_1"],
    rate: Float[Array, ""],
    t: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    r"""Exponentially decayed integral of a piecewise-constant function.

    For :math:`v(u)` piecewise constant on the buckets and a rate
    :math:`k > 0`,

    .. math::

        I(t) = \int_0^t v(u)\,e^{-k(t-u)}\,du, \qquad
        J(t) = \int_0^t I(s)\,ds .

    Inside bucket :math:`j`, with :math:`\delta = t - \tau_j`, :math:`I` solves
    :math:`I' = v_j - kI`, so

    .. math::

        I(t) = I(\tau_j)\,e^{-k\delta} + \frac{v_j}{k}\bigl(1 - e^{-k\delta}\bigr),
        \qquad
        J(t) = J(\tau_j) + \frac{I(\tau_j)}{k}\bigl(1 - e^{-k\delta}\bigr)
              + \frac{v_j}{k^2}\bigl(k\delta - 1 + e^{-k\delta}\bigr),

    and the breakpoint values follow from the same recurrence. Everything is
    closed form; the bucket count is static, the breakpoints may be traced.

    Hull-White's variance is :math:`I` with :math:`v = \sigma^2, k = 2a`; its
    exact-fit convexity is :math:`(I_{k=a} - I_{k=2a})/a`, whose step average
    needs :math:`J`. G2++ uses the same primitive at rates :math:`a`,
    :math:`b`, :math:`a + b` with products of its two volatilities.

    Args:
        values: Integrand value per bucket.
        times: Interior breakpoints, increasing, in year fractions.
        rate: Decay rate :math:`k` (positive).
        t: Evaluation times (non-negative).

    Returns:
        ``(I(t), J(t))``, each shaped like ``t``.
    """
    n = values.shape[-1]
    starts = jnp.concatenate([jnp.zeros((1,), dtype=times.dtype), times])
    i_table = [jnp.zeros((), dtype=values.dtype)]
    j_table = [jnp.zeros((), dtype=values.dtype)]
    for k in range(n - 1):
        x = rate * (starts[k + 1] - starts[k])
        i_k, j_k = i_table[-1], j_table[-1]
        i_table.append(i_k * jnp.exp(-x) + values[k] * _one_minus_exp(x) / rate)
        j_table.append(
            j_k + i_k * _one_minus_exp(x) / rate
            + values[k] * _x_minus_one_minus_exp(x) / rate**2
        )
    i_tab, j_tab = jnp.stack(i_table), jnp.stack(j_table)

    t = jnp.asarray(t)
    j = bucket_index(times, t)
    x = rate * (t - starts[j])
    i0, v = i_tab[j], values[j]
    integral = i0 * jnp.exp(-x) + v * _one_minus_exp(x) / rate
    antiderivative = (
        j_tab[j] + i0 * _one_minus_exp(x) / rate
        + v * _x_minus_one_minus_exp(x) / rate**2
    )
    return integral, antiderivative


def piecewise_integral(
    values: Float[Array, " n_buckets"],
    times: Float[Array, " n_buckets_minus_1"],
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    r""":math:`\int_0^t v(u)\,du` of a piecewise-constant function (no decay).

    Used for the time average of :math:`\sigma^2` over a solver step.

    Args:
        values: Integrand value per bucket.
        times: Interior breakpoints, increasing, in year fractions.
        t: Evaluation times (non-negative).

    Returns:
        The integral, shaped like ``t``.
    """
    starts = jnp.concatenate([jnp.zeros((1,), dtype=times.dtype), times])
    widths = starts[1:] - starts[:-1]
    cumulative = jnp.concatenate(
        [jnp.zeros((1,), dtype=values.dtype), jnp.cumsum(values[:-1] * widths)]
    )
    t = jnp.asarray(t)
    j = bucket_index(times, t)
    return cumulative[j] + values[j] * (t - starts[j])


__all__ = [
    "as_buckets",
    "bucket_index",
    "check_buckets",
    "decayed_integral",
    "default_bucket_times",
    "is_constant",
    "piecewise_integral",
    "require_initial_curve",
    "volatility_bucket_fields",
]
