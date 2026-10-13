r"""Smoothed positive part: a convergence remedy for kinked payoffs.

A finite-difference grid samples a payoff at its nodes. Kinked payoffs
include :math:`\max(f, 0)` (a swaption's terminal value, a caplet's fixing
value) and the exercise projection :math:`\max(C, E) = C + \max(E - C, 0)`.
Sampling one pointwise makes the discrete price only **piecewise** smooth in
the model inputs. As a parameter moves, the exercise boundary slides
continuously, but each node flips from one branch to the other all at once,
so the price picks up a kink every time a node crosses. Autodiff is exact on
each piece, but finite-difference risk that straddles a crossing is not. In
1-D the convergence order in the grid spacing is also erratic.

The remedy (Pooley, Vetzal & Forsyth 2003) replaces :math:`f^+` at each node
by a :math:`C^1` function of :math:`f` and of :math:`\delta`, the
half-variation of :math:`f` over the node's cell: :math:`f` ranges over
:math:`[f_i - \delta, f_i + \delta]` if it is linear across the cell. Both
kernels below equal :math:`\max(f, 0)` for :math:`|f| \ge \delta`, so only
nodes whose cell straddles the kink change, by :math:`O(\delta) = O(h)` on a
band of width :math:`O(h)`. The price becomes :math:`C^1` in its inputs at an
:math:`O(h^2)` bias.

**1-D: the exact cell average** (:func:`smooth_positive_part`). For a linear
:math:`f` the average of :math:`f^+` over the cell is exactly

.. math::

    \overline{f^+}(f, \delta) =
    \begin{cases}
        \max(f, 0) & |f| \ge \delta, \\[2pt]
        \dfrac{(f + \delta)^2}{4\delta} & |f| < \delta,
    \end{cases}

which is :math:`C^1` jointly in :math:`(f, \delta)`. On the Hull-White grids
it turns the erratic pointwise convergence of a European swaption into a
clean second order: the error ratio is 4.0 per refinement, against
:math:`-63`, :math:`212` and :math:`0.12` pointwise.

**Tensor-product (2-D) grids: the bias-neutral kernel**
(:func:`bias_neutral_positive_part`). The cell average is one-signed: it
exceeds :math:`f^+` by :math:`\int (\overline{f^+} - f^+)\,df = \delta^2/6`
per crossing. In 1-D that is what removes the offset-dependent error of
pointwise sampling. In 2-D an oblique exercise boundary already crosses the
nodes at every cell offset, so pointwise sampling is self-averaging, and the
extra :math:`\delta^2/6` is pure added bias. On the G2++ ADI grid it measured
3x the pointwise spatial error on an OTM receiver (61x61 nodes:
:math:`+8.9\times10^{-3}` against :math:`+2.7\times10^{-3}` relative). The
2-D recipes therefore use the :math:`C^1` kernel with the same support and
**zero integrated bias**:

.. math::

    s(f, \delta) = \delta\Bigl[\tfrac14 (u + 1)^2
        - \tfrac{5}{32}(1 - u^2)^2\Bigr], \qquad u = f/\delta,\ |u| < 1,

and :math:`\max(f, 0)` outside. The quartic correction vanishes with its
derivative at :math:`u = \pm 1`, so :math:`C^1` is kept. Its integral
cancels the :math:`\delta^2/6`. The price this gives tracks the pointwise
price to within a few percent of its error on every G2++ swaption tested,
while being :math:`C^1`. The cost: :math:`s` dips to
:math:`-0.0256\,\delta` near :math:`u = -0.52`. That is a slightly negative
nodal payoff, or, in an exercise projection, :math:`V` slightly below
:math:`C` at nodes just inside the continuation region. Each is :math:`O(h)`
locally and zero in the integral.

**The half-variation** is read off the grid. Along one axis, the cell of node
:math:`i` is :math:`[x_{i-1/2}, x_{i+1/2}]`, and the central slope gives

.. math::

    \delta_i = \tfrac{1}{4}\,|f_{i+1} - f_{i-1}|
             = \tfrac{1}{2}\,|\partial_x f|\,\bar h_i ,

independent of the spacing, so it also holds on non-uniform grids. Edge
nodes use the one-sided :math:`\tfrac12|f_1 - f_0|`. On a tensor-product grid
the per-axis values are combined **in quadrature**,
:math:`\delta = \sqrt{\delta_x^2 + \delta_y^2}`. This is a separable
approximation: over a rectangle a linear :math:`f` is a sum of two uniforms
(trapezoidal), and the formula swaps that for the uniform with the same mean
and variance (:math:`(\delta_x^2 + \delta_y^2)/3`). The support-matching
:math:`\delta_x + \delta_y` over-smooths by the cross term.

The short-rate recipes (:mod:`~valax.pricing.pde.hull_white`,
:mod:`~valax.pricing.pde.g2pp`, :mod:`~valax.pricing.pde.cheyette`) apply this
to their terminal payoffs and exercise projections when
``config.smooth_payoff`` is set (the default), through :func:`positive_part`,
:func:`project_max` and :func:`project_min`. These pick the kernel from the
field's rank.

References:
    D. M. Pooley, K. R. Vetzal and P. A. Forsyth, "Convergence remedies for
    non-smooth payoffs in option pricing", *Journal of Computational Finance*
    6(4), 2003.
"""

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

# Below this half-variation the cell is treated as having no spread at all and
# the plain positive part is returned. Far below any price scale in use
# (prices are O(notional * 1e-2)), it only guards the division.
DELTA_FLOOR = 1e-30


def _guarded(
    f: Float[Array, "..."], delta: Float[Array, "..."]
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Band mask and double-``where``-safe ``(f, delta)`` for the inner branch."""
    delta = jnp.broadcast_to(delta, jnp.shape(f))
    inside = (jnp.abs(f) < delta) & (delta > DELTA_FLOOR)
    return inside, jnp.where(inside, f, 0.0), jnp.where(inside, delta, 1.0)


def smooth_positive_part(
    f: Float[Array, "..."],
    delta: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Exact cell average of :math:`\max(f, 0)` for :math:`f` linear on the cell.

    .. math::

        \overline{f^+}(f, \delta) = \max(f, 0) \;\; (|f| \ge \delta), \qquad
        \frac{(f + \delta)^2}{4\delta} \;\; (|f| < \delta).

    :math:`C^1` jointly in ``f`` and ``delta``; never below
    :math:`\max(f, 0)`, by at most :math:`\delta/4`. The quadratic branch is
    evaluated with a guarded denominator (the double-``where`` trick), so
    gradients are finite everywhere, including at ``f = 0`` and as
    ``delta -> 0``.

    Args:
        f: Values whose positive part is wanted.
        delta: Half-variation of ``f`` over each cell, broadcastable against
            ``f``; non-negative. ``0`` recovers ``max(f, 0)``.

    Returns:
        The cell-averaged positive part, same shape as ``f``.
    """
    inside, safe_f, safe_delta = _guarded(f, delta)
    quadratic = (safe_f + safe_delta) ** 2 / (4.0 * safe_delta)
    return jnp.where(inside, quadratic, jnp.maximum(f, 0.0))


def bias_neutral_positive_part(
    f: Float[Array, "..."],
    delta: Float[Array, "..."],
) -> Float[Array, "..."]:
    r""":math:`C^1` positive part with zero integrated bias over the band.

    .. math::

        s(f, \delta) = \delta\Bigl[\tfrac14 (u + 1)^2
            - \tfrac{5}{32}(1 - u^2)^2\Bigr] \;\; (|u| < 1,\ u = f/\delta),
        \qquad \max(f, 0) \;\; \text{otherwise},

    so that :math:`\int_{-\delta}^{\delta} (s - f^+)\,df = 0`: the cell
    average :func:`smooth_positive_part` minus a :math:`C^1` quartic bump
    that cancels its :math:`\delta^2/6` excess. :math:`C^1` jointly in ``f``
    and ``delta``; lies within :math:`[-0.0256\,\delta,\ 0.094\,\delta]` of
    :math:`\max(f, 0)`. Used on tensor-product grids, where pointwise sampling
    of an oblique kink is already self-averaging (see the module docstring).

    Args:
        f: Values whose positive part is wanted.
        delta: Half-variation of ``f`` over each cell, broadcastable against
            ``f``; non-negative. ``0`` recovers ``max(f, 0)``.

    Returns:
        The smoothed positive part, same shape as ``f``.
    """
    inside, safe_f, safe_delta = _guarded(f, delta)
    u = safe_f / safe_delta
    quartic = safe_delta * (0.25 * (u + 1.0) ** 2 - (5.0 / 32.0) * (1.0 - u * u) ** 2)
    return jnp.where(inside, quartic, jnp.maximum(f, 0.0))


def _axis_half_variation(f: Float[Array, "..."], axis: int) -> Float[Array, "..."]:
    """Half the variation of ``f`` over each node's cell along one axis."""
    f = jnp.moveaxis(f, axis, 0)
    if f.shape[0] < 2:
        return jnp.moveaxis(jnp.zeros_like(f), 0, axis)
    lo = 0.5 * jnp.abs(f[1] - f[0])[None]
    hi = 0.5 * jnp.abs(f[-1] - f[-2])[None]
    interior = 0.25 * jnp.abs(f[2:] - f[:-2])
    return jnp.moveaxis(jnp.concatenate([lo, interior, hi], axis=0), 0, axis)


def cell_half_variation(f: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Half-variation :math:`\delta` of a node field over each node's cell.

    Per axis, :math:`\tfrac14|f_{i+1} - f_{i-1}|` at interior nodes (half
    the change of the central-difference linearisation across one cell) and
    :math:`\tfrac12|f_1 - f_0|` (resp. the last interval) at the edges;
    combined in quadrature, :math:`\sqrt{\sum_k \delta_k^2}`, over the axes of
    a tensor-product grid (the variance-matched separable approximation
    described in the module docstring). Every axis of ``f`` is a spatial axis
    (``vmap`` over any batch axis).

    Args:
        f: Field values at the grid nodes (1-D or tensor-product N-D).

    Returns:
        :math:`\delta` at every node, same shape as ``f``.
    """
    if f.ndim == 0:
        return jnp.zeros_like(f)
    if f.ndim == 1:
        return _axis_half_variation(f, 0)
    sq = sum(_axis_half_variation(f, ax) ** 2 for ax in range(f.ndim))
    # Double-where: sqrt has an infinite derivative at 0.
    positive = sq > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, sq, 1.0)), 0.0)


def positive_part(f: Float[Array, "..."], smooth: bool) -> Float[Array, "..."]:
    """Positive part of a node field, smoothed over each node's cell when ``smooth``.

    Args:
        f: Field values at the grid nodes. Every axis is a grid axis.
        smooth: Static switch. ``False`` returns the pointwise
            ``max(f, 0)``. ``True`` returns the exact cell average
            (:func:`smooth_positive_part`) on a 1-D grid, and the
            bias-neutral kernel (:func:`bias_neutral_positive_part`) on a
            tensor-product grid. Both use :func:`cell_half_variation`.

    Returns:
        The (smoothed) positive part at every node.
    """
    if not smooth:
        return jnp.maximum(f, 0.0)
    kernel = smooth_positive_part if f.ndim <= 1 else bias_neutral_positive_part
    return kernel(f, cell_half_variation(f))


def project_max(
    continuation: Float[Array, "..."],
    obstacle: Float[Array, "..."],
    smooth: bool,
) -> Float[Array, "..."]:
    r"""Holder-optimal exercise projection :math:`\max(C, E) = C + (E - C)^+`.

    With ``smooth`` the positive part is smoothed by :func:`positive_part`,
    with :math:`\delta` read off :math:`E - C` on the grid.

    Args:
        continuation: Continuation value :math:`C` at the nodes.
        obstacle: Exercise value :math:`E` at the nodes (finite).
        smooth: Static switch for smoothing.

    Returns:
        The projected value at every node.
    """
    if not smooth:
        return jnp.maximum(continuation, obstacle)
    return continuation + positive_part(obstacle - continuation, True)


def project_min(
    continuation: Float[Array, "..."],
    obstacle: Float[Array, "..."],
    smooth: bool,
) -> Float[Array, "..."]:
    r"""Issuer-optimal exercise projection :math:`\min(C, E) = C - (C - E)^+`.

    Args:
        continuation: Continuation value :math:`C` at the nodes.
        obstacle: Exercise value :math:`E` at the nodes (finite).
        smooth: Static switch for smoothing.

    Returns:
        The projected value at every node.
    """
    if not smooth:
        return jnp.minimum(continuation, obstacle)
    return continuation - positive_part(continuation - obstacle, True)
