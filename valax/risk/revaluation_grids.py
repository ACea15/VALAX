r"""Revaluation grids (ladders): P&L to all orders along chosen directions.

Bucketed sensitivities (:mod:`valax.risk.state_sensitivities`) are
derivatives **at the base market**; a P&L built from them is a Taylor
expansion truncated at second order. A **revaluation grid** instead reprices
the book at a set of *shifted* levels along each chosen direction of the
factor space, recording the present value **and** the directional delta at
every level. A one-dimensional grid is what desks call a *ladder* (a spot
ladder, a DV01 ladder at shifted curve levels); each level is a *rung*.

For a direction :math:`d` in the coordinates of a
:class:`~valax.risk.scenarios.KeyedFactorLayout` and levels
:math:`h_1 < \dots < h_m` (zero among them), the grid holds

.. math::

    V_j = \mathrm{PV}\bigl(\mathrm{shock}(\text{market}, h_j d)\bigr),
    \qquad
    \delta_j = \frac{\partial}{\partial h}
      \mathrm{PV}\bigl(\mathrm{shock}(\text{market}, h d)\bigr)\Big|_{h_j}.

The P&L of a move :math:`m` along :math:`d` is the integral of the delta
along the direction,

.. math::

    \mathrm{P\&L}(m) = \int_0^m \delta(h)\,dh = V(m) - V(0),

read off the grid by interpolation. Because :math:`V_j` and
:math:`\delta_j` are exact full revaluations, the only error is the
interpolation error between rungs — **every order** of the P&L along the
direction is retained, not only the first two. Two interpolants are offered:

* ``"hermite"`` (default): the cubic Hermite interpolant through
  :math:`(V_j, \delta_j)` on each interval — exact at every rung, error
  :math:`O(w^4)` in the rung spacing :math:`w`;
* ``"delta"``: integrate the piecewise-linear interpolant of the delta rungs
  alone (the classical "sum the delta ladder" recipe) — error
  :math:`O(w^2)`; it does not use the PVs.

A move beyond the outer rung is extrapolated linearly from that rung's PV
and delta and flagged in :attr:`GridPnL.out_of_range`.

**Cross terms.** Summing per-direction ladders captures each direction to
all orders but **no** interaction between directions
(:math:`\partial^2 V/\partial x_i \partial x_j`, :math:`i \ne j`, and
higher mixed terms). :func:`revaluation_grid_pnl` can add the second-order
cross terms from a :class:`~valax.risk.state_sensitivities.KeyedSensitivities`
computed with ``second_order=True``; mixed terms of third and higher order
remain outside the number (a two-dimensional product grid is the planned
remedy).

Everything goes through :func:`valax.core.pricing.price` (via
:func:`valax.risk.state_var.group_pv`) on the same shock map the
sensitivities differentiate, so a grid, a sensitivity set and a scenario
P&L are expressed in one coordinate system. ``key`` is threaded unchanged
into every node, so an MC group's grid is a response to the shifts rather
than to fresh randomness.

References:
    Basel Committee on Banking Supervision, *Minimum capital requirements
    for market risk* (MAR21.5, curvature risk), for the use of full
    revaluation at shifted levels in the standardised approach.
"""

import warnings
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float
from jax import Array

from valax.core.market_state import MarketState
from valax.market.keyed_scenario import KeyedScenario
from valax.risk.scenarios import KeyedFactorLayout
from valax.risk.state_sensitivities import KeyedSensitivities, _pv_of_factors
from valax.risk.state_var import _warn_shadowed_shocks

#: Interpolants accepted by :func:`revaluation_grid_pnl`.
INTERPOLATIONS = ("hermite", "delta")


# ── Pytrees ──────────────────────────────────────────────────────────


class RevaluationGrid(eqx.Module):
    """PV and directional delta on a grid of shift levels per direction.

    Attributes:
        layout: Factor coordinates the directions live in. Static.
        labels: One name per direction (e.g. a factor id or block prefix).
            Static.
        disjoint: Whether the directions have non-overlapping support, so
            per-direction P&Ls add up to a total. Static.
        directions: One factor-space vector per direction, shape
            ``(n_dirs, n_factors)``.
        levels: Shift levels shared by every direction, strictly increasing
            and containing zero.
        base_pv: Present value at the base market.
        pv: Present value at every node, ``pv[k, j]`` for direction ``k`` at
            level ``j``.
        delta: Directional derivative ``d PV / d h`` at every node.
    """

    layout: KeyedFactorLayout = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    disjoint: bool = eqx.field(static=True)
    directions: Float[Array, "n_dirs n_factors"]
    levels: Float[Array, " n_levels"]
    base_pv: Float[Array, ""]
    pv: Float[Array, "n_dirs n_levels"]
    delta: Float[Array, "n_dirs n_levels"]

    def index(self, label: str) -> int:
        """Position of a direction.

        Args:
            label: Direction name.

        Returns:
            Row index into :attr:`pv` / :attr:`delta`.

        Raises:
            KeyError: If no direction has that name.
        """
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(
                f"no direction {label!r} in this grid; have {self.labels[:5]}..."
            ) from None

    def ladder(self, label: str) -> Float[Array, " n_levels"]:
        """Full-revaluation P&L at each rung of one direction.

        Args:
            label: Direction name.

        Returns:
            ``pv - base_pv`` at every level of that direction.
        """
        return self.pv[self.index(label)] - self.base_pv

    def delta_ladder(self, label: str) -> Float[Array, " n_levels"]:
        """Directional delta at each rung of one direction.

        Args:
            label: Direction name.

        Returns:
            ``d PV / d h`` at every level of that direction.
        """
        return self.delta[self.index(label)]


class GridPnL(eqx.Module):
    """P&L read off a revaluation grid.

    Leading ``batch`` dimensions follow the shape of the moves passed in.

    Attributes:
        labels: Direction names, in the order of :attr:`by_direction`.
            Static.
        by_direction: P&L of each direction's move, to all orders along it
            (up to interpolation error).
        cross: Second-order cross-direction P&L from a supplied
            ``KeyedSensitivities``; zero when none was supplied, NaN when
            the directions overlap.
        total: ``sum(by_direction) + cross``; NaN when the directions
            overlap (a sum would double count).
        out_of_range: Whether each move lies outside ``[levels[0],
            levels[-1]]`` (linearly extrapolated).
    """

    labels: tuple[str, ...] = eqx.field(static=True)
    by_direction: Float[Array, "*batch n_dirs"]
    cross: Float[Array, "*batch"]
    total: Float[Array, "*batch"]
    out_of_range: Bool[Array, "*batch n_dirs"]


# ── Direction helpers ────────────────────────────────────────────────


def key_rate_directions(
    layout: KeyedFactorLayout,
    curve_id: str,
) -> dict[str, Float[Array, " n_factors"]]:
    """One direction per pillar of a named curve (a key-rate ladder each).

    A level ``h`` along a direction is a zero-rate move of ``h`` at that
    pillar alone (``0.01`` = 100 bp), in the convention of
    :func:`valax.risk.shocks.bump_graph_curve`.

    Args:
        layout: Factor coordinates.
        curve_id: Curve (or basis) block identifier.

    Returns:
        ``factor id -> unit vector`` for every pillar of the curve.

    Raises:
        KeyError: If the layout has no block for ``curve_id``.
    """
    sl = layout.index("curve", curve_id)
    ids = layout.factor_ids()
    eye = jnp.eye(layout.n_factors)
    return {ids[i]: eye[i] for i in range(sl.start, sl.stop)}


def parallel_direction(
    layout: KeyedFactorLayout,
    kind: str,
    key: str,
) -> dict[str, Float[Array, " n_factors"]]:
    """One direction moving a whole block in parallel.

    For a curve block a level ``h`` is a parallel zero-rate move of ``h``;
    for a spot block it is a spot move (absolute, or relative when the
    layout is ``multiplicative``); for a vol block a parallel vol shift.

    Args:
        layout: Factor coordinates.
        kind: Block kind (one of :data:`~valax.risk.scenarios.FACTOR_KINDS`).
        key: Block identifier.

    Returns:
        ``block prefix -> indicator vector`` of that block.

    Raises:
        KeyError: If the layout has no such block.
    """
    sl = layout.index(kind, key)
    vec = jnp.zeros((layout.n_factors,)).at[sl].set(1.0)
    return {layout.block(kind, key).prefix: vec}


# ── Construction ─────────────────────────────────────────────────────


def _checked_levels(levels: Any) -> np.ndarray:
    """Validate the shift levels at the user boundary.

    Args:
        levels: Candidate levels (concrete).

    Returns:
        The levels as a float64 NumPy array.

    Raises:
        ValueError: If fewer than two levels, not strictly increasing, or
            zero is not one of them.
    """
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"levels must be a 1-D sequence of at least 2, got {arr.shape}.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"levels must be strictly increasing, got {arr}.")
    if not np.any(arr == 0.0):
        raise ValueError(
            f"levels must contain 0.0 (the base market), got {arr}; the P&L "
            "is measured from the base."
        )
    return arr


def _direction_matrix(
    directions: Mapping[str, Any],
    layout: KeyedFactorLayout,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Stack directions into a matrix, resolving keyed scenarios.

    Args:
        directions: ``label -> factor-space vector`` or ``KeyedScenario``.
        layout: Factor coordinates.

    Returns:
        ``(labels, matrix)`` with the matrix of shape ``(n_dirs, n_factors)``.

    Raises:
        ValueError: If empty, or a vector has the wrong length or is zero.
    """
    if not directions:
        raise ValueError("revaluation_grid needs at least one direction.")
    labels, rows = [], []
    for label, d in directions.items():
        vec = layout.flatten(d) if isinstance(d, KeyedScenario) else d
        vec = np.asarray(vec, dtype=np.float64)
        if vec.shape != (layout.n_factors,):
            raise ValueError(
                f"direction {label!r} has shape {vec.shape}; the layout has "
                f"{layout.n_factors} factors."
            )
        if not np.any(vec):
            raise ValueError(f"direction {label!r} is the zero vector.")
        labels.append(str(label))
        rows.append(vec)
    return tuple(labels), np.stack(rows)


def revaluation_grid(
    instruments: Any,
    model: Any,
    market: MarketState,
    directions: Mapping[str, Any],
    levels: Any,
    *,
    layout: KeyedFactorLayout | None = None,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> RevaluationGrid:
    """Reprice one homogeneous group on a grid of shift levels per direction.

    Every node is a full revaluation through ``price()``: the PV and, by one
    forward-mode ``jax.jvp``, the exact directional delta. Nodes are
    ``vmap``-ped over levels and directions, so the cost is
    ``n_dirs × n_levels`` priced-with-tangent evaluations.

    Args:
        instruments: Batched instrument pytree (leading batch dim), as for
            :func:`valax.risk.state_var.group_pv`.
        model: Model pytree shared by the group, or ``None``.
        market: The calibrated market.
        directions: ``label -> direction``; each direction is a vector in
            ``layout`` coordinates (see :func:`key_rate_directions`,
            :func:`parallel_direction`) or a ``KeyedScenario`` flattened by
            the layout. Must be concrete.
        levels: Shift levels shared by every direction, strictly increasing
            and containing ``0.0``, in the units of the direction (a level
            of ``0.01`` along a unit curve direction is 100 bp). Must be
            concrete.
        layout: Factor coordinates; defaults to
            :meth:`~valax.risk.scenarios.KeyedFactorLayout.from_market_state`.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key (required for ``method="mc"``), reused at every node.

    Returns:
        A :class:`RevaluationGrid`.

    Raises:
        ValueError: If the levels or directions are invalid.

    Warns:
        UserWarning: If the directions overlap (their per-direction P&Ls
            then cannot be added), or if ``model`` shadows a shocked market
            object.
    """
    if layout is None:
        layout = KeyedFactorLayout.from_market_state(market)
    lv = _checked_levels(levels)
    labels, mat = _direction_matrix(directions, layout)

    support = mat != 0.0
    disjoint = bool(np.all(support.sum(axis=0) <= 1))
    if not disjoint:
        warnings.warn(
            "revaluation_grid: the directions overlap, so their P&Ls cannot be "
            "summed; GridPnL.total and .cross will be NaN. Use disjoint "
            "directions (e.g. key-rate directions of different curves) for a "
            "total.",
            UserWarning,
            stacklevel=2,
        )
    _warn_shadowed_shocks(model, layout.zero(), "revaluation_grid")

    pv_fn = _pv_of_factors(instruments, model, market, layout, method, config, key)
    D = jnp.asarray(mat)
    H = jnp.asarray(lv)

    def node(d: Float[Array, " n_factors"], h: Float[Array, ""]):
        return jax.jvp(lambda t: pv_fn(t * d), (h,), (jnp.ones_like(h),))

    pv, delta = jax.vmap(lambda d: jax.vmap(lambda h: node(d, h))(H))(D)
    base_pv = pv_fn(jnp.zeros((layout.n_factors,)))

    return RevaluationGrid(
        layout=layout,
        labels=labels,
        disjoint=disjoint,
        directions=D,
        levels=H,
        base_pv=base_pv,
        pv=pv,
        delta=delta,
    )


# ── Reading P&L off the grid ─────────────────────────────────────────


def _interp_hermite(
    levels: Float[Array, " n_levels"],
    pv: Float[Array, " n_levels"],
    delta: Float[Array, " n_levels"],
    m: Float[Array, ""],
) -> Float[Array, ""]:
    """Cubic Hermite interpolation of PV, linear extrapolation outside."""
    n = levels.shape[0]
    i = jnp.clip(jnp.searchsorted(levels, m, side="right") - 1, 0, n - 2)
    h0, h1 = levels[i], levels[i + 1]
    w = h1 - h0
    t = (m - h0) / w
    t2, t3 = t * t, t * t * t
    inside = (
        (2 * t3 - 3 * t2 + 1) * pv[i]
        + (t3 - 2 * t2 + t) * w * delta[i]
        + (-2 * t3 + 3 * t2) * pv[i + 1]
        + (t3 - t2) * w * delta[i + 1]
    )
    below = pv[0] + delta[0] * (m - levels[0])
    above = pv[-1] + delta[-1] * (m - levels[-1])
    return jnp.where(m < levels[0], below, jnp.where(m > levels[-1], above, inside))


def _interp_delta(
    levels: Float[Array, " n_levels"],
    delta: Float[Array, " n_levels"],
    m: Float[Array, ""],
) -> Float[Array, ""]:
    """Integral from 0 of the piecewise-linear delta, linear outside."""
    n = levels.shape[0]
    w = jnp.diff(levels)
    cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(0.5 * (delta[:-1] + delta[1:]) * w)])
    cum = cum - cum[jnp.argmin(jnp.abs(levels))]  # zero at the base level
    i = jnp.clip(jnp.searchsorted(levels, m, side="right") - 1, 0, n - 2)
    s = m - levels[i]
    slope = (delta[i + 1] - delta[i]) / w[i]
    inside = cum[i] + delta[i] * s + 0.5 * slope * s * s
    below = cum[0] + delta[0] * (m - levels[0])
    above = cum[-1] + delta[-1] * (m - levels[-1])
    return jnp.where(m < levels[0], below, jnp.where(m > levels[-1], above, inside))


def _moves_array(
    grid: RevaluationGrid,
    moves: Any,
) -> Float[Array, "*batch n_dirs"]:
    """Normalise moves to an array whose last axis indexes the directions.

    Args:
        grid: The grid.
        moves: ``label -> move`` (scalars or equally shaped arrays; missing
            labels move by zero) or an array of shape ``(..., n_dirs)``.

    Returns:
        Moves as an array of shape ``(..., n_dirs)``.

    Raises:
        ValueError: On an unknown label or a wrong trailing dimension.
    """
    n = len(grid.labels)
    if isinstance(moves, Mapping):
        unknown = set(moves) - set(grid.labels)
        if unknown:
            raise ValueError(f"moves name unknown directions: {sorted(unknown)}")
        shape = jnp.broadcast_shapes(*(jnp.shape(v) for v in moves.values())) if moves else ()
        cols = [
            jnp.broadcast_to(jnp.asarray(moves.get(lab, 0.0), dtype=grid.levels.dtype), shape)
            for lab in grid.labels
        ]
        return jnp.stack(cols, axis=-1)
    arr = jnp.asarray(moves, dtype=grid.levels.dtype)
    if arr.ndim == 0 or arr.shape[-1] != n:
        raise ValueError(
            f"moves array must have trailing dimension {n} (one per direction), "
            f"got shape {arr.shape}."
        )
    return arr


def revaluation_grid_pnl(
    grid: RevaluationGrid,
    moves: Any,
    *,
    interpolation: str = "hermite",
    sensitivities: KeyedSensitivities | None = None,
) -> GridPnL:
    r"""P&L of a move along each direction, read off a revaluation grid.

    Per direction :math:`k` with move :math:`m_k`,

    .. math::

        \mathrm{P\&L}_k = \int_0^{m_k} \delta_k(h)\,dh,

    interpolated between rungs — to all orders along the direction, with
    only the interpolation error (``"hermite"``: :math:`O(w^4)`;
    ``"delta"``: :math:`O(w^2)` in the rung spacing). With disjoint
    directions the total is the sum, plus, if ``sensitivities`` (with a
    Hessian, same layout) is given, the second-order cross-direction term

    .. math::

        \tfrac12 x^{\top} H x - \sum_k \tfrac12 m_k^2\, d_k^{\top} H d_k,
        \qquad x = \sum_k m_k d_k .

    Args:
        grid: The revaluation grid.
        moves: ``label -> move`` (missing labels move by zero), or an array
            of shape ``(..., n_dirs)``; leading dimensions are a batch of
            scenarios.
        interpolation: ``"hermite"`` (PV and delta, default) or ``"delta"``
            (integrate the delta rungs only).
        sensitivities: Optional second-order keyed sensitivities on the
            grid's layout, for the cross-direction term.

    Returns:
        A :class:`GridPnL`.

    Raises:
        ValueError: On an unknown interpolation, unknown direction labels,
            a wrongly shaped move array, or sensitivities without a Hessian
            or on a different layout.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}.")
    if sensitivities is not None:
        if sensitivities.gamma is None:
            raise ValueError(
                "cross terms need second-order sensitivities: build them with "
                "keyed_sensitivities(..., second_order=True)."
            )
        if sensitivities.layout.n_factors != grid.layout.n_factors:
            raise ValueError(
                f"sensitivities have {sensitivities.layout.n_factors} factors, "
                f"the grid's layout {grid.layout.n_factors}."
            )

    m = _moves_array(grid, moves)
    batch = m.shape[:-1]
    flat = m.reshape((-1, m.shape[-1]))
    lv = grid.levels

    if interpolation == "hermite":
        one = jax.vmap(_interp_hermite, in_axes=(None, 0, 0, 0))
        per = jax.vmap(lambda row: one(lv, grid.pv, grid.delta, row) - grid.base_pv)(flat)
    else:
        one = jax.vmap(_interp_delta, in_axes=(None, 0, 0))
        per = jax.vmap(lambda row: one(lv, grid.delta, row))(flat)

    if sensitivities is not None:
        H = sensitivities.gamma
        D = grid.directions
        x = flat @ D
        full = 0.5 * jnp.einsum("bi,ij,bj->b", x, H, x)
        own = 0.5 * jnp.sum(flat**2 * jnp.einsum("ki,ij,kj->k", D, H, D), axis=-1)
        cross = full - own
    else:
        cross = jnp.zeros(flat.shape[:1])

    total = jnp.sum(per, axis=-1) + cross
    if not grid.disjoint:
        nan = jnp.full_like(total, jnp.nan)
        cross, total = nan, nan

    out = (flat < lv[0]) | (flat > lv[-1])
    return GridPnL(
        labels=grid.labels,
        by_direction=per.reshape(m.shape),
        cross=cross.reshape(batch),
        total=total.reshape(batch),
        out_of_range=out.reshape(m.shape),
    )


__all__ = [
    "INTERPOLATIONS",
    "GridPnL",
    "RevaluationGrid",
    "key_rate_directions",
    "parallel_direction",
    "revaluation_grid",
    "revaluation_grid_pnl",
]
