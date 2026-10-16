"""Hull-White trinomial tree for callable and puttable bond pricing.

Builds a recombining trinomial tree for the Hull-White one-factor model
and prices bonds with embedded call/put options via backward induction.

The tree construction follows Hull & White (1994):

1. Build a symmetric tree for the *x*-process where :math:`r = x + \\alpha`.
2. At each time step, calibrate :math:`\\alpha_i` so that the
   tree-implied zero-coupon bond price matches the initial curve
   (Arrow-Debreu forward propagation).
3. For callable/puttable bonds, roll back from maturity applying
   exercise decisions at each call/put date.

The implementation is fully JAX-compatible:

- ``jax.lax.fori_loop`` for the forward and backward sweeps.
- Fixed-size arrays (no Python loops over time steps inside traced code).
- ``jax.jit`` and ``jax.grad`` work on the bond pricers.

References:
    Hull & White (1994), "Numerical Procedures for Implementing Term
        Structure Models: Single-Factor Models".
    Hull (2018), *Options, Futures, and Other Derivatives*, ch. 32.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jaxtyping import Bool, Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.instruments.bonds import CallableBond, PuttableBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.hull_white import (
    HullWhiteModel,
    _market_df,
    hw_alpha,
    hw_bond_price,
    hw_state_variance,
)
from valax.models.short_rate import is_constant
from valax.pricing.multicurve import model_swap_leg_basis
from valax.pricing.pde.payoff_smoothing import positive_part, project_max
from valax.pricing.pde.rates_payoffs import snap_to_levels, swap_tail_values
from valax.dates.accrual import (
    bond_accrued_interest_at_dates,
    fixed_coupon_amounts,
)
from valax.dates.daycounts import year_fraction


# ─────────────────────────────────────────────────────────────────────
# Tree data structure
# ─────────────────────────────────────────────────────────────────────

class HullWhiteTree(eqx.Module):
    """Pre-built Hull-White trinomial tree.

    All arrays have static shapes determined by ``n_steps`` and
    ``j_max``.  ``n_states = 2 * j_max + 1``.

    Attributes:
        dt: Time step in year fractions.
        dx: State-variable step.
        n_steps: Number of time steps (static).
        j_max: Maximum state index (static).
        alpha: Curve-fitting shifts, shape ``(n_steps,)``.
        rates: Short rates at each node,
            shape ``(n_steps + 1, n_states)``.
        probs: Transition probabilities ``(p_u, p_m, p_d)`` for each
            state, shape ``(n_states, 3)``.
        targets: Target state indices for each source state,
            shape ``(n_states, 3)``.  ``targets[j, k]`` is the index
            into the state dimension that source *j* transitions to
            for branch *k* ∈ {up, mid, down}.
    """

    dt: Float[Array, ""]
    dx: Float[Array, ""]
    n_steps: int = eqx.field(static=True)
    j_max: int = eqx.field(static=True)
    alpha: Float[Array, " n_steps"]
    rates: Float[Array, "n_steps_plus1 n_states"]
    probs: Float[Array, "n_states 3"]
    targets: Int[Array, "n_states 3"]


# ─────────────────────────────────────────────────────────────────────
# Tree construction
# ─────────────────────────────────────────────────────────────────────
# Tree width utility
# ─────────────────────────────────────────────────────────────────────


def hw_tree_j_max(mean_reversion: float, dt: float) -> int:
    """Minimum tree half-width for valid Hull-White trinomial branching.

    Returns the smallest :math:`j_{\\max}` such that all branching
    probabilities remain non-negative at every node, including the boundary
    rows where non-standard (up- or down-shifted) branching is used.

    Hull & White (1994) show the tight lower bound is

    .. math::

        j_{\\max} = \\max\\!\\left(
            \\left\\lceil \\frac{0.1835}{a \\,\\Delta t} \\right\\rceil,\\; 1
        \\right)

    The constant 0.1835 is derived from the condition that the shifted
    branching probabilities at the boundary rows are non-negative (see
    Hull & White 1994, §3).

    Both arguments must be **concrete** Python/NumPy scalars — this function
    is intentionally kept outside any JAX trace because the result determines
    array shapes.  Typical usage before entering a JAX gradient::

        dt    = T / n_steps
        j_max = hw_tree_j_max(float(model.mean_reversion), dt)
        grad  = eqx.filter_grad(
            lambda m: callable_bond_price(bond, m, n_steps=n_steps, j_max=j_max)
        )(model)

    Args:
        mean_reversion: Hull-White mean-reversion speed :math:`a` (positive).
        dt: Tree time step in year fractions (:math:`T / n_{\\text{steps}}`).

    Returns:
        Minimum valid tree half-width as a Python ``int``.

    References:
        Hull & White (1994), "Numerical Procedures for Implementing Term
            Structure Models: Single-Factor Models", §3.
    """
    return max(int(jnp.ceil(0.1835 / (float(mean_reversion) * float(dt)))), 1)


# ─────────────────────────────────────────────────────────────────────


class TreeConfig(eqx.Module):
    """Settings of the Hull-White tree routes of :func:`valax.price`.

    All-static, like every pricing config. ``price(..., method="lattice")``
    prices on the **general** tree (:func:`build_general_hull_white_tree`),
    for a constant volatility too: its nearest-node branching is valid at any
    half-width, so the default ``j_max=None`` sizes the tree from the config
    alone, :math:`J = \\lceil w\\, n_\\text{steps} \\rceil` with
    ``width_factor`` :math:`w`. That works on a traced model (under ``jit`` /
    ``grad``); the outer nodes beyond the reachable width carry no
    probability mass, so they cost time, not accuracy.

    Why :math:`w = 2`: a step moves at most one level, but the level
    spacing follows :math:`\\sigma(t)`, so where the volatility drops the
    finer levels need more nodes to span the same rates — the reachable
    width is roughly :math:`n_\\text{steps}\\,\\sigma_\\max/\\sigma_\\min`.
    Measured with :func:`hw_tree_width` (width / ``n_steps``): constant
    :math:`\\sigma` ≤ 1.00, a 0.012 → 0.009 drop 1.17, a ×1.5 zig-zag 1.25,
    a ×2 hump 1.30, a ×3 drop 2.00. A tree too narrow for its model is
    flagged invalid and prices NaN, never silently wrong: raise
    ``width_factor`` or pass ``j_max`` (e.g. the tight
    :func:`hw_tree_width`).

    Attributes:
        n_steps: Number of time steps to the horizon.
        j_max: Tree half-width; ``None`` means
            ``ceil(width_factor * n_steps)``.
        width_factor: Half-width per time step used when ``j_max`` is
            ``None`` (at least 1).
        smooth_payoff: Cell-average kinked payoffs and exercise projections
            (swaption routes; the bond routes take no smoothing).

    Raises:
        ValueError: If ``width_factor < 1``.
    """

    n_steps: int = eqx.field(static=True, default=100)
    j_max: int | None = eqx.field(static=True, default=None)
    width_factor: float = eqx.field(static=True, default=2.0)
    smooth_payoff: bool = eqx.field(static=True, default=True)

    def __check_init__(self):
        if not self.width_factor >= 1.0:
            raise ValueError(
                f"TreeConfig.width_factor must be >= 1, got {self.width_factor}"
            )

    @property
    def half_width(self) -> int:
        """The tree half-width used: ``j_max``, or ``ceil(width_factor * n_steps)``."""
        if self.j_max is not None:
            return self.j_max
        return int(np.ceil(self.width_factor * self.n_steps))


def build_hull_white_tree(
    model: HullWhiteModel,
    T: float,
    n_steps: int = 100,
    j_max: int | None = None,
    *,
    general: bool = False,
) -> "HullWhiteTree | GeneralHullWhiteTree":
    """Build a Hull-White trinomial tree from *t* = 0 to *T*.

    A constant (one-bucket) volatility builds the classical Hull & White
    (1994) tree below. A piecewise-constant :math:`\\sigma(t)` — or
    ``general=True`` — builds the general tree of
    :func:`build_general_hull_white_tree` instead, whose level spacing
    follows the volatility and whose nearest-node branching is valid at any
    half-width (so a static ``j_max`` sized from the config alone, see
    :class:`TreeConfig`, works on a traced model).

    Args:
        model: Hull-White model (carries initial curve and parameters).
        T: Horizon in year fractions.
        n_steps: Number of time steps.
        j_max: Tree half-width — see :func:`hw_tree_j_max`.  When ``None``
            (default) it is computed automatically from the concrete model
            parameters via :func:`hw_tree_j_max`.  Pass an explicit ``int``
            (pre-computed outside any JAX trace) to enable
            ``eqx.filter_grad`` or ``eqx.filter_jit`` through the pricing
            functions.

    Returns:
        A :class:`HullWhiteTree` ready for backward induction.
    """
    if not is_constant(model.volatility):
        return build_general_hull_white_tree(model, T, n_steps, j_max=j_max)
    a = model.mean_reversion
    sigma = model.volatility[0]

    dt = T / n_steps          # Python float — always concrete
    dx = sigma * jnp.sqrt(3.0 * dt)

    if j_max is None:
        j_max = hw_tree_j_max(float(a), dt)
    n_states = 2 * j_max + 1

    # State indices: j ∈ {-j_max, …, +j_max} stored as array indices
    # 0 … n_states-1 mapping to j_val = idx - j_max.
    j_vals = jnp.arange(n_states) - j_max  # (n_states,)

    # ── Branching probabilities ──────────────────────────────────────
    # η_j = -a * j * dt  (mean drift in units of dx)
    eta = -a * j_vals.astype(jnp.float64) * dt

    # Normal branching probabilities (always computed).
    p_u_normal = 1.0 / 6.0 + (eta**2 + eta) / 2.0
    p_m_normal = 2.0 / 3.0 - eta**2
    p_d_normal = 1.0 / 6.0 + (eta**2 - eta) / 2.0

    # Target offsets from source index for the 3 branches.
    # Normal: (j+1, j, j-1) → target indices: (idx+1, idx, idx-1).
    # Up:     (j+2, j+1, j) → target indices: (idx+2, idx+1, idx).
    # Down:   (j, j-1, j-2) → target indices: (idx, idx-1, idx-2).
    # We select by branching type and clip to valid range.

    # Up-branching probabilities (for very negative j, large positive η).
    eta_up = eta - 1.0  # shift for up branching
    p_u_up = 1.0 / 6.0 + (eta_up**2 + eta_up) / 2.0
    p_m_up = 2.0 / 3.0 - eta_up**2
    p_d_up = 1.0 / 6.0 + (eta_up**2 - eta_up) / 2.0

    # Down-branching probabilities (for very positive j, large negative η).
    eta_dn = eta + 1.0
    p_u_dn = 1.0 / 6.0 + (eta_dn**2 + eta_dn) / 2.0
    p_m_dn = 2.0 / 3.0 - eta_dn**2
    p_d_dn = 1.0 / 6.0 + (eta_dn**2 - eta_dn) / 2.0

    # Select branching type: up when j < -j_max, down when j > j_max.
    # (In practice j_max is chosen so the boundary states are exactly
    # ±j_max, so only the top and bottom rows switch.)
    use_up = j_vals < -j_max + 1  # numpy-style bool
    use_dn = j_vals > j_max - 1

    p_u = jnp.where(use_up, p_u_up, jnp.where(use_dn, p_u_dn, p_u_normal))
    p_m = jnp.where(use_up, p_m_up, jnp.where(use_dn, p_m_dn, p_m_normal))
    p_d = jnp.where(use_up, p_d_up, jnp.where(use_dn, p_d_dn, p_d_normal))

    probs = jnp.stack([p_u, p_m, p_d], axis=1)  # (n_states, 3)

    # Target indices (clamped to [0, n_states-1]).
    idx = jnp.arange(n_states, dtype=jnp.int32)
    t_normal = jnp.stack([idx + 1, idx, idx - 1], axis=1)
    t_up = jnp.stack([idx + 2, idx + 1, idx], axis=1)
    t_dn = jnp.stack([idx, idx - 1, idx - 2], axis=1)

    targets = jnp.where(
        use_up[:, None], t_up,
        jnp.where(use_dn[:, None], t_dn, t_normal),
    )
    targets = jnp.clip(targets, 0, n_states - 1)

    # ── Forward propagation of Arrow-Debreu prices ───────────────────
    x = j_vals.astype(jnp.float64) * dx  # (n_states,)

    # We propagate Q (Arrow-Debreu) prices and compute alpha at each
    # step.  We also store the resulting rates.
    # Q_0 = delta at the center node.
    Q_init = jnp.zeros(n_states)
    Q_init = Q_init.at[j_max].set(1.0)

    # Market DFs at each step boundary: P^M(0, t_i) for i = 1 … n_steps.
    step_times = jnp.arange(1, n_steps + 1) * dt
    mkt_dfs = _market_df(model, step_times)

    def forward_step(carry, i):
        """Propagate Q from step i to step i+1 and compute alpha_i."""
        Q = carry
        # alpha_i: (1/dt) * ln( sum_j Q_j * exp(-x_j * dt) / P^M(0, t_{i+1}) )
        sum_weighted = jnp.sum(Q * jnp.exp(-x * dt))
        alpha_i = jnp.log(sum_weighted / mkt_dfs[i]) / dt

        # Short rates at step i.
        rates_i = x + alpha_i

        # Discount factors at step i.
        disc_i = jnp.exp(-rates_i * dt)

        # Propagate Q to step i+1.
        contrib = Q * disc_i  # (n_states,)
        Q_next = jnp.zeros(n_states)
        for k in range(3):
            Q_next = Q_next.at[targets[:, k]].add(contrib * probs[:, k])

        return Q_next, (alpha_i, rates_i)

    _, (alphas, rates_body) = jax.lax.scan(forward_step, Q_init, jnp.arange(n_steps))

    # Terminal rates (step n_steps) — use last alpha as a reasonable
    # approximation (they are only needed if we want to inspect the
    # full tree; bond pricing starts backward from maturity payoff).
    rates_terminal = (x + alphas[-1])[None, :]  # (1, n_states)
    rates_all = jnp.concatenate([rates_body, rates_terminal], axis=0)

    return HullWhiteTree(
        dt=dt,
        dx=dx,
        n_steps=n_steps,
        j_max=j_max,
        alpha=alphas,
        rates=rates_all,
        probs=probs,
        targets=targets,
    )


# ─────────────────────────────────────────────────────────────────────
# General tree: time-dependent volatility
# ─────────────────────────────────────────────────────────────────────


class GeneralHullWhiteTree(eqx.Module):
    """Hull-White trinomial tree with level-dependent spacing (σ(t)).

    Built by :func:`build_general_hull_white_tree`. Unlike
    :class:`HullWhiteTree`, the state spacing, branching probabilities and
    targets vary by level, so they carry a leading step axis.

    Attributes:
        dt: Time step in year fractions.
        n_steps: Number of time steps (static).
        j_max: Tree half-width (static); ``n_states = 2 * j_max + 1``.
        nodes: Centred state :math:`x` at every node,
            shape ``(n_steps + 1, n_states)``.
        alpha: Curve-fitting shifts per step, shape ``(n_steps,)``.
        rates: Short rates at each node, shape ``(n_steps + 1, n_states)``.
        probs: Branching probabilities ``(p_u, p_m, p_d)`` per step and
            state, shape ``(n_steps, n_states, 3)``.
        targets: Target state indices per step, state and branch,
            shape ``(n_steps, n_states, 3)``.
        valid: ``False`` if any node reached with positive probability has a
            negative branching probability (``j_max`` too small); prices off
            an invalid tree are NaN.
    """

    dt: Float[Array, ""]
    n_steps: int = eqx.field(static=True)
    j_max: int = eqx.field(static=True)
    nodes: Float[Array, "n_steps_plus1 n_states"]
    alpha: Float[Array, " n_steps"]
    rates: Float[Array, "n_steps_plus1 n_states"]
    probs: Float[Array, "n_steps n_states 3"]
    targets: Int[Array, "n_steps n_states 3"]
    valid: Bool[Array, ""]


def _level_spacing(
    model: HullWhiteModel, dt: Float[Array, ""], n_steps: int
) -> Float[Array, " n_steps_plus1"]:
    """State spacing of each level: :math:`\\Delta x_{i+1} = \\sqrt{3V(t_i, t_{i+1})}`.

    Level 0 holds only the root, so its spacing is immaterial; it reuses the
    first step's.
    """
    times = jnp.arange(n_steps + 1) * dt
    spacing = jnp.sqrt(3.0 * hw_state_variance(model, times[:-1], times[1:]))
    return jnp.concatenate([spacing[:1], spacing])


def hw_tree_width(model: HullWhiteModel, T: float, n_steps: int) -> int:
    """Half-width ``j_max`` the general tree needs for a concrete model (eager).

    Propagates the reachable width level by level: from a level of
    half-width :math:`w` every node branches around
    :math:`k = \\mathrm{round}(j\\,\\Delta x_i e^{-a\\Delta t} / \\Delta x_{i+1})`,
    so the next level reaches :math:`\\max|k| + 1`. Mean reversion makes the
    width saturate; a drop in :math:`\\sigma(t)` (a finer next level) widens
    it. The result is a static shape, so it must be computed outside any
    trace — pass it as ``j_max`` to differentiate or ``jit`` a tree price.

    Args:
        model: Hull-White model with concrete parameters.
        T: Tree horizon in year fractions.
        n_steps: Number of time steps.

    Returns:
        The smallest half-width with every target inside the tree.
    """
    a = float(model.mean_reversion)
    dt = float(T) / n_steps
    spacing = np.asarray(_level_spacing(model, jnp.asarray(dt), n_steps))
    decay = np.exp(-a * dt)
    width, widest = 0, 1
    for i in range(n_steps):
        j = np.arange(-width, width + 1)
        k = np.rint(j * spacing[i] * decay / spacing[i + 1])
        width = int(np.max(np.abs(k))) + 1
        widest = max(widest, width)
    return widest


def build_general_hull_white_tree(
    model: HullWhiteModel,
    T: Float[Array, ""] | float,
    n_steps: int = 100,
    j_max: int | None = None,
) -> GeneralHullWhiteTree:
    """Build a Hull-White trinomial tree for a piecewise-constant σ(t).

    The general Hull & White (2001) construction on the centred state
    :math:`x` (:math:`r = x + \\alpha`):

    * level :math:`i+1` has spacing :math:`\\Delta x_{i+1} = \\sqrt{3V_i}` with
      :math:`V_i = V(t_i, t_{i+1})` the exact conditional variance of
      :math:`x` over the step (:func:`~valax.models.hull_white.hw_state_variance`),
      so the spacing follows :math:`\\sigma(t)`;
    * node :math:`j` of level :math:`i` (state :math:`x = j\\Delta x_i`) has
      exact conditional mean :math:`M = x e^{-a\\Delta t}`; its middle branch
      targets the nearest node :math:`k = \\mathrm{round}(M/\\Delta x_{i+1})`
      and, with :math:`\\varepsilon = M/\\Delta x_{i+1} - k`,

      .. math::

          p_u = \\tfrac16 + \\tfrac{\\varepsilon^2 + \\varepsilon}{2},\\quad
          p_m = \\tfrac23 - \\varepsilon^2,\\quad
          p_d = \\tfrac16 + \\tfrac{\\varepsilon^2 - \\varepsilon}{2},

      which match the conditional mean and variance exactly and are positive
      for :math:`|\\varepsilon| \\le \\tfrac12`. Nearest-node branching is also
      what makes the width saturate under mean reversion, so no separate edge
      rule is needed;
    * :math:`\\alpha_i` is fitted by Arrow-Debreu forward induction so the tree
      reprices the initial curve exactly, as in the constant-σ tree.

    A static ``j_max`` below the reachable width clips targets; a clipped node
    whose probabilities turn negative and that carries probability mass makes
    the tree invalid (``valid=False``) and its prices NaN, never silently
    wrong.

    Args:
        model: Hull-White model bound to its initial curve.
        T: Horizon in year fractions (may be traced when ``j_max`` is given).
        n_steps: Number of time steps.
        j_max: Static half-width; ``None`` computes it eagerly with
            :func:`hw_tree_width` (concrete model and horizon required).

    Returns:
        A :class:`GeneralHullWhiteTree`.

    References:
        Hull & White (2001), "The General Hull-White Model and
            Supercalibration", *Financial Analysts Journal* 57(6).
    """
    if j_max is None:
        j_max = hw_tree_width(model, float(T), n_steps)
    n_states = 2 * j_max + 1
    a = model.mean_reversion
    dt = jnp.asarray(T) / n_steps

    spacing = _level_spacing(model, dt, n_steps)                  # (n+1,)
    j_vals = (jnp.arange(n_states) - j_max).astype(jnp.float64)
    nodes = spacing[:, None] * j_vals[None, :]                    # (n+1, S)

    ratio = nodes[:-1] * jnp.exp(-a * dt) / spacing[1:, None]     # (n, S)
    k = jnp.clip(jnp.round(ratio), -(j_max - 1), j_max - 1)
    eps = ratio - k
    probs = jnp.stack([
        1.0 / 6.0 + 0.5 * (eps**2 + eps),
        2.0 / 3.0 - eps**2,
        1.0 / 6.0 + 0.5 * (eps**2 - eps),
    ], axis=-1)                                                   # (n, S, 3)
    k_idx = k.astype(jnp.int32) + j_max
    targets = jnp.stack([k_idx + 1, k_idx, k_idx - 1], axis=-1)   # (n, S, 3)

    mkt_dfs = _market_df(model, jnp.arange(1, n_steps + 1) * dt)
    q_init = jnp.zeros(n_states).at[j_max].set(1.0)

    def forward_step(carry, xs):
        q, ok = carry
        x_i, p_i, tg_i, df_next = xs
        alpha_i = jnp.log(jnp.sum(q * jnp.exp(-x_i * dt)) / df_next) / dt
        rates_i = x_i + alpha_i
        contrib = q * jnp.exp(-rates_i * dt)
        q_next = jnp.zeros(n_states)
        for b in range(3):
            q_next = q_next.at[tg_i[:, b]].add(contrib * p_i[:, b])
        ok = ok & ~jnp.any((q > 0.0) & (jnp.min(p_i, axis=1) < 0.0))
        return (q_next, ok), (alpha_i, rates_i)

    (_, valid), (alphas, rates_body) = jax.lax.scan(
        forward_step, (q_init, jnp.asarray(True)),
        (nodes[:-1], probs, targets, mkt_dfs),
    )
    # The terminal level is only read for its node coordinates; carry the last
    # shift for inspection, as the constant-σ tree does.
    rates = jnp.concatenate([rates_body, (nodes[-1] + alphas[-1])[None, :]], axis=0)

    return GeneralHullWhiteTree(
        dt=dt,
        n_steps=n_steps,
        j_max=j_max,
        nodes=nodes,
        alpha=alphas,
        rates=rates,
        probs=probs,
        targets=targets,
        valid=valid,
    )


# ─────────────────────────────────────────────────────────────────────
# Backward induction helpers
# ─────────────────────────────────────────────────────────────────────


def _branching(
    tree: HullWhiteTree | GeneralHullWhiteTree, step: Int[Array, ""]
) -> tuple[Float[Array, "n_states 3"], Int[Array, "n_states 3"]]:
    """Branching probabilities and targets used at ``step`` (either tree)."""
    if isinstance(tree, GeneralHullWhiteTree):
        return tree.probs[step], tree.targets[step]
    return tree.probs, tree.targets


def _level_nodes(
    tree: HullWhiteTree | GeneralHullWhiteTree, level: Int[Array, ""]
) -> Float[Array, " n_states"]:
    """Centred state :math:`x` at every node of ``level`` (either tree)."""
    if isinstance(tree, GeneralHullWhiteTree):
        return tree.nodes[level]
    return (jnp.arange(2 * tree.j_max + 1) - tree.j_max) * tree.dx


def _tree_price(
    tree: HullWhiteTree | GeneralHullWhiteTree, root_value: Float[Array, ""]
) -> Float[Array, ""]:
    """The root value, or NaN off an invalid general tree."""
    if isinstance(tree, GeneralHullWhiteTree):
        return jnp.where(tree.valid, root_value, jnp.nan)
    return root_value


def _backward_one_step(
    values: Float[Array, " n_states"],
    rates_i: Float[Array, " n_states"],
    dt: Float[Array, ""],
    probs: Float[Array, "n_states 3"],
    targets: Int[Array, "n_states 3"],
) -> Float[Array, " n_states"]:
    """Roll back one time step: expected discounted continuation value."""
    target_vals = values[targets]  # (n_states, 3)
    expected = jnp.sum(target_vals * probs, axis=1)  # (n_states,)
    return jnp.exp(-rates_i * dt) * expected


def _snap_dates_to_steps(
    dates: Int[Array, " m"],
    ref_date: Int[Array, ""],
    dt: Float[Array, ""],
    day_count: str,
) -> Int[Array, " m"]:
    """Map ordinal dates to nearest tree-step indices."""
    times = year_fraction(ref_date, dates, day_count)
    return jnp.round(times / dt).astype(jnp.int32)


def _on_tree(steps: Int[Array, " m"], n_steps: int) -> Int[Array, " m"]:
    """Step indices with off-tree ones sent past the end (``mode="drop"``).

    Negative indices would otherwise wrap around in a scatter.
    """
    return jnp.where((steps >= 0) & (steps <= n_steps), steps, n_steps + 1)


def _horizon(t: Float[Array, ""]) -> Float[Array, ""] | float:
    """A concrete horizon as a Python ``float``, a traced one unchanged.

    The classical tree was built from a Python-float horizon; keeping that
    eagerly keeps its prices and gradients bit-identical, while a traced
    horizon (under ``jit``) still flows through.
    """
    return t if isinstance(t, jax.core.Tracer) else float(t)


# ─────────────────────────────────────────────────────────────────────
# Callable bond pricing
# ─────────────────────────────────────────────────────────────────────

def callable_bond_price(
    bond: CallableBond,
    model: HullWhiteModel,
    n_steps: int = 100,
    j_max: int | None = None,
    *,
    general: bool = False,
) -> Float[Array, ""]:
    """Price a callable fixed-rate bond on a Hull-White trinomial tree.

    The issuer exercises the call (redeeming the bond at ``call_price``)
    whenever the bond's continuation value exceeds the call price —
    equivalently, whenever rates have fallen enough to make refinancing
    attractive.  This **reduces** the bond's value to the holder
    relative to an otherwise identical non-callable ("bullet") bond.

    Traceable (event dates are scattered onto steps, never ``int()``-ed);
    pass a static ``j_max`` to ``jit`` or differentiate it.

    Args:
        bond: Callable bond instrument.
        model: Hull-White model (initial curve provides discounting).
        n_steps: Number of tree time steps.
        j_max: Tree half-width — see :func:`hw_tree_j_max`.  Pass a
            pre-computed concrete ``int`` to enable ``eqx.filter_grad``
            through this function.
        general: Static switch: price on the general tree even for a
            constant volatility (as ``price()``'s lattice route does).

    Returns:
        Callable bond price (dirty price at ``settlement_date``).
    """
    ref = model.initial_curve.reference_date
    maturity_time = _horizon(year_fraction(ref, bond.payment_dates[-1], bond.day_count))
    if j_max is None:
        j_max = _eager_j_max(model, maturity_time, n_steps, general)
    tree = build_hull_white_tree(
        model, maturity_time, n_steps, j_max=j_max, general=general
    )

    n_states = 2 * tree.j_max + 1
    coupons = fixed_coupon_amounts(bond)

    # Map event dates to tree steps.
    cpn_steps = _snap_dates_to_steps(
        bond.payment_dates, ref, tree.dt, bond.day_count
    )
    call_steps = _snap_dates_to_steps(
        bond.call_dates, ref, tree.dt, bond.day_count
    )

    # Pre-build event arrays indexed by tree step:
    # coupon_at_step[i] = coupon amount if step i is a coupon date, else 0.
    # call_price_at_step[i] = call price if step i is a call date, else +inf.
    n = tree.n_steps
    # Scatter each coupon onto its tree step.  Accumulating the traced
    # `coupons[k]` values directly (rather than a binary indicator times a
    # scalar) keeps jax.grad flowing through coupon_rate / face_value while
    # allowing the amounts to differ period by period, as day-count accrual
    # requires.  Dates off the tree are dropped (no `int()` on a date, so
    # the pricer traces).
    coupon_at_step = jnp.zeros(n + 1).at[_on_tree(cpn_steps, n)].add(
        coupons, mode="drop"
    )

    # Call prices are quoted *clean*; the continuation value they are compared
    # against is a PV of remaining cashflows and is therefore *dirty*. An
    # issuer calling mid-period must hand over the redemption amount plus the
    # interest accrued since the last coupon, so adding accrual to the
    # obstacle puts both sides on the same footing. On a coupon date accrual
    # is zero and nothing changes — which is why every existing test, all of
    # which align call and coupon dates, could not see the difference.
    #
    # Accrual is measured on the true call date, not at the step it snaps to,
    # so the obstacle is independent of `n_steps` and the tree agrees with
    # the PDE rather than drifting from it as the grids diverge.
    accrued_at_call = bond_accrued_interest_at_dates(bond, bond.call_dates)

    # Two call dates snapping to one step keep the cheaper call (the one the
    # issuer would choose).
    call_price_at_step = jnp.full(n + 1, jnp.inf).at[_on_tree(call_steps, n)].min(
        bond.call_prices * bond.face_value + accrued_at_call, mode="drop"
    )

    # Terminal value: face + final coupon (if maturity is a coupon date).
    values = jnp.full(n_states, 1.0) * bond.face_value + coupon_at_step[n]

    # Backward induction.
    def step_fn(i_fwd, values):
        """i_fwd counts 0 … n_steps-1 forward; actual step = n - 1 - i_fwd."""
        step = n - 1 - i_fwd
        rates_step = tree.rates[step]
        cont = _backward_one_step(values, rates_step, tree.dt, *_branching(tree, step))

        # Call exercise: the issuer calls if the *ex-coupon* continuation
        # value exceeds the call price.  A holder called on a coupon date
        # still receives that date's coupon, hence the decision is taken
        # before the coupon is added.  `call_price_at_step` already carries
        # accrued interest, so this is a like-for-like comparison.
        cont = jnp.minimum(cont, call_price_at_step[step])

        # Coupon is paid whether or not the bond is called.
        cont = cont + coupon_at_step[step]

        return cont

    values = jax.lax.fori_loop(0, n, step_fn, values)
    return _tree_price(tree, values[tree.j_max])  # root node


# ─────────────────────────────────────────────────────────────────────
# Puttable bond pricing
# ─────────────────────────────────────────────────────────────────────

def puttable_bond_price(
    bond: PuttableBond,
    model: HullWhiteModel,
    n_steps: int = 100,
    j_max: int | None = None,
    *,
    general: bool = False,
) -> Float[Array, ""]:
    """Price a puttable fixed-rate bond on a Hull-White trinomial tree.

    The bondholder exercises the put (forcing early redemption at
    ``put_price``) whenever the continuation value falls below the put
    price — i.e. when rates have risen enough to make the bond less
    valuable than the guaranteed redemption.  This **increases** the
    bond's value relative to a non-puttable bond.

    Traceable like :func:`callable_bond_price`; pass a static ``j_max`` to
    ``jit`` or differentiate it.

    Args:
        bond: Puttable bond instrument.
        model: Hull-White model.
        n_steps: Number of tree time steps.
        j_max: Tree half-width — see :func:`hw_tree_j_max`.  Pass a
            pre-computed concrete ``int`` to enable ``eqx.filter_grad``
            through this function.
        general: Static switch: price on the general tree even for a
            constant volatility (as ``price()``'s lattice route does).

    Returns:
        Puttable bond price (dirty price at ``settlement_date``).
    """
    ref = model.initial_curve.reference_date
    maturity_time = _horizon(year_fraction(ref, bond.payment_dates[-1], bond.day_count))
    if j_max is None:
        j_max = _eager_j_max(model, maturity_time, n_steps, general)
    tree = build_hull_white_tree(
        model, maturity_time, n_steps, j_max=j_max, general=general
    )

    n_states = 2 * tree.j_max + 1
    coupons = fixed_coupon_amounts(bond)

    cpn_steps = _snap_dates_to_steps(
        bond.payment_dates, ref, tree.dt, bond.day_count
    )
    put_steps = _snap_dates_to_steps(
        bond.put_dates, ref, tree.dt, bond.day_count
    )

    n = tree.n_steps
    coupon_at_step = jnp.zeros(n + 1).at[_on_tree(cpn_steps, n)].add(
        coupons, mode="drop"
    )

    # Put prices are quoted clean; see `callable_bond_price` for why accrual
    # is added to the obstacle and why it is measured on the true put date.
    accrued_at_put = bond_accrued_interest_at_dates(bond, bond.put_dates)

    # Two put dates snapping to one step keep the richer put.
    put_price_at_step = jnp.full(n + 1, -jnp.inf).at[_on_tree(put_steps, n)].max(
        bond.put_prices * bond.face_value + accrued_at_put, mode="drop"
    )

    # Terminal value.
    values = jnp.full(n_states, 1.0) * bond.face_value + coupon_at_step[n]

    def step_fn(i_fwd, values):
        step = n - 1 - i_fwd
        rates_step = tree.rates[step]
        cont = _backward_one_step(values, rates_step, tree.dt, *_branching(tree, step))
        # Put exercise on the *ex-coupon* continuation value: a holder who
        # puts on a coupon date still receives that date's coupon.  The
        # obstacle already carries accrued interest.  See
        # `callable_bond_price`.
        cont = jnp.maximum(cont, put_price_at_step[step])
        # Coupon is paid whether or not the bond is put.
        cont = cont + coupon_at_step[step]
        return cont

    values = jax.lax.fori_loop(0, n, step_fn, values)
    return _tree_price(tree, values[tree.j_max])


# ─────────────────────────────────────────────────────────────────────
# Swaptions
# ─────────────────────────────────────────────────────────────────────


def _tail_swap_values(
    model: HullWhiteModel,
    nodes: Float[Array, " n_states"],
    exercise_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
    instrument,
    basis: Float[Array, " n_cf"] | None,
) -> Float[Array, " n_states"]:
    """Underlying swap value at ``exercise_date`` at every node (analytic).

    The same replication and dual-curve fold as the PDE recipes
    (:func:`~valax.pricing.pde.rates_payoffs.swap_tail_values`), valued with
    the affine bond at :math:`r = x + \\alpha(t)` — the continuous exact-fit
    shift, consistent with the tree's centred state :math:`x`.
    """

    def bond_portfolio(t, payment_times, cashflows):
        rates = nodes + hw_alpha(model, t)
        bonds = hw_bond_price(model, rates[:, None], t, payment_times)
        return bonds @ cashflows

    return swap_tail_values(
        bond_portfolio,
        model.initial_curve.reference_date,
        exercise_date,
        payment_dates,
        instrument.strike,
        instrument.notional,
        instrument.day_count,
        instrument.is_payer,
        basis,
    )


def _roll_back(
    tree: HullWhiteTree | GeneralHullWhiteTree,
    terminal: Float[Array, " n_states"],
    exercise: tuple[Bool[Array, " n_steps_plus1"], Float[Array, "n_steps_plus1 n_states"]] | None,
    smooth: bool,
) -> Float[Array, ""]:
    """Backward induction from the horizon, projecting on exercise levels.

    ``exercise`` is ``(is_exercise_level, exercise_value)``; at an exercise
    level the value becomes :math:`\\max(C, E)`, cell-averaged when
    ``smooth`` (an explicit mask, never a ``-inf`` obstacle, whose cell
    variation would be NaN).
    """
    n = tree.n_steps

    def step_fn(i_fwd, values):
        step = n - 1 - i_fwd
        cont = _backward_one_step(
            values, tree.rates[step], tree.dt, *_branching(tree, step)
        )
        if exercise is not None:
            is_exercise, exercise_value = exercise
            projected = project_max(cont, exercise_value[step], smooth)
            cont = jnp.where(is_exercise[step], projected, cont)
        return cont

    values = jax.lax.fori_loop(0, n, step_fn, terminal)
    return _tree_price(tree, values[tree.j_max])


def swaption_tree_price(
    swaption: Swaption,
    model: HullWhiteModel,
    n_steps: int = 100,
    j_max: int | None = None,
    forward_curve: DiscountCurve | None = None,
    smooth_payoff: bool = True,
    *,
    general: bool = False,
) -> Float[Array, ""]:
    """European swaption on the Hull-White trinomial tree.

    Redundant with the exact Jamshidian closed form
    (:func:`~valax.pricing.analytic.hull_white_swaptions.hw_swaption_price`)
    and kept as its independent check: the tree's horizon is the expiry, the
    payoff is the underlying swap valued analytically at each terminal node,
    and the value is rolled back through the lattice. With a piecewise
    :math:`\\sigma(t)` it runs on the general tree.

    Traceable (no ``int()`` on dates); pass a static ``j_max``
    (:func:`hw_tree_j_max` / :func:`hw_tree_width`) to ``jit`` or
    differentiate it.

    Args:
        swaption: The swaption.
        model: Hull-White model bound to its initial (discount) curve.
        n_steps: Number of tree steps to the expiry.
        j_max: Static tree half-width, or ``None`` to size it eagerly.
        forward_curve: Projection curve of the floating leg (frozen tenor
            basis), or ``None`` for single-curve.
        smooth_payoff: Static switch: sample the kinked payoff as its cell
            average over each terminal node's cell
            (:mod:`~valax.pricing.pde.payoff_smoothing`, as the PDE does)
            instead of pointwise, which removes the lattice's oscillating
            convergence in ``n_steps``.
        general: Static switch: price on the general tree even for a
            constant volatility (as ``price()``'s lattice route does).

    Returns:
        The swaption price.
    """
    ref = model.initial_curve.reference_date
    horizon = year_fraction(ref, swaption.expiry_date, swaption.day_count)
    if j_max is None:
        j_max = _eager_j_max(model, horizon, n_steps, general)
    tree = build_hull_white_tree(model, horizon, n_steps, j_max=j_max, general=general)
    basis = model_swap_leg_basis(
        model, forward_curve, swaption.expiry_date, swaption.fixed_dates
    )
    swap = _tail_swap_values(
        model, _level_nodes(tree, n_steps), swaption.expiry_date,
        swaption.fixed_dates, swaption, basis,
    )
    return _roll_back(tree, positive_part(swap, smooth_payoff), None, smooth_payoff)


def bermudan_swaption_tree_price(
    bermudan: BermudanSwaption,
    model: HullWhiteModel,
    n_steps: int = 100,
    j_max: int | None = None,
    forward_curve: DiscountCurve | None = None,
    smooth_payoff: bool = True,
    *,
    general: bool = False,
) -> Float[Array, ""]:
    """Bermudan swaption on the Hull-White trinomial tree.

    The same contract as the PDE recipe: the horizon is the last exercise
    date, exercise ``e`` enters the tail swap on ``fixed_dates[e:]``, valued
    analytically at the true exercise date on the nodes of the level the date
    snaps to (:func:`~valax.pricing.pde.rates_payoffs.snap_to_levels`), and
    the holder takes ``max(continuation, tail)`` there. With a piecewise
    :math:`\\sigma(t)` it runs on the general tree.

    Traceable; pass a static ``j_max`` to ``jit`` or differentiate it.
    Choose ``n_steps`` so the exercise dates fall on (or very near) tree
    levels: a date between levels is exercised at the nearest one, which on a
    6Y Bermudan with 400 steps moved the Richardson-extrapolated price by
    ~4e-4 relative (600 steps put the annual dates on levels).

    Args:
        bermudan: The Bermudan swaption.
        model: Hull-White model bound to its initial (discount) curve.
        n_steps: Number of tree steps to the last exercise date.
        j_max: Static tree half-width, or ``None`` to size it eagerly.
        forward_curve: Projection curve of the floating leg, or ``None``.
        smooth_payoff: Static switch: cell-average the terminal payoff and
            the exercise projection ``max(continuation, tail)`` over each
            node's cell (:mod:`~valax.pricing.pde.payoff_smoothing`, as the
            PDE does) instead of sampling them pointwise, which gives clean
            first-order convergence in ``n_steps``.
        general: Static switch: price on the general tree even for a
            constant volatility (as ``price()``'s lattice route does).

    Returns:
        The Bermudan swaption price.
    """
    ref = model.initial_curve.reference_date
    day_count = bermudan.day_count
    horizon = year_fraction(ref, bermudan.exercise_dates[-1], day_count)
    if j_max is None:
        j_max = _eager_j_max(model, horizon, n_steps, general)
    tree = build_hull_white_tree(model, horizon, n_steps, j_max=j_max, general=general)
    levels = snap_to_levels(bermudan.exercise_dates, ref, day_count, tree.dt, n_steps)

    n_states = 2 * j_max + 1
    is_exercise = jnp.zeros(n_steps + 1, dtype=bool).at[levels].set(True)
    exercise_value = jnp.zeros((n_steps + 1, n_states))
    tails = []
    for e in range(bermudan.exercise_dates.shape[0]):
        basis = model_swap_leg_basis(
            model, forward_curve, bermudan.exercise_dates[e], bermudan.fixed_dates[e:]
        )
        tail = _tail_swap_values(
            model, _level_nodes(tree, levels[e]), bermudan.exercise_dates[e],
            bermudan.fixed_dates[e:], bermudan, basis,
        )
        tails.append(tail)
        exercise_value = exercise_value.at[levels[e]].set(tail)
    # The last exercise is the horizon: exercise or expire worthless.
    terminal = positive_part(tails[-1], smooth_payoff)
    return _roll_back(tree, terminal, (is_exercise, exercise_value), smooth_payoff)


def _eager_j_max(
    model: HullWhiteModel, horizon, n_steps: int, general: bool = False
) -> int:
    """Concrete tree half-width for an eager call (either tree)."""
    if is_constant(model.volatility) and not general:
        return hw_tree_j_max(float(model.mean_reversion), float(horizon) / n_steps)
    return hw_tree_width(model, float(horizon), n_steps)
