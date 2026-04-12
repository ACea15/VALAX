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
from jaxtyping import Float, Int
from jax import Array

from valax.instruments.bonds import CallableBond, PuttableBond
from valax.models.hull_white import HullWhiteModel, _market_df, _pillar_times
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

def build_hull_white_tree(
    model: HullWhiteModel,
    T: float,
    n_steps: int = 100,
) -> HullWhiteTree:
    """Build a Hull-White trinomial tree from *t* = 0 to *T*.

    Args:
        model: Hull-White model (carries initial curve and parameters).
        T: Horizon in year fractions.
        n_steps: Number of time steps.

    Returns:
        A :class:`HullWhiteTree` ready for backward induction.
    """
    a = model.mean_reversion
    sigma = model.volatility

    dt = T / n_steps
    dx = sigma * jnp.sqrt(3.0 * dt)

    # Maximum number of states above/below zero before switching
    # to non-standard (up/down) branching.
    j_max = max(int(jnp.ceil(0.1835 / (float(a) * float(dt)))), 1)
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
# Backward induction helpers
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# Callable bond pricing
# ─────────────────────────────────────────────────────────────────────

def callable_bond_price(
    bond: CallableBond,
    model: HullWhiteModel,
    n_steps: int = 100,
) -> Float[Array, ""]:
    """Price a callable fixed-rate bond on a Hull-White trinomial tree.

    The issuer exercises the call (redeeming the bond at ``call_price``)
    whenever the bond's continuation value exceeds the call price —
    equivalently, whenever rates have fallen enough to make refinancing
    attractive.  This **reduces** the bond's value to the holder
    relative to an otherwise identical non-callable ("bullet") bond.

    Args:
        bond: Callable bond instrument.
        model: Hull-White model (initial curve provides discounting).
        n_steps: Number of tree time steps.

    Returns:
        Callable bond price (dirty price at ``settlement_date``).
    """
    ref = model.initial_curve.reference_date
    maturity_time = float(year_fraction(ref, bond.payment_dates[-1], bond.day_count))
    tree = build_hull_white_tree(model, maturity_time, n_steps)

    n_states = 2 * tree.j_max + 1
    coupon = bond.face_value * bond.coupon_rate / bond.frequency

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
    # Build a binary indicator for coupon steps and multiply by the
    # coupon amount.  This keeps `coupon` as a traced JAX value so that
    # jax.grad can flow through coupon_rate / face_value.
    cpn_indicator = jnp.zeros(n + 1)
    for k in range(bond.payment_dates.shape[0]):
        step_k = int(cpn_steps[k])
        if 0 <= step_k <= n:
            cpn_indicator = cpn_indicator.at[step_k].add(1.0)
    coupon_at_step = cpn_indicator * coupon

    call_price_at_step = jnp.full(n + 1, jnp.inf)
    for k in range(bond.call_dates.shape[0]):
        step_k = int(call_steps[k])
        if 0 <= step_k <= n:
            call_price_at_step = call_price_at_step.at[step_k].set(
                bond.call_prices[k] * bond.face_value
            )

    # Terminal value: face + final coupon (if maturity is a coupon date).
    values = jnp.full(n_states, 1.0) * bond.face_value + coupon_at_step[n]

    # Backward induction.
    def step_fn(i_fwd, values):
        """i_fwd counts 0 … n_steps-1 forward; actual step = n - 1 - i_fwd."""
        step = n - 1 - i_fwd
        rates_step = tree.rates[step]
        cont = _backward_one_step(values, rates_step, tree.dt, tree.probs, tree.targets)

        # Add coupon at this step.
        cont = cont + coupon_at_step[step]

        # Call exercise: issuer calls if cont > call_price.
        cont = jnp.minimum(cont, call_price_at_step[step])

        return cont

    values = jax.lax.fori_loop(0, n, step_fn, values)
    return values[tree.j_max]  # root node


# ─────────────────────────────────────────────────────────────────────
# Puttable bond pricing
# ─────────────────────────────────────────────────────────────────────

def puttable_bond_price(
    bond: PuttableBond,
    model: HullWhiteModel,
    n_steps: int = 100,
) -> Float[Array, ""]:
    """Price a puttable fixed-rate bond on a Hull-White trinomial tree.

    The bondholder exercises the put (forcing early redemption at
    ``put_price``) whenever the continuation value falls below the put
    price — i.e. when rates have risen enough to make the bond less
    valuable than the guaranteed redemption.  This **increases** the
    bond's value relative to a non-puttable bond.

    Args:
        bond: Puttable bond instrument.
        model: Hull-White model.
        n_steps: Number of tree time steps.

    Returns:
        Puttable bond price (dirty price at ``settlement_date``).
    """
    ref = model.initial_curve.reference_date
    maturity_time = float(year_fraction(ref, bond.payment_dates[-1], bond.day_count))
    tree = build_hull_white_tree(model, maturity_time, n_steps)

    n_states = 2 * tree.j_max + 1
    coupon = bond.face_value * bond.coupon_rate / bond.frequency

    cpn_steps = _snap_dates_to_steps(
        bond.payment_dates, ref, tree.dt, bond.day_count
    )
    put_steps = _snap_dates_to_steps(
        bond.put_dates, ref, tree.dt, bond.day_count
    )

    n = tree.n_steps
    cpn_indicator = jnp.zeros(n + 1)
    for k in range(bond.payment_dates.shape[0]):
        step_k = int(cpn_steps[k])
        if 0 <= step_k <= n:
            cpn_indicator = cpn_indicator.at[step_k].add(1.0)
    coupon_at_step = cpn_indicator * coupon

    put_price_at_step = jnp.full(n + 1, -jnp.inf)
    for k in range(bond.put_dates.shape[0]):
        step_k = int(put_steps[k])
        if 0 <= step_k <= n:
            put_price_at_step = put_price_at_step.at[step_k].set(
                bond.put_prices[k] * bond.face_value
            )

    # Terminal value.
    values = jnp.full(n_states, 1.0) * bond.face_value + coupon_at_step[n]

    def step_fn(i_fwd, values):
        step = n - 1 - i_fwd
        rates_step = tree.rates[step]
        cont = _backward_one_step(values, rates_step, tree.dt, tree.probs, tree.targets)
        cont = cont + coupon_at_step[step]
        # Put exercise: holder puts if cont < put_price.
        cont = jnp.maximum(cont, put_price_at_step[step])
        return cont

    values = jax.lax.fori_loop(0, n, step_fn, values)
    return values[tree.j_max]
