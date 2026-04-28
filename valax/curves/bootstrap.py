"""Curve bootstrapping: construct a DiscountCurve from market quotes.

Two approaches are provided:

* **Sequential bootstrap** — processes instruments one at a time in
  maturity order, solving analytically for each new discount factor.
  Fast, simple, and differentiable (gradients flow through standard
  JAX ops).  Only the three classic quote types
  (:class:`DepositRate`, :class:`FRA`, :class:`SwapRate`) are
  supported sequentially because each must admit a closed-form
  per-pillar update.

* **Simultaneous bootstrap** — solves for all discount factors at
  once via ``optimistix.root_find``.  More robust when intermediate
  payment dates don't coincide with pillars, and supports
  ``ImplicitAdjoint`` for differentiable curve construction.  The
  solver iterates over the input list of instruments, asking each
  for its residual via the
  :class:`~valax.curves.bootstrap_proto.BootstrapInstrument` protocol.

Both return a standard :class:`DiscountCurve` that is fully
compatible with all existing pricing functions.

Migration note (MC-Curves-1, task #3): the simultaneous bootstrap was
refactored to call ``inst.residual(graph, fixings, ref_date)`` rather
than dispatching on instrument type via a packed-arrays representation.
This removes the previous ``_PackedInstruments`` machinery and lets
new instrument types plug in with no edits to this file.  Public
function signatures are unchanged.
"""

import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.curves.fixings import FixingHistory, empty_fixing_history
from valax.curves.graph import CurveGraph
from valax.curves.instruments import DepositRate, FRA, SwapRate
from valax.dates.daycounts import year_fraction


_DEFAULT_CURVE_ID = "_default_"


# ── Sequential bootstrap ─────────────────────────────────────────────


def bootstrap_sequential(
    reference_date: Int[Array, ""],
    instruments: list,
    day_count: str = "act_365",
    fixings: FixingHistory | None = None,
) -> DiscountCurve:
    """Build a discount curve by sequentially solving for each pillar DF.

    Instruments must be pre-sorted by maturity.  Each instrument adds
    exactly one new pillar to the curve.  The function uses standard
    Python control flow (not JIT-able as a whole), but each step uses
    JAX operations so ``jax.grad`` flows through the result.

    Sequential bootstrapping requires a closed-form per-pillar update,
    which only the three classic quote types
    (:class:`DepositRate`, :class:`FRA`, :class:`SwapRate`) admit.
    Other instruments raise :class:`TypeError` here and should be
    bootstrapped with :func:`bootstrap_simultaneous` or the joint
    multi-curve solver instead.

    Args:
        reference_date: Valuation date (ordinal).
        instruments: List of bootstrap quotes, sorted by maturity.
        day_count: Day count convention for the output curve.
        fixings: Optional realised-fixings registry.  Unused by the
            three classic quote types but accepted for forward
            compatibility.

    Returns:
        A :class:`DiscountCurve` with one pillar per instrument plus
        the reference date (DF = 1).
    """
    del fixings  # the three classic quote types ignore fixings

    pillar_dates: list = [reference_date]
    dfs: list = [jnp.array(1.0)]

    for inst in instruments:
        if isinstance(inst, (DepositRate, FRA)):
            df_new = _bootstrap_deposit_or_fra(
                inst, pillar_dates, dfs, reference_date, day_count,
            )
            pillar_dates.append(inst.end_date)
            dfs.append(df_new)

        elif isinstance(inst, SwapRate):
            df_new = _bootstrap_swap(
                inst, pillar_dates, dfs, reference_date, day_count,
            )
            pillar_dates.append(inst.fixed_dates[-1])
            dfs.append(df_new)

        else:
            raise TypeError(
                f"Sequential bootstrap supports DepositRate, FRA, and "
                f"SwapRate only.  Got {type(inst).__name__}.  Use "
                f"bootstrap_simultaneous (or, when MC-Curves-2 lands, "
                f"bootstrap_curve_graph) for other instrument types."
            )

    all_dates = jnp.stack(
        [jnp.asarray(d, dtype=jnp.int32) for d in pillar_dates]
    )
    all_dfs = jnp.stack(dfs)

    return DiscountCurve(
        pillar_dates=all_dates,
        discount_factors=all_dfs,
        reference_date=jnp.asarray(reference_date, dtype=jnp.int32),
        day_count=day_count,
    )


def _bootstrap_deposit_or_fra(
    inst,
    pillar_dates: list,
    dfs: list,
    reference_date,
    day_count: str,
) -> Float[Array, ""]:
    """Solve for DF(end) from a deposit or FRA quote."""
    tau = year_fraction(inst.start_date, inst.end_date, inst.day_count)
    df_start = _lookup_df(
        pillar_dates, dfs, inst.start_date, reference_date, day_count
    )
    return df_start / (1.0 + inst.rate * tau)


def _bootstrap_swap(
    inst: SwapRate,
    pillar_dates: list,
    dfs: list,
    reference_date,
    day_count: str,
) -> Float[Array, ""]:
    """Solve for DF(maturity) from a par swap rate quote.

    par_rate * sum(tau_i * DF_i) = DF(start) - DF(maturity)

    Rearranging for DF(maturity):
        DF_last = (DF_start - rate * sum_{known} tau_i * DF_i)
                  / (1 + rate * tau_last)
    """
    n_fixed = inst.fixed_dates.shape[0]

    # Accrual fractions for each fixed period.
    starts = jnp.concatenate([inst.start_date[None], inst.fixed_dates[:-1]])
    taus = year_fraction(starts, inst.fixed_dates, inst.day_count)

    df_start = _lookup_df(
        pillar_dates, dfs, inst.start_date, reference_date, day_count,
    )

    # Look up DFs at intermediate payment dates (all but last).
    if n_fixed > 1:
        known_dfs = jnp.stack(
            [
                _lookup_df(
                    pillar_dates,
                    dfs,
                    inst.fixed_dates[i],
                    reference_date,
                    day_count,
                )
                for i in range(n_fixed - 1)
            ]
        )
        known_annuity = jnp.sum(taus[:-1] * known_dfs)
    else:
        known_annuity = jnp.array(0.0)

    numerator = df_start - inst.rate * known_annuity
    df_last = numerator / (1.0 + inst.rate * taus[-1])
    return df_last


def _lookup_df(
    pillar_dates: list,
    dfs: list,
    date,
    reference_date,
    day_count: str,
) -> Float[Array, ""]:
    """Look up or interpolate a DF from the partially-built curve."""
    dates_arr = jnp.stack(
        [jnp.asarray(d, dtype=jnp.int32) for d in pillar_dates]
    )
    dfs_arr = jnp.stack(dfs)
    temp_curve = DiscountCurve(
        pillar_dates=dates_arr,
        discount_factors=dfs_arr,
        reference_date=jnp.asarray(reference_date, dtype=jnp.int32),
        day_count=day_count,
    )
    return temp_curve(jnp.asarray(date, dtype=jnp.int32))


# ── Simultaneous bootstrap ───────────────────────────────────────────


def bootstrap_simultaneous(
    reference_date: Int[Array, ""],
    pillar_dates: Int[Array, " n_pillars"],
    instruments: list,
    day_count: str = "act_365",
    initial_guess: Float[Array, " n_pillars"] | None = None,
    solver: optx.AbstractRootFinder | None = None,
    max_steps: int = 256,
    fixings: FixingHistory | None = None,
) -> DiscountCurve:
    """Build a discount curve by solving all pillar DFs simultaneously.

    Uses ``optimistix.root_find`` to find the log-DFs that make every
    instrument reprice to par.  The system must be square: one
    instrument per pillar.

    Working in log-DF space guarantees positive discount factors
    throughout the iteration and matches the log-linear interpolation
    of :class:`DiscountCurve`.

    Each instrument is asked for its residual via the
    :class:`BootstrapInstrument` protocol — the solver is agnostic to
    instrument type.  New instrument types only need to implement
    ``residual(graph, fixings, ref_date)``; no edits to this file are
    required.

    Args:
        reference_date: Valuation date (ordinal).
        pillar_dates: Sorted ordinal dates for curve nodes (shape ``n``).
        instruments: List of :class:`BootstrapInstrument` (length must
            equal ``n``).
        day_count: Day count convention for the output curve.
        initial_guess: Starting log-DFs (shape ``n``).  Defaults to a
            flat 4% curve.
        solver: An ``optimistix.AbstractRootFinder``.  Defaults to
            ``optimistix.Newton``.
        max_steps: Maximum solver iterations.
        fixings: Realised-fixings registry passed to each instrument's
            residual.  Defaults to an empty registry, which is
            sufficient for the three classic quote types.

    Returns:
        A :class:`DiscountCurve` with the specified pillars plus
        ``DF = 1`` at the reference date.
    """
    n = pillar_dates.shape[0]
    if len(instruments) != n:
        raise ValueError(
            f"Need exactly one instrument per pillar: got {len(instruments)} "
            f"instruments for {n} pillars."
        )

    if fixings is None:
        fixings = empty_fixing_history()

    if initial_guess is None:
        pillar_times = year_fraction(reference_date, pillar_dates, day_count)
        initial_guess = -0.04 * pillar_times  # log-DF for ~4% flat

    if solver is None:
        solver = optx.Newton(rtol=1e-10, atol=1e-10)

    def residual_fn(log_dfs, args):
        ref, pillars, instruments_inner, dc, fixings_inner = args
        all_dates = jnp.concatenate([ref[None], pillars])
        all_dfs = jnp.concatenate([jnp.ones(1), jnp.exp(log_dfs)])
        curve = DiscountCurve(
            pillar_dates=all_dates.astype(jnp.int32),
            discount_factors=all_dfs,
            reference_date=ref.astype(jnp.int32),
            day_count=dc,
        )
        # Single-curve bootstrap: wrap the in-progress curve in a graph
        # keyed by the sentinel id that the classic instrument types
        # default to.  Multi-curve users (MC-Curves-2+) drive the joint
        # solver directly and never enter this path.
        graph = CurveGraph(curves={_DEFAULT_CURVE_ID: curve})
        ref_int = ref.astype(jnp.int32)
        return jnp.stack(
            [
                inst.residual(graph, fixings_inner, ref_int)
                for inst in instruments_inner
            ]
        )

    args = (
        jnp.asarray(reference_date, dtype=jnp.float64),
        jnp.asarray(pillar_dates, dtype=jnp.float64),
        instruments,
        day_count,
        fixings,
    )

    sol = optx.root_find(
        residual_fn, solver, initial_guess, args=args, max_steps=max_steps,
    )

    log_dfs = sol.value
    all_dates = jnp.concatenate([reference_date[None], pillar_dates])
    all_dfs = jnp.concatenate([jnp.ones(1), jnp.exp(log_dfs)])

    return DiscountCurve(
        pillar_dates=all_dates.astype(jnp.int32),
        discount_factors=all_dfs,
        reference_date=jnp.asarray(reference_date, dtype=jnp.int32),
        day_count=day_count,
    )
