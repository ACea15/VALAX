"""Curve bootstrapping: construct a DiscountCurve from market quotes.

Two approaches are provided:

* **Sequential bootstrap** — processes instruments one at a time in maturity
  order, solving analytically for each new discount factor.  Fast, simple,
  and differentiable (gradients flow through standard JAX ops).

* **Simultaneous bootstrap** — solves for all discount factors at once via
  ``optimistix.root_find``.  More robust when intermediate payment dates
  don't coincide with pillars, and supports ``ImplicitAdjoint`` for
  differentiable curve construction.

Both return a standard ``DiscountCurve`` that is fully compatible with all
existing pricing functions.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.curves.instruments import DepositRate, FRA, SwapRate
from valax.dates.daycounts import year_fraction


# ── Sequential bootstrap ─────────────────────────────────────────────


def bootstrap_sequential(
    reference_date: Int[Array, ""],
    instruments: list,
    day_count: str = "act_365",
) -> DiscountCurve:
    """Build a discount curve by sequentially solving for each pillar DF.

    Instruments must be pre-sorted by maturity.  Each instrument adds
    exactly one new pillar to the curve.  The function uses standard
    Python control flow (not JIT-able as a whole), but each step uses
    JAX operations so ``jax.grad`` flows through the result.

    Args:
        reference_date: Valuation date (ordinal).
        instruments: List of ``DepositRate``, ``FRA``, or ``SwapRate``
            objects, sorted by maturity.
        day_count: Day count convention for the output curve.

    Returns:
        A ``DiscountCurve`` with one pillar per instrument plus the
        reference date (DF = 1).
    """
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
            raise TypeError(f"Unknown instrument type: {type(inst)}")

    all_dates = jnp.stack([jnp.asarray(d, dtype=jnp.int32) for d in pillar_dates])
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
    df_start = _lookup_df(pillar_dates, dfs, inst.start_date, reference_date, day_count)
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
        DF_last = (DF_start - rate * sum_{known} tau_i * DF_i) / (1 + rate * tau_last)
    """
    n_fixed = inst.fixed_dates.shape[0]

    # Accrual fractions for each fixed period
    starts = jnp.concatenate([inst.start_date[None], inst.fixed_dates[:-1]])
    taus = year_fraction(starts, inst.fixed_dates, inst.day_count)

    df_start = _lookup_df(
        pillar_dates, dfs, inst.start_date, reference_date, day_count,
    )

    # Look up DFs at intermediate payment dates (all but last)
    if n_fixed > 1:
        known_dfs = jnp.stack([
            _lookup_df(pillar_dates, dfs, inst.fixed_dates[i], reference_date, day_count)
            for i in range(n_fixed - 1)
        ])
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
    # Build a temporary curve from accumulated pillars
    dates_arr = jnp.stack([jnp.asarray(d, dtype=jnp.int32) for d in pillar_dates])
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
) -> DiscountCurve:
    """Build a discount curve by solving all pillar DFs simultaneously.

    Uses ``optimistix.root_find`` to find the log-DFs that make every
    instrument reprice to par.  The system must be square: one instrument
    per pillar.

    Working in log-DF space guarantees positive discount factors throughout
    the iteration and matches the log-linear interpolation of
    ``DiscountCurve``.

    Args:
        reference_date: Valuation date (ordinal).
        pillar_dates: Sorted ordinal dates for curve nodes (shape n).
        instruments: List of ``DepositRate``, ``FRA``, or ``SwapRate``
            (length must equal n).
        day_count: Day count convention.
        initial_guess: Starting log-DFs (shape n).  Defaults to a flat
            4% curve.
        solver: An ``optimistix.AbstractRootFinder``.  Defaults to
            ``optimistix.Newton``.
        max_steps: Maximum solver iterations.

    Returns:
        A ``DiscountCurve`` with the specified pillars plus DF=1 at
        reference_date.
    """
    n = pillar_dates.shape[0]
    if len(instruments) != n:
        raise ValueError(
            f"Need exactly one instrument per pillar: got {len(instruments)} "
            f"instruments for {n} pillars."
        )

    # Pre-compute instrument data as JAX arrays for JIT compatibility
    packed = _pack_instruments(instruments)

    if initial_guess is None:
        pillar_times = year_fraction(reference_date, pillar_dates, day_count)
        initial_guess = -0.04 * pillar_times  # log-DF for ~4% flat

    if solver is None:
        solver = optx.Newton(rtol=1e-10, atol=1e-10)

    def residual_fn(log_dfs, args):
        ref, pillars, pack, dc = args
        all_dates = jnp.concatenate([ref[None], pillars])
        all_dfs = jnp.concatenate([jnp.ones(1), jnp.exp(log_dfs)])
        curve = DiscountCurve(
            pillar_dates=all_dates.astype(jnp.int32),
            discount_factors=all_dfs,
            reference_date=ref.astype(jnp.int32),
            day_count=dc,
        )
        return _compute_residuals(curve, pack)

    args = (
        jnp.asarray(reference_date, dtype=jnp.float64),
        jnp.asarray(pillar_dates, dtype=jnp.float64),
        packed,
        day_count,
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


# ── Packed instrument representation for JIT ─────────────────────────

# Instrument type tags
_DEPOSIT = 0
_FRA = 1
_SWAP = 2


class _PackedInstruments(eqx.Module):
    """Pre-packed instrument data for the simultaneous bootstrapper.

    All instruments are encoded into fixed-size arrays so that the
    residual function is JIT-compatible (no Python-level dispatch).

    The ``types`` field is static so ``jax.lax.switch`` can branch
    on instrument type at trace time.
    """

    types: tuple = eqx.field(static=True)  # tuple of ints
    start_dates: Int[Array, " n"]
    end_dates: Int[Array, " n"]
    rates: Float[Array, " n"]
    day_counts: tuple = eqx.field(static=True)  # tuple of str
    # For swaps: padded fixed-date schedules
    swap_fixed_dates: Int[Array, "n max_fixed"]
    swap_n_fixed: tuple = eqx.field(static=True)  # tuple of ints


def _pack_instruments(instruments: list) -> _PackedInstruments:
    """Convert a heterogeneous instrument list into packed arrays."""
    n = len(instruments)

    types = []
    start_dates = []
    end_dates = []
    rates = []
    day_counts = []
    fixed_date_lists = []
    n_fixed_list = []

    max_fixed = 1  # minimum pad width

    for inst in instruments:
        if isinstance(inst, DepositRate):
            types.append(_DEPOSIT)
            start_dates.append(inst.start_date)
            end_dates.append(inst.end_date)
            rates.append(inst.rate)
            day_counts.append(inst.day_count)
            fixed_date_lists.append([])
            n_fixed_list.append(0)
        elif isinstance(inst, FRA):
            types.append(_FRA)
            start_dates.append(inst.start_date)
            end_dates.append(inst.end_date)
            rates.append(inst.rate)
            day_counts.append(inst.day_count)
            fixed_date_lists.append([])
            n_fixed_list.append(0)
        elif isinstance(inst, SwapRate):
            types.append(_SWAP)
            start_dates.append(inst.start_date)
            fd = inst.fixed_dates
            end_dates.append(fd[-1])
            rates.append(inst.rate)
            day_counts.append(inst.day_count)
            fd_list = [int(fd[i]) for i in range(fd.shape[0])]
            fixed_date_lists.append(fd_list)
            n_fixed_list.append(len(fd_list))
            max_fixed = max(max_fixed, len(fd_list))
        else:
            raise TypeError(f"Unknown instrument type: {type(inst)}")

    # Pad fixed date arrays
    swap_fixed = jnp.zeros((n, max_fixed), dtype=jnp.int32)
    for i, fd_list in enumerate(fixed_date_lists):
        for j, d in enumerate(fd_list):
            swap_fixed = swap_fixed.at[i, j].set(d)

    return _PackedInstruments(
        types=tuple(types),
        start_dates=jnp.stack([jnp.asarray(d, dtype=jnp.int32) for d in start_dates]),
        end_dates=jnp.stack([jnp.asarray(d, dtype=jnp.int32) for d in end_dates]),
        rates=jnp.stack([jnp.asarray(r) for r in rates]),
        day_counts=tuple(day_counts),
        swap_fixed_dates=swap_fixed,
        swap_n_fixed=tuple(n_fixed_list),
    )


def _compute_residuals(
    curve: DiscountCurve,
    packed: _PackedInstruments,
) -> Float[Array, " n"]:
    """Compute pricing residuals for all packed instruments.

    Uses Python-level dispatch on the static ``types`` tuple (unrolled
    at trace time).  Each residual should be zero when the curve is
    correctly calibrated.
    """
    residuals = []

    for i, itype in enumerate(packed.types):
        start = packed.start_dates[i]
        end = packed.end_dates[i]
        rate = packed.rates[i]
        dc = packed.day_counts[i]

        if itype == _DEPOSIT or itype == _FRA:
            # DF(end) * (1 + rate * tau) - DF(start) = 0
            tau = year_fraction(start, end, dc)
            df_start = curve(start)
            df_end = curve(end)
            residuals.append(df_end * (1.0 + rate * tau) - df_start)

        elif itype == _SWAP:
            # rate * annuity - (DF(start) - DF(maturity)) = 0
            n_fixed = packed.swap_n_fixed[i]
            fixed_dates = packed.swap_fixed_dates[i, :n_fixed]

            starts_arr = jnp.concatenate([start[None], fixed_dates[:-1]])
            taus = year_fraction(starts_arr, fixed_dates, dc)
            fixed_dfs = curve(fixed_dates)
            annuity = jnp.sum(taus * fixed_dfs)

            df_start = curve(start)
            df_mat = curve(fixed_dates[-1])
            residuals.append(rate * annuity - (df_start - df_mat))

    return jnp.stack(residuals)
