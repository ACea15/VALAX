"""Multi-curve framework: separate discount and forward projection curves.

Post-crisis, OIS discounting + tenor-specific forward curves is the standard.
For example, SOFR OIS for discounting and separate 1M/3M SOFR curves for
forward rate projection.

The dual-curve bootstrap first builds the OIS discount curve, then
bootstraps each forward curve using the OIS curve for discounting.
"""

import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.curves.instruments import DepositRate, FRA, SwapRate
from valax.curves.bootstrap import bootstrap_sequential
from valax.dates.daycounts import year_fraction


class MultiCurveSet(eqx.Module):
    """Container for a set of related yield curves.

    Attributes:
        discount_curve: OIS discount curve used for present-value discounting.
        forward_curves: Tenor-keyed forward projection curves
            (e.g., ``{"3M": curve_3m, "6M": curve_6m}``).
    """

    discount_curve: DiscountCurve
    forward_curves: dict  # str -> DiscountCurve (keys are static in the pytree)


def bootstrap_multi_curve(
    reference_date: Int[Array, ""],
    discount_instruments: list,
    forward_instruments: dict[str, list],
    day_count: str = "act_365",
    solver: optx.AbstractRootFinder | None = None,
    max_steps: int = 256,
) -> MultiCurveSet:
    """Build a multi-curve set from market quotes.

    1. Bootstraps the OIS discount curve from ``discount_instruments``
       using the sequential method.
    2. For each tenor in ``forward_instruments``, bootstraps a forward
       projection curve where swaps use the OIS curve for discounting
       but the forward curve for rate projection.

    For deposits and FRAs in the forward instrument set, the bootstrap
    is identical to single-curve (the DF relationship is independent of
    the discount curve).  For swaps, the dual-curve par condition is:

        sum_i [ (DF_fwd(T_{i-1}) / DF_fwd(T_i) - 1) * DF_ois(T_i) ]
        = swap_rate * sum_i [ tau_i * DF_ois(T_i) ]

    Args:
        reference_date: Valuation date (ordinal).
        discount_instruments: Instruments for the OIS discount curve.
        forward_instruments: Dict mapping tenor label to instrument list
            for each forward curve.
        day_count: Day count convention.
        solver: Root finder for the forward curve simultaneous bootstrap.
        max_steps: Maximum solver iterations.

    Returns:
        A ``MultiCurveSet`` with the discount curve and all forward curves.
    """
    # Step 1: Build OIS discount curve (sequential — straightforward)
    discount_curve = bootstrap_sequential(
        reference_date, discount_instruments, day_count,
    )

    # Step 2: Build each forward curve
    forward_curves = {}
    for tenor, fwd_instruments in forward_instruments.items():
        fwd_curve = _bootstrap_forward_curve(
            reference_date=reference_date,
            instruments=fwd_instruments,
            discount_curve=discount_curve,
            day_count=day_count,
            solver=solver,
            max_steps=max_steps,
        )
        forward_curves[tenor] = fwd_curve

    return MultiCurveSet(
        discount_curve=discount_curve,
        forward_curves=forward_curves,
    )


def _bootstrap_forward_curve(
    reference_date: Int[Array, ""],
    instruments: list,
    discount_curve: DiscountCurve,
    day_count: str,
    solver: optx.AbstractRootFinder | None,
    max_steps: int,
) -> DiscountCurve:
    """Bootstrap a forward projection curve using OIS discounting.

    Deposits and FRAs are bootstrapped sequentially (their DF formula
    doesn't depend on the discount curve).  If any swaps are present,
    falls back to a simultaneous solve for the swap pillars.
    """
    # Separate deposits/FRAs (can be done sequentially) from swaps
    simple_insts = []
    swap_insts = []
    for inst in instruments:
        if isinstance(inst, (DepositRate, FRA)):
            simple_insts.append(inst)
        elif isinstance(inst, SwapRate):
            swap_insts.append(inst)
        else:
            raise TypeError(f"Unknown instrument type: {type(inst)}")

    # Bootstrap the short end sequentially
    pillar_dates: list = [reference_date]
    dfs: list = [jnp.array(1.0)]

    for inst in simple_insts:
        tau = year_fraction(inst.start_date, inst.end_date, inst.day_count)
        df_start = _lookup_df_from_lists(
            pillar_dates, dfs, inst.start_date, reference_date, day_count,
        )
        df_new = df_start / (1.0 + inst.rate * tau)
        pillar_dates.append(inst.end_date)
        dfs.append(df_new)

    if not swap_insts:
        # No swaps — we're done
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

    # For swaps: simultaneous solve for the swap-pillar DFs
    swap_pillar_dates = jnp.stack(
        [jnp.asarray(s.fixed_dates[-1], dtype=jnp.int32) for s in swap_insts]
    )

    # Build the known (short-end) curve as a base
    known_dates = jnp.stack(
        [jnp.asarray(d, dtype=jnp.int32) for d in pillar_dates]
    )
    known_dfs = jnp.stack(dfs)

    # Initial guess: flat 4% from known curve end
    swap_times = year_fraction(reference_date, swap_pillar_dates, day_count)
    initial_log_dfs = -0.04 * swap_times

    if solver is None:
        solver = optx.Newton(rtol=1e-10, atol=1e-10)

    def residual_fn(log_dfs, args):
        kd, kdf, sp, ref, dc_curve, dc, swaps_data = args
        # Build full forward curve: known pillars + swap pillars
        all_dates = jnp.concatenate([kd, sp]).astype(jnp.int32)
        all_dfs_val = jnp.concatenate([kdf, jnp.exp(log_dfs)])
        fwd_curve = DiscountCurve(
            pillar_dates=all_dates,
            discount_factors=all_dfs_val,
            reference_date=ref.astype(jnp.int32),
            day_count=dc,
        )
        return _dual_curve_swap_residuals(fwd_curve, dc_curve, swaps_data, dc)

    # Pack swap data for JIT
    swaps_data = _pack_swap_data(swap_insts)

    args = (
        known_dates.astype(jnp.float64),
        known_dfs,
        swap_pillar_dates.astype(jnp.float64),
        jnp.asarray(reference_date, dtype=jnp.float64),
        discount_curve,
        day_count,
        swaps_data,
    )

    sol = optx.root_find(
        residual_fn, solver, initial_log_dfs, args=args, max_steps=max_steps,
    )

    all_dates = jnp.concatenate([known_dates, swap_pillar_dates]).astype(jnp.int32)
    all_dfs_final = jnp.concatenate([known_dfs, jnp.exp(sol.value)])

    return DiscountCurve(
        pillar_dates=all_dates,
        discount_factors=all_dfs_final,
        reference_date=jnp.asarray(reference_date, dtype=jnp.int32),
        day_count=day_count,
    )


class _SwapData(eqx.Module):
    """Packed swap data for the dual-curve residual function."""

    start_dates: Int[Array, " n"]
    rates: Float[Array, " n"]
    fixed_dates: Int[Array, "n max_fixed"]
    n_fixed: tuple = eqx.field(static=True)
    day_counts: tuple = eqx.field(static=True)


def _pack_swap_data(swap_insts: list[SwapRate]) -> _SwapData:
    """Pack swap instruments into arrays."""
    n = len(swap_insts)
    max_fixed = max(s.fixed_dates.shape[0] for s in swap_insts)

    start_dates = jnp.stack(
        [jnp.asarray(s.start_date, dtype=jnp.int32) for s in swap_insts]
    )
    rates = jnp.stack([jnp.asarray(s.rate) for s in swap_insts])

    fixed_dates = jnp.zeros((n, max_fixed), dtype=jnp.int32)
    n_fixed = []
    day_counts = []
    for i, s in enumerate(swap_insts):
        nf = s.fixed_dates.shape[0]
        n_fixed.append(nf)
        day_counts.append(s.day_count)
        for j in range(nf):
            fixed_dates = fixed_dates.at[i, j].set(s.fixed_dates[j])

    return _SwapData(
        start_dates=start_dates,
        rates=rates,
        fixed_dates=fixed_dates,
        n_fixed=tuple(n_fixed),
        day_counts=tuple(day_counts),
    )


def _dual_curve_swap_residuals(
    forward_curve: DiscountCurve,
    discount_curve: DiscountCurve,
    swaps: _SwapData,
    day_count: str,
) -> Float[Array, " n"]:
    """Residuals for dual-curve swap repricing.

    The par condition under dual-curve is:

        float_leg = sum_i (DF_fwd(T_{i-1})/DF_fwd(T_i) - 1) * DF_ois(T_i)
        fixed_leg = rate * sum_i tau_i * DF_ois(T_i)
        residual  = float_leg - fixed_leg

    This is zero when the forward curve correctly projects the floating
    leg and the OIS curve discounts it.
    """
    residuals = []

    for i, nf in enumerate(swaps.n_fixed):
        start = swaps.start_dates[i]
        rate = swaps.rates[i]
        dc = swaps.day_counts[i]
        fixed_dates = swaps.fixed_dates[i, :nf]

        # Accrual fractions
        starts_arr = jnp.concatenate([start[None], fixed_dates[:-1]])
        taus = year_fraction(starts_arr, fixed_dates, dc)

        # Fixed leg (OIS discounted)
        ois_dfs = discount_curve(fixed_dates)
        fixed_leg = rate * jnp.sum(taus * ois_dfs)

        # Floating leg (forward projected, OIS discounted)
        # Forward rate for period i: (DF_fwd(T_{i-1}) / DF_fwd(T_i) - 1)
        fwd_starts = jnp.concatenate([start[None], fixed_dates[:-1]])
        fwd_df_starts = forward_curve(fwd_starts)
        fwd_df_ends = forward_curve(fixed_dates)
        fwd_rates = fwd_df_starts / fwd_df_ends - 1.0
        float_leg = jnp.sum(fwd_rates * ois_dfs)

        residuals.append(float_leg - fixed_leg)

    return jnp.stack(residuals)


def _lookup_df_from_lists(
    pillar_dates: list,
    dfs: list,
    date,
    reference_date,
    day_count: str,
) -> Float[Array, ""]:
    """Look up or interpolate a DF from lists of accumulated pillars."""
    dates_arr = jnp.stack([jnp.asarray(d, dtype=jnp.int32) for d in pillar_dates])
    dfs_arr = jnp.stack(dfs)
    temp_curve = DiscountCurve(
        pillar_dates=dates_arr,
        discount_factors=dfs_arr,
        reference_date=jnp.asarray(reference_date, dtype=jnp.int32),
        day_count=day_count,
    )
    return temp_curve(jnp.asarray(date, dtype=jnp.int32))
