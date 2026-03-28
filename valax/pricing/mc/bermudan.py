"""Bermudan swaption pricing via Longstaff-Schwartz Monte Carlo.

Prices Bermudan swaptions under the LMM by backward induction through
exercise dates, using least-squares regression to estimate continuation
values. All operations are JIT-compatible via jax.lax.fori_loop.

Reference: Longstaff & Schwartz (2001), "Valuing American Options by
Simulation: A Simple Least-Squares Approach".
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.instruments.rates import BermudanSwaption
from valax.pricing.mc.lmm_paths import LMMPathResult


class LSMConfig(eqx.Module):
    """Longstaff-Schwartz regression configuration.

    Attributes:
        poly_degree: Maximum polynomial degree for the regression basis.
                     Basis is [1, x, x^2, ..., x^poly_degree] where x = swap rate.
    """

    poly_degree: int = eqx.field(static=True, default=3)


def _tail_swap_value(
    forwards_at_tenor: Float[Array, "n_paths N"],
    taus: Float[Array, " N"],
    exercise_idx: Int[Array, ""],
    strike: Float[Array, ""],
    notional: Float[Array, ""],
    is_payer: bool,
) -> tuple[Float[Array, " n_paths"], Float[Array, " n_paths"]]:
    """Compute tail swap exercise value and swap rate at a given exercise date.

    At exercise date T_e, the tail swap runs from T_e to T_N with forwards
    F_j(T_e) for j = e, ..., N-1.

    Args:
        forwards_at_tenor: Forward rates at time T_e, shape (n_paths, N).
                           Dead forwards (j < e) should be zero.
        taus: Accrual fractions for each forward period, shape (N,).
        exercise_idx: Tenor index e for this exercise date.
        strike: Fixed rate K.
        notional: Notional principal.
        is_payer: True for payer swaption.

    Returns:
        (exercise_value, swap_rate) each shape (n_paths,).
    """
    N = taus.shape[0]
    j_indices = jnp.arange(N)
    alive = (j_indices >= exercise_idx).astype(forwards_at_tenor.dtype)

    # Accrual terms: (1 + tau_j * F_j) for alive forwards, 1.0 for dead
    accrual = jnp.where(alive[None, :], 1.0 + taus[None, :] * forwards_at_tenor, 1.0)

    # Cumulative product → relative discount factors from T_e
    cum_accrual = jnp.cumprod(accrual, axis=1)
    rel_df = 1.0 / cum_accrual  # (n_paths, N)

    # Annuity: sum_{j>=e} tau_j * DF(T_e, T_{j+1})
    annuity = jnp.sum(alive[None, :] * taus[None, :] * rel_df, axis=1)

    # Terminal DF: DF(T_e, T_N) = last alive rel_df
    df_TN = rel_df[:, -1]

    # Swap rate
    swap_rate = (1.0 - df_TN) / jnp.maximum(annuity, 1e-12)

    # Intrinsic value
    if is_payer:
        intrinsic = jnp.maximum(swap_rate - strike, 0.0)
    else:
        intrinsic = jnp.maximum(strike - swap_rate, 0.0)

    exercise_value = notional * annuity * intrinsic
    return exercise_value, swap_rate


def _discount_between(
    forwards_at_tenor: Float[Array, "n_paths N"],
    taus: Float[Array, " N"],
    from_idx: Int[Array, ""],
    to_idx: Int[Array, ""],
) -> Float[Array, " n_paths"]:
    """Compute DF(T_from, T_to) from forwards observed at T_from.

    DF(T_from, T_to) = 1 / prod_{j=from}^{to-1} (1 + tau_j * F_j(T_from))
    """
    N = taus.shape[0]
    j_indices = jnp.arange(N)
    active = ((j_indices >= from_idx) & (j_indices < to_idx)).astype(taus.dtype)

    accrual = jnp.where(active[None, :], 1.0 + taus[None, :] * forwards_at_tenor, 1.0)
    return 1.0 / jnp.prod(accrual, axis=1)


def _build_basis(
    swap_rate: Float[Array, " n_paths"],
    poly_degree: int,
) -> Float[Array, "n_paths n_basis"]:
    """Polynomial basis matrix: [1, x, x^2, ..., x^poly_degree]."""
    powers = jnp.arange(poly_degree + 1)
    return swap_rate[:, None] ** powers[None, :]


def bermudan_swaption_lsm(
    result: LMMPathResult,
    swaption: BermudanSwaption,
    exercise_indices: Int[Array, " n_exercise"],
    taus: Float[Array, " N"],
    config: LSMConfig = LSMConfig(),
) -> Float[Array, ""]:
    """Price a Bermudan swaption via Longstaff-Schwartz Monte Carlo.

    Uses backward induction through exercise dates with polynomial
    regression for continuation value estimation.

    Args:
        result: LMM path simulation result (must include forwards_at_tenors).
        swaption: Bermudan swaption instrument.
        exercise_indices: Indices into the LMM tenor structure for each exercise
                          date, shape (n_exercise,). Must be sorted ascending.
        taus: Accrual fractions for each forward period, shape (N,).
        config: LS regression configuration.

    Returns:
        Bermudan swaption price (scalar), discounted to time 0.
    """
    n_ex = exercise_indices.shape[0]
    fwd_tenors = result.forwards_at_tenors  # (n_paths, N, N)

    # --- Last exercise date: exercise iff positive ---
    last_idx = exercise_indices[n_ex - 1]
    ex_val_last, _ = _tail_swap_value(
        fwd_tenors[:, last_idx, :], taus, last_idx,
        swaption.strike, swaption.notional, swaption.is_payer,
    )
    # cashflow_pv: the undiscounted exercise value at the current exercise date
    # (will be discounted backward step by step)
    cashflow_pv = ex_val_last

    # --- Backward loop from e = n_ex-2 down to 0 ---
    def body_fn(loop_idx, cashflow_pv):
        e = n_ex - 2 - loop_idx  # backward: n_ex-2, n_ex-3, ..., 0
        ex_idx = exercise_indices[e]
        next_ex_idx = exercise_indices[e + 1]

        # Discount cashflow from T_{next} back to T_e
        fwd_e = fwd_tenors[:, ex_idx, :]
        df_e_to_next = _discount_between(fwd_e, taus, ex_idx, next_ex_idx)
        discounted_future = cashflow_pv * df_e_to_next

        # Exercise value at T_e
        ex_val_e, swap_rate_e = _tail_swap_value(
            fwd_e, taus, ex_idx,
            swaption.strike, swaption.notional, swaption.is_payer,
        )

        # LS regression: estimate E[discounted_future | swap_rate_e]
        # Only use ITM paths for regression (standard LS trick)
        itm = ex_val_e > 0.0
        basis = _build_basis(swap_rate_e, config.poly_degree)

        # Weighted least squares with Tikhonov regularization
        W = itm.astype(basis.dtype)
        XtWX = (basis * W[:, None]).T @ basis
        XtWy = (basis * W[:, None]).T @ discounted_future
        reg = 1e-8 * jnp.eye(config.poly_degree + 1)
        beta = jnp.linalg.solve(XtWX + reg, XtWy)

        continuation = basis @ beta

        # Exercise if ITM and exercise value > continuation value
        exercise_now = itm & (ex_val_e > continuation)
        cashflow_pv = jnp.where(exercise_now, ex_val_e, discounted_future)

        return cashflow_pv

    cashflow_pv = jax.lax.fori_loop(0, n_ex - 1, body_fn, cashflow_pv)

    # Discount from first exercise date to time 0
    first_ex_idx = exercise_indices[0]
    df_0_to_first = result.discount_factors[:, first_ex_idx]
    price = jnp.mean(cashflow_pv * df_0_to_first)

    return price
