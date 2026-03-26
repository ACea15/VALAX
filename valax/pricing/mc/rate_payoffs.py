"""Payoff functions for interest rate derivatives under LMM Monte Carlo.

Each payoff takes an LMMPathResult and an instrument, returning per-path
cashflows already discounted to time 0. The discount factors are derived
from the realized forward rates on each path.
"""

import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.instruments.rates import Caplet, Cap, Swaption
from valax.pricing.mc.lmm_paths import LMMPathResult


def caplet_mc_payoff(
    result: LMMPathResult,
    caplet: Caplet,
    forward_index: int,
    tau: Float[Array, ""],
) -> Float[Array, " n_paths"]:
    """Caplet/floorlet payoff under LMM MC, discounted to time 0.

    payoff = DF(0, T_{i+1}) * tau * notional * max(F_i(T_i) - K, 0)

    Args:
        result: LMM path simulation result.
        caplet: Caplet/floorlet instrument.
        forward_index: Index i of the forward rate in the tenor structure.
        tau: Accrual fraction for the caplet period.

    Returns:
        Discounted payoff per path, shape (n_paths,).
    """
    F_i = result.forwards_at_fixing[:, forward_index]
    # DF(0, T_{i+1}) = result.discount_factors[:, forward_index + 1]
    df = result.discount_factors[:, forward_index + 1]

    if caplet.is_cap:
        intrinsic = jnp.maximum(F_i - caplet.strike, 0.0)
    else:
        intrinsic = jnp.maximum(caplet.strike - F_i, 0.0)

    return df * tau * caplet.notional * intrinsic


def cap_mc_payoff(
    result: LMMPathResult,
    cap: Cap,
    forward_indices: Int[Array, " n_caplets"],
    taus: Float[Array, " n_caplets"],
) -> Float[Array, " n_paths"]:
    """Cap/floor payoff: sum of discounted caplet payoffs.

    Args:
        result: LMM path result.
        cap: Cap/floor instrument.
        forward_indices: Indices of forwards for each caplet period.
        taus: Accrual fractions for each period, shape (n_caplets,).

    Returns:
        Discounted total payoff per path, shape (n_paths,).
    """
    F = result.forwards_at_fixing[:, forward_indices]
    df = result.discount_factors[:, forward_indices + 1]

    if cap.is_cap:
        intrinsic = jnp.maximum(F - cap.strike, 0.0)
    else:
        intrinsic = jnp.maximum(cap.strike - F, 0.0)

    caplet_pvs = df * taus[None, :] * cap.notional * intrinsic
    return jnp.sum(caplet_pvs, axis=1)


def swaption_mc_payoff(
    result: LMMPathResult,
    swaption: Swaption,
    forward_indices: Int[Array, " n_periods"],
    taus: Float[Array, " n_periods"],
) -> Float[Array, " n_paths"]:
    """European swaption payoff under LMM MC, discounted to time 0.

    At the swaption expiry (= swap start T_0), computes the swap rate
    from realized forwards:

        annuity = sum_i tau_i * DF(T_0, T_{i+1})
        swap_rate = (1 - DF(T_0, T_N)) / annuity

    Payoff = notional * annuity * max(+/-(swap_rate - K), 0) * DF(0, T_0)

    Args:
        result: LMM path result.
        swaption: Swaption instrument.
        forward_indices: Indices of forwards spanning the underlying swap.
        taus: Accrual fractions for each swap period.

    Returns:
        Discounted payoff per path, shape (n_paths,).
    """
    # Forward rates at their fixing dates
    F = result.forwards_at_fixing[:, forward_indices]

    # Relative discount factors from swap start to each payment date:
    # DF(T_0, T_{i+1}) = 1 / prod_{j=0}^{i} (1 + tau_j * F_j(T_j))
    accrual = 1.0 + taus[None, :] * F
    cum_accrual = jnp.cumprod(accrual, axis=1)
    rel_df = 1.0 / cum_accrual

    # Annuity: sum_i tau_i * DF(T_0, T_{i+1})
    annuity = jnp.sum(taus[None, :] * rel_df, axis=1)

    # Swap rate: (1 - DF(T_0, T_N)) / annuity
    df_TN = rel_df[:, -1]
    swap_rate_val = (1.0 - df_TN) / annuity

    # Exercise value
    K = swaption.strike
    if swaption.is_payer:
        exercise = jnp.maximum(swap_rate_val - K, 0.0)
    else:
        exercise = jnp.maximum(K - swap_rate_val, 0.0)

    # Discount to time 0: DF(0, T_0) where T_0 is the swap start
    df_0_T0 = result.discount_factors[:, forward_indices[0]]

    return swaption.notional * annuity * exercise * df_0_T0
