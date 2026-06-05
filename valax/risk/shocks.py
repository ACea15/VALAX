"""Shock application: apply scenarios to market data.

Core operation for VaR and stress testing. All functions are pure,
JIT-compatible, and differentiable — so ``jax.grad`` through a shocked
repricing gives sensitivities to the shock magnitudes themselves.

The module groups primitives by factor category:

- **IR (single curve)**: ``bump_curve_zero_rates``, ``parallel_shift``,
  ``key_rate_bump``.
- **IR (multi-curve)**: ``bump_discount_curve``, ``bump_forward_curve``,
  ``parallel_basis_shift``.
- **Credit**: ``bump_hazard_rates``, ``parallel_credit_spread_shift``,
  ``key_rate_hazard_bump``.
- **Market-data composite**: ``apply_scenario``.

See ``docs/risk-factors.md`` for the full factor catalogue.
"""

import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.curves.multi_curve import MultiCurveSet
from valax.curves.survival import SurvivalCurve
from valax.dates.daycounts import year_fraction
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario


# ── Curve shocks ─────────────────────────────────────────────────────


def bump_curve_zero_rates(
    curve: DiscountCurve,
    rate_bumps: Float[Array, " n_pillars"],
) -> DiscountCurve:
    """Apply additive zero-rate bumps to each pillar of a discount curve.

    Given continuously-compounded zero rates ``r_i`` at year fractions
    ``t_i``, bumping by ``dr_i`` gives::

        new_df[i] = exp(-(r_i + dr_i) * t_i) = old_df[i] * exp(-dr_i * t_i)

    The returned curve has the same pillar structure; its log-linear
    interpolation naturally produces smooth shocked rates between pillars.

    Follows the same construction pattern as ``key_rate_durations`` in
    ``valax/pricing/analytic/bonds.py``.
    """
    pillar_times = year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count
    )
    adjustment = jnp.exp(-rate_bumps * pillar_times)
    new_dfs = curve.discount_factors * adjustment
    return DiscountCurve(
        pillar_dates=curve.pillar_dates,
        discount_factors=new_dfs,
        reference_date=curve.reference_date,
        day_count=curve.day_count,
    )


def parallel_shift(
    curve: DiscountCurve,
    bump: Float[Array, ""],
) -> DiscountCurve:
    """Parallel shift: bump all pillar zero rates by the same amount."""
    n = curve.pillar_dates.shape[0]
    return bump_curve_zero_rates(curve, jnp.full(n, bump))


def key_rate_bump(
    curve: DiscountCurve,
    pillar_index: int,
    bump: Float[Array, ""],
) -> DiscountCurve:
    """Bump a single pillar's zero rate."""
    n = curve.pillar_dates.shape[0]
    bumps = jnp.zeros(n).at[pillar_index].set(bump)
    return bump_curve_zero_rates(curve, bumps)


# ── Full scenario application ────────────────────────────────────────


def apply_scenario(
    base: MarketData,
    scenario: MarketScenario,
) -> MarketData:
    """Apply a scenario (risk factor deltas) to a base market state.

    Returns a new MarketData with shocked values. Fully differentiable
    with respect to both the base market data and the shock magnitudes.

    Spot shocks are multiplicative when ``scenario.multiplicative`` is True
    (``new = old * (1 + shock)``), otherwise additive (``new = old + shock``).
    Vol, dividend, and rate shocks are always additive.
    """
    if scenario.multiplicative:
        new_spots = base.spots * (1.0 + scenario.spot_shocks)
    else:
        new_spots = base.spots + scenario.spot_shocks

    new_vols = base.vols + scenario.vol_shocks
    new_dividends = base.dividends + scenario.dividend_shocks
    new_curve = bump_curve_zero_rates(base.discount_curve, scenario.rate_shocks)

    return MarketData(
        spots=new_spots,
        vols=new_vols,
        dividends=new_dividends,
        discount_curve=new_curve,
    )


# ── Multi-curve shocks ───────────────────────────────────────────────


def bump_discount_curve(
    mcs: MultiCurveSet,
    rate_bumps: Float[Array, " n_pillars"],
) -> MultiCurveSet:
    """Bump the OIS / discount leg of a :class:`MultiCurveSet`.

    The forward curves are left **completely untouched**, so the effect
    is a pure discount-curve shock (``IR.OIS.<ccy>`` in the factor
    registry).  Combine with ``bump_forward_curve`` to express joint or
    basis moves.
    """
    new_discount = bump_curve_zero_rates(mcs.discount_curve, rate_bumps)
    return MultiCurveSet(
        discount_curve=new_discount,
        forward_curves=mcs.forward_curves,
    )


def bump_forward_curve(
    mcs: MultiCurveSet,
    tenor: str,
    rate_bumps: Float[Array, " n_pillars"],
) -> MultiCurveSet:
    """Bump zero rates of a named forward curve inside a multi-curve set.

    The discount curve and all *other* forward curves remain untouched,
    so the result is a clean basis move between the targeted forward
    curve and everything else (``IR.FWD.<ccy>.<tenor>`` factor).  When
    the targeted curve is the only one used by an instrument's pricer,
    this is also the natural way to perturb a single tenor curve.

    Args:
        mcs: Source multi-curve set.
        tenor: Key into ``mcs.forward_curves`` (e.g. ``"3M"``).
        rate_bumps: Per-pillar additive zero-rate bumps for that curve.

    Returns:
        A new ``MultiCurveSet`` with the named forward curve bumped.

    Raises:
        KeyError: If ``tenor`` is not in ``mcs.forward_curves``.
    """
    if tenor not in mcs.forward_curves:
        raise KeyError(
            f"tenor {tenor!r} not in MultiCurveSet.forward_curves "
            f"(available: {sorted(mcs.forward_curves)})"
        )
    new_forwards = dict(mcs.forward_curves)
    new_forwards[tenor] = bump_curve_zero_rates(
        mcs.forward_curves[tenor], rate_bumps,
    )
    return MultiCurveSet(
        discount_curve=mcs.discount_curve,
        forward_curves=new_forwards,
    )


def parallel_basis_shift(
    mcs: MultiCurveSet,
    tenor: str,
    bump: Float[Array, ""],
) -> MultiCurveSet:
    """Parallel shift the named forward curve, discount held fixed.

    Equivalent to a pure ``IR.BASIS.<ccy>.<tenor>.OIS`` parallel move:
    every pillar of the forward curve moves by ``bump`` while the
    discount curve is unchanged.
    """
    n_pillars = mcs.forward_curves[tenor].pillar_dates.shape[0]
    return bump_forward_curve(mcs, tenor, jnp.full(n_pillars, bump))


# ── Credit shocks (SurvivalCurve) ────────────────────────────────────


def bump_hazard_rates(
    curve: SurvivalCurve,
    hazard_bumps: Float[Array, " n_pillars"],
) -> SurvivalCurve:
    """Apply additive piecewise-constant hazard-rate bumps.

    The bump at index ``i`` is added to the constant hazard on the
    interval ``(pillar_{i-1}, pillar_i]``; the first entry is ignored.
    Survival probabilities update by

    .. math::

        S_{\\text{new}}(t_i) = S_{\\text{old}}(t_i)
            \\cdot \\exp\\!\\left(-\\sum_{j \\le i} \\Delta h_j \\, \\Delta t_j\\right).

    Equivalent in form to :func:`bump_curve_zero_rates` for a
    :class:`DiscountCurve`: a per-pillar additive perturbation in
    log-space.

    Args:
        curve: Base survival curve.
        hazard_bumps: Additive hazard bumps per pillar (length =
            ``n_pillars``).  Entry 0 is ignored.

    Returns:
        New :class:`SurvivalCurve` with shocked survival probabilities.
    """
    pillar_times = year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count,
    )
    dt = jnp.diff(pillar_times)
    # First slot: zero contribution; remaining slots accumulate Δh·Δt.
    cum_extra_haz = jnp.concatenate([
        jnp.zeros((1,)),
        jnp.cumsum(hazard_bumps[1:] * dt),
    ])
    new_S = curve.survival_probabilities * jnp.exp(-cum_extra_haz)
    return SurvivalCurve(
        pillar_dates=curve.pillar_dates,
        survival_probabilities=new_S,
        reference_date=curve.reference_date,
        day_count=curve.day_count,
    )


def parallel_credit_spread_shift(
    curve: SurvivalCurve,
    spread_bump: Float[Array, ""],
    recovery_rate: Float[Array, ""] = jnp.asarray(0.4),
) -> SurvivalCurve:
    """Parallel CDS-spread shift, converted to a hazard bump.

    Uses the credit-triangle approximation ``Δh ≈ Δs / (1 − R)`` to
    convert a parallel CDS-spread move into a parallel hazard-rate
    move applied to every interval.  Default recovery is 40%.

    Args:
        curve: Base survival curve.
        spread_bump: Additive CDS-spread bump (e.g. ``0.0010`` = +10 bp).
        recovery_rate: Assumed recovery rate (default 0.4).

    Returns:
        New :class:`SurvivalCurve` with hazards uniformly shifted.
    """
    n = curve.pillar_dates.shape[0]
    hazard_bump = spread_bump / jnp.maximum(1.0 - recovery_rate, 1e-12)
    bumps = jnp.full((n,), hazard_bump)
    bumps = bumps.at[0].set(0.0)
    return bump_hazard_rates(curve, bumps)


def key_rate_hazard_bump(
    curve: SurvivalCurve,
    pillar_index: int,
    bump: Float[Array, ""],
) -> SurvivalCurve:
    """Bump the hazard rate on a single pillar interval.

    The bump affects only the hazard constant on the interval
    ``(pillar_{pillar_index-1}, pillar_{pillar_index}]``; survival
    probabilities at all pillars at or beyond that interval move by the
    corresponding multiplicative factor, while earlier pillars are
    unchanged.  This is the credit analogue of a key-rate IR bump.

    Args:
        curve: Base survival curve.
        pillar_index: Interval to bump (must be ≥ 1).
        bump: Additive hazard-rate bump on that interval.

    Returns:
        New :class:`SurvivalCurve` with one interval's hazard bumped.
    """
    n = curve.pillar_dates.shape[0]
    bumps = jnp.zeros((n,)).at[pillar_index].set(bump)
    return bump_hazard_rates(curve, bumps)
