"""Survival probability curve for credit risk.

A :class:`SurvivalCurve` is the credit-side analogue of
:class:`valax.curves.discount.DiscountCurve`: a pytree storing
pillar-date / survival-probability pairs with log-linear interpolation.
Log-linear in survival is equivalent to **piecewise-constant hazard
rate** between pillars, which is the standard market convention.

Three quantities are derived from the curve:

1. :math:`S(t)` — probability that the reference entity has not
   defaulted by ``t``.  Stored at pillars, interpolated elsewhere.
2. :math:`h(t)` — instantaneous default intensity (hazard rate), with
   :math:`S(t)=\\exp(-\\int_0^t h(s)\\,\\mathrm{d}s)`.
3. Par CDS spreads :math:`s\\approx h\\,(1-R)` (the *credit triangle*)
   used to bootstrap the curve from market CDS quotes.

Because the curve is an :class:`equinox.Module`, ``jax.grad`` through
any pricing function that takes it yields per-pillar credit-delta
sensitivities (the credit analogue of KRD / DV01).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.dates.daycounts import year_fraction


class SurvivalCurve(eqx.Module):
    """Term structure of survival probabilities for a single credit entity.

    Mirrors :class:`valax.curves.discount.DiscountCurve` in shape so the
    same shock primitives, ladder builders, and key-rate routines can
    be reused with minimal duplication.

    Attributes:
        pillar_dates: Sorted ordinal dates for curve nodes.  First
            entry must equal ``reference_date`` (so the first
            survival probability is 1.0).
        survival_probabilities: ``P(τ > t)`` at each pillar.  Monotone
            non-increasing in time.  First entry should be 1.0.
        reference_date: Valuation date (ordinal).
        day_count: Day-count convention used to convert dates to year
            fractions when computing hazard rates and interpolating.
    """

    pillar_dates: Int[Array, " n_pillars"]
    survival_probabilities: Float[Array, " n_pillars"]
    reference_date: Int[Array, ""]
    day_count: str = eqx.field(static=True, default="act_365")

    def __call__(self, dates: Int[Array, "..."]) -> Float[Array, "..."]:
        """Interpolate survival probabilities at arbitrary dates.

        Linear interpolation is performed in **log-survival space**,
        which is equivalent to piecewise-constant hazard between
        pillars.  Flat extrapolation is applied beyond the curve range.
        """
        pillar_times = year_fraction(
            self.reference_date, self.pillar_dates, self.day_count,
        )
        query_times = year_fraction(self.reference_date, dates, self.day_count)
        log_S = jnp.log(self.survival_probabilities)
        log_S_interp = jnp.interp(query_times, pillar_times, log_S)
        return jnp.exp(log_S_interp)


# ── Derived quantities ──────────────────────────────────────────────


def survival_probability(
    curve: SurvivalCurve,
    date: Int[Array, ""],
) -> Float[Array, ""]:
    """Survival probability to a single date — alias for ``curve(date)``."""
    return curve(date)


def hazard_rate(
    curve: SurvivalCurve,
    date: Int[Array, ""],
) -> Float[Array, ""]:
    """Average hazard rate from the reference date to ``date``.

    .. math::

        \\bar h(0,t) = -\\frac{\\ln S(t)}{t}

    For a piecewise-constant hazard curve this equals the instantaneous
    hazard inside the first pillar interval; beyond that it averages
    the per-interval hazards in proportion to time.  For the
    interval-by-interval hazard between two pillars, see
    :func:`piecewise_hazards`.
    """
    S = curve(date)
    tau = year_fraction(curve.reference_date, date, curve.day_count)
    tau_safe = jnp.maximum(tau, 1e-12)
    return -jnp.log(S) / tau_safe


def piecewise_hazards(curve: SurvivalCurve) -> Float[Array, " n_pillars"]:
    """Piecewise-constant hazard rate on each pillar interval.

    Returns an array of the same length as the pillars; entry ``i``
    is the constant hazard on the interval ``(pillar_{i-1}, pillar_i]``.
    The first entry is set to 0.0 (no interval before the reference).
    """
    pillar_times = year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count,
    )
    log_S = jnp.log(curve.survival_probabilities)
    dt = jnp.diff(pillar_times)
    dlog = jnp.diff(log_S)
    h_per_interval = -dlog / jnp.maximum(dt, 1e-12)
    # Prepend a 0 so the array length matches the pillar count.
    return jnp.concatenate([jnp.zeros((1,)), h_per_interval])


# ── Constructors ────────────────────────────────────────────────────


def from_hazard_rates(
    reference_date: Int[Array, ""],
    pillar_dates: Int[Array, " n_pillars"],
    hazards: Float[Array, " n_pillars"],
    day_count: str = "act_365",
) -> SurvivalCurve:
    """Build a :class:`SurvivalCurve` from per-pillar hazard rates.

    The hazard at index ``i`` is interpreted as the *constant* hazard on
    the interval ``(pillar_{i-1}, pillar_i]``.  Entry 0 is ignored
    (there is no interval before the reference date) and the first
    survival probability is set to 1.0.

    Args:
        reference_date: Valuation date (ordinal).  Must equal
            ``pillar_dates[0]``.
        pillar_dates: Sorted ordinal dates for curve nodes (first =
            ``reference_date``).
        hazards: Hazard rate on each interval; ``hazards[0]`` ignored.
        day_count: Day-count convention.

    Returns:
        SurvivalCurve with survival probabilities computed as
        ``S(t_i) = exp(-Σ_{j≤i} h_j · Δt_j)``.
    """
    pillar_times = year_fraction(reference_date, pillar_dates, day_count)
    dt = jnp.diff(pillar_times)
    # Hazards beyond the first are the interval hazards.
    h_intervals = hazards[1:]
    cum_haz = jnp.concatenate([
        jnp.zeros((1,)),
        jnp.cumsum(h_intervals * dt),
    ])
    survival = jnp.exp(-cum_haz)
    return SurvivalCurve(
        pillar_dates=pillar_dates,
        survival_probabilities=survival,
        reference_date=reference_date,
        day_count=day_count,
    )


def from_cds_spreads(
    reference_date: Int[Array, ""],
    pillar_dates: Int[Array, " n_pillars"],
    spreads: Float[Array, " n_pillars"],
    recovery_rate: Float[Array, ""] = jnp.asarray(0.4),
    day_count: str = "act_365",
) -> SurvivalCurve:
    """Build a :class:`SurvivalCurve` from CDS spreads via the credit triangle.

    Uses the standard market approximation

    .. math::

        h(t) \\approx \\frac{s(t)}{1 - R}

    where ``s(t)`` is the par CDS spread at maturity ``t`` and ``R`` is
    the assumed recovery rate.  This is the same first-order
    bootstrapping approximation used as a starting point in the ISDA
    CDS Standard Model.  For tighter calibration, the resulting curve
    can be refined by solving for hazards that exactly reprice the
    CDS protection-leg / premium-leg condition (not done here).

    Args:
        reference_date: Valuation date (ordinal).
        pillar_dates: CDS maturity dates as ordinals (first =
            ``reference_date``).
        spreads: Par CDS spreads at each pillar (e.g. ``0.01`` = 100 bps).
            ``spreads[0]`` is ignored.
        recovery_rate: Assumed recovery rate (default 0.4 = 40%).
        day_count: Day-count convention.

    Returns:
        A SurvivalCurve consistent with ``h_i = s_i / (1 - R)``.
    """
    hazards = spreads / jnp.maximum(1.0 - recovery_rate, 1e-12)
    # Ensure first slot is 0 — there is no interval before the reference.
    hazards = hazards.at[0].set(0.0)
    return from_hazard_rates(
        reference_date=reference_date,
        pillar_dates=pillar_dates,
        hazards=hazards,
        day_count=day_count,
    )
