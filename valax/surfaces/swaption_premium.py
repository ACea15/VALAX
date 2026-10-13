r"""Swaption premium → normal (Bachelier) volatility, dual-curve.

Broker swaption matrices are sometimes quoted as **premiums** rather than vols.
Converting one to the other is not a detached unit change: the premium is quoted
*against a numeraire*, and that numeraire is a function of the curves. Writing
:math:`\sigma_N` for the normal volatility, :math:`q` for the quoted premium and
:math:`\mathcal{N}` for the numeraire, the conversion is the root of

.. math::

    R(\sigma_N;\, \theta_c, q)
      = N \cdot \mathcal{N}(\theta_c)\,
        \mathrm{Bachelier}\!\left(S_x(\theta_c), K, \sigma_N, T\right) - q
      = 0,

solved by a scalar ``optimistix.Newton``. Because the solve is implicit,
``jax.grad`` of the resulting vol with respect to a discount or projection
pillar works without unrolling the iteration — so a premium-quoted vol surface
carries genuine curve risk, which is the whole point: move the curve and the
*vol* the premium implies moves too.

Why this needs the dual-curve split
-----------------------------------
Both curve-dependent inputs enter:

* the forward par swap rate :math:`S_x` is **projected** (off the tenor curve
  when one is supplied), and
* the numeraire is **discounted** off the OIS / collateral curve.

So the same premium converts to different vols under different bases, exactly
as on the cap side (:mod:`valax.surfaces.caplet_stripping`). Nothing about the
payoff is re-derived here: the Bachelier core is
:func:`valax.pricing.analytic.swaptions.swaption_bachelier_from_forward`, and
:math:`(S_x, A_d)` come from
:func:`valax.pricing.analytic.swaptions.swaption_forward_and_annuity` — the
same functions the pricer uses.

Settlement
----------
The numeraire depends on how the option settles, and the two conventions are
*not* interchangeable:

``"physical"``
    Exercise delivers the swap, so the payoff is valued under the physical
    annuity measure and :math:`\mathcal{N} = A_d = \sum_i \tau_i P_d(0, T_i)`.
    Exact.

``"cash"``
    Exercise pays a single cash amount at the settlement date, computed from a
    **flat-yield** annuity at the settlement rate:
    :math:`\mathcal{N} = P_d(0, T^p)\, A_{\text{cash}}(S_x)` with
    :math:`A_{\text{cash}}` the street-standard
    :func:`~valax.pricing.analytic.cms_convexity.cash_settled_annuity`.

.. warning::

    The cash branch is the standard **approximation**: it evaluates the
    flat-yield annuity at the forward rather than taking the expectation of
    :math:`A_{\text{cash}}(S)\,(S-K)^+`, so the cash-settled convexity
    correction (a smile integral, as in
    :mod:`valax.pricing.analytic.cms_convexity`) is **not** applied. The
    difference is second order in vol but is real; it is recorded as a
    documented deviation in the
    [audit conformance map](../architecture/aad-audit-conformance.md).

References:
    Mercurio, F. (2008), "Cash-Settled Swaptions and No-Arbitrage", Risk.
    Hagan, P. (2003), "Convexity Conundrums", Wilmott Magazine.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from jax import Array

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationResult
from valax.calibration.settings import SolverSettings
from valax.calibration.solve import solve_level
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import year_fraction
from valax.instruments.rates import Swaption
from valax.pricing.analytic.cms_convexity import cash_settled_annuity
from valax.pricing.analytic.swaptions import (
    swaption_bachelier_from_forward,
    swaption_forward_and_annuity,
)


#: Settlement conventions understood by this module.
SETTLEMENTS = ("physical", "cash")

#: Floor on the inferred initial vol guess, in absolute (normal) vol units.
_MIN_VOL_GUESS = 1e-8

#: Floor on the normalised premium used to seed the root-find.
_MIN_NORMALISED = 1e-14

#: Nominal days per year used only to turn a schedule span into a frequency.
_DAYS_PER_YEAR = 365.0


def _infer_freq(swaption: Swaption) -> int:
    """Fixed-leg frequency implied by a swaption's schedule.

    Counts the fixed payments (a *static* shape) and divides by the schedule's
    span in years, rounding to the nearest integer -- recovering 1, 2, 4 or 12
    for the conventional schedules.

    The span is read with NumPy rather than ``jnp`` so the inference is
    genuinely pre-trace: the schedule is metadata, not a number to be
    differentiated. When the dates really are traced (the caller jitted over
    the schedule itself) the answer cannot be known at trace time, so this
    raises and asks for ``freq`` explicitly rather than guessing.

    Args:
        swaption: Swaption contract.

    Returns:
        Payments per year.

    Raises:
        ValueError: If the schedule dates are traced, so no concrete frequency
            can be determined.
    """
    n_periods = int(jnp.asarray(swaption.fixed_dates).shape[0])
    try:
        span_days = float(
            np.asarray(swaption.fixed_dates[-1])
            - np.asarray(swaption.expiry_date)
        )
    except Exception as exc:  # traced schedule
        raise ValueError(
            "Cash-settled conversion needs the fixed-leg frequency, which is "
            "inferred from the schedule -- but the schedule dates are traced, "
            "so it cannot be determined at trace time. Pass freq=... "
            "explicitly (it is a convention, not a differentiable input)."
        ) from exc
    years = max(span_days / _DAYS_PER_YEAR, 1.0 / _DAYS_PER_YEAR)
    return max(int(round(n_periods / years)), 1)


def swaption_forward_and_numeraire(
    swaption: Swaption,
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
    *,
    settlement: str = "physical",
    freq: int | None = None,
    settlement_date: Float[Array, ""] | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    r"""Forward par swap rate, premium numeraire, and time to expiry.

    The three curve-derived quantities a premium quote is expressed in terms
    of. Separated out so the residual, the forward conversion and the
    :class:`~valax.greeks.adjoint.CalibrationLevel` wrapper all read the same
    numbers.

    Args:
        swaption: Swaption contract.
        curve: Discount (OIS / collateral) curve.
        forward_curve: Optional tenor / projection curve for the underlying
            swap's floating leg. Defaults to ``curve`` (single-curve).
        settlement: ``"physical"`` (numeraire :math:`A_d`) or ``"cash"``
            (numeraire :math:`P_d(0, T^p) A_{\text{cash}}(S_x)`).
        freq: Fixed-leg payments per year, used only by the cash branch.
            Inferred from the schedule when omitted.
        settlement_date: Cash-settlement payment date (day ordinal), used only
            by the cash branch. Defaults to the option expiry, the market
            convention.

    Returns:
        Tuple ``(forward, numeraire, expiry)`` — the forward par swap rate, the
        numeraire per unit notional, and the time to expiry in years.

    Raises:
        ValueError: If ``settlement`` is not in :data:`SETTLEMENTS`.
    """
    if settlement not in SETTLEMENTS:
        raise ValueError(
            f"settlement must be one of {SETTLEMENTS}, got {settlement!r}."
        )

    expiry = year_fraction(
        curve.reference_date, swaption.expiry_date, swaption.day_count
    )
    forward, annuity_d = swaption_forward_and_annuity(
        swaption, curve, forward_curve
    )

    if settlement == "physical":
        return forward, annuity_d, expiry

    q = _infer_freq(swaption) if freq is None else freq
    n_periods = int(jnp.asarray(swaption.fixed_dates).shape[0])
    pay_date = (
        swaption.expiry_date if settlement_date is None else settlement_date
    )
    numeraire = curve(pay_date) * cash_settled_annuity(forward, n_periods, q)
    return forward, numeraire, expiry


def swaption_premium_from_normal_vol(
    swaption: Swaption,
    vol: Float[Array, ""],
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
    *,
    settlement: str = "physical",
    freq: int | None = None,
    settlement_date: Float[Array, ""] | None = None,
) -> Float[Array, ""]:
    """Premium implied by a normal (Bachelier) volatility.

    The forward direction of :func:`swaption_implied_normal_vol`, and the
    function whose root that solver finds. Delegates the payoff to
    :func:`valax.pricing.analytic.swaptions.swaption_bachelier_from_forward`
    so the two directions cannot disagree.

    Args:
        swaption: Swaption contract.
        vol: Normal (Bachelier) volatility.
        curve: Discount curve.
        forward_curve: Optional projection curve.
        settlement: ``"physical"`` or ``"cash"``.
        freq: Fixed-leg frequency for the cash branch; inferred when omitted.
        settlement_date: Cash-settlement date; defaults to the expiry.

    Returns:
        Swaption premium.
    """
    forward, numeraire, expiry = swaption_forward_and_numeraire(
        swaption, curve, forward_curve,
        settlement=settlement, freq=freq, settlement_date=settlement_date,
    )
    return swaption_bachelier_from_forward(
        swaption, forward, numeraire, vol, expiry
    )


def swaption_premium_residual(
    vol: Float[Array, ""],
    swaption: Swaption,
    premium: Float[Array, ""],
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
    *,
    settlement: str = "physical",
    freq: int | None = None,
    settlement_date: Float[Array, ""] | None = None,
) -> Float[Array, ""]:
    r"""The premium-to-vol residual :math:`R(\sigma_N;\, \theta_c, q)`.

    Zero exactly when ``vol`` reprices ``premium`` off the given curves. Stated
    as a standalone function (``vol`` first) so it can be handed to a root
    finder, differentiated directly, or wrapped as a
    :class:`~valax.greeks.adjoint.CalibrationLevel`.

    Args:
        vol: Candidate normal (Bachelier) volatility.
        swaption: Swaption contract.
        premium: Quoted premium.
        curve: Discount curve.
        forward_curve: Optional projection curve.
        settlement: ``"physical"`` or ``"cash"``.
        freq: Fixed-leg frequency for the cash branch; inferred when omitted.
        settlement_date: Cash-settlement date; defaults to the expiry.

    Returns:
        Residual in premium (currency) units.
    """
    model_premium = swaption_premium_from_normal_vol(
        swaption, vol, curve, forward_curve,
        settlement=settlement, freq=freq, settlement_date=settlement_date,
    )
    return model_premium - premium


def swaption_implied_normal_vol(
    swaption: Swaption,
    premium: Float[Array, ""],
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
    *,
    settlement: str = "physical",
    freq: int | None = None,
    settlement_date: Float[Array, ""] | None = None,
    initial_vol: Float[Array, ""] | float | None = None,
    settings: SolverSettings | None = None,
    name: str = "swaption_implied_normal_vol",
) -> CalibrationResult:
    r"""Normal (Bachelier) volatility implied by a quoted swaption premium.

    Solves :func:`swaption_premium_residual` for :math:`\sigma_N` as a scalar
    root level (Newton by default). The premium is the level's quote and the
    curves its upstream input, so the level adjoint and the solve's implicit
    differentiation both give

    * ``jax.grad`` w.r.t. ``premium`` gives the reciprocal vega, and
    * ``jax.grad`` w.r.t. the curves' pillars gives the curve sensitivity of
      the *implied vol* — the term that is identically zero if a premium is
      converted with a frozen numeraire.

    The initial guess inverts the **straddle** asymptote rather than the
    at-the-money one. Writing :math:`p = q / (N\mathcal{N})` for the normalised
    premium and :math:`x` for the signed moneyness, put-call symmetry gives
    :math:`p - x/2 = \tfrac12(\text{straddle}) \ge v/\sqrt{2\pi}` with
    :math:`v = \sigma\sqrt{T}`, so

    .. math::

        v_0 = \sqrt{2\pi}\,\bigl(p - x/2\bigr) \;\ge\; v,

    a guaranteed upper bound that stays well-conditioned far from the money.
    The naive at-the-money seed
    :math:`\sigma \approx \text{time value}\sqrt{2\pi/T}` collapses there --
    it underestimates by an order of magnitude on a 100 bp-out option, landing
    Newton in a region where the Bachelier vega has underflowed and the step
    is undefined.

    Args:
        swaption: Swaption contract.
        premium: Quoted premium (same currency and notional basis as
            ``swaption.notional``).
        curve: Discount (OIS / collateral) curve.
        forward_curve: Optional tenor / projection curve.
        settlement: ``"physical"`` or ``"cash"``; see the module docstring.
        freq: Fixed-leg frequency for the cash branch; inferred from the
            schedule when omitted.
        settlement_date: Cash-settlement date; defaults to the expiry.
        initial_vol: Override for the first Newton iterate.
        settings: Solver settings; defaults to :data:`INVERSION_SETTINGS`
            (Newton to 1e-12, 100 steps).
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the normal
        volatility that reprices ``premium`` is ``result.value``.

    Raises:
        ValueError: If ``settlement`` is not in :data:`SETTLEMENTS`.
    """
    settings = INVERSION_SETTINGS if settings is None else settings
    if settlement == "cash" and freq is None:
        # A convention, resolved once pre-trace: inside the solve the
        # schedule is closure-converted into traced arguments.
        freq = _infer_freq(swaption)

    def anchors(inputs: LevelInputs):
        disc, fwd = inputs.upstream
        return swaption_forward_and_numeraire(
            swaption, disc, fwd,
            settlement=settlement, freq=freq, settlement_date=settlement_date,
        )

    inputs = LevelInputs(
        quotes=jnp.asarray(premium, dtype=jnp.float64),
        upstream=(curve, forward_curve),
    )
    forward, numeraire, expiry = anchors(inputs)

    if initial_vol is not None:
        guess = jnp.asarray(initial_vol, dtype=float)
    else:
        moneyness = forward - swaption.strike
        x = moneyness if swaption.is_payer else -moneyness
        scale = swaption.notional * numeraire
        normalised = premium / scale
        half_straddle = jnp.maximum(normalised - 0.5 * x, _MIN_NORMALISED)
        guess = jnp.maximum(
            jnp.sqrt(2.0 * jnp.pi) * half_straddle / jnp.sqrt(expiry),
            _MIN_VOL_GUESS,
        )

    def residual(sigma: Float[Array, ""], inp: LevelInputs) -> Float[Array, " 1"]:
        fwd, num, t = anchors(inp)
        model = swaption_bachelier_from_forward(swaption, fwd, num, sigma, t)
        return jnp.reshape(model - inp.quotes, (1,))

    return solve_level(
        name, "root",
        lambda th, inp: residual(th[0], inp),
        jnp.reshape(jax.lax.stop_gradient(guess), (1,)),
        inputs,
        lambda th, _inp: th[0],
        settings,
    )


def swaption_implied_normal_vol_from_graph(
    swaption: Swaption,
    premium: Float[Array, ""],
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None = None,
    **kwargs: Any,
) -> CalibrationResult:
    """:func:`swaption_implied_normal_vol` resolved on a curve graph.

    Args:
        swaption: Swaption contract.
        premium: Quoted premium.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        forward_id: Optional id of the projection curve.
        **kwargs: Forwarded to :func:`swaption_implied_normal_vol`.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult` of the
        inversion.
    """
    discount_curve = graph[discount_id]
    forward_curve = None if forward_id is None else graph[forward_id]
    return swaption_implied_normal_vol(
        swaption, premium, discount_curve, forward_curve, **kwargs
    )


#: Default settings of the premium → vol inversion: Newton to 1e-12.
INVERSION_SETTINGS = SolverSettings(
    method="newton", rtol=1e-12, atol=1e-12, max_steps=100
)


__all__ = [
    "INVERSION_SETTINGS",
    "SETTLEMENTS",
    "swaption_forward_and_numeraire",
    "swaption_implied_normal_vol",
    "swaption_implied_normal_vol_from_graph",
    "swaption_premium_from_normal_vol",
    "swaption_premium_residual",
]
