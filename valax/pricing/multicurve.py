r"""Deterministic tenor-basis overlay for the short-rate solvers.

The analytic rates pricers have been dual-curve for a long time — they take
a discount curve *and* an optional projection curve
(:func:`~valax.pricing.analytic.floating.ois_swap_price`,
:mod:`~valax.pricing.analytic.graph_pricing`).  The Monte-Carlo and PDE
solvers were not: a Hull-White or G2++ recipe discounted **and** projected
off the single curve embedded in the model.  This module is the seam that
closes that gap, and it is deliberately small because the modelling content
is one assumption.

Why an overlay and not "just pass a second curve"
-------------------------------------------------
One-factor Hull-White and two-factor G2++ are **exact-fit-to-one-curve**
short-rate models: they produce a single term structure :math:`P(t, T)`.
Handing a recipe a second curve therefore cannot, on its own, mean anything
— there is no second stochastic driver for it to attach to.  What is needed
is a statement about how the projection curve moves relative to the discount
curve, and the simplest defensible one is that it does not:

.. math::

    P^{\text{fwd}}(t, T)
      = P^{\text{disc}}(t, T)\; \beta(t, T),
    \qquad
    \beta(t, T) = \frac{P^{\text{fwd}}_0(T) / P^{\text{fwd}}_0(t)}
                       {P^{\text{disc}}_0(T) / P^{\text{disc}}_0(t)} ,

i.e. the **multiplicative tenor basis is frozen at its time-zero value** and
the stochastic factors do not touch it.  Cash flows are discounted with the
model (fitted to the discount/OIS curve); floating rates are projected off
the discount-curve bond scaled by the deterministic :math:`\beta`.

This is the standard "deterministic basis" approximation.  It reproduces
today's forward curve exactly, keeps every affine formula in the recipes
intact (the basis is a constant multiplier at each node), and prices linear
products *exactly* — a swap or FRN NPV under the overlay equals the analytic
dual-curve price, because a linear product's PV does not depend on the
dynamics at all.  What it does **not** capture is basis *volatility*: a
spread option on the OIS/IBOR basis is worth zero under this assumption.  A
stochastic-basis multi-curve HJM setup is out of scope (see
``docs/theory/multi-curve-short-rate.md``).

The period factor the recipes actually use
------------------------------------------
Every payoff in :mod:`valax.pricing.pde.rates_payoffs` needs :math:`\beta`
only through the **ratio over one accrual period**, which is independent of
the valuation time :math:`t`:

.. math::

    b(T_s, T_e)
      \;=\; \frac{\beta(t, T_s)}{\beta(t, T_e)}
      \;=\; \frac{P^{\text{fwd}}_0(T_s) / P^{\text{fwd}}_0(T_e)}
                 {P^{\text{disc}}_0(T_s) / P^{\text{disc}}_0(T_e)}
      \;=\; \frac{1 + \tau F^{\text{fwd}}(T_s, T_e)}
                 {1 + \tau F^{\text{disc}}(T_s, T_e)} ,

the ratio of the two curves' simply-compounded growth factors over the
period.  :func:`tenor_basis` computes exactly that.  ``b == 1`` is the
single-curve case, and the recipes encode it as ``basis=None`` so the
arithmetic is *identical*, not merely algebraically equal — the
non-regression gate of the migration report.

See ``docs/architecture/pricing-ontology.md`` §12 for the phase plan this
module implements.
"""

from __future__ import annotations

from typing import TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve

#: Any exact-fit short-rate model carrying an ``initial_curve`` field
#: (``HullWhiteModel``, ``G2PPModel``, ``CheyetteModel``, …).
ExactFitModel = TypeVar("ExactFitModel")


__all__ = [
    "with_discount_curve",
    "tenor_basis",
    "swap_leg_basis",
    "apply_swap_basis",
    "require_single_curve",
]


def with_discount_curve(
    model: ExactFitModel,
    discount_curve: DiscountCurve | None,
) -> ExactFitModel:
    """Re-fit an exact-fit short-rate model to a trade's discount curve.

    The model's ``initial_curve`` *is* its discount curve: the exact-fit
    shift (:math:`\\alpha` for Hull-White, :math:`\\varphi` for G2++) is
    built so that the model reprices it.  Honouring an instrument's
    ``discount_id`` therefore means re-binding that field — the same
    operation :meth:`~valax.models.hull_white.HullWhiteModel.with_market_curve`
    performs from a :class:`~valax.core.market_state.MarketState`, expressed
    here against an already-resolved curve.

    Args:
        model: A model pytree carrying an ``initial_curve`` field.
        discount_curve: The trade's resolved discount curve, or ``None``
            to leave the model untouched.

    Returns:
        ``model`` unchanged when ``discount_curve`` is ``None`` or is
        already the model's own curve (so a single-curve call is
        bit-for-bit unaffected, and the curve object's *identity* is
        preserved for the ``is``-checks in :func:`tenor_basis`), otherwise
        a copy fitted to ``discount_curve``.

    Raises:
        AttributeError: If ``model`` has no ``initial_curve`` field, i.e.
            it is not an exact-fit short-rate model.
    """
    if discount_curve is None:
        return model
    if not hasattr(model, "initial_curve"):
        raise AttributeError(
            f"{type(model).__name__} has no `initial_curve`, so it cannot be "
            f"re-fitted to a resolved discount curve.  Only exact-fit "
            f"short-rate models (Hull-White, G2++, Cheyette) support the "
            f"discount/projection split."
        )
    if discount_curve is model.initial_curve:
        return model
    return eqx.tree_at(lambda m: m.initial_curve, model, discount_curve)


def tenor_basis(
    discount_curve: DiscountCurve,
    forward_curve: DiscountCurve | None,
    start_dates: Int[Array, "..."],
    end_dates: Int[Array, "..."],
) -> Float[Array, "..."] | None:
    r"""Frozen multiplicative tenor basis over each accrual period.

    Returns :math:`b(T_s, T_e)` of the module docstring — the ratio of the
    projection curve's growth factor to the discount curve's over the same
    period.  It enters every dual-curve payoff as a plain multiplier, which
    is why the affine machinery in the recipes needs no other change.

    Args:
        discount_curve: The curve the trade discounts on (the curve the
            short-rate model is fitted to).
        forward_curve: The projection curve, or ``None`` for single-curve
            projection.
        start_dates: Accrual-period start dates as integer ordinals.
        end_dates: Accrual-period end dates as integer ordinals.

    Returns:
        ``None`` when ``forward_curve`` is ``None`` or is the very same
        object as ``discount_curve`` — the signal the payoff builders read
        as "use the single-curve expression unchanged".  Otherwise the
        per-period basis factor, broadcast over the date arrays.

    Examples:
        >>> starts, ends = swap.float_dates[:-1], swap.float_dates[1:]
        >>> tenor_basis(ois, ois, starts, ends) is None     # single-curve
        True
        >>> b = tenor_basis(ois, projection, starts, ends)  # dual-curve
        >>> b.shape == starts.shape
        True
    """
    if forward_curve is None or forward_curve is discount_curve:
        return None
    forward_growth = forward_curve(start_dates) / forward_curve(end_dates)
    discount_growth = discount_curve(start_dates) / discount_curve(end_dates)
    return forward_growth / discount_growth


def swap_leg_basis(
    discount_curve: DiscountCurve,
    forward_curve: DiscountCurve | None,
    start_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
) -> Float[Array, " n_cf"] | None:
    """Per-period tenor basis for a swap leg paying on ``payment_dates``.

    Builds the accrual periods the way
    :func:`~valax.pricing.pde.rates_payoffs.swap_accruals` does — the first
    runs from ``start_date``, the rest between consecutive payment dates —
    and returns :func:`tenor_basis` over them.  The floating leg is assumed
    to pay on the fixed leg's schedule, matching the single-curve
    replication the short-rate recipes already use.

    Args:
        discount_curve: The curve the trade discounts on.
        forward_curve: The projection curve, or ``None`` for single-curve.
        start_date: Swap effective date / option exercise date, as an
            integer ordinal.
        payment_dates: Leg payment dates as integer ordinals.

    Returns:
        ``None`` for single-curve projection, otherwise the basis factor
        of each accrual period.
    """
    if forward_curve is None or forward_curve is discount_curve:
        return None
    starts = jnp.concatenate([jnp.asarray(start_date)[jnp.newaxis], payment_dates[:-1]])
    return tenor_basis(discount_curve, forward_curve, starts, payment_dates)


def apply_swap_basis(
    cashflows: Float[Array, " n_cf"],
    basis: Float[Array, " n_cf"] | None,
) -> tuple[Float[Array, ""] | float, Float[Array, " n_cf"]]:
    r"""Fold a swap's frozen tenor basis into its cash-flow vector.

    The single-curve replication of a payer swap at its start :math:`T_0` is
    :math:`1 - \sum_i c_i P(T_0, T_i)`. With a distinct projection curve the
    floating leg becomes
    :math:`\sum_j (b_j P(T_0, T_{j-1}) - P(T_0, T_j))`, which — using
    :math:`P(T_0, T_0) = 1` — is :math:`b_1 + \sum_{j<n}(b_{j+1} - 1)
    P(T_0, T_j) - P(T_0, T_n)`. Re-arranged, the swap is still
    ``float_start - Σ c_i P(T_0, T_i)`` for an adjusted constant and an
    adjusted cash-flow vector, so every affine bond-portfolio evaluation in
    the short-rate solvers keeps working unchanged.

    This is the *one* implementation of that algebra; the PDE payoff
    (:func:`~valax.pricing.pde.rates_payoffs.swap_tail_values`) and the MC
    swaption recipes both go through it.

    Args:
        cashflows: Single-curve cash-flow vector — ``K * tau_i`` with the
            unit principal folded into the final flow.
        basis: Per-period frozen tenor basis, or ``None`` for single-curve.

    Returns:
        ``(float_start, cashflows)``. For ``basis=None`` this is
        ``(1.0, cashflows)`` unchanged, so the single-curve arithmetic is
        untouched bit-for-bit.
    """
    if basis is None:
        return 1.0, cashflows
    # Period j+1 accrues over (T_j, T_{j+1}), so its basis attaches to the flow
    # at T_j; the final period has no successor, hence the slice.
    return basis[0], cashflows.at[:-1].add(1.0 - basis[1:])


def require_single_curve(
    discount_curve: DiscountCurve,
    forward_curve: DiscountCurve | None,
    what: str,
) -> None:
    """Reject a distinct projection curve for a not-yet-dual-curve payoff.

    Used at the recipe boundary (pre-trace) by payoffs whose projection
    has not been migrated to the overlay.  Failing loudly is the point:
    silently discounting *and* projecting off the same curve after the
    caller explicitly named a second one would be a wrong price with no
    signal.

    Args:
        discount_curve: The resolved discount curve.
        forward_curve: The resolved projection curve, possibly ``None``.
        what: Human-readable name of the payoff, used in the message.

    Raises:
        ValueError: If ``forward_curve`` is a distinct curve object.
    """
    if forward_curve is None or forward_curve is discount_curve:
        return
    raise ValueError(
        f"{what} does not yet support a projection curve distinct from its "
        f"discount curve under the deterministic tenor-basis overlay.  Drop "
        f"the instrument's `forward_id` (or point it at the discount curve) "
        f"to price it single-curve, or use the analytic route.  See "
        f"docs/architecture/pricing-ontology.md §12."
    )
