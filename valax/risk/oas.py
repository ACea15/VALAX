r"""Option-adjusted spread (OAS) and Z-spread for bonds.

The **option-adjusted spread** is the constant parallel shift :math:`s` applied
to a model's discount curve that makes the *model* price equal the observed
market dirty price.  For a bond with embedded optionality (callable / puttable)
the short-rate model reprices the option at every trial spread, so the recovered
:math:`s` isolates the pure credit / liquidity component with the option value
stripped out.  For an option-free bond the same construction collapses to the
**Z-spread** — there is no option to adjust for — which this module also exposes
directly through the closed-form curve pricer as an independent oracle.

**Why this is clean under Hull-White.**  A parallel shift sends every pillar
zero rate :math:`r_i \to r_i + s`; because :math:`-s\,t` is exactly linear the
shift holds at *every* continuous time :math:`t`, so

.. math::

    f^M(0, t) \to f^M(0, t) + s \quad\Longrightarrow\quad \alpha(t) \to \alpha(t) + s,

while the convexity term of :math:`\alpha` (which depends only on :math:`a` and
:math:`\sigma`) is untouched.  The PDE therefore sees ``r = x + alpha(t) + s`` —
a pure constant shift of the discount coefficient with the :math:`x`-dynamics
unchanged — so the two conventional definitions of OAS ("shift discounting only"
versus "re-fit the whole model") coincide exactly.

Everything here is a pure function of pytrees.  The OAS root-find is implicitly
differentiable through :func:`optimistix.root_find`, and the effective-risk
helpers take exact autodiff derivatives of the model repricing map — never
finite differences.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, §3.3.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.instruments.bonds import FixedRateBond
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.pde.dispatch import pde_price_dispatch
from valax.risk.shocks import parallel_shift


# ── Spread-convention conversions ────────────────────────────────────


def continuous_to_compounded_spread(
    spread: Float[Array, ""],
    frequency: int,
) -> Float[Array, ""]:
    r"""Convert a continuously-compounded spread to a periodically-compounded one.

    VALAX curves discount continuously, so an OAS produced by this module is a
    *continuous* spread.  External conventions (notably QuantLib's
    ``CallableFixedRateBond.OAS``) quote the spread on a periodic-compounding
    basis with a stated frequency.  The two are related by matching one period's
    growth factor, :math:`(1 + s_m / f)^f = e^{s_c}`, giving

    .. math::

        s_m = f\bigl(e^{s_c / f} - 1\bigr).

    Args:
        spread: Continuously-compounded spread.
        frequency: Compounding frequency of the target convention (periods/yr).

    Returns:
        The equivalent periodically-compounded spread.
    """
    f = jnp.asarray(frequency, dtype=jnp.float64)
    return f * jnp.expm1(spread / f)


def compounded_to_continuous_spread(
    spread: Float[Array, ""],
    frequency: int,
) -> Float[Array, ""]:
    r"""Convert a periodically-compounded spread to a continuous one.

    Inverse of :func:`continuous_to_compounded_spread`:

    .. math::

        s_c = f \ln\!\bigl(1 + s_m / f\bigr).

    Use this to bring an externally-quoted (e.g. QuantLib) compounded OAS onto
    the continuous convention that :func:`callable_bond_oas` returns before
    comparing the two.

    Args:
        spread: Periodically-compounded spread.
        frequency: Compounding frequency of the source convention (periods/yr).

    Returns:
        The equivalent continuously-compounded spread.
    """
    f = jnp.asarray(frequency, dtype=jnp.float64)
    return f * jnp.log1p(spread / f)


# ── Model repricing under a parallel spread ──────────────────────────


def _shifted_model(
    model: HullWhiteModel,
    spread: Float[Array, ""],
) -> HullWhiteModel:
    """Hull-White model with its initial curve parallel-shifted by ``spread``.

    Rebuilds the model rather than mutating it: the mean-reversion and
    volatility are carried through unchanged, and only the initial curve is
    replaced, so the exact-fit :math:`\\alpha(t)` picks up a pure constant
    ``spread`` with the :math:`x`-dynamics untouched.
    """
    return HullWhiteModel(
        mean_reversion=model.mean_reversion,
        volatility=model.volatility,
        initial_curve=parallel_shift(model.initial_curve, spread),
    )


def price_under_spread(
    bond: Any,
    model: HullWhiteModel,
    config: Any,
    spread: Float[Array, ""],
) -> Float[Array, ""]:
    """Model price of a bond with a parallel ``spread`` on the discount curve.

    Thin wrapper over :func:`~valax.pricing.pde.dispatch.pde_price_dispatch`
    that first shifts the model's initial curve.  This is the repricing map that
    :func:`callable_bond_oas` inverts and that the effective-risk helpers
    differentiate.

    Args:
        bond: A bond instrument with a registered Hull-White PDE recipe
            (``FixedRateBond``, ``CallableBond``, ``PuttableBond``).
        model: Base Hull-White model.
        config: A :class:`~valax.pricing.pde.config.PDEConfig`.
        spread: Continuously-compounded parallel shift on the discount curve.

    Returns:
        The model dirty price at the reference date.
    """
    shifted = _shifted_model(model, spread)
    return pde_price_dispatch(bond, shifted, config).price


# ── OAS root-find ────────────────────────────────────────────────────


def callable_bond_oas(
    bond: Any,
    model: HullWhiteModel,
    market_price: Float[Array, ""],
    config: Any,
    *,
    x0: Float[Array, ""] | float = 0.0,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    max_steps: int = 64,
) -> Float[Array, ""]:
    r"""Solve for the option-adjusted spread of a (callable/puttable) bond.

    Finds the continuously-compounded parallel shift :math:`s` on the model's
    discount curve such that the model price equals ``market_price``:

    .. math::

        \text{price}\bigl(\text{bond}, \text{model}(s)\bigr) - \text{market\_price} = 0.

    A one-dimensional Newton root-find (:class:`optimistix.Newton`) drives the
    residual to zero; because the PDE pricer is differentiable the Jacobian is
    exact autodiff, and the solve is implicitly differentiable, so
    ``jax.grad(callable_bond_oas, ...)`` propagates sensitivities of the
    recovered spread without unrolling the iteration.

    Works unchanged for an option-free :class:`~valax.instruments.bonds.FixedRateBond`
    (where the result is the Z-spread), a
    :class:`~valax.instruments.bonds.CallableBond`, or a
    :class:`~valax.instruments.bonds.PuttableBond` — dispatch is on the
    instrument type.

    Args:
        bond: Bond instrument with a registered Hull-White PDE recipe.
        model: Base Hull-White model (its initial curve is the un-shifted curve).
        market_price: Observed market dirty price to match.
        config: A :class:`~valax.pricing.pde.config.PDEConfig`.
        x0: Initial spread guess (default ``0.0``).
        rtol: Relative tolerance for the Newton solver.
        atol: Absolute tolerance for the Newton solver.
        max_steps: Maximum Newton iterations.

    Returns:
        The option-adjusted spread :math:`s` (continuous, scalar array).
    """
    target = jnp.asarray(market_price)

    def residual(spread: Float[Array, ""], args: Any) -> Float[Array, ""]:
        return price_under_spread(bond, model, config, spread) - target

    solver = optimistix.Newton(rtol=rtol, atol=atol)
    sol = optimistix.root_find(
        residual,
        solver,
        jnp.asarray(x0, dtype=jnp.float64),
        max_steps=max_steps,
        throw=False,
    )
    return sol.value


def bond_z_spread(
    bond: FixedRateBond,
    curve: DiscountCurve,
    market_price: Float[Array, ""],
    *,
    x0: Float[Array, ""] | float = 0.0,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    max_steps: int = 64,
) -> Float[Array, ""]:
    r"""Z-spread of an option-free fixed-rate bond via the closed-form pricer.

    The Z-spread is the constant continuous spread :math:`s` on the discount
    curve that reprices a bond to its market price.  For an option-free bond
    this is computed here through the analytic curve pricer
    (:func:`~valax.pricing.analytic.bonds.fixed_rate_bond_price`) — a code path
    completely independent of the Hull-White PDE — which makes it the natural
    oracle for the identity *OAS ≡ Z-spread on an option-free bond*.

    Args:
        bond: An option-free fixed-rate bond.
        curve: Base discount curve.
        market_price: Observed market dirty price to match.
        x0: Initial spread guess (default ``0.0``).
        rtol: Relative tolerance for the Newton solver.
        atol: Absolute tolerance for the Newton solver.
        max_steps: Maximum Newton iterations.

    Returns:
        The Z-spread :math:`s` (continuous, scalar array).
    """
    target = jnp.asarray(market_price)

    def residual(spread: Float[Array, ""], args: Any) -> Float[Array, ""]:
        shifted = parallel_shift(curve, spread)
        return fixed_rate_bond_price(bond, shifted) - target

    solver = optimistix.Newton(rtol=rtol, atol=atol)
    sol = optimistix.root_find(
        residual,
        solver,
        jnp.asarray(x0, dtype=jnp.float64),
        max_steps=max_steps,
        throw=False,
    )
    return sol.value


# ── Effective risk (spread-based Greeks) ─────────────────────────────


def effective_duration(
    bond: Any,
    model: HullWhiteModel,
    config: Any,
    spread: Float[Array, ""] | float = 0.0,
) -> Float[Array, ""]:
    r"""Effective duration of a bond under the short-rate model.

    .. math::

        D_{\text{eff}} = -\frac{1}{P}\frac{\partial P}{\partial s},

    where :math:`P(s)` is the model price under a parallel discount-curve shift
    :math:`s` (see :func:`price_under_spread`).  The derivative is exact autodiff
    through the PDE exercise projection, so an embedded call — which truncates
    the price upside — correctly compresses duration relative to the bullet.

    Args:
        bond: Bond instrument with a registered Hull-White PDE recipe.
        model: Base Hull-White model.
        config: A :class:`~valax.pricing.pde.config.PDEConfig`.
        spread: Spread level at which to evaluate the derivative (default ``0``).

    Returns:
        The effective duration (scalar array).
    """
    s = jnp.asarray(spread, dtype=jnp.float64)
    price_fn = lambda x: price_under_spread(bond, model, config, x)
    price = price_fn(s)
    dprice = jax.grad(price_fn)(s)
    return -dprice / price


def effective_convexity(
    bond: Any,
    model: HullWhiteModel,
    config: Any,
    spread: Float[Array, ""] | float = 0.0,
) -> Float[Array, ""]:
    r"""Effective convexity of a bond under the short-rate model.

    .. math::

        C_{\text{eff}} = \frac{1}{P}\frac{\partial^2 P}{\partial s^2},

    computed as a second autodiff derivative of :func:`price_under_spread`.  The
    embedded call **compresses** convexity: as rates fall and the call bites, a
    callable's convexity collapses from the bullet's large positive value toward
    the small positive convexity of the near-term call stub, whereas an
    option-free bond's convexity keeps rising.  Note that, measured as this
    parallel-spread second derivative in the exact-fit Hull-White framework, a
    callable's effective convexity does **not** go strictly negative — the capped
    redemption is still discounted convexly (see the risk-measures theory notes,
    §7.9).  The textbook "negative convexity" of a callable is a statement about
    its price-vs-yield-to-call curve, not this term-structure OAS derivative.

    Args:
        bond: Bond instrument with a registered Hull-White PDE recipe.
        model: Base Hull-White model.
        config: A :class:`~valax.pricing.pde.config.PDEConfig`.
        spread: Spread level at which to evaluate the derivative (default ``0``).

    Returns:
        The effective convexity (scalar array).
    """
    s = jnp.asarray(spread, dtype=jnp.float64)
    price_fn = lambda x: price_under_spread(bond, model, config, x)
    price = price_fn(s)
    d2price = jax.grad(jax.grad(price_fn))(s)
    return d2price / price


# ── Z-spread risk (option-free bonds) ────────────────────────────────

# One basis point, the standard quoting unit for DV01.
BASIS_POINT = 1.0e-4


def _bond_price_at_spread(
    bond: FixedRateBond,
    curve: DiscountCurve,
    spread: Float[Array, ""],
) -> Float[Array, ""]:
    """Analytic price of an option-free bond under a parallel curve spread."""
    return fixed_rate_bond_price(bond, parallel_shift(curve, spread))


def z_spread_duration(
    bond: FixedRateBond,
    curve: DiscountCurve,
    spread: Float[Array, ""] | float = 0.0,
) -> Float[Array, ""]:
    r"""Spread duration of an option-free bond.

    .. math::

        D_z = -\frac{1}{P}\frac{\partial P}{\partial z},

    the normalised sensitivity of the analytic price to the Z-spread :math:`z`,
    evaluated at the supplied spread level.  Pass the bond's fitted spread
    (:func:`bond_z_spread`) to get the duration at its market quote, or leave the
    default ``0`` to measure it at the base curve.

    For a bond discounted off a single curve a Z-spread shift is identical to a
    parallel curve shift, so this equals the bond's parallel-curve (IR) duration;
    the *spread* framing is what separates credit/liquidity risk from pure rate
    risk in a risk report.

    Args:
        bond: Option-free fixed-rate bond.
        curve: Base discount curve.
        spread: Z-spread level at which to evaluate the derivative (default ``0``).

    Returns:
        The spread duration (scalar array).
    """
    z = jnp.asarray(spread, dtype=jnp.float64)
    price_fn = lambda x: _bond_price_at_spread(bond, curve, x)
    price = price_fn(z)
    dprice = jax.grad(price_fn)(z)
    return -dprice / price


def z_spread_dv01(
    bond: FixedRateBond,
    curve: DiscountCurve,
    spread: Float[Array, ""] | float = 0.0,
) -> Float[Array, ""]:
    r"""Spread DV01 of an option-free bond — cash P&L per 1 bp of Z-spread.

    .. math::

        \text{DV01}_z = -\frac{\partial P}{\partial z}\times 10^{-4},

    the change in price for a one-basis-point widening of the Z-spread, in the
    bond's price/face currency.  Reported positive for a long position (price
    falls as the spread widens).  Related to :func:`z_spread_duration` by
    :math:`\text{DV01}_z = D_z\,P\,\times 10^{-4}`.

    Args:
        bond: Option-free fixed-rate bond.
        curve: Base discount curve.
        spread: Z-spread level at which to evaluate the derivative (default ``0``).

    Returns:
        The spread DV01 (scalar array), in price/face currency per basis point.
    """
    z = jnp.asarray(spread, dtype=jnp.float64)
    price_fn = lambda x: _bond_price_at_spread(bond, curve, x)
    dprice = jax.grad(price_fn)(z)
    return -dprice * BASIS_POINT


def z_spread_convexity(
    bond: FixedRateBond,
    curve: DiscountCurve,
    spread: Float[Array, ""] | float = 0.0,
) -> Float[Array, ""]:
    r"""Spread convexity of an option-free bond.

    .. math::

        C_z = \frac{1}{P}\frac{\partial^2 P}{\partial z^2},

    the normalised second derivative of the analytic price with respect to the
    Z-spread.  Strictly positive for a bond with fixed positive cash flows, since
    :math:`P(z)=\sum_i a_i e^{-z t_i}` is convex.

    Args:
        bond: Option-free fixed-rate bond.
        curve: Base discount curve.
        spread: Z-spread level at which to evaluate the derivative (default ``0``).

    Returns:
        The spread convexity (scalar array).
    """
    z = jnp.asarray(spread, dtype=jnp.float64)
    price_fn = lambda x: _bond_price_at_spread(bond, curve, x)
    price = price_fn(z)
    d2price = jax.grad(jax.grad(price_fn))(z)
    return d2price / price
