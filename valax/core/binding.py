"""Late-binding a model's market-sourced parameters — the middle layer.

VALAX pricing is four layers, and this module is the third:

1. **Kernel** — the textbook closed form, pure scalars, no market and no
   model: ``black_scholes_price(option, spot, vol, rate, dividend)``.
2. **Model** — calibrated parameters as a pytree:
   ``BlackScholesModel(vol=0.2, rate=0.05, dividend=0.0)``.  A model whose
   fields are all given is *self-contained*, which is what makes it a
   textbook example.
3. **Binding** (here) — fills any field left as ``None`` from the
   :class:`~valax.core.resolve.ResolvedMarket`, returning a model of the
   **same class** with every parameter concrete.
4. **Front door** — :func:`valax.core.pricing.price`, which calls
   ``resolve`` then :func:`bind_model` and hands the *bound* model to the
   pricer.

Why a bind step rather than optional reads at each pricer
--------------------------------------------------------
``model.rate`` / ``.vol`` / ``.dividend`` are read at roughly thirty sites
across the MC path generators, the PDE coefficient builders, the MC engine
and the PDE boundary conditions — several of which receive the model whole
and one of which uses ``model.vol`` to *size the spot grid*.  Teaching all
of them about ``None`` would be a large change with a lot of surface for
silent bugs.  Binding once, above them, means **every one of those sites
is untouched**: they always see a concrete model, exactly as before.

It also puts the projection *inside* the differentiated path, which is the
whole point: a market-sourced ``rate`` traces back to the curve, so a
curve shock moves the price and
:func:`valax.risk.state_var.group_market_gradient` reports pillar deltas
for an equity group instead of a structural zero.

Two modes, one model class
--------------------------
::

    BlackScholesModel(vol=0.2, rate=0.05, dividend=0.0)  # all constant
    BlackScholesModel(vol=0.2)                # rate + dividend from market
    BlackScholesModel(rate=0.05)              # vol from the surface
    BlackScholesModel()                       # everything from the market

A fully-specified model is returned **unchanged** — same object, not a
copy — so constant-mode prices are bit-identical to the pre-binding
behaviour.

Note:
    A constant-rate model with a *projected* rate is still a constant-rate
    model.  Projecting at each trade's own maturity makes its discount
    factor right for that maturity, so a book of mixed expiries uses
    different rates — correct per trade, and strictly better than one
    pillar for everything, but it is not a term-structure model.  See
    ``PRICING_ONTOLOGY.md`` § 3.1.
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve, zero_rate
from valax.dates.daycounts import ordinal_from_year_fraction, year_fraction
from valax.surfaces.leverage import LeverageGrid
from valax.surfaces.swaption_cube import SwaptionCube


#: Model fields this module can source from the market, in bind order.
BINDABLE_FIELDS = ("rate", "dividend", "vol")

#: Smallest horizon used for a zero-rate projection.  A zero-maturity
#: trade has no well-defined zero rate (``-ln DF / t`` at ``t = 0``); the
#: floor keeps the expression finite instead of raising inside traced code.
_MIN_HORIZON = 1e-8

#: Surface types that expose a 2-argument ``__call__`` but do **not** mean
#: ``(strike, expiry) -> implied vol``, and so would be silently
#: mis-sampled.  ``LeverageGrid`` takes ``(log_moneyness, time)`` — same
#: arity, different units.  ``SwaptionCube`` is 3-arity and would raise,
#: but naming it gives a better message than a ``TypeError``.
_NOT_AN_IMPLIED_VOL_SURFACE = (LeverageGrid, SwaptionCube)


# ── Horizon ──────────────────────────────────────────────────────────


def instrument_horizon(
    instrument: Any,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """The year fraction a constant parameter should be projected at.

    Equity contracts carry ``expiry`` as a year fraction; rates contracts
    carry ``expiry_date`` / ``maturity`` as date ordinals.  Both are
    accepted and reduced to a year fraction in the *curve's* day count.

    Args:
        instrument: Any instrument pytree.
        curve: The curve the projection will read, whose ``day_count``
            defines the time axis.

    Returns:
        Year fraction from the curve's reference date, floored away from
        zero so a projected rate stays finite.

    Raises:
        ValueError: If the instrument exposes no horizon this function
            recognises.  Supply the model parameter explicitly instead.
    """
    horizon = getattr(instrument, "expiry", None)
    if horizon is None:
        for field in ("expiry_date", "maturity"):
            ordinal = getattr(instrument, field, None)
            if ordinal is not None:
                horizon = year_fraction(
                    curve.reference_date, ordinal, curve.day_count,
                )
                break

    if horizon is None:
        raise ValueError(
            f"{type(instrument).__name__} exposes no 'expiry' (year "
            f"fraction) nor 'expiry_date' / 'maturity' (date ordinal), so "
            f"a market-sourced model parameter cannot be projected to its "
            f"horizon. Give the model an explicit value instead of "
            f"leaving it None."
        )
    return jnp.maximum(jnp.asarray(horizon, dtype=jnp.float64), _MIN_HORIZON)


# ── Projections ──────────────────────────────────────────────────────


def project_rate(
    curve: DiscountCurve,
    horizon: Float[Array, ""],
) -> Float[Array, ""]:
    """Continuously-compounded zero rate to ``horizon``.

    The honest version of the collapse that
    ``valax.risk.var._extract_short_rate`` performs: the rate is read at
    the trade's *own* maturity rather than at a fixed pillar, and the
    year-fraction → ordinal conversion uses the curve's ``day_count``
    rather than a hard-coded 365.  Differentiable in the curve, so a
    curve shock flows through to the price.

    Args:
        curve: Discount curve to read.
        horizon: Year fraction, in the curve's day count.

    Returns:
        The continuously-compounded zero rate ``-ln P(0, T) / T``.
    """
    ordinal = ordinal_from_year_fraction(
        curve.reference_date, horizon, curve.day_count,
    )
    return zero_rate(curve, ordinal)


def sample_vol(
    surface: Any,
    strike: Float[Array, ""],
    horizon: Float[Array, ""],
) -> Float[Array, ""]:
    """Sample an implied-vol surface at ``(strike, expiry)``.

    The equity surface protocol is ``__call__(strike, expiry)`` with an
    **absolute** strike and expiry as a year fraction;
    :class:`~valax.surfaces.constant.ConstantVol` and
    :class:`~valax.surfaces.shifted.ShiftedVolSurface` are variadic and so
    satisfy it trivially.

    Two families are rejected rather than sampled, because they would be
    wrong quietly:

    * anything in :data:`_NOT_AN_IMPLIED_VOL_SURFACE` — ``LeverageGrid``
      shares the arity but is quoted in log-moneyness, and ``SwaptionCube``
      needs a tenor.  ``resolve_surface`` does no kind checking and falls
      back to the market's default surface whatever it is, so this check
      is the only thing standing between a leverage grid and a
      Black-Scholes vol;
    * a surface quoting **normal** (Bachelier) vols, which must not be fed
      to a lognormal closed form.

    Args:
        surface: A duck-typed vol surface.
        strike: Absolute strike.
        horizon: Expiry as a year fraction.

    Returns:
        Implied (lognormal) volatility.

    Raises:
        ValueError: If ``surface`` is ``None``, is not an implied-vol
            surface, or quotes normal vols.
    """
    if surface is None:
        raise ValueError(
            "A model left `vol=None` but the MarketState carries no "
            "surface for this instrument's underlying. Add one (e.g. "
            "MarketState.single_curve(curve, vol=0.2)) or give the model "
            "an explicit vol."
        )
    if isinstance(surface, _NOT_AN_IMPLIED_VOL_SURFACE):
        raise ValueError(
            f"{type(surface).__name__} is not an implied-vol surface "
            f"quoted in (strike, expiry), so sampling it for a model vol "
            f"would silently use the wrong units or axes. File an "
            f"implied-vol surface under the '<UNDERLYING>.VOL' key, or "
            f"give the model an explicit vol."
        )
    if getattr(surface, "is_normal", False):
        raise ValueError(
            f"{type(surface).__name__} quotes normal (Bachelier) vols, "
            f"which cannot be used as a lognormal model vol. Convert it "
            f"first, or give the model an explicit vol."
        )
    return jnp.asarray(surface(strike, horizon))


# ── Binding ──────────────────────────────────────────────────────────


def _bound_value(
    field: str,
    instrument: Any,
    resolved: Any,
) -> Float[Array, ""]:
    """Compute one market-sourced parameter.

    Args:
        field: One of :data:`BINDABLE_FIELDS`.
        instrument: The instrument being priced (supplies the horizon and,
            for ``vol``, the strike).
        resolved: The bound market objects.

    Returns:
        The concrete parameter value.

    Raises:
        ValueError: If the market cannot supply it.
    """
    horizon = instrument_horizon(instrument, resolved.discount_curve)

    if field == "rate":
        return project_rate(resolved.discount_curve, horizon)

    if field == "dividend":
        if resolved.dividend is None:
            raise ValueError(
                "A model left `dividend=None` but the MarketState carries "
                "no assets, so there is no dividend yield to read. Give "
                "the model an explicit dividend."
            )
        return jnp.asarray(resolved.dividend)

    strike = getattr(instrument, "strike", None)
    if strike is None:
        raise ValueError(
            f"A model left `vol=None` but "
            f"{type(instrument).__name__} has no `strike`, so its surface "
            f"cannot be sampled. Give the model an explicit vol."
        )
    return sample_vol(resolved.surface, jnp.asarray(strike), horizon)


def bind_model(model: Any, instrument: Any, resolved: Any) -> Any:
    """Fill a model's ``None`` parameters from the resolved market.

    Called once by :func:`valax.core.pricing.price`, between ``resolve``
    and pricer dispatch.  Returns a model of the **same class**, so
    dispatch on ``type(model)`` is unaffected.

    A model with nothing left to bind is returned **as-is**, not copied,
    which is what guarantees that a fully-specified (textbook) model
    prices bit-identically to before this layer existed.

    Args:
        model: A VALAX model pytree, or ``None`` for curve-discounted
            pricing (returned unchanged).
        instrument: The instrument being priced; supplies the projection
            horizon and, for ``vol``, the strike.
        resolved: The :class:`~valax.core.resolve.ResolvedMarket` for this
            ``(instrument, model, market)`` triple.

    Returns:
        The model with every bindable parameter concrete.

    Raises:
        ValueError: If a parameter is ``None`` and the market cannot
            supply it, or if the instrument exposes no usable horizon.
    """
    if model is None:
        return model

    pending = [
        field for field in BINDABLE_FIELDS
        if getattr(model, field, None) is None and hasattr(model, field)
    ]
    if not pending:
        return model

    bound = model
    for field in pending:
        value = _bound_value(field, instrument, resolved)
        bound = eqx.tree_at(
            lambda m, _f=field: getattr(m, _f),
            bound,
            value,
            is_leaf=lambda x: x is None,
        )
    return bound
