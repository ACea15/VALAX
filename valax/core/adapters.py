"""Adapters registering the existing pricers behind :func:`price`.

Imported for its side effects by :func:`valax.core.pricing.price`.  Every
function here is a **thin shim**: it unpacks the
:class:`~valax.core.resolve.ResolvedMarket` and calls a pricer that
already existed, unchanged.  No numerics live in this module — that is
the whole point of P2 of the ontology migration ("core types + unified
dispatch, adapters only; no pricer rewrites").

Two families are registered:

* **Model-driven** — the closed forms and the tree: Black-Scholes and
  Heston-COS European, Black-Scholes digital, and the binomial American.
  The model supplies vol / rate / dividend; the market supplies spot.
* **Model-free** (``model=None``) — instruments priced by curve
  discounting alone: fixed-rate bonds, vanilla and OIS swaps, FRNs.
  These take their discount curve (and optional projection curve) from
  the market via the instrument's own selectors, which is what makes a
  multi-curve trade a one-line change from a single-curve one.

MC and PDE need no adapters at all: :func:`valax.core.pricing.price`
bridges their existing registries directly, so every recipe in
``valax/pricing/{mc,pde}/`` is reachable through the unified signature
the moment it is registered there.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from valax.core.pricing import PricingResult, register_pricer, set_auto_method
from valax.core.resolve import ResolvedMarket
from valax.instruments.bonds import FixedRateBond, FloatingRateBond
from valax.instruments.options import (
    AmericanOption,
    DigitalOption,
    EuropeanOption,
)
from valax.instruments.rates import InterestRateSwap, OISSwap
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.analytic.digital import digital_option_price
from valax.pricing.analytic.floating import floating_rate_bond_price, ois_swap_price
from valax.pricing.analytic.heston import heston_cos_price
from valax.pricing.analytic.swaptions import swap_price
from valax.pricing.lattice.binomial import BinomialConfig, binomial_price


def _require_spot(resolved: ResolvedMarket, what: str):
    """Return the resolved spot or explain what the market is missing."""
    if resolved.spot is None:
        raise ValueError(
            f"{what} needs a spot: the MarketState carries none for this "
            f"instrument's underlying.  Build it with "
            f"MarketState.single_curve(curve, spot=...) or add the asset "
            f"to MarketState.spots."
        )
    return resolved.spot


# ── Equity closed forms ──────────────────────────────────────────────


@register_pricer(EuropeanOption, BlackScholesModel, "analytic")
def _european_bs_analytic(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Black-Scholes-Merton closed form."""
    del market, config, key
    return PricingResult(
        price=black_scholes_price(
            instrument,
            _require_spot(resolved, "Black-Scholes pricing"),
            model.vol,
            model.rate,
            model.dividend,
        ),
        method="analytic",
    )


@register_pricer(DigitalOption, BlackScholesModel, "analytic")
def _digital_bs_analytic(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Cash-or-nothing / asset-or-nothing closed form."""
    del market, config, key
    return PricingResult(
        price=digital_option_price(
            instrument,
            _require_spot(resolved, "Digital option pricing"),
            model.vol,
            model.rate,
            model.dividend,
        ),
        method="analytic",
    )


@register_pricer(EuropeanOption, HestonModel, "analytic")
def _european_heston_analytic(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Fourier-cosine (COS) expansion under Heston."""
    del market, config, key
    return PricingResult(
        price=heston_cos_price(
            instrument,
            _require_spot(resolved, "Heston COS pricing"),
            model=model,
        ),
        method="analytic",
    )


@register_pricer(EuropeanOption, BlackScholesModel, "lattice")
def _european_bs_lattice(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """CRR binomial tree, European exercise."""
    del market, key
    return PricingResult(
        price=binomial_price(
            instrument,
            _require_spot(resolved, "Binomial pricing"),
            model.vol,
            model.rate,
            model.dividend,
            config if config is not None else BinomialConfig(american=False),
        ),
        method="lattice",
    )


@register_pricer(AmericanOption, BlackScholesModel, "lattice")
def _american_bs_lattice(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """CRR binomial tree with early exercise.

    ``config`` defaults to ``BinomialConfig(american=True)``; an explicit
    config is honoured as given (the caller owns the exercise flag).
    """
    del market, key
    return PricingResult(
        price=binomial_price(
            instrument,
            _require_spot(resolved, "Binomial pricing"),
            model.vol,
            model.rate,
            model.dividend,
            config if config is not None else BinomialConfig(american=True),
        ),
        method="lattice",
    )


# ── Curve-discounted instruments (model-free) ────────────────────────


@register_pricer(FixedRateBond, type(None), "analytic")
def _fixed_rate_bond(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Deterministic cashflow discounting — no model required."""
    del model, market, config, key
    return PricingResult(
        price=fixed_rate_bond_price(instrument, resolved.discount_curve),
        method="analytic",
    )


@register_pricer(InterestRateSwap, type(None), "analytic")
def _vanilla_swap(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Dual-curve vanilla swap NPV (single-curve when no projection id)."""
    del model, market, config, key
    return PricingResult(
        price=swap_price(
            instrument, resolved.discount_curve, resolved.forward_curve,
        ),
        method="analytic",
    )


@register_pricer(OISSwap, type(None), "analytic")
def _ois_swap(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """OIS swap NPV."""
    del model, market, config, key
    return PricingResult(
        price=ois_swap_price(
            instrument, resolved.discount_curve, resolved.forward_curve,
        ),
        method="analytic",
    )


@register_pricer(FloatingRateBond, type(None), "analytic")
def _floating_rate_bond(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """FRN present value."""
    del model, market, config, key
    return PricingResult(
        price=floating_rate_bond_price(
            instrument, resolved.discount_curve, resolved.forward_curve,
        ),
        method="analytic",
    )


# ── ``method="auto"`` policy (Q2) ────────────────────────────────────
#
# Only pairs whose default differs from the priority order, or whose
# default is worth stating explicitly, need an entry; everything else
# falls through to analytic → lattice → pde → mc.

set_auto_method(EuropeanOption, BlackScholesModel, "analytic")
set_auto_method(DigitalOption, BlackScholesModel, "analytic")
set_auto_method(EuropeanOption, HestonModel, "analytic")
set_auto_method(AmericanOption, BlackScholesModel, "lattice")
set_auto_method(FixedRateBond, type(None), "analytic")
set_auto_method(InterestRateSwap, type(None), "analytic")
set_auto_method(OISSwap, type(None), "analytic")
set_auto_method(FloatingRateBond, type(None), "analytic")
