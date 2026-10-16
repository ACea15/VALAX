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
* **Short-rate closed forms** — the European swaption under Hull-White
  (Jamshidian) and G2++ (Brigo-Mercurio); the model's curve is bound from
  the market by ``price``.
* **Market implied vol** — caplets, caps and European swaptions under
  :class:`~valax.models.implied_vol.ImpliedVolModel`, which names the vol
  source (swaption cube, optionlet surface, flat vol) in the market.
* **Model-free** (``model=None``) — instruments priced by curve
  discounting alone: fixed-rate bonds, vanilla and OIS swaps, FRNs, and
  trades expressed as legs (``MultiLegSwap``: tenor basis, amortising,
  seasoned), which read every curve they name from ``resolved.curves``.
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

import jax
import jax.numpy as jnp

from valax.core.binding import sample_vol
from valax.core.pricing import PricingResult, register_pricer
from valax.core.resolve import ResolvedMarket
from valax.dates.daycounts import year_fraction
from valax.instruments.bonds import (
    CallableBond,
    FixedRateBond,
    FloatingRateBond,
    PuttableBond,
)
from valax.instruments.inflation import (
    InflationCapFloor,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)
from valax.instruments.legs import MultiLegSwap
from valax.instruments.options import (
    AmericanOption,
    DigitalOption,
    EuropeanOption,
)
from valax.instruments.rates import (
    BermudanSwaption,
    Cap,
    Caplet,
    CrossCurrencySwap,
    InterestRateSwap,
    OISSwap,
    Swaption,
)
from valax.models.black_scholes import BlackScholesModel
from valax.models.g2pp import G2PPModel
from valax.models.heston import HestonModel
from valax.models.hull_white import HullWhiteModel
from valax.models.implied_vol import ImpliedVolModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.analytic.digital import digital_option_price
from valax.pricing.analytic.floating import floating_rate_bond_price, ois_swap_price
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.heston import heston_cos_price
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.analytic.inflation import (
    inflation_cap_floor_price_black76,
    yyis_price,
    zcis_price,
)
from valax.pricing.analytic.legs import swap_pv
from valax.pricing.analytic.rates_exotics import cross_currency_swap_price
from valax.pricing.analytic.rates_smile import (
    cap_price as smile_cap_price,
    caplet_price as smile_caplet_price,
    swaption_price as smile_swaption_price,
)
from valax.pricing.analytic.swaptions import swap_price
from valax.pricing.lattice.binomial import BinomialConfig, binomial_price
from valax.pricing.lattice.hull_white_tree import (
    bermudan_swaption_tree_price,
    callable_bond_price,
    puttable_bond_price,
    swaption_tree_price,
)


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


@register_pricer(MultiLegSwap, type(None), "analytic")
def _multi_leg_swap(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """A trade as a tuple of legs, each on its own curves (D16).

    ``resolved.curves`` holds every curve the legs name — e.g. the OIS
    discount curve and the 3M and 6M projection curves of a tenor basis
    swap — and the leg engine reads each leg's own ids from it; realised
    fixings come from the market by each leg's index id.
    """
    del model, market, config, key
    return PricingResult(
        price=swap_pv(
            instrument, resolved.curves, resolved.fixings,
            resolved.reference_date,
        ),
        method="analytic",
    )


@register_pricer(CrossCurrencySwap, type(None), "analytic")
def _cross_currency_swap(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Cross-currency basis swap: two discount curves and the FX spot.

    The domestic and foreign curves are the trade's ``discount_id`` and
    ``foreign_discount_id`` in ``resolved.curves``; the spot (domestic per
    foreign) is ``resolved.fx_spots`` under the pair without the ``/``.
    """
    del model, market, config, key
    if instrument.forward_id is not None:
        raise ValueError(
            "CrossCurrencySwap projects each currency's floating leg off its "
            "own discount curve; a distinct forward_id would be ignored, so "
            "it is refused. Leave forward_id=None."
        )
    (fx_id,) = instrument.fx_ids
    return PricingResult(
        price=cross_currency_swap_price(
            instrument,
            resolved.curves[instrument.discount_id],
            resolved.curves[instrument.foreign_discount_id],
            resolved.fx_spots[fx_id],
        ),
        method="analytic",
    )


# ── Inflation (keyed inflation curves) ───────────────────────────────


@register_pricer(ZeroCouponInflationSwap, type(None), "analytic")
def _zcis(*, instrument, model, market, resolved, config, key) -> PricingResult:
    """Zero-coupon inflation swap off its inflation and discount curves."""
    del model, market, config, key
    return PricingResult(
        price=zcis_price(
            instrument, resolved.inflation_curves[instrument.inflation_id],
            resolved.discount_curve,
        ),
        method="analytic",
    )


@register_pricer(YearOnYearInflationSwap, type(None), "analytic")
def _yyis(*, instrument, model, market, resolved, config, key) -> PricingResult:
    """Year-on-year inflation swap (forward-CPI ratios, no convexity)."""
    del model, market, config, key
    return PricingResult(
        price=yyis_price(
            instrument, resolved.inflation_curves[instrument.inflation_id],
            resolved.discount_curve,
        ),
        method="analytic",
    )


@register_pricer(InflationCapFloor, ImpliedVolModel, "analytic")
def _inflation_cap_floor(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Inflation cap / floor: Black-76 on the forward YoY rate.

    Each caplet reads the lognormal vol from the surface named by the
    ``ImpliedVolModel`` at ``(strike, its fixing time)`` — the time the
    Black-76 kernel uses; a normal-vol source or a non-(strike, expiry)
    surface is refused (:func:`~valax.core.binding.sample_vol`).
    """
    del market, config, key
    starts = jnp.concatenate(
        [instrument.effective_date[None], instrument.payment_dates[:-1]]
    )
    fixing_times = year_fraction(
        resolved.discount_curve.reference_date, starts, instrument.day_count
    )
    vols = jax.vmap(
        lambda t: sample_vol(resolved.surface, instrument.strike, t)
    )(fixing_times)
    return PricingResult(
        price=inflation_cap_floor_price_black76(
            instrument, resolved.inflation_curves[instrument.inflation_id],
            resolved.discount_curve, vols,
        ),
        method="analytic",
    )


# ── Short-rate closed forms ──────────────────────────────────────────


@register_pricer(Swaption, HullWhiteModel, "analytic")
def _swaption_hw_analytic(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """European swaption under Hull-White: Jamshidian's decomposition.

    ``price`` has already bound the model's ``initial_curve`` (the discount
    curve the model is exact-fitted to); the projection curve, if the
    instrument names one, enters through the frozen tenor-basis overlay.
    Exact, so it is the default engine — and the one per-trade basket
    calibrations price their basket with.
    """
    del market, config, key
    return PricingResult(
        price=hw_swaption_price(instrument, model, resolved.forward_curve),
        method="analytic",
    )


@register_pricer(Swaption, G2PPModel, "analytic")
def _swaption_g2pp_analytic(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """European swaption under G2++: the Brigo-Mercurio one-dimensional integral.

    As for Hull-White, ``price`` has bound the model's ``initial_curve`` and
    the projection curve enters through the frozen tenor-basis overlay. Exact
    to the 64-node Gauss-Hermite quadrature (well below 1e-8 relative), smooth
    in the parameters, so it is the default engine and the one per-trade
    basket calibrations price their basket with.
    """
    del market, config, key
    return PricingResult(
        price=g2pp_swaption_price(instrument, model, resolved.forward_curve),
        method="analytic",
    )


# ── Market implied vol (cube / optionlet surface / flat) ─────────────
#
# ``ImpliedVolModel(surface_id)`` names the vol source; ``resolve`` has bound
# it to ``resolved.surface``. The kernels read the vol at the instrument's
# coordinate and live forward and pick Black-76 / Bachelier from the source.


def _vol_source(resolved: ResolvedMarket, model: ImpliedVolModel):
    """The resolved surface, checked to be a rates vol source."""
    surface = resolved.surface
    if not (callable(surface) and hasattr(surface, "is_normal")):
        raise ValueError(
            f"ImpliedVolModel({model.surface_id!r}) resolved a "
            f"{type(surface).__name__}, which is not a rates vol source (a "
            f"callable with an `is_normal` flag, e.g. SwaptionCube, "
            f"OptionletVolSurface or ConstantVol)."
        )
    return surface


@register_pricer(Caplet, ImpliedVolModel, "analytic")
def _caplet_implied_vol(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Caplet / floorlet off the market optionlet vol."""
    del market, config, key
    return PricingResult(
        price=smile_caplet_price(
            instrument, resolved.discount_curve, _vol_source(resolved, model),
            resolved.forward_curve,
        ),
        method="analytic",
    )


@register_pricer(Cap, ImpliedVolModel, "analytic")
def _cap_implied_vol(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Cap / floor as a strip of optionlets, each at its own market vol."""
    del market, config, key
    return PricingResult(
        price=smile_cap_price(
            instrument, resolved.discount_curve, _vol_source(resolved, model),
            resolved.forward_curve,
        ),
        method="analytic",
    )


@register_pricer(Swaption, ImpliedVolModel, "analytic")
def _swaption_implied_vol(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """European swaption off the market swaption vol (cube or flat).

    The route per-trade basket calibrations price their targets with.
    """
    del market, config, key
    return PricingResult(
        price=smile_swaption_price(
            instrument, resolved.discount_curve, _vol_source(resolved, model),
            resolved.forward_curve,
        ),
        method="analytic",
    )


# ── Hull-White trinomial tree ────────────────────────────────────────
#
# On the general tree with a config-sized half-width (``TreeConfig``), so the
# routes run on a traced model under ``jit`` / ``grad``. ``auto`` stays the
# PDE / closed form for these
# pairs (:mod:`valax.core.policy`); the tree is the independent check.


def _single_curve_tree(resolved: ResolvedMarket, what: str) -> None:
    """Refuse a projection curve the bond tree cannot consume."""
    if (
        resolved.forward_curve is not None
        and resolved.forward_curve is not resolved.discount_curve
    ):
        raise ValueError(
            f"The lattice route for {what} prices single-curve, but the "
            f"instrument resolves a projection curve distinct from its "
            f"discount curve."
        )


@register_pricer(CallableBond, HullWhiteModel, "lattice")
def _callable_bond_hw_tree(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Callable bond on the general Hull-White tree."""
    del market, key
    _single_curve_tree(resolved, "CallableBond")
    return PricingResult(
        price=callable_bond_price(
            instrument, model, n_steps=config.n_steps,
            j_max=config.half_width, general=True,
        ),
        method="lattice",
    )


@register_pricer(PuttableBond, HullWhiteModel, "lattice")
def _puttable_bond_hw_tree(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Puttable bond on the general Hull-White tree."""
    del market, key
    _single_curve_tree(resolved, "PuttableBond")
    return PricingResult(
        price=puttable_bond_price(
            instrument, model, n_steps=config.n_steps,
            j_max=config.half_width, general=True,
        ),
        method="lattice",
    )


@register_pricer(Swaption, HullWhiteModel, "lattice")
def _swaption_hw_tree(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """European swaption on the general Hull-White tree (Jamshidian's check)."""
    del market, key
    return PricingResult(
        price=swaption_tree_price(
            instrument, model, n_steps=config.n_steps, j_max=config.half_width,
            forward_curve=resolved.forward_curve,
            smooth_payoff=config.smooth_payoff, general=True,
        ),
        method="lattice",
    )


@register_pricer(BermudanSwaption, HullWhiteModel, "lattice")
def _bermudan_swaption_hw_tree(
    *, instrument, model, market, resolved, config, key,
) -> PricingResult:
    """Bermudan swaption on the general Hull-White tree."""
    del market, key
    return PricingResult(
        price=bermudan_swaption_tree_price(
            instrument, model, n_steps=config.n_steps, j_max=config.half_width,
            forward_curve=resolved.forward_curve,
            smooth_payoff=config.smooth_payoff, general=True,
        ),
        method="lattice",
    )


# The ``method="auto"`` policy of every pair lives in
# :mod:`valax.core.policy`, next to the default configs.
