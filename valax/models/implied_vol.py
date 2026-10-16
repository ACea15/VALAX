"""The market-quoted implied-volatility "model" of the rates options."""

import equinox as eqx


class ImpliedVolModel(eqx.Module):
    """Price off a market implied-vol surface, by its id.

    The quoting model of the vanilla rates options: no dynamics, just the
    convention the market quotes them in. A cap, caplet or European swaption
    priced under ``ImpliedVolModel(surface_id)`` reads the vol source filed
    under ``surface_id`` in the :class:`~valax.core.market_state.MarketState`
    — a :class:`~valax.surfaces.swaption_cube.SwaptionCube`, an
    :class:`~valax.surfaces.optionlet_surface.OptionletVolSurface` or a
    :class:`~valax.surfaces.constant.ConstantVol` — at the instrument's own
    coordinate and live forward, and prices Black-76 or Bachelier per the
    source's ``is_normal`` flag
    (:mod:`~valax.pricing.analytic.rates_smile`). An inflation cap / floor
    under it reads a lognormal vol at ``(strike, fixing time)`` for each
    caplet and prices Black-76 on the forward year-on-year rate. The model
    holds no numbers, so every sensitivity lands on the curves and the
    surface.

    Example::

        res = price(swaption, ImpliedVolModel("USD.SWPN.VOL"), market)

    Attributes:
        surface_id: Key of the vol source in ``MarketState.surfaces``
            (``"_default_"`` for a single-surface market).
    """

    surface_id: str = eqx.field(static=True)
