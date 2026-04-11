"""FX derivative instrument definitions (data-only pytrees).

FX instruments are quoted in terms of a **currency pair** ``FOR/DOM``
(e.g., EUR/USD means 1 EUR = X USD).  The **foreign** currency is the
asset (numeraire of the option payoff), and the **domestic** currency
is the pricing currency.  In Garman-Kohlhagen terms:

- ``spot`` is the price of 1 unit of foreign currency in domestic terms
- ``r_domestic`` is the domestic risk-free rate
- ``r_foreign`` is the foreign risk-free rate (acts like a dividend yield)
"""

import equinox as eqx
from jaxtyping import Float, Int
from jax import Array


class FXForward(eqx.Module):
    """FX forward contract.

    Agreement to exchange ``notional_foreign`` units of foreign currency
    for ``notional_foreign * strike`` units of domestic currency at
    maturity.  ``is_buy`` = True means buying foreign / selling domestic.

    The fair forward rate is: ``F = S * exp((r_dom - r_for) * T)``.
    NPV is zero at inception when ``strike = F``.

    Attributes:
        strike: Delivery FX rate (domestic per foreign).
        expiry: Time to maturity in year fractions.
        notional_foreign: Amount in foreign currency.
        is_buy: True = buy foreign / sell domestic.
        currency_pair: E.g., ``"EUR/USD"`` (static, not traced).
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    notional_foreign: Float[Array, ""]
    is_buy: bool = eqx.field(static=True, default=True)
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")


class FXVanillaOption(eqx.Module):
    """European FX vanilla option (Garman-Kohlhagen).

    A call gives the right to **buy** foreign currency at the strike
    rate.  A put gives the right to **sell** foreign currency.

    The payoff in domestic terms is:

    - Call: ``notional * max(S_T - K, 0)``
    - Put:  ``notional * max(K - S_T, 0)``

    ``premium_currency`` controls whether the option premium is quoted
    in domestic or foreign currency units.  This affects the
    **premium-adjusted delta** used in the FX market.

    Attributes:
        strike: Strike FX rate (domestic per foreign).
        expiry: Time to expiry in year fractions.
        notional_foreign: Notional in foreign currency.
        is_call: True for call (buy foreign), False for put (sell foreign).
        premium_currency: ``"domestic"`` or ``"foreign"`` (static).
        currency_pair: E.g., ``"EUR/USD"`` (static).
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    notional_foreign: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)
    premium_currency: str = eqx.field(static=True, default="domestic")
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")


class FXBarrierOption(eqx.Module):
    """FX single-barrier option.

    European option that is activated (knock-in) or deactivated
    (knock-out) if the spot rate breaches a barrier level during
    the option's life.

    - Up-and-in / Up-and-out: barrier above current spot
    - Down-and-in / Down-and-out: barrier below current spot

    For analytical pricing of continuous barriers, closed-form solutions
    exist.  For discrete monitoring, Monte Carlo is required.

    Attributes:
        strike: Strike FX rate.
        expiry: Time to expiry in year fractions.
        notional_foreign: Notional in foreign currency.
        barrier: Barrier FX rate level.
        is_call: True for call, False for put.
        is_up: True if barrier is above spot (up-barrier).
        is_knock_in: True for knock-in, False for knock-out.
        currency_pair: E.g., ``"EUR/USD"`` (static).
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    notional_foreign: Float[Array, ""]
    barrier: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)
    is_up: bool = eqx.field(static=True, default=True)
    is_knock_in: bool = eqx.field(static=True, default=True)
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")


class QuantoOption(eqx.Module):
    """Quanto (quantity-adjusted) option — foreign asset, domestic payout.

    A European option on a foreign underlying asset where the payoff
    is converted to the domestic currency at a **fixed** (pre-agreed)
    FX rate rather than the prevailing spot rate.

    The quanto payoff in domestic terms is:

    - Call: ``Q * max(S_T - K, 0)`` (in domestic currency)
    - Put:  ``Q * max(K - S_T, 0)`` (in domestic currency)

    where ``S_T`` is the foreign-currency asset price and ``Q`` is the
    fixed quanto FX rate.

    The **quanto adjustment** modifies the drift of the foreign asset
    by ``-ρ * σ_S * σ_FX`` (correlation between asset and FX rate
    times the product of their volatilities), reducing the forward.

    Attributes:
        strike: Strike in foreign currency.
        expiry: Time to expiry in year fractions.
        notional: Notional amount (in domestic currency terms).
        quanto_fx_rate: Fixed FX conversion rate (domestic per foreign).
        is_call: True for call, False for put.
        currency_pair: Underlying FX pair, e.g., ``"EUR/USD"`` (static).
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    notional: Float[Array, ""]
    quanto_fx_rate: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")


class TARF(eqx.Module):
    """Target Accrual Range Forward (TARF).

    An FX structured product consisting of a series of forward-like
    settlements with **early termination** once cumulative gains reach
    a target (the "target accrual").

    At each fixing date, if spot is above strike (for a buy TARF):

    - Gain = ``(spot - strike) * notional_per_fixing``
    - Accumulated gains increase; if they reach ``target``, the
      contract terminates.

    If spot is below strike:

    - Loss = ``(strike - spot) * notional_per_fixing * leverage``
    - Losses are typically leveraged (2x is common), making the
      downside asymmetric.

    TARFs are popular in Asian FX markets and are highly
    path-dependent — pricing requires Monte Carlo simulation.

    Attributes:
        fixing_dates: Observation/fixing dates (ordinals, shape n).
        strike: Strike FX rate.
        target: Target accrual level (accumulated gains trigger termination).
        notional_per_fixing: Notional for each fixing period.
        leverage: Leverage multiplier on losses (e.g., 2.0).
        is_buy: True = client buys foreign at strike (benefits from
            higher spot).
        currency_pair: E.g., ``"USD/CNH"`` (static).
    """

    fixing_dates: Int[Array, " n"]
    strike: Float[Array, ""]
    target: Float[Array, ""]
    notional_per_fixing: Float[Array, ""]
    leverage: Float[Array, ""]
    is_buy: bool = eqx.field(static=True, default=True)
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")


class FXSwap(eqx.Module):
    """FX swap — simultaneous spot and forward FX transactions.

    An FX swap consists of two legs:

    1. **Near leg**: exchange currencies at the spot rate on the near date.
    2. **Far leg**: reverse the exchange at the forward rate on the far date.

    The **swap points** (forward points) reflect the interest rate
    differential between the two currencies.

    FX swaps are the most traded instrument in the FX market by
    volume, used primarily for funding and hedging.

    Attributes:
        near_date: Near leg settlement date as ordinal.
        far_date: Far leg settlement date as ordinal.
        spot_rate: Agreed spot FX rate (domestic per foreign) for near leg.
        forward_rate: Agreed forward FX rate for far leg.
        notional_foreign: Amount in foreign currency.
        is_buy_near: True = buy foreign on near leg / sell on far leg.
        currency_pair: E.g., ``"EUR/USD"`` (static).
    """

    near_date: Int[Array, ""]
    far_date: Int[Array, ""]
    spot_rate: Float[Array, ""]
    forward_rate: Float[Array, ""]
    notional_foreign: Float[Array, ""]
    is_buy_near: bool = eqx.field(static=True, default=True)
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
