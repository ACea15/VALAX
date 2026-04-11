"""Equity option and exotic instrument definitions (data-only pytrees)."""

from typing import Optional

import equinox as eqx
from jaxtyping import Float, Int
from jax import Array


class EuropeanOption(eqx.Module):
    """European option contract.

    Data-only pytree — no pricing logic. Pass to a pricing function
    along with market data to get a price.
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]  # time to expiry in year fractions
    is_call: bool = eqx.field(static=True, default=True)


class AmericanOption(eqx.Module):
    """American option contract — exercisable at any time up to expiry.

    Structurally identical to :class:`EuropeanOption` but semantically
    distinct: pricing functions that accept an ``AmericanOption`` must
    use methods that handle early exercise (binomial trees, PDE with
    free boundary, or Longstaff-Schwartz MC).

    For calls on non-dividend-paying stocks, the American price equals
    the European price (early exercise is never optimal). For puts or
    when dividends are present, the early exercise premium is positive.

    Attributes:
        strike: Exercise price.
        expiry: Time to expiry in year fractions.
        is_call: True for call, False for put.
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)


class EquityBarrierOption(eqx.Module):
    """Equity single-barrier option.

    European option that activates (knock-in) or deactivates (knock-out)
    if the underlying spot price breaches a barrier during the option's
    life.  Barrier monitoring is assumed continuous for analytical
    pricing and discrete (per time step) for Monte Carlo.

    **Knock-in / knock-out parity:** For the same strike, barrier,
    and type, ``knock_in_price + knock_out_price = vanilla_price``.

    Attributes:
        strike: Exercise price.
        expiry: Time to expiry in year fractions.
        barrier: Barrier price level.
        is_call: True for call, False for put.
        is_up: True if barrier is above current spot (up barrier).
        is_knock_in: True for knock-in, False for knock-out.
        smoothing: Sigmoid smoothing width for differentiable MC payoffs.
            Set to 0.0 for hard barriers (non-differentiable).
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    barrier: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)
    is_up: bool = eqx.field(static=True, default=True)
    is_knock_in: bool = eqx.field(static=True, default=True)
    smoothing: float = eqx.field(static=True, default=0.0)


class AsianOption(eqx.Module):
    """Asian (average price) option.

    The payoff depends on the **average** spot price over a set of
    observation dates, rather than the terminal spot:

    - Arithmetic Asian call: ``max(A - K, 0)``
    - Arithmetic Asian put:  ``max(K - A, 0)``

    where ``A`` is the arithmetic (or geometric) average of observed
    spot prices.

    Asian options have lower vega and gamma than equivalent vanillas
    because averaging reduces the effective volatility.  No closed-form
    price exists for arithmetic averages under BSM (but geometric
    averages have a closed form).

    Attributes:
        strike: Exercise price.
        expiry: Time to expiry in year fractions.
        is_call: True for call, False for put.
        averaging: ``"arithmetic"`` or ``"geometric"`` (static).
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)
    averaging: str = eqx.field(static=True, default="arithmetic")


class LookbackOption(eqx.Module):
    """Lookback option — payoff depends on the path extremum.

    **Floating strike** (``is_fixed_strike=False``):

    - Call payoff: ``S_T - min(S_t)``  (buy at the lowest price)
    - Put payoff:  ``max(S_t) - S_T``  (sell at the highest price)

    **Fixed strike** (``is_fixed_strike=True``):

    - Call payoff: ``max(max(S_t) - K, 0)``
    - Put payoff:  ``max(K - min(S_t), 0)``

    Floating-strike lookbacks are always in the money (the payoff is
    non-negative by construction), so they are more expensive than
    vanillas.

    Attributes:
        expiry: Time to expiry in year fractions.
        is_call: True for call, False for put.
        is_fixed_strike: True for fixed-strike, False for floating-strike.
        strike: Exercise price (used only when ``is_fixed_strike=True``).
    """

    expiry: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)
    is_fixed_strike: bool = eqx.field(static=True, default=False)
    strike: Float[Array, ""] = eqx.field(default=None)


class VarianceSwap(eqx.Module):
    """Variance swap — pays the difference between realized and strike variance.

    The payoff at expiry is:

    .. math::

        N_{\\text{var}} \\cdot (\\sigma_{\\text{realized}}^2 - K_{\\text{var}})

    where :math:`\\sigma_{\\text{realized}}^2` is the annualized realized
    variance computed from discrete log returns, and :math:`K_{\\text{var}}`
    is the variance strike (squared vol strike).

    Under Black-Scholes, the fair variance strike equals the squared
    implied volatility: :math:`K_{\\text{var}} = \\sigma^2`.

    The **vega notional** (more intuitive) relates to the variance
    notional as: :math:`N_{\\text{vega}} = 2 \\sigma \\cdot N_{\\text{var}}`,
    so a 1-vol-point move in implied vol produces a P&L of approximately
    :math:`N_{\\text{vega}}` in dollar terms.

    Attributes:
        expiry: Time to expiry / observation period in year fractions.
        strike_var: Variance strike :math:`K_{\\text{var}} = K_{\\text{vol}}^2`.
        notional_var: Variance notional (for vega notional, use
            :math:`N_{\\text{var}} = N_{\\text{vega}} / (2 K_{\\text{vol}})`).
    """

    expiry: Float[Array, ""]
    strike_var: Float[Array, ""]
    notional_var: Float[Array, ""]


class CompoundOption(eqx.Module):
    """Compound option — an option on an option.

    The holder has the right to buy (or sell) an underlying vanilla
    European option at a future date.  There are four types:

    - Call on call, call on put, put on call, put on put.

    At the **outer expiry**, the holder decides whether to pay the
    **outer strike** to acquire the underlying option.  The underlying
    option then lives from the outer expiry to the **inner expiry**
    with the **inner strike**.

    Compound options are sensitive to the term structure of volatility
    and are priced via bivariate normal integrals (Geske's formula)
    or, more generally, via PDE/MC.

    Attributes:
        outer_expiry: Expiry of the compound option (year fractions).
        outer_strike: Premium to acquire the underlying option.
        inner_expiry: Expiry of the underlying option (year fractions).
            Must be > outer_expiry.
        inner_strike: Strike of the underlying option.
        outer_is_call: True if the compound option is a call (right to buy).
        inner_is_call: True if the underlying option is a call.
    """

    outer_expiry: Float[Array, ""]
    outer_strike: Float[Array, ""]
    inner_expiry: Float[Array, ""]
    inner_strike: Float[Array, ""]
    outer_is_call: bool = eqx.field(static=True, default=True)
    inner_is_call: bool = eqx.field(static=True, default=True)


class ChooserOption(eqx.Module):
    """Chooser option — holder chooses call or put at a future date.

    At the **choose date**, the holder selects whether the option
    becomes a European call or put.  After the choice, the selected
    option lives until the common expiry.

    A simple chooser (same strike and expiry for call and put) can be
    decomposed using put-call parity into a call plus a put with
    adjusted parameters.  Complex choosers (different strikes/expiries
    for call and put) require numerical methods.

    Attributes:
        choose_date: Date when call/put choice is made (year fractions).
        expiry: Final expiry of the resulting option (year fractions).
        strike: Strike price.
    """

    choose_date: Float[Array, ""]
    expiry: Float[Array, ""]
    strike: Float[Array, ""]


class Autocallable(eqx.Module):
    """Autocallable / phoenix structured note.

    A path-dependent structured product with:

    - **Autocall barrier**: if the underlying is above this level on
      an observation date, the note redeems early at par plus coupon.
    - **Coupon barrier**: if the underlying is above this (lower)
      level, a coupon is paid.  In "phoenix" variants, missed coupons
      can be paid later (memory feature).
    - **Knock-in put barrier**: if the underlying breaches this level
      at any point (or at maturity), the investor is exposed to
      downside (receives shares or cash below strike).

    Autocallables are among the most traded structured products
    globally, particularly in Asia and Europe.

    Attributes:
        observation_dates: Autocall/coupon observation dates (ordinals, shape n).
        autocall_barrier: Autocall trigger level as fraction of initial spot
            (e.g., 1.0 = at-the-money).
        coupon_barrier: Coupon payment trigger level as fraction of initial.
        coupon_rate: Periodic coupon rate (e.g., 0.08 for 8% p.a.).
        ki_barrier: Knock-in put barrier as fraction of initial spot.
        strike: Put strike as fraction of initial (if knocked in).
        notional: Note face value / notional.
        has_memory: True if missed coupons are paid when barrier is
            subsequently breached (phoenix/memory feature).
    """

    observation_dates: Int[Array, " n"]
    autocall_barrier: Float[Array, ""]
    coupon_barrier: Float[Array, ""]
    coupon_rate: Float[Array, ""]
    ki_barrier: Float[Array, ""]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    has_memory: bool = eqx.field(static=True, default=False)


class WorstOfBasketOption(eqx.Module):
    """Worst-of basket option — payoff depends on the worst performer.

    A European option whose payoff is determined by the **worst-
    performing** asset in a basket of underlyings:

    - Call: ``max(min_i(S_i(T) / S_i(0)) - K, 0) * notional``
    - Put:  ``max(K - min_i(S_i(T) / S_i(0)), 0) * notional``

    where returns are measured relative to initial spot levels.

    These are highly **correlation-sensitive**: lower correlation
    increases the probability that at least one asset performs poorly,
    making the option cheaper (for calls) or more expensive (for puts).

    Pricing requires correlated multi-asset Monte Carlo with a
    correlation matrix (Cholesky decomposition).

    Attributes:
        expiry: Time to expiry in year fractions.
        strike: Strike expressed as return level (e.g., 1.0 = at-the-money).
        notional: Notional amount.
        n_assets: Number of assets in the basket.
        is_call: True for call, False for put.
    """

    expiry: Float[Array, ""]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    n_assets: int = eqx.field(static=True, default=2)
    is_call: bool = eqx.field(static=True, default=False)


class Cliquet(eqx.Module):
    """Cliquet (ratchet) option — series of forward-starting options.

    The payoff is based on the **sum of capped and floored periodic
    returns** of the underlying:

    .. math::

        \\text{Payoff} = N \\cdot \\max\\left(\\sum_{i=1}^{n}
        \\min\\left(\\max\\left(\\frac{S(t_i)}{S(t_{i-1})} - 1,\\;
        f\\right),\\; c\\right),\\; g\\right)

    where :math:`c` is the local cap, :math:`f` is the local floor,
    and :math:`g` is the global floor (minimum total return).

    Cliquets are popular in structured notes and insurance products.
    They are sensitive to the **forward volatility smile** — local
    volatility or SLV models are required for accurate pricing.

    Attributes:
        observation_dates: Reset / observation dates (ordinals, shape n).
        local_cap: Maximum credited return per period (e.g., 0.05 for 5%).
        local_floor: Minimum credited return per period (e.g., 0.0 for floored at zero).
        global_floor: Minimum total accumulated return (e.g., 0.0).
        notional: Notional amount.
    """

    observation_dates: Int[Array, " n"]
    local_cap: Float[Array, ""]
    local_floor: Float[Array, ""]
    global_floor: Float[Array, ""]
    notional: Float[Array, ""]


class DigitalOption(eqx.Module):
    """Digital (binary) option — pays a fixed amount if in-the-money.

    A European digital option pays a **fixed cash amount** (or one
    unit of asset) if the underlying is above (call) or below (put)
    the strike at expiry:

    - Cash-or-nothing call: pays ``payout`` if ``S_T > K``
    - Cash-or-nothing put: pays ``payout`` if ``S_T < K``

    Digital options have **discontinuous payoffs**, which makes Greeks
    (especially gamma and vega) spiky near the strike.  In practice,
    they are often hedged as tight call/put spreads.

    Attributes:
        strike: Strike price.
        expiry: Time to expiry in year fractions.
        payout: Fixed payout amount if in-the-money.
        is_call: True for digital call, False for digital put.
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    payout: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)


class SpreadOption(eqx.Module):
    """Spread option — option on the difference between two assets.

    The payoff is:

    - Call: ``max(S1(T) - S2(T) - K, 0) * notional``
    - Put:  ``max(K - (S1(T) - S2(T)), 0) * notional``

    When ``K = 0``, this reduces to a **Margrabe** (exchange) option
    with a closed-form solution.  For ``K ≠ 0``, Kirk's approximation
    or 2D Monte Carlo is used.

    Spread options are common in commodities (crack spreads, spark
    spreads), equities (relative value), and rates (CMS spreads).

    Attributes:
        expiry: Time to expiry in year fractions.
        strike: Spread strike (often zero for exchange options).
        notional: Notional amount.
        is_call: True for call on the spread, False for put.
    """

    expiry: Float[Array, ""]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)
