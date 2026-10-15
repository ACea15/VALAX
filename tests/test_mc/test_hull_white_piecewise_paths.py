"""Hull-White Monte Carlo with piecewise-constant σ(t).

Each step samples the exact conditional variance
:math:`V(t_i, t_{i+1}) = \\int \\sigma(u)^2 e^{-2a(t_{i+1}-u)} du`, so the
marginal law of :math:`r(t)` is exact on *any* grid — including steps that
straddle bucket breakpoints. Checked by the moments of :math:`r(T)` on a
deliberately coarse, misaligned grid, by the zero-coupon bond, and by a
European swaption priced from the paths against the σ(t) Jamshidian closed
form, all within two standard errors.
"""

import jax
import jax.numpy as jnp
import numpy as np

from valax.core.binding import bind_curves
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.instruments.rates import Swaption
from valax.models.hull_white import (
    HullWhiteModel,
    hw_alpha,
    hw_bond_price,
    hw_market_df,
    hw_short_rate_variance,
)
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.mc.hull_white_paths import generate_hull_white_paths

REF = int(ymd_to_ordinal(2026, 1, 1))
RATE = 0.035
N_PATHS = 40_000


def _curve() -> DiscountCurve:
    years = [0.0] + [float(k) for k in range(1, 20)]
    pillars = jnp.array([REF + int(round(y * 365)) for y in years], dtype=jnp.int32)
    t = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-RATE * t),
        reference_date=jnp.int32(REF),
        day_count="act_365",
    )


MODEL = bind_curves(
    HullWhiteModel(
        mean_reversion=jnp.asarray(0.05),
        volatility=jnp.asarray([0.012, 0.006, 0.018, 0.009]),
        volatility_times=jnp.asarray([0.7, 1.3, 1.8]),
    ),
    _curve(),
)


def test_short_rate_moments_exact_on_misaligned_grid():
    """7 steps over 2Y: every step straddles or misses a breakpoint."""
    T = 2.0
    paths = generate_hull_white_paths(MODEL, T, 7, N_PATHS, jax.random.PRNGKey(11))
    r_T = np.asarray(paths.short_rates[:, -1])
    var = float(hw_short_rate_variance(MODEL, jnp.asarray(T)))
    mean_se = np.sqrt(var / N_PATHS)
    assert abs(r_T.mean() - float(hw_alpha(MODEL, jnp.asarray(T)))) < 2 * mean_se
    var_se = var * np.sqrt(2.0 / (N_PATHS - 1))
    assert abs(r_T.var(ddof=1) - var) < 2 * var_se


def test_zero_coupon_bond():
    """Pooled over 10 keys (400k paths).

    A single 40k-path key landed 2.25 stderr out; a 10-key study showed
    z-scores distributed as N(0, 1) for both σ(t) and constant σ (mean +0.15
    and +0.19), i.e. no bias, so the check pools the keys instead of
    picking a seed.
    """
    T = 3.0
    dfs = np.concatenate([
        np.exp(np.asarray(generate_hull_white_paths(
            MODEL, T, 150, N_PATHS, jax.random.PRNGKey(100 + s)
        ).log_discount_factors[:, -1]))
        for s in range(10)
    ])
    se = dfs.std(ddof=1) / np.sqrt(dfs.size)
    assert abs(dfs.mean() - float(hw_market_df(MODEL, jnp.asarray(T)))) < 2 * se


def test_european_swaption_from_paths():
    """Payoff at expiry from the affine bond, discounted along the path."""
    expiry_days, n = 730, 4
    sw = Swaption(
        expiry_date=jnp.int32(REF + expiry_days),
        fixed_dates=jnp.array([REF + expiry_days + 365 * k for k in range(1, n + 1)], dtype=jnp.int32),
        strike=jnp.asarray(RATE),
        notional=jnp.asarray(1.0),
        is_payer=True,
        day_count="act_365",
    )
    T = float(year_fraction(jnp.int32(REF), sw.expiry_date, "act_365"))
    pay_t = year_fraction(jnp.int32(REF), sw.fixed_dates, "act_365")
    tau = jnp.diff(jnp.concatenate([jnp.asarray([T]), pay_t]))
    cash = (sw.strike * tau).at[-1].add(1.0)

    paths = generate_hull_white_paths(MODEL, T, 200, N_PATHS, jax.random.PRNGKey(13))
    r_T = paths.short_rates[:, -1]
    bonds = jax.vmap(
        lambda r: jnp.sum(cash * hw_bond_price(MODEL, r, jnp.asarray(T), pay_t))
    )(r_T)
    payoff = np.asarray(jnp.maximum(1.0 - bonds, 0.0) * jnp.exp(paths.log_discount_factors[:, -1]))
    se = payoff.std(ddof=1) / np.sqrt(N_PATHS)
    exact = float(hw_swaption_price(sw, MODEL))
    assert abs(payoff.mean() - exact) < 2 * se
