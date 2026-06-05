"""Synthetic discount curve generators.

Two flavours:

- :func:`flat_discount_curve` and :func:`sample_flat_curve` build a
  two-pillar curve at a single rate.  Useful for replicating the
  flat-rate world of ``examples/comparisons/01_european_options.py``.
- :func:`sample_nss_curve` draws a Nelson-Siegel-Svensson parameter
  vector and evaluates it on the maturity grid declared in the config.

NSS form
--------
::

    r(tau) = beta0
           + beta1 * (1 - exp(-tau/tau1)) / (tau/tau1)
           + beta2 * ((1 - exp(-tau/tau1)) / (tau/tau1) - exp(-tau/tau1))
           + beta3 * ((1 - exp(-tau/tau2)) / (tau/tau2) - exp(-tau/tau2))

The resulting zero rates ``r(tau)`` are converted to continuously-
compounded discount factors ``exp(-r(tau) * tau)``.  Discount factors
are clipped to ``(0, 1]`` so a wildly negative draw cannot produce a
DF above 1, which the rest of the library treats as a contract
violation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve

from valax.market.synthetic.config import SyntheticMarketConfig
from valax.market.synthetic.seeds import SeedRegistry


# ── Flat curve ─────────────────────────────────────────────────────


def flat_discount_curve(
    rate: float | Float[Array, ""],
    reference_date: Int[Array, ""],
    horizon_years: float = 50.0,
    day_count: str = "act_365",
) -> DiscountCurve:
    """Build a two-pillar flat-rate discount curve.

    Deterministic helper — no RNG.  Encodes the same flat-rate market
    used by ``examples/comparisons/01_european_options.py`` in a form
    the rest of the library (which always expects a ``DiscountCurve``)
    can consume.

    Args:
        rate: Continuously-compounded zero rate.
        reference_date: Valuation date ordinal.
        horizon_years: Far-pillar maturity in years.  Anything past the
            far pillar is flat-extrapolated, so this only needs to
            exceed the longest instrument expiry you'll evaluate.
        day_count: Day count convention; must match what the rest of
            the curve infra expects.

    Returns:
        ``DiscountCurve`` with two pillars at ``reference_date`` and
        ``reference_date + horizon_years * 365``.
    """
    rate_arr = jnp.asarray(rate, dtype=jnp.float64)
    ref = jnp.asarray(reference_date, dtype=jnp.int32)
    far_ordinal = ref + jnp.int32(int(horizon_years * 365.0))

    pillars = jnp.stack([ref, far_ordinal])
    times = jnp.array([0.0, float(horizon_years)], dtype=jnp.float64)
    dfs = jnp.exp(-rate_arr * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref,
        day_count=day_count,
    )


def sample_flat_curve(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
) -> DiscountCurve:
    """Draw a single flat short-rate from ``cfg.rate_range`` and build
    the corresponding two-pillar discount curve.

    Stream name: ``"synthetic.curve.flat.rate"``.
    """
    key = registry.key("synthetic.curve.flat.rate")
    low, high = cfg.rate_range
    rate = jax.random.uniform(
        key, shape=(), dtype=jnp.float64, minval=low, maxval=high
    )
    return flat_discount_curve(
        rate=rate,
        reference_date=cfg.reference_date,
        day_count=cfg.day_count,
    )


# ── Nelson-Siegel-Svensson curve ───────────────────────────────────


def _nss_zero_rate(
    tau: Float[Array, " n"],
    beta0: Float[Array, ""],
    beta1: Float[Array, ""],
    beta2: Float[Array, ""],
    beta3: Float[Array, ""],
    tau1: Float[Array, ""],
    tau2: Float[Array, ""],
) -> Float[Array, " n"]:
    """Vectorised Nelson-Siegel-Svensson zero rate at maturities ``tau``."""
    # Guard against tau == 0 by replacing with a tiny floor; we never
    # query the curve at tau == 0 because the reference pillar is
    # constructed separately.
    tau_safe = jnp.maximum(tau, 1e-8)
    x1 = tau_safe / tau1
    x2 = tau_safe / tau2
    f1 = (1.0 - jnp.exp(-x1)) / x1
    f2 = f1 - jnp.exp(-x1)
    f3 = (1.0 - jnp.exp(-x2)) / x2 - jnp.exp(-x2)
    return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3


def _sample_nss_params(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
) -> tuple[Float[Array, ""], ...]:
    """Draw the six NSS parameters in their configured ranges.

    Stream name: ``"synthetic.curve.nss.params"``.
    """
    key = registry.key("synthetic.curve.nss.params")
    keys = jax.random.split(key, 6)
    out = []
    for k, (low, high) in zip(keys, cfg.nss_param_ranges, strict=True):
        out.append(
            jax.random.uniform(
                k, shape=(), dtype=jnp.float64, minval=low, maxval=high
            )
        )
    return tuple(out)


def sample_nss_curve(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
) -> DiscountCurve:
    """Draw an NSS curve and return it as a :class:`DiscountCurve`.

    The pillar grid is ``cfg.nss_pillars_years`` (in years from the
    reference date), plus the reference date itself as the zero pillar
    (DF == 1.0 by construction).

    The discount factors are clipped to ``(eps, 1.0]`` so a numerically
    extreme parameter draw cannot break the contract that DFs lie in
    ``(0, 1]``.  This is a regularisation, not a hard error — callers
    that need to detect such draws should inspect ``cfg.nss_param_ranges``.

    Stream name: ``"synthetic.curve.nss.params"``.
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = _sample_nss_params(registry, cfg)

    years = jnp.asarray(cfg.nss_pillars_years, dtype=jnp.float64)
    rates = _nss_zero_rate(years, beta0, beta1, beta2, beta3, tau1, tau2)

    # Build pillar ordinals: reference_date + round(years * 365).
    ref = jnp.asarray(cfg.reference_date, dtype=jnp.int32)
    offsets = jnp.round(years * 365.0).astype(jnp.int32)
    body_pillars = ref + offsets
    pillars = jnp.concatenate([ref[None], body_pillars])

    # DFs: DF(ref) = 1; DF(t) = exp(-r(t) * tau).
    body_dfs = jnp.exp(-rates * years)
    dfs = jnp.concatenate([jnp.ones((1,), dtype=jnp.float64), body_dfs])
    dfs = jnp.clip(dfs, 1e-12, 1.0)

    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref,
        day_count=cfg.day_count,
    )


def sample_discount_curve(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
) -> DiscountCurve:
    """Dispatch to :func:`sample_flat_curve` or :func:`sample_nss_curve`
    based on ``cfg.curve_kind``."""
    if cfg.curve_kind == "flat":
        return sample_flat_curve(registry, cfg)
    if cfg.curve_kind == "nss":
        return sample_nss_curve(registry, cfg)
    raise ValueError(f"Unknown curve_kind: {cfg.curve_kind!r}")


__all__ = [
    "flat_discount_curve",
    "sample_flat_curve",
    "sample_nss_curve",
    "sample_discount_curve",
]
