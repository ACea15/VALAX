"""Ground-truth model parameter samplers.

These produce the ``θ_true`` that an end-to-end test then tries to
*recover indirectly* via observations + calibration.  Sampling lives
here rather than in ``valax/models/`` so the runtime model classes
stay free of any RNG dependency.

Each function returns a fully-formed ``eqx.Module`` from
``valax/models/`` with parameters drawn from documented ranges that
respect each model's domain constraints (e.g., Feller's condition for
Heston, ``|rho| < 1`` for SABR).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.models.hull_white import HullWhiteModel
from valax.models.multi_asset import MultiAssetGBMModel
from valax.models.sabr import SABRModel

from valax.market.synthetic.config import SyntheticMarketConfig
from valax.market.synthetic.correlations import sample_correlation_from_config
from valax.market.synthetic.seeds import SeedRegistry


def _u(key: Array, low: float, high: float) -> Float[Array, ""]:
    return jax.random.uniform(
        key, shape=(), dtype=jnp.float64, minval=low, maxval=high
    )


# ── Black-Scholes ─────────────────────────────────────────────────


def sample_bs_params(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
) -> BlackScholesModel:
    """Draw a :class:`BlackScholesModel` with parameters in ``cfg``'s ranges.

    Stream prefix: ``synthetic.model.bs.*``.
    """
    return BlackScholesModel(
        vol=_u(registry.key("synthetic.model.bs.vol"), *cfg.vol_range),
        rate=_u(registry.key("synthetic.model.bs.rate"), *cfg.rate_range),
        dividend=_u(
            registry.key("synthetic.model.bs.dividend"), *cfg.div_range
        ),
    )


# ── Heston ────────────────────────────────────────────────────────

# Ranges chosen so that the Feller condition 2*kappa*theta >= xi^2 is
# typically satisfied; rho biased negative to match the equity leverage
# effect.
_HESTON_RANGES: dict[str, tuple[float, float]] = {
    "v0": (0.01, 0.10),       # initial variance: sigma ~ 10%..32%
    "kappa": (0.5, 4.0),      # mean-reversion speed
    "theta": (0.02, 0.09),    # long-run variance
    "rho": (-0.85, -0.10),    # leverage correlation (equity)
}


def sample_heston_params(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    *,
    enforce_feller: bool = True,
) -> HestonModel:
    """Draw a :class:`HestonModel` with finite, in-domain parameters.

    Args:
        registry: Seed registry.
        cfg: Used only for ``rate`` and ``dividend`` ranges.
        enforce_feller: If True, ``xi`` is sampled from ``[xi_min,
            xi_feller]`` where ``xi_feller = sqrt(2*kappa*theta)`` so
            the Feller condition is satisfied with probability one.

    Stream prefix: ``synthetic.model.heston.*``.
    """
    v0 = _u(
        registry.key("synthetic.model.heston.v0"), *_HESTON_RANGES["v0"]
    )
    kappa = _u(
        registry.key("synthetic.model.heston.kappa"),
        *_HESTON_RANGES["kappa"],
    )
    theta = _u(
        registry.key("synthetic.model.heston.theta"),
        *_HESTON_RANGES["theta"],
    )
    rho = _u(
        registry.key("synthetic.model.heston.rho"), *_HESTON_RANGES["rho"]
    )
    xi_min = 0.05
    xi_max_default = 1.0
    if enforce_feller:
        # 2*kappa*theta >= xi^2  =>  xi <= sqrt(2*kappa*theta).
        feller_cap = jnp.sqrt(2.0 * kappa * theta)
        xi_high = jnp.minimum(feller_cap, xi_max_default)
        # Guard against degenerate ranges by collapsing to feller_cap.
        xi_high = jnp.maximum(xi_high, xi_min + 1e-3)
        u = jax.random.uniform(
            registry.key("synthetic.model.heston.xi"),
            shape=(),
            dtype=jnp.float64,
        )
        xi = xi_min + (xi_high - xi_min) * u
    else:
        xi = _u(
            registry.key("synthetic.model.heston.xi"),
            xi_min,
            xi_max_default,
        )

    rate = _u(registry.key("synthetic.model.heston.rate"), *cfg.rate_range)
    dividend = _u(
        registry.key("synthetic.model.heston.dividend"), *cfg.div_range
    )
    return HestonModel(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        rate=rate, dividend=dividend,
    )


# ── SABR ──────────────────────────────────────────────────────────


_SABR_RANGES: dict[str, tuple[float, float]] = {
    "alpha": (0.05, 0.40),
    "rho": (-0.70, 0.30),
    "nu": (0.10, 0.80),
}


def sample_sabr_params(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    *,
    fixed_beta: float | None = 0.5,
) -> SABRModel:
    """Draw a :class:`SABRModel`.

    Args:
        registry: Seed registry.
        cfg: Unused (kept for API symmetry).
        fixed_beta: If not None, ``beta`` is set to this constant
            (common choices: ``0.5`` for equities, ``0.0`` for
            normal-rates).  If None, ``beta`` is sampled uniformly
            from ``[0.0, 1.0]``.

    Stream prefix: ``synthetic.model.sabr.*``.
    """
    del cfg  # not needed; signature kept for consistency
    alpha = _u(
        registry.key("synthetic.model.sabr.alpha"),
        *_SABR_RANGES["alpha"],
    )
    rho = _u(
        registry.key("synthetic.model.sabr.rho"), *_SABR_RANGES["rho"]
    )
    nu = _u(
        registry.key("synthetic.model.sabr.nu"), *_SABR_RANGES["nu"]
    )
    if fixed_beta is None:
        beta = _u(registry.key("synthetic.model.sabr.beta"), 0.0, 1.0)
    else:
        beta = jnp.asarray(fixed_beta, dtype=jnp.float64)
    return SABRModel(alpha=alpha, beta=beta, rho=rho, nu=nu)


# ── Hull-White ────────────────────────────────────────────────────


_HW_RANGES: dict[str, tuple[float, float]] = {
    "mean_reversion": (0.01, 0.20),
    "volatility": (0.003, 0.020),
}


def sample_hull_white_params(
    registry: SeedRegistry,
    initial_curve: DiscountCurve,
) -> HullWhiteModel:
    """Draw a :class:`HullWhiteModel` calibrated to a supplied curve.

    The initial curve is *not* sampled here — pass one produced by
    :func:`~valax.market.synthetic.curves.sample_nss_curve` (or any
    other source) so the model exactly reproduces it.

    Stream prefix: ``synthetic.model.hw.*``.
    """
    a = _u(
        registry.key("synthetic.model.hw.mean_reversion"),
        *_HW_RANGES["mean_reversion"],
    )
    sigma = _u(
        registry.key("synthetic.model.hw.volatility"),
        *_HW_RANGES["volatility"],
    )
    return HullWhiteModel(
        mean_reversion=a, volatility=sigma, initial_curve=initial_curve
    )


# ── Multi-asset GBM ───────────────────────────────────────────────


def sample_multi_asset_gbm_params(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    *,
    block_sizes: tuple[int, ...] | None = None,
) -> MultiAssetGBMModel:
    """Draw a :class:`MultiAssetGBMModel`.

    The risk-free rate is scalar (uniform draw from ``cfg.rate_range``).
    Per-asset vols, dividends, and the correlation matrix follow the
    config's ranges and ``correlation_kind``.

    Stream prefix: ``synthetic.model.mgbm.*`` plus whatever the
    correlation sampler consumes.
    """
    n = cfg.n_assets
    vols = jax.random.uniform(
        registry.key("synthetic.model.mgbm.vols"),
        shape=(n,),
        dtype=jnp.float64,
        minval=cfg.vol_range[0],
        maxval=cfg.vol_range[1],
    )
    dividends = jax.random.uniform(
        registry.key("synthetic.model.mgbm.dividends"),
        shape=(n,),
        dtype=jnp.float64,
        minval=cfg.div_range[0],
        maxval=cfg.div_range[1],
    )
    rate = _u(
        registry.key("synthetic.model.mgbm.rate"), *cfg.rate_range
    )
    correlation = sample_correlation_from_config(registry, cfg, block_sizes)
    return MultiAssetGBMModel(
        vols=vols, rate=rate, dividends=dividends, correlation=correlation
    )


__all__ = [
    "sample_bs_params",
    "sample_heston_params",
    "sample_sabr_params",
    "sample_hull_white_params",
    "sample_multi_asset_gbm_params",
]
