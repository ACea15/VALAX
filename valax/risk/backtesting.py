"""VaR backtesting and FRTB P&L Attribution test statistics.

Two regulatory validations sit on top of a P&L-vector engine:

1. **VaR backtest** (Basel / FRTB SBA): compare daily VaR forecasts to
   realised one-day P&L across a rolling 250-day window.  Score the
   breach count (Basel traffic light), and apply the Kupiec
   proportion-of-failures test (unconditional coverage) and the
   Christoffersen independence / conditional-coverage tests.

2. **FRTB PLA test** (BCBS d558 §MAR32): compare each desk's
   risk-theoretical P&L (RTPL) and hypothetical P&L (HPL) over the same
   250-day window via Spearman rank correlation and a two-sample
   Kolmogorov–Smirnov statistic, then read the green/amber/red zone.

All functions take plain ``jax.numpy`` arrays.  Where the statistic has a
known asymptotic distribution under the null we return both the test
statistic and the right-tail p-value.  Zone classifications return
Python strings (``"green"`` / ``"yellow"`` / ``"amber"`` / ``"red"``)
and are therefore *not* JIT-friendly — they are intended for
post-processing on small scalar inputs, not inside hot loops.

See :doc:`docs/theory.md` §7.6 and §7.7 for the underlying statistics
and threshold derivations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jaxtyping import Bool, Float, Int
from jax import Array


# ── Breach detection ─────────────────────────────────────────────────


def var_breaches(
    actual_pnl: Float[Array, " n_days"],
    var_forecast: Float[Array, " n_days"],
) -> Bool[Array, " n_days"]:
    """Boolean breach sequence: ``loss > VaR`` on each day.

    Convention: ``actual_pnl`` is signed P&L (negative ⇒ loss); ``var_forecast``
    is a non-negative loss threshold (the output of
    :func:`valax.risk.var.value_at_risk`).  Day ``t`` is a breach iff
    ``-actual_pnl[t] > var_forecast[t]``.

    Args:
        actual_pnl: Signed daily P&L (negative for losses).
        var_forecast: Daily VaR forecasts (positive numbers).

    Returns:
        Boolean array of length ``n_days``.
    """
    return -actual_pnl > var_forecast


# ── Kupiec proportion-of-failures (unconditional coverage) ───────────


def kupiec_pof(
    breaches: Bool[Array, " n_days"],
    confidence: float = 0.99,
) -> dict[str, Float[Array, ""]]:
    """Kupiec POF likelihood-ratio test for unconditional coverage.

    Under H₀ the breach indicator is i.i.d. Bernoulli with probability
    ``p = 1 - confidence``.  The likelihood ratio

    .. math::

        \\mathrm{LR}_{uc} = -2 \\ln \\frac{(1-p)^{n-x} p^{x}}{(1-\\hat p)^{n-x} \\hat p^{x}}

    where ``x`` is the breach count and ``\\hat p = x/n``, is asymptotically
    ``χ²₁`` under H₀.

    Args:
        breaches: Boolean breach sequence.
        confidence: VaR confidence level (e.g. 0.99 ⇒ expected p = 0.01).

    Returns:
        Dict with keys ``"n"``, ``"x"``, ``"lr_uc"``, ``"p_value"``.
    """
    n = breaches.shape[0]
    x = jnp.sum(breaches.astype(jnp.float64))
    n_f = jnp.float64(n)
    p = 1.0 - confidence
    p_hat = x / n_f

    # Use jnp.where to avoid log(0) when boundary p_hat hits 0 or 1.
    eps = 1e-12
    p_hat_safe = jnp.clip(p_hat, eps, 1.0 - eps)

    log_l_null = (n_f - x) * jnp.log(1.0 - p) + x * jnp.log(p)
    log_l_alt = (n_f - x) * jnp.log(1.0 - p_hat_safe) + x * jnp.log(p_hat_safe)

    # When p_hat is exactly 0 (no breaches), the unrestricted MLE
    # concentrates at p̂ = 0 which sends the alt log-likelihood to 0
    # for the x·log(p̂) term (0·log 0 = 0) and gives finite (n-x)·log(1).
    log_l_alt = jnp.where(x == 0.0, 0.0, log_l_alt)
    log_l_alt = jnp.where(x == n_f, 0.0, log_l_alt)

    lr_uc = -2.0 * (log_l_null - log_l_alt)
    p_value = jstats.chi2.sf(lr_uc, df=1)

    return {
        "n": jnp.float64(n_f),
        "x": x,
        "lr_uc": lr_uc,
        "p_value": p_value,
    }


# ── Christoffersen independence test ────────────────────────────────


def christoffersen_independence(
    breaches: Bool[Array, " n_days"],
) -> dict[str, Float[Array, ""]]:
    """Christoffersen LR test for independence of breaches.

    Treats breaches as a first-order Markov chain and tests
    ``π_{01} = π_{11}`` (no memory) against the unrestricted alternative,
    using the LR statistic

    .. math::

        \\mathrm{LR}_{ind} = -2 \\ln
        \\frac{(1-\\hat\\pi)^{n_{00}+n_{10}} \\hat\\pi^{n_{01}+n_{11}}}
              {(1-\\hat\\pi_{01})^{n_{00}} \\hat\\pi_{01}^{n_{01}}
               (1-\\hat\\pi_{11})^{n_{10}} \\hat\\pi_{11}^{n_{11}}}

    asymptotically ``χ²₁`` under H₀.  When the sample has no breaches
    (or no non-breaches), the LR statistic is defined to be zero.

    Returns:
        Dict with keys ``"lr_ind"``, ``"p_value"``, plus the transition
        counts ``"n00"``, ``"n01"``, ``"n10"``, ``"n11"``.
    """
    b = breaches.astype(jnp.float64)
    prev = b[:-1]
    curr = b[1:]

    n00 = jnp.sum((1.0 - prev) * (1.0 - curr))
    n01 = jnp.sum((1.0 - prev) * curr)
    n10 = jnp.sum(prev * (1.0 - curr))
    n11 = jnp.sum(prev * curr)

    eps = 1e-12
    pi01 = n01 / jnp.maximum(n00 + n01, eps)
    pi11 = n11 / jnp.maximum(n10 + n11, eps)
    pi = (n01 + n11) / jnp.maximum(n00 + n01 + n10 + n11, eps)

    def safe_log(x):
        return jnp.log(jnp.clip(x, eps, 1.0))

    log_l_null = (
        (n00 + n10) * safe_log(1.0 - pi) + (n01 + n11) * safe_log(pi)
    )
    log_l_alt = (
        n00 * safe_log(1.0 - pi01)
        + n01 * safe_log(pi01)
        + n10 * safe_log(1.0 - pi11)
        + n11 * safe_log(pi11)
    )

    lr_ind = -2.0 * (log_l_null - log_l_alt)
    # Guard against tiny negative round-off (LR is nonnegative under H0).
    lr_ind = jnp.maximum(lr_ind, 0.0)

    # If there are no breaches at all (or no non-breaches), there is no
    # transition information; report zero.
    total_breaches = n01 + n11
    total_non = n00 + n10
    no_info = (total_breaches == 0.0) | (total_non == 0.0)
    lr_ind = jnp.where(no_info, 0.0, lr_ind)

    p_value = jstats.chi2.sf(lr_ind, df=1)
    return {
        "lr_ind": lr_ind,
        "p_value": p_value,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


# ── Christoffersen conditional coverage ─────────────────────────────


def christoffersen_conditional_coverage(
    breaches: Bool[Array, " n_days"],
    confidence: float = 0.99,
) -> dict[str, Float[Array, ""]]:
    """Christoffersen joint test of correct coverage AND independence.

    ``LR_cc = LR_uc + LR_ind``, asymptotically ``χ²₂`` under H₀.
    """
    uc = kupiec_pof(breaches, confidence=confidence)
    ind = christoffersen_independence(breaches)
    lr_cc = uc["lr_uc"] + ind["lr_ind"]
    p_value = jstats.chi2.sf(lr_cc, df=2)
    return {
        "lr_cc": lr_cc,
        "p_value": p_value,
        "lr_uc": uc["lr_uc"],
        "lr_ind": ind["lr_ind"],
    }


# ── Basel traffic light ─────────────────────────────────────────────


def basel_traffic_light(
    n_breaches: int,
    n_obs: int = 250,
    confidence: float = 0.99,
) -> str:
    """Basel/FRTB SBA traffic-light zone for a backtesting window.

    The default thresholds (4 and 9) are the regulatory zones for a
    250-day window at 99% VaR; these correspond to the 95% and 99.99%
    quantiles of ``Binomial(250, 0.01)``.

    For non-standard ``(n_obs, confidence)`` combinations the thresholds
    are recomputed dynamically from the cumulative binomial
    distribution: green up to the 95% quantile, red beyond the 99.99%
    quantile, yellow in between.

    Args:
        n_breaches: Observed number of VaR breaches.
        n_obs: Length of the backtesting window (default 250).
        confidence: VaR confidence level (default 0.99).

    Returns:
        ``"green"``, ``"yellow"`` or ``"red"``.
    """
    p = 1.0 - confidence
    # Standard regulatory thresholds for the canonical (250, 99%) case.
    if n_obs == 250 and abs(confidence - 0.99) < 1e-9:
        green_max, yellow_max = 4, 9
    else:
        # Recompute thresholds via the binomial survival function.
        # We use a python-level loop here because (n_obs, confidence)
        # are static inputs and the call is cheap.
        from math import comb
        cum = 0.0
        green_max = 0
        yellow_max = 0
        for x in range(n_obs + 1):
            cum += comb(n_obs, x) * (p ** x) * ((1.0 - p) ** (n_obs - x))
            if cum < 0.95:
                green_max = x
            if cum < 0.9999:
                yellow_max = x

    if n_breaches <= green_max:
        return "green"
    if n_breaches <= yellow_max:
        return "yellow"
    return "red"


# ── Helpers: Spearman correlation, two-sample KS ────────────────────


def _ranks(x: Float[Array, " n"]) -> Float[Array, " n"]:
    """Average ranks of ``x`` (1-based, with mean over ties).

    Implemented via two argsorts; the second argsort gives the
    inverse-permutation which is the rank vector.  Ties are not broken
    using midranks (we use simple ordinal ranks here, which is
    sufficient for FRTB-style Spearman where ties in continuous P&L
    series are vanishingly rare).
    """
    order = jnp.argsort(x)
    ranks = jnp.argsort(order).astype(jnp.float64) + 1.0
    return ranks


def pla_spearman(
    rtpl: Float[Array, " n"],
    hpl: Float[Array, " n"],
) -> Float[Array, ""]:
    """Spearman rank correlation between two P&L series.

    Equal to the Pearson correlation of the rank-transformed series.
    Returns a scalar in ``[-1, 1]``.
    """
    rx = _ranks(rtpl)
    ry = _ranks(hpl)
    rx = rx - jnp.mean(rx)
    ry = ry - jnp.mean(ry)
    num = jnp.sum(rx * ry)
    den = jnp.sqrt(jnp.sum(rx ** 2) * jnp.sum(ry ** 2))
    return num / jnp.maximum(den, 1e-12)


def ks_statistic(
    x: Float[Array, " n"],
    y: Float[Array, " m"],
) -> Float[Array, ""]:
    """Two-sample Kolmogorov–Smirnov statistic.

    ``D = sup_t |F_x(t) - F_y(t)|`` where ``F_x``, ``F_y`` are the
    right-continuous empirical CDFs of the two samples.  Evaluates the
    CDFs at every value in the joint sample via ``jnp.searchsorted``
    (sort + binary search), which is ``O((n+m) log(n+m))`` and handles
    ties between the two samples correctly.
    """
    n = x.shape[0]
    m = y.shape[0]
    x_sorted = jnp.sort(x)
    y_sorted = jnp.sort(y)
    joint = jnp.concatenate([x_sorted, y_sorted])
    fx = jnp.searchsorted(x_sorted, joint, side="right") / jnp.float64(n)
    fy = jnp.searchsorted(y_sorted, joint, side="right") / jnp.float64(m)
    return jnp.max(jnp.abs(fx - fy))


def pla_ks(
    rtpl: Float[Array, " n"],
    hpl: Float[Array, " n"],
) -> Float[Array, ""]:
    """Kolmogorov–Smirnov distance between RTPL and HPL empirical CDFs.

    Thin wrapper around :func:`ks_statistic` exposed under the FRTB name.
    """
    return ks_statistic(rtpl, hpl)


def _ks_pvalue(D: Float[Array, ""], n: int, m: int) -> Float[Array, ""]:
    """Asymptotic two-sample KS right-tail p-value.

    Uses the Kolmogorov distribution series

    .. math::

        \\Pr(\\sqrt{n_e} D > \\lambda) = 2 \\sum_{k=1}^\\infty
            (-1)^{k-1} e^{-2 k^2 \\lambda^2}

    with ``n_e = n m / (n + m)``.  Truncated at ``k = 100`` (more than
    enough for any practical ``λ``).
    """
    ne = (n * m) / (n + m)
    lam = (jnp.sqrt(ne) + 0.12 + 0.11 / jnp.sqrt(ne)) * D
    k = jnp.arange(1, 101, dtype=jnp.float64)
    terms = 2.0 * ((-1.0) ** (k - 1.0)) * jnp.exp(-2.0 * k ** 2 * lam ** 2)
    p = jnp.sum(terms)
    # The alternating series Q(λ) = 2Σ(-1)^(j-1) exp(-2 j² λ²) tends to 1 as
    # λ → 0, but the truncated 100-term sum oscillates around 0 for tiny λ.
    # Clamp to 1.0 for λ below the convergence radius of the series.
    p = jnp.where(lam < 0.04, 1.0, p)
    return jnp.clip(p, 0.0, 1.0)


# ── PLA traffic light (BCBS d558) ───────────────────────────────────


def pla_traffic_light(
    spearman: float,
    ks_stat: float,
    n_obs: int = 250,
    spearman_green: float = 0.80,
    spearman_amber: float = 0.70,
    ks_green_p: float = 0.264,
    ks_amber_p: float = 0.055,
) -> str:
    """FRTB PLA traffic-light zone using BCBS d558 thresholds.

    The PLA zone is the *worse* of the Spearman zone and the KS zone:

    | Test           | Green             | Amber             | Red               |
    |----------------|-------------------|-------------------|-------------------|
    | Spearman ρ     | ≥ 0.80            | ≥ 0.70            | < 0.70            |
    | KS p-value     | ≥ 0.264           | ≥ 0.055           | < 0.055           |

    The KS test is converted to a p-value via the asymptotic
    Kolmogorov distribution with the same-sample-size correction
    ``n_e = n²/(2n) = n/2``.

    Args:
        spearman: Spearman rank correlation between RTPL and HPL.
        ks_stat: Two-sample KS distance between RTPL and HPL.
        n_obs: Sample size (per series) used to compute the KS p-value.
        spearman_green: Spearman threshold for the green zone.
        spearman_amber: Spearman threshold for the amber zone (below
            this the zone is red).
        ks_green_p: KS p-value threshold for the green zone.
        ks_amber_p: KS p-value threshold for the amber zone (below
            this the zone is red).

    Returns:
        ``"green"``, ``"amber"`` or ``"red"``.
    """
    if spearman >= spearman_green:
        spearman_zone = "green"
    elif spearman >= spearman_amber:
        spearman_zone = "amber"
    else:
        spearman_zone = "red"

    p_val = float(_ks_pvalue(jnp.asarray(ks_stat), n_obs, n_obs))
    if p_val >= ks_green_p:
        ks_zone = "green"
    elif p_val >= ks_amber_p:
        ks_zone = "amber"
    else:
        ks_zone = "red"

    order = {"green": 0, "amber": 1, "red": 2}
    return spearman_zone if order[spearman_zone] >= order[ks_zone] else ks_zone
