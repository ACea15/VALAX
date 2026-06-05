"""Example 08 — End-to-end workflow on synthetic data.

Walks the six-stage workflow:

  1. Ground-truth world (synthetic config, snapshot, SABR truth).
  2. Observation layer (noisy SABR smile + noisy price strip).
  3. Calibration (recover a SABR fit to the noisy smile; smile-residual
     check only — NOT a parameter-recovery check).
  4. Portfolio construction (stacked European options).
  5. Pricing and Greeks via ``valax.portfolio.batch``.
  6. Arbitrage stress test — inject calendar arb in a total-variance
     vector and report what the (currently missing) detector would
     have flagged.

Run::

    python examples/08_end_to_end_workflow.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import valax
from valax.calibration import calibrate_sabr
from valax.greeks.autodiff import greeks as ad_greeks
from valax.instruments.options import EuropeanOption
from valax.market.synthetic import (
    OptionPortfolioSpec,
    SeedRegistry,
    SyntheticMarketConfig,
    inject_calendar_arb,
    sample_market_with_correlation,
    sample_option_portfolio,
    sample_sabr_params,
    synthesize_sabr_smile,
)
from valax.portfolio.batch import batch_price
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.sabr import sabr_implied_vol


def header(s: str) -> None:
    print(f"\n{'=' * 70}\n  {s}\n{'=' * 70}")


def main() -> None:
    registry = SeedRegistry(
        master_seed=20260101, library_version=valax.__version__
    )
    cfg = SyntheticMarketConfig(n_assets=3)

    # ── Stage 1: ground-truth world ──────────────────────────────────
    header("STAGE 1 — ground-truth world")
    md, corr = sample_market_with_correlation(registry, cfg)
    sabr_truth = sample_sabr_params(registry, cfg)
    print(f"spots      = {md.spots}")
    print(f"vols       = {md.vols}")
    print(f"sabr_truth = α={float(sabr_truth.alpha):.4f}  "
          f"β={float(sabr_truth.beta):.2f}  "
          f"ρ={float(sabr_truth.rho):+.4f}  "
          f"ν={float(sabr_truth.nu):.4f}")

    # ── Stage 2: observation layer ───────────────────────────────────
    header("STAGE 2 — noisy observations")
    F = jnp.array(100.0)
    T = jnp.array(1.0)
    strikes = jnp.linspace(70.0, 130.0, 13)
    smile_clean = synthesize_sabr_smile(
        registry, sabr_truth, F, T, strikes, vol_bp_noise=0.0,
    )
    smile_noisy = synthesize_sabr_smile(
        registry, sabr_truth, F, T, strikes, vol_bp_noise=15.0,
    )
    noise_rms_bp = float(
        jnp.sqrt(jnp.mean((smile_noisy - smile_clean) ** 2))
    ) * 1e4
    print(f"clean vols  (head): {smile_clean[:5]}")
    print(f"noisy vols  (head): {smile_noisy[:5]}")
    print(f"observation noise (RMS): {noise_rms_bp:.2f} bp")

    # ── Stage 3: calibration (smile-residual check, NOT param recovery)
    header("STAGE 3 — SABR calibration to noisy smile")
    sabr_fit, sol = calibrate_sabr(
        strikes=strikes,
        market_vols=smile_noisy,
        forward=F,
        expiry=T,
        fixed_beta=sabr_truth.beta,  # beta is conventionally fixed
    )
    smile_fitted = jax.vmap(
        lambda K: sabr_implied_vol(sabr_fit, F, K, T)
    )(strikes)
    fit_rms_bp = float(
        jnp.sqrt(jnp.mean((smile_fitted - smile_noisy) ** 2))
    ) * 1e4
    print(f"sabr_fit   = α={float(sabr_fit.alpha):.4f}  "
          f"β={float(sabr_fit.beta):.2f}  "
          f"ρ={float(sabr_fit.rho):+.4f}  "
          f"ν={float(sabr_fit.nu):.4f}")
    print(f"smile fit residual (RMS): {fit_rms_bp:.2f} bp")
    print(f"(compare with observation noise {noise_rms_bp:.2f} bp — "
          f"a good fit means residual ~ noise)")

    # ── Stage 4: portfolio ───────────────────────────────────────────
    header("STAGE 4 — option portfolio")
    port = sample_option_portfolio(
        registry, md, OptionPortfolioSpec(n_per_asset=4, call_probability=1.0),
    )
    calls, idx = port["calls"]
    print(f"{calls.strike.shape[0]} call legs across "
          f"{cfg.n_assets} underlyings")

    # ── Stage 5: pricing + Greeks ────────────────────────────────────
    header("STAGE 5 — batched pricing & Greeks")
    n = calls.strike.shape[0]
    spots = md.spots[idx]
    vols = md.vols[idx]
    rates = jnp.full((n,), 0.03)
    divs = md.dividends[idx]
    prices = batch_price(black_scholes_price, calls, spots, vols, rates, divs)
    print(f"total portfolio PV: {float(jnp.sum(prices)):.4f}")
    # Single-leg Greeks (autodiff) for the first call.
    g0 = ad_greeks(
        black_scholes_price,
        EuropeanOption(
            strike=calls.strike[0], expiry=calls.expiry[0], is_call=True,
        ),
        spots[0], vols[0], rates[0], divs[0],
    )
    print(f"leg-0 greeks: "
          f"Δ={float(g0['delta']):+.4f}  "
          f"Γ={float(g0['gamma']):+.4f}  "
          f"Vega={float(g0['vega']):+.4f}  "
          f"Θ={float(g0['theta']):+.4f}  "
          f"ρ={float(g0['rho']):+.4f}")

    # ── Stage 6: arbitrage stress test ───────────────────────────────
    header("STAGE 6 — arbitrage stress test (calendar arb)")
    # Build a small synthetic total-variance vector and inject calendar arb.
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0])
    sigma = 0.20
    w_clean = sigma**2 * expiries
    w_bad, diag = inject_calendar_arb(w_clean, i=1, j=3)
    print(f"clean total variance: {w_clean}")
    print(f"after injection     : {w_bad}")
    print(f"diagnosis: {diag}")
    print("Library detector status:")
    print("  • No surface-level calendar-arb checker exists yet.")
    print("  • See tests/test_market/test_arbitrage_handling.py "
          "for the xfail backlog.")


if (__name__ == "__main__"):
    main()
