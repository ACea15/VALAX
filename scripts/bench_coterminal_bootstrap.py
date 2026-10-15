"""Measure the co-terminal bootstrap (evidence for D14 and the register).

``swaption_basket_node(mode="coterminal_bootstrap")`` solves its square,
lower-triangular level bucket by bucket (one scalar Newton root per bucket,
seeded at the co-terminal's cube normal vol). This script measures, on
Hull-White with the closed-form (Jamshidian) basket for Bermudans with 4, 6
and 10 exercises, and on Cheyette with the frozen-y PDE basket:

* compile + first run, and steady-state run time (``eqx.filter_jit``);
* Newton iterations per bucket, and the pricings they imply — each step of
  bucket ``k`` prices one swaption with one tangent, ``2`` pricings;
* failure reporting on an infeasible co-terminal (bucket 1's target cut to a
  tenth): flagged buckets, the named root cause, and whether bucket 0 — which
  is healthy — stays exact.

**A joint solver was measured and removed** (one Newton solve of the whole
system, Jacobian tagged lower triangular so ``lineax`` substitutes; each step
prices the residual plus an ``n``-column forward Jacobian, ``(n + 1) n``
pricings). Same machine, same baskets, after guarding the nested root-finds:

====  ==================  ===========  ==========  ============
 n    engine              pricings     compile     run
====  ==================  ===========  ==========  ============
 4    HW closed form      120 vs 40    9.3 / 7.6s  0.3 / 0.2 ms
 6    HW closed form      252 vs 58    16.8/13.0s  0.3 / 0.3 ms
 10   HW closed form      770 vs 108   38.5/25.4s  1.4 / 0.9 ms
 3    Cheyette PDE        60 vs 28     5.6 / 6.1s  6.1 / 5.2 ms
 5    Cheyette PDE        150 vs 48    11.7/11.2s  12.0/ 9.6 ms
====  ==================  ===========  ==========  ============

(joint / sequential; roots agreed to ≤ 1.4e-13. The sequential solver then
started each bucket from the previous bucket's root; seeding every bucket at
its own co-terminal's cube vol, as now, costs 42 / 66 / 118 HW and 28 / 46
PDE pricings — within 10% either way.) On the infeasible
co-terminal the joint solver reported *every* bucket failed with the wrong
root cause (bucket 0, NaN everywhere) — before the guard it raised at
runtime — while the sequential solver names bucket 1 and keeps bucket 0 exact.

(Measured before the per-bucket cube seed existed: a fixed seed of σ = 0.3%
failed at every ``n`` — the first, in-the-money co-terminal sits on its
zero-vega intrinsic plateau there and Newton overshoots to σ ≈ 100%, outside
the Jamshidian pricer's domain; 1% and 3% converged.)

Run::

    python scripts/bench_coterminal_bootstrap.py
"""

from __future__ import annotations

import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from valax.calibration import (  # noqa: E402
    CalibrationChain,
    calibrate_market,
    coterminal_swaptions,
    positive,
    swaption_basket_node,
)
from valax.core.pricing import price  # noqa: E402
from valax.instruments.rates import BermudanSwaption  # noqa: E402
from valax.market.synthetic import (  # noqa: E402
    RatesVolSpec,
    RatesWorldSpec,
    SeedRegistry,
    SyntheticMarketConfig,
    rates_vol_snapshot,
)
from valax.models.hull_white import HullWhiteModel  # noqa: E402

A_KEY = "HW.MEAN_REVERSION"


def _market():
    snap, world = rates_vol_snapshot(
        SeedRegistry(3, "bench"), SyntheticMarketConfig(),
        RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10, 12, 15)),
        RatesVolSpec(expiry_years=(1, 2, 5, 10), tenor_years=(2, 5, 10), cap_years=(1,)),
        bp_noise=0.5, vol_bp_noise=0.5,
    )
    chain = CalibrationChain(nodes=snap.chain.nodes[:2])
    quotes = snap.quotes.select(chain.quote_keys)
    fixed = snap.fixed.with_values({A_KEY: jnp.asarray(0.05)})
    return world, calibrate_market(chain, quotes, fixed, strict=True).market, quotes, fixed


def _bermudan(world, n_ex):
    ref = int(world.rates.reference_date)
    return BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in range(1, n_ex + 1)], dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(2, n_ex + 3)], dtype=jnp.int32),
        strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
        day_count=world.rates.day_count,
        discount_id=world.discount_id, forward_id=world.forward_id,
    )


def _node(world, basket, seed=0.01):
    keys = tuple(f"T.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        "hw", template=HullWhiteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(seed)),
        basket=basket, cube_id=world.spec.cube_id, transforms={"volatility": positive()},
        fixed_fields={"mean_reversion": A_KEY}, target_keys=keys,
        mode="coterminal_bootstrap",
    )
    return node, keys


def _time(fit, args, reps):
    t0 = time.perf_counter()
    res = fit(*args)
    jax.block_until_ready(res.theta)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fit(*args).theta)
    return res, t_first, (time.perf_counter() - t0) / reps


def main() -> None:
    world, market, quotes, fixed = _market()
    print(f"{'n':>3} {'compile+run':>12} {'run':>9} {'iters':>22} {'pricings':>9} {'max|R|':>9}")
    for n_ex in (4, 6, 10):
        basket = coterminal_swaptions(_bermudan(world, n_ex))
        node, keys = _node(world, basket)
        p = fixed.with_values({k: jnp.asarray(0.0) for k in keys})
        fit = eqx.filter_jit(lambda q, m, pp, _node=node: _node.calibrate(q, m, pp))
        res, t_first, t_run = _time(fit, (quotes, market, p), 5)
        steps = np.asarray(res.diagnostics.n_steps)
        print(f"{len(basket):>3} {t_first:>11.2f}s {t_run * 1e3:>7.1f}ms "
              f"{'+'.join(str(int(s)) for s in steps):>22} {int(2 * steps.sum()):>9} "
              f"{float(np.max(np.abs(res.diagnostics.residuals))):>9.1e}  "
              f"converged={bool(res.converged)}")

    print("\nInfeasible co-terminal (bucket 1 target cut to a tenth), n = 6:")
    basket = coterminal_swaptions(_bermudan(world, 6))
    node, keys = _node(world, basket)
    p = fixed.with_values({k: jnp.asarray(0.0) for k in keys})
    good = node.calibrate(quotes, market, p)
    t1 = float(price(basket[1], good.value, market).price)
    res = node.calibrate(quotes, market, p.with_values({keys[1]: jnp.asarray(-0.9 * t1)}))
    r = np.asarray(res.diagnostics.residuals).ravel()
    print(f"  converged={bool(res.converged)} failed={list(res.failed_labels())} "
          f"root cause={list(res.root_cause_labels())} |R_0|={abs(r[0]):.1e} "
          f"θ_0 vs healthy fit Δ={abs(float(res.theta[0]) - float(good.theta[0])):.1e}")


def main_pde() -> None:
    """The expensive basket: Cheyette (β = 0.3) priced on the frozen-y PDE.

    Every pricing is a PDE solve, so the pricing count (``2`` per bucket
    Newton step) is the cost.
    """
    from valax.models.cheyette import CheyetteModel
    from valax.pricing.pde.config import PDEConfig

    cfg = PDEConfig(n_spot=101, n_time=60, spot_range=5.0)
    snap, world = rates_vol_snapshot(
        SeedRegistry(3, "bench"), SyntheticMarketConfig(),
        RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10, 12, 15), projection_tenors=()),
        RatesVolSpec(expiry_years=(1, 2, 5, 10), tenor_years=(2, 5, 10), cap_years=(1,)),
        bp_noise=0.5, vol_bp_noise=0.5,
    )
    chain = CalibrationChain(nodes=snap.chain.nodes[:2])
    quotes = snap.quotes.select(chain.quote_keys)
    base = snap.fixed.with_values({"CH.KAPPA": jnp.asarray(0.03), "CH.BETA": jnp.asarray(0.3)})
    market = calibrate_market(chain, quotes, base, strict=True).market
    ref = int(world.rates.reference_date)

    print("\nCheyette frozen-y PDE basket (101 x 60 grid):")
    print(f"{'n':>3} {'compile+run':>12} {'run':>9} {'iters':>14} {'PDE solves':>11}")
    for n_ex in (3, 5):
        trade = BermudanSwaption(
            exercise_dates=jnp.array([ref + 365 * k for k in range(1, n_ex + 1)], dtype=jnp.int32),
            fixed_dates=jnp.array([ref + 365 * k for k in range(2, n_ex + 3)], dtype=jnp.int32),
            strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
            day_count=world.rates.day_count, discount_id=world.discount_id, forward_id=None,
        )
        basket = coterminal_swaptions(trade)
        keys = tuple(f"T.{i}" for i in range(len(basket)))
        node = swaption_basket_node(
            "ch", template=CheyetteModel(mean_reversion=jnp.asarray(0.03),
                                         volatility=jnp.asarray(0.008), skew=jnp.asarray(0.3)),
            basket=basket, cube_id=world.spec.cube_id, transforms={"volatility": positive()},
            fixed_fields={"mean_reversion": "CH.KAPPA", "skew": "CH.BETA"}, target_keys=keys,
            mode="coterminal_bootstrap", method="pde", config=cfg,
        )
        p = base.with_values({k: jnp.asarray(0.0) for k in keys})
        fit = eqx.filter_jit(lambda q, m, pp, _node=node: _node.calibrate(q, m, pp))
        res, t_first, t_run = _time(fit, (quotes, market, p), 3)
        steps = np.asarray(res.diagnostics.n_steps)
        print(f"{len(basket):>3} {t_first:>11.2f}s {t_run * 1e3:>7.1f}ms "
              f"{'+'.join(str(int(s)) for s in steps):>14} {int(2 * steps.sum()):>11}"
              f"  converged={bool(res.converged)}")


if __name__ == "__main__":
    main()
    main_pde()
