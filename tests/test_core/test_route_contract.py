"""The ``price()`` route contract: every registered route, one set of checks.

For each registered library ``(instrument, model, method)`` the sample trade
of :mod:`route_samples` must:

1. price to a finite number (with a cheap version of the method's default
   config);
2. price with the **instrument as a traced** ``eqx.filter_jit`` argument —
   which entries it reads is decided by static selectors, never by traced
   data;
3. have a finite market gradient that is **exactly zero outside**
   :func:`~valax.core.resolve.declared_entries` (no hidden market
   dependency) and non-zero on the entries the sample names (anti-vacuity);
4. batch under ``eqx.filter_vmap`` over a stack of two trades.

A second test prices every route with ``config=None`` (the pair's default
config; ≤ 7 s per route measured). A registered pair without a sample fails
``test_every_registered_pair_has_a_sample``; adding a route therefore means
adding its sample (``docs/guide/tutorial-unified-pricing.md`` §10).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_core.route_samples import (
    DECOYS,
    KNOWN_UNPRICEABLE,
    MARKET,
    SAMPLES,
    cheap,
)
from valax.core import pricing as P
from valax.core.pricing import default_config, price
from valax.core.resolve import declared_entries

jax.config.update("jax_enable_x64", True)

KEY = jax.random.PRNGKey(0)


def _library(cls: type) -> bool:
    return cls is type(None) or cls.__module__.startswith("valax.")


def _routes() -> list[tuple[type, type, str]]:
    P._ensure_backends()
    pairs = {(i, m) for (i, m, _) in P._REGISTRY}
    for method in ("pde", "mc"):
        pairs |= set(P._legacy_registry(method))
    out = []
    for (i, m) in pairs:
        if not (_library(i) and _library(m)) or (i, m) in KNOWN_UNPRICEABLE:
            continue
        for method in P.available_methods(i, m):
            out.append((i, m, method))
    return sorted(out, key=lambda r: (r[0].__name__, r[1].__name__, r[2]))


ROUTES = _routes()
IDS = [f"{i.__name__}-{m.__name__}-{meth}" for i, m, meth in ROUTES]

#: Routes that genuinely break a contract clause, each with its reason.
#: Empty means every route honours the whole contract.
KNOWN_DEVIATIONS: dict[tuple[str, str, str, str], str] = {}


def _deviation(route, clause):
    i, m, meth = route
    return KNOWN_DEVIATIONS.get((i.__name__, m.__name__, meth, clause))


def test_every_registered_pair_has_a_sample():
    missing = sorted(
        f"{i.__name__} x {m.__name__}"
        for (i, m) in {(i, m) for (i, m, _) in ROUTES}
        if (i, m) not in SAMPLES
    )
    assert not missing, f"registered routes without a route sample: {missing}"


def _config(route):
    i, m, meth = route
    return cheap(default_config(i, m, meth))


def _price(instrument, model, market, route, config):
    _, _, meth = route
    return price(instrument, model, market, method=meth, config=config,
                 key=KEY if meth == "mc" else None).price


def _slot_leaves(grad, slot: str) -> dict:
    """``id -> list of gradient leaves`` of one market slot."""
    table = grad.curves.curves if slot == "curves" else getattr(grad, slot)
    return {k: [x for x in jax.tree.leaves(v) if eqx.is_inexact_array(x)]
            for k, v in table.items()}


@pytest.mark.parametrize("route", ROUTES, ids=IDS)
def test_route_contract(route):
    sample = SAMPLES[(route[0], route[1])]()
    inst, model = sample.instrument, sample.model
    cfg = _config(route)

    # 1. Prices.
    value = _price(inst, model, MARKET, route, cfg)
    assert np.isfinite(float(value)), "price is not finite"

    # 2. Traced instrument under jit.
    if not _deviation(route, "traced"):
        traced = eqx.filter_jit(
            lambda i, m: _price(i, model, m, route, cfg))(inst, MARKET)
        assert np.isfinite(float(traced)), "traced-instrument price is not finite"

    # 3. Gradient: zero outside the declared entries, non-zero where named.
    grad = eqx.filter_grad(lambda m: _price(inst, model, m, route, cfg))(MARKET)
    declared = declared_entries(inst, model, MARKET)._asdict()
    for slot in ("curves", "spots", "dividends", "fx_spots", "inflation_curves",
                 "surfaces"):
        for key, leaves in _slot_leaves(grad, slot).items():
            for leaf in leaves:
                assert np.all(np.isfinite(np.asarray(leaf))), (slot, key, "non-finite")
                if key not in declared[slot]:
                    assert np.all(np.asarray(leaf) == 0.0), (
                        f"gradient on undeclared {slot}[{key!r}]")
    for slot, key in sample.nonzero:
        assert key in declared[slot], f"sample names undeclared {slot}[{key!r}]"
        assert any(np.any(np.asarray(x) != 0.0)
                   for x in _slot_leaves(grad, slot)[key]), (
            f"price does not move with {slot}[{key!r}]")
    for slot, key in DECOYS.items():
        assert key not in declared[slot]

    # 4. A batch of two trades.
    if not _deviation(route, "vmap"):
        batch = jax.tree.map(
            lambda x: jnp.stack([x, x]) if eqx.is_array(x) else x, inst)
        out = eqx.filter_vmap(lambda i: _price(i, model, MARKET, route, cfg))(batch)
        assert out.shape == (2,) and np.all(np.isfinite(np.asarray(out)))


class _Leaky(eqx.Module):
    notional: jnp.ndarray


@pytest.fixture
def leaky_route():
    """A pricer that reads a curve its instrument does not declare."""
    key = (_Leaky, type(None), "analytic")

    @P.register_pricer(*key)
    def _adapter(*, instrument, model, market, resolved, config, key):
        hidden = market.curve(DECOYS["curves"])(resolved.reference_date + 365)
        return P.PricingResult(price=instrument.notional * hidden, method="analytic")

    yield key
    P._REGISTRY.pop(key, None)


def test_contract_catches_a_hidden_market_read(leaky_route):
    """Anti-vacuity: the 'zero outside declared' clause sees a leaky adapter."""
    inst = _Leaky(notional=jnp.asarray(1.0))
    grad = eqx.filter_grad(
        lambda m: price(inst, None, m, method="analytic").price)(MARKET)
    decoy = DECOYS["curves"]
    assert decoy not in declared_entries(inst, None, MARKET).curves
    assert np.any(np.asarray(_slot_leaves(grad, "curves")[decoy][0]) != 0.0)


@pytest.mark.parametrize("route", ROUTES, ids=IDS)
def test_route_prices_with_its_default_config(route):
    """``config=None`` uses the pair's default (D15) and prices finitely."""
    i, m, meth = route
    sample = SAMPLES[(i, m)]()
    res = price(sample.instrument, sample.model, MARKET, method=meth,
                key=KEY if meth == "mc" else None)
    assert res.config == default_config(i, m, meth)
    assert np.isfinite(float(res.price))
