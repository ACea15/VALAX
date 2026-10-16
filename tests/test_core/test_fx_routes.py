"""FX forward and FX vanilla through ``price()`` (onboarded by the checklist).

Each route equals its kernel called on the same inputs: the FX spot keyed by
the pair, and each currency's continuously-compounded zero rate to the
expiry read off its own curve (so the kernels' ``exp(-r T)`` is the curve's
discount factor). The generic clauses — traced instrument under ``jit``,
gradient only on declared entries, ``vmap`` — are the route contract's
(``test_route_contract.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.binding import project_rate
from valax.core.pricing import auto_method, price
from valax.instruments.fx import FXForward, FXVanillaOption
from valax.models import ImpliedVolModel
from valax.pricing.analytic.fx import fx_forward_price, garman_kohlhagen_price
from tests.test_core.route_samples import EUR, FX_VOL, MARKET

jax.config.update("jax_enable_x64", True)

DOM = MARKET.curve("_default_")
FOR = MARKET.curve(EUR)
SPOT = MARKET.fx_spots["EURUSD"]


def _rates(expiry):
    return project_rate(DOM, expiry), project_rate(FOR, expiry)


@pytest.mark.parametrize("is_buy", [True, False])
def test_forward_is_the_cip_kernel(is_buy):
    fwd = FXForward(strike=jnp.asarray(1.08), expiry=jnp.asarray(1.5),
                    notional_foreign=jnp.asarray(1e6), is_buy=is_buy,
                    currency_pair="EUR/USD", foreign_discount_id=EUR)
    got = price(fwd, None, MARKET)
    r_d, r_f = _rates(fwd.expiry)
    assert got.method == "analytic" and auto_method(fwd, None) == "analytic"
    np.testing.assert_array_equal(np.asarray(got.price),
                                  np.asarray(fx_forward_price(fwd, SPOT, r_d, r_f)))


@pytest.mark.parametrize("is_call", [True, False])
def test_vanilla_is_garman_kohlhagen_at_the_surface_vol(is_call):
    opt = FXVanillaOption(strike=jnp.asarray(1.12), expiry=jnp.asarray(1.0),
                          notional_foreign=jnp.asarray(1e6), is_call=is_call,
                          currency_pair="EUR/USD", foreign_discount_id=EUR)
    got = price(opt, ImpliedVolModel(FX_VOL), MARKET).price
    r_d, r_f = _rates(opt.expiry)
    want = garman_kohlhagen_price(opt, SPOT, MARKET.surfaces[FX_VOL].vol, r_d, r_f)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert float(got) > 0.0


def test_foreign_curve_is_not_read_as_a_projection_curve():
    """``(domestic, foreign)`` is not a ``(discount, forward)`` pair."""
    from valax.core.resolve import forward_id_of
    fwd = FXForward(strike=jnp.asarray(1.08), expiry=jnp.asarray(1.0),
                    notional_foreign=jnp.asarray(1.0), currency_pair="EUR/USD",
                    foreign_discount_id=EUR)
    assert forward_id_of(fwd) is None


def test_missing_fx_spot_is_named():
    fwd = FXForward(strike=jnp.asarray(1.0), expiry=jnp.asarray(1.0),
                    notional_foreign=jnp.asarray(1.0), currency_pair="AUD/USD")
    with pytest.raises(KeyError, match="FX spot 'AUDUSD'"):
        price(fwd, None, MARKET)
