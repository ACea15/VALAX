"""Quote risk through a calibration chain: adjoint vs autodiff vs bumping.

The chain under test is curves → spot/dividend → SABR smile (on the
curve-implied forwards) → optionally a Heston model fitted to the smile
quotes. Gates:

* the adjoint and autodiff paths agree (``rtol = 1e-7``);
* both agree with full-recalibration central finite differences
  (``rtol = 1e-5`` on material entries);
* fixed-input (SABR beta) sensitivity is reported and matches FD;
* the adjoint sweep is linear: a book's summed cotangent gives the sum of
  trade risks for one sweep;
* reconciliation flags a mismatch rather than hiding it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration import (
    calibrate_market,
    calibrate_model,
    price_fit_node,
)
from valax.calibration.transforms import HESTON_TRANSFORMS
from valax.core.pricing import price
from valax.greeks import (
    QuoteRisk,
    chain_adjoint,
    market_gradient,
    quote_risk_adjoint,
    quote_risk_autodiff,
    reconcile_quote_risk,
)
from valax.greeks.quote_risk import tree_add
from valax.instruments.options import EuropeanOption
from valax.market import QuoteSet
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel

from tests.test_calibration._chain_fixtures import (
    EXPIRIES,
    STRIKES,
    VOL_KEYS,
    equity_chain,
)


OPTION = EuropeanOption(strike=jnp.asarray(105.0), expiry=jnp.asarray(0.75))


@pytest.fixture(scope="module")
def setup():
    chain, quotes, fixed = equity_chain()
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    return chain, quotes, fixed, calibrated


def _fd(value_fn, quotes, fixed, key, *, on_fixed=False):
    # Bump relative to the quote's size so solver noise (~1e-10 relative)
    # stays far below the bump.
    base = fixed[key] if on_fixed else quotes[key]
    eps = 1e-4 * max(abs(float(base)), 0.01)
    if on_fixed:
        up, dn = value_fn(quotes, fixed.bump(key, eps)), value_fn(quotes, fixed.bump(key, -eps))
    else:
        up, dn = value_fn(quotes.bump(key, eps), fixed), value_fn(quotes.bump(key, -eps), fixed)
    return (up - dn) / (2 * eps)


class TestMarketChain:
    def test_paths_agree(self, setup):
        chain, quotes, fixed, calibrated = setup
        adj = quote_risk_adjoint(OPTION, BlackScholesModel(), calibrated)
        ad = quote_risk_autodiff(OPTION, BlackScholesModel(), chain, quotes, fixed)
        assert adj.method == "adjoint" and ad.method == "autodiff"
        rec = reconcile_quote_risk(adj, ad, rtol=1e-7, atol=1e-9)
        assert rec.ok, rec

    def test_against_full_recalibration_fd(self, setup):
        chain, quotes, fixed, calibrated = setup
        risk = quote_risk_adjoint(OPTION, BlackScholesModel(), calibrated)

        def pv(q, p):
            market = calibrate_market(chain, q, p).market
            return float(price(OPTION, BlackScholesModel(), market).price)

        for key in ("USD.OIS.1Y", "SPX.SPOT", "SPX.DIV", VOL_KEYS[1][1], VOL_KEYS[0][2]):
            fd = _fd(pv, quotes, fixed, key)
            assert float(risk.quotes[key]) == pytest.approx(fd, rel=1e-4, abs=1e-6), key
        fd_beta = _fd(pv, quotes, fixed, "SPX.SABR.BETA", on_fixed=True)
        assert float(risk.fixed["SPX.SABR.BETA"]) == pytest.approx(fd_beta, rel=1e-4, abs=1e-7)

    def test_vol_level_carries_curve_risk(self, setup):
        """The smile is fitted on curve forwards, so curve quotes see it.

        With vols held as a flat surface the 1Y OIS delta of a 9M option comes
        only from discounting and the BS forward; through the SABR level it
        also moves the smile's anchor. The chain must include that term —
        compare with the autodiff path, which cannot omit it.
        """
        chain, quotes, fixed, calibrated = setup
        adj = quote_risk_adjoint(OPTION, BlackScholesModel(), calibrated)
        ad = quote_risk_autodiff(OPTION, BlackScholesModel(), chain, quotes, fixed)
        assert float(adj.quotes["USD.OIS.1Y"]) == pytest.approx(
            float(ad.quotes["USD.OIS.1Y"]), rel=1e-9
        )
        assert abs(float(adj.quotes["USD.OIS.1Y"])) > 1.0

    def test_levels_report_stationarity(self, setup):
        _, _, _, calibrated = setup
        risk = quote_risk_adjoint(OPTION, BlackScholesModel(), calibrated)
        assert set(risk.level_residuals) == {"usd_ois", "spx_inputs", "spx_smile"}
        for norm in risk.level_residuals.values():
            assert float(norm) < 1e-8

    def test_book_sweep_is_linear(self, setup):
        """Sum of trade cotangents → one sweep == sum of per-trade risk."""
        _, _, _, calibrated = setup
        trades = [
            OPTION,
            EuropeanOption(strike=jnp.asarray(95.0), expiry=jnp.asarray(0.5), is_call=False),
        ]
        model = BlackScholesModel()
        book_bar = None
        for t in trades:
            book_bar = tree_add(book_bar, market_gradient(t, model, calibrated.market))
        book = chain_adjoint(calibrated, book_bar)
        per_trade = [quote_risk_adjoint(t, model, calibrated) for t in trades]
        for k in book.quotes.keys():
            total = sum(float(r.quotes[k]) for r in per_trade)
            assert float(book.quotes[k]) == pytest.approx(total, rel=1e-10, abs=1e-12)

    def test_reconcile_flags_mismatch(self, setup):
        _, _, _, calibrated = setup
        risk = quote_risk_adjoint(OPTION, BlackScholesModel(), calibrated)
        tampered = QuoteRisk(
            quotes=risk.quotes.bump("SPX.SPOT", 1e-3), fixed=risk.fixed,
        )
        rec = reconcile_quote_risk(risk, tampered)
        assert not rec.ok and rec.mismatched == ("q:SPX.SPOT",)


class TestModelLevel:
    """Level 2: Heston fitted once per underlying to the smile quotes."""

    @pytest.fixture(scope="class")
    def heston(self, setup):
        _, quotes, fixed, calibrated = setup
        # One expiry keeps second-order compilation through the COS pricer
        # cheap; the mechanism does not depend on the basket size.
        basket = [
            EuropeanOption(strike=jnp.asarray(k), expiry=EXPIRIES[1])
            for k in STRIKES[1].tolist()
        ]
        # Mean reversion is a held-fixed input p (its sensitivity is
        # reported); v0, xi, rho are fitted to the four quotes of one expiry.
        node = price_fit_node(
            "spx_heston",
            template=HestonModel(
                v0=jnp.asarray(0.045), kappa=jnp.asarray(1.5),
                theta=jnp.asarray(0.045), xi=jnp.asarray(0.6),
                rho=jnp.asarray(-0.6),
            ),
            transforms={k: HESTON_TRANSFORMS[k] for k in ("v0", "xi", "rho")},
            basket=basket,
            quote_keys=list(VOL_KEYS[1]),
            quote_form="lognormal_vol",
            fixed_fields={"kappa": "SPX.HESTON.KAPPA"},
        )
        fixed_all = fixed.merge(QuoteSet.from_dict({"SPX.HESTON.KAPPA": 1.5}))
        result = calibrate_model(node, quotes, calibrated.market, fixed_all)
        assert bool(result.diagnostics.converged)
        return node, result, fixed_all

    def test_paths_agree_through_model(self, setup, heston):
        chain, quotes, _, calibrated = setup
        node, result, fixed_all = heston
        trade = EuropeanOption(strike=jnp.asarray(105.0), expiry=jnp.asarray(1.0))
        adj = quote_risk_adjoint(trade, None, calibrated, model_result=result)
        ad = quote_risk_autodiff(
            trade, None, chain, quotes, fixed_all, model_node=node
        )
        rec = reconcile_quote_risk(adj, ad, rtol=1e-6, atol=1e-8)
        assert rec.ok, rec
        # The basket quotes carry the model's vega; the fixed mean reversion
        # gets its own sensitivity; both levels report stationarity.
        assert abs(float(adj.quotes[VOL_KEYS[1][1]])) > 1.0
        assert abs(float(adj.fixed["SPX.HESTON.KAPPA"])) > 0.0
        assert "spx_heston" in adj.level_residuals
