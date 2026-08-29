"""Stage 3 — CMS convexity: VALAX Hagan model vs QuantLib Hagan pricers.

Validates both VALAX convexity routes against their QuantLib counterparts:

- ``method="analytic"``  vs  ``ql.AnalyticHaganPricer``
- ``method="replication"`` vs ``ql.NumericHaganPricer``

**Design.** :func:`ql_cms_convexity_setup` builds an annual, Act/365
``ql.SwapIndex`` matching VALAX's annual synthetic swap, a flat
``ConstantSwaptionVolatility``, and a *near-natural* CMS coupon (payment one
day after fixing) so QuantLib prices essentially the natural swaplet VALAX
models.  VALAX is fed QuantLib's own forward swap rate, so the test isolates
the **convexity model** from any forward/annuity construction difference.

**Tolerance.** VALAX uses the street-standard *flat-yield* G-function
(:math:`A(S)=(1-(1+S)^{-n})/S`), whereas QuantLib's ``GFunctionStandard``
builds the numeraire ratio from the actual (here flat) discount curve.  These
are genuinely different closed forms, so agreement is at the few-percent level,
not floating-point.  The tolerances below bound the observed gap across the
scenario grid; the tight *internal* analytic-vs-replication check lives in
``tests/test_pricing/test_cms_convexity.py``.
"""

import jax.numpy as jnp
import pytest

from valax.pricing.analytic.cms_convexity import cms_convexity_adjustment

from tests.test_quantlib_comparison._ql_adapters import ql_cms_convexity_setup


# (rate, tenor_years, expiry_years, flat_vol)
_CASES = [
    (0.03, 5, 2.0, 0.20),
    (0.03, 10, 5.0, 0.25),
    (0.02, 5, 1.0, 0.15),
    (0.05, 10, 3.0, 0.30),
    (0.04, 2, 5.0, 0.25),
    (0.03, 10, 1.0, 0.20),
]


@pytest.fixture(params=_CASES, ids=[f"c{i}" for i in range(len(_CASES))])
def scenario(request):
    rate, tenor, expiry, vol = request.param
    ql_out = ql_cms_convexity_setup(rate, tenor, expiry, vol)
    return {
        "tenor": tenor, "expiry": expiry, "vol": vol, "ql": ql_out,
        "forward": ql_out["forward"],
    }


class TestCMSConvexityQL:
    """VALAX Hagan convexity vs QuantLib's analytic & numeric Hagan pricers."""

    def _valax_adj(self, scenario, method):
        return float(cms_convexity_adjustment(
            jnp.asarray(scenario["forward"]),
            jnp.asarray(float(scenario["expiry"])),
            scenario["tenor"],
            jnp.asarray(scenario["vol"]),
            method=method,
        ))

    def test_analytic_matches_ql_analytic_hagan(self, scenario):
        v = self._valax_adj(scenario, "analytic")
        q = scenario["ql"]["analytic_adj"]
        assert v > 0.0 and q > 0.0
        rel = abs(v - q) / abs(q)
        assert rel < 0.08, f"analytic: VALAX={v:.6e} QL={q:.6e} rel={rel:.3f}"

    def test_replication_matches_ql_numeric_hagan(self, scenario):
        v = self._valax_adj(scenario, "replication")
        q = scenario["ql"]["numeric_adj"]
        assert v > 0.0 and q > 0.0
        rel = abs(v - q) / abs(q)
        assert rel < 0.15, f"numeric: VALAX={v:.6e} QL={q:.6e} rel={rel:.3f}"

    def test_adjusted_rate_within_a_few_bp(self, scenario):
        # The adjusted CMS rate should land within a couple of basis points
        # of QuantLib's, across both engines.
        for method, key in [("analytic", "analytic_rate"), ("replication", "numeric_rate")]:
            v_rate = scenario["forward"] + self._valax_adj(scenario, method)
            q_rate = scenario["ql"][key]
            assert abs(v_rate - q_rate) < 5e-4, (
                f"{method}: VALAX rate={v_rate:.6f} QL rate={q_rate:.6f}"
            )
