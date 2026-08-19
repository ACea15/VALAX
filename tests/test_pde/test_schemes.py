"""Tests for the 1-D time-stepping scheme selection (``theta_for_scheme``).

Guards the 1-D/2-D split: the 2-D ADI schemes (Douglas, Craig-Sneyd, HV) must
be rejected by the 1-D theta-scheme solver rather than silently degrading to
Crank-Nicolson.
"""

import pytest

from valax.pricing.pde.config import Scheme
from valax.pricing.pde.schemes import theta_for_scheme


class TestSchemeIsAdi:
    @pytest.mark.parametrize(
        "scheme", [Scheme.DOUGLAS, Scheme.CRAIG_SNEYD, Scheme.HV]
    )
    def test_adi_schemes_flagged(self, scheme):
        assert scheme.is_adi() is True

    @pytest.mark.parametrize(
        "scheme", [Scheme.IMPLICIT, Scheme.CRANK_NICOLSON]
    )
    def test_one_dimensional_schemes_not_flagged(self, scheme):
        assert scheme.is_adi() is False


class TestThetaForScheme:
    def test_implicit_is_one(self):
        assert theta_for_scheme(Scheme.IMPLICIT) == 1.0

    def test_crank_nicolson_is_half(self):
        assert theta_for_scheme(Scheme.CRANK_NICOLSON) == 0.5

    @pytest.mark.parametrize(
        "scheme", [Scheme.DOUGLAS, Scheme.CRAIG_SNEYD, Scheme.HV]
    )
    def test_adi_schemes_rejected(self, scheme):
        # A 2-D ADI scheme has no 1-D theta interpretation: it must raise at
        # the user boundary rather than silently degrade to Crank-Nicolson.
        with pytest.raises(ValueError, match="ADI"):
            theta_for_scheme(scheme)
