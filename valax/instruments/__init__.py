"""Financial instrument definitions (data-only pytrees)."""

# Equity options and exotics
from valax.instruments.options import (
    EuropeanOption,
    AmericanOption,
    EquityBarrierOption,
    AsianOption,
    LookbackOption,
    VarianceSwap,
    CompoundOption,
    ChooserOption,
    Autocallable,
    WorstOfBasketOption,
    Cliquet,
    DigitalOption,
    SpreadOption,
)

# Fixed income
from valax.instruments.bonds import (
    ZeroCouponBond,
    FixedRateBond,
    FloatingRateBond,
    CallableBond,
    PuttableBond,
    ConvertibleBond,
)

# Interest rate derivatives
from valax.instruments.rates import (
    Caplet,
    Cap,
    InterestRateSwap,
    Swaption,
    BermudanSwaption,
    OISSwap,
    CrossCurrencySwap,
    TotalReturnSwap,
    CMSSwap,
    CMSSpreadSwap,
    CMSCapFloor,
    RangeAccrual,
)

# Legs and multi-leg trade representation
from valax.instruments.legs import (
    COMPOUNDING_CONVENTIONS,
    FixedLeg,
    FloatingLeg,
    Leg,
    MultiLegSwap,
)

# FX derivatives
from valax.instruments.fx import (
    FXForward,
    FXVanillaOption,
    FXBarrierOption,
    QuantoOption,
    TARF,
    FXSwap,
)

# Credit derivatives
from valax.instruments.credit import (
    CDS,
    CDOTranche,
)

# Inflation derivatives
from valax.instruments.inflation import (
    ZeroCouponInflationSwap,
    YearOnYearInflationSwap,
    InflationCapFloor,
)
