"""Make the QuantLib comparison suite truly optional.

Guarding collection here keeps the suite optional without touching each module:
when QuantLib cannot be imported, every ``*_ql.py`` module in this directory is
skipped from collection entirely.  When QuantLib *is* installed, nothing is
ignored and the full comparison suite runs as normal.
"""

import importlib.util

if importlib.util.find_spec("QuantLib") is None:
    # Skip collecting the QuantLib comparison modules (all named ``*_ql.py``)
    # so a missing optional dependency never breaks the wider test session.
    collect_ignore_glob = ["*_ql.py"]
