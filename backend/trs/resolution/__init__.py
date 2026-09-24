"""Entity resolution: linking incoming names to canonical biomedical entities.

Every link is recorded in ``entity_resolution_decision`` with the method and
score that produced it, so a merge can be explained and reversed. Thresholds
are set to prefer a missed merge over a false one — see
:mod:`trs.resolution.resolver` for why that asymmetry drives the design.
"""

from trs.resolution.normalise import (
    normalise_compound,
    normalise_general,
    normalise_sponsor,
    normalise_target,
)
from trs.resolution.resolver import (
    AUTO_ACCEPT_SCORE,
    RESOLVER_VERSION,
    REVIEW_SCORE,
    EntityResolver,
    Resolution,
)

__all__ = [
    "AUTO_ACCEPT_SCORE", "EntityResolver", "RESOLVER_VERSION", "REVIEW_SCORE",
    "Resolution", "normalise_compound", "normalise_general",
    "normalise_sponsor", "normalise_target",
]
