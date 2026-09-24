"""Controlled vocabularies for the translational evidence model.

Every categorical value that enters a feature vector, a database row or an API
payload is drawn from one of the enumerations defined here. Free text is
permitted only in explicitly designated note/description fields, which never
feed the statistical model.

Rationale: translational evidence is heterogeneous and largely narrative in its
source form. Forcing it through a closed vocabulary is what makes the evidence
base queryable, the feature schema versionable, and the model auditable.
"""

from trs.ontology.enums import *  # noqa: F401,F403
from trs.ontology.enums import __all__ as _enum_all

__all__ = list(_enum_all)
