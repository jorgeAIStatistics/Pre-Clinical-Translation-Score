"""All ORM models. Importing this package registers every table on ``Base``."""

from trs.db.models.analytics import (
    CohortEntry,
    CohortVersion,
    ComparatorLink,
    ConnectorRun,
    FeatureSnapshot,
    ModelArtifact,
    Scenario,
    ScoreRecord,
)
from trs.db.models.core import (
    Biomarker,
    Compound,
    CompoundAlias,
    EntityResolutionDecision,
    Indication,
    IndicationAlias,
    Pathway,
    Program,
    ProgramAlias,
    ProgramIndication,
    ProgramSponsorHistory,
    ProgramTarget,
    Sponsor,
    SponsorAlias,
    Target,
    TargetAlias,
    TargetPathway,
)
from trs.db.models.evidence import (
    EvidenceFieldProvenance,
    EvidenceItem,
    PkPdBridge,
    ProgramBiomarker,
    RawRecord,
    SafetyFinding,
    Source,
    TargetAssessment,
    Trial,
)
from trs.db.models.tenancy import (
    ApiKey,
    AuditEvent,
    Membership,
    Organization,
    Project,
    User,
)

__all__ = [
    "ApiKey", "AuditEvent", "Biomarker", "CohortEntry", "CohortVersion",
    "ComparatorLink", "Compound", "CompoundAlias", "ConnectorRun",
    "EntityResolutionDecision", "EvidenceFieldProvenance", "EvidenceItem",
    "FeatureSnapshot", "Indication", "IndicationAlias", "Membership",
    "ModelArtifact", "Organization", "Pathway", "PkPdBridge", "Program",
    "ProgramAlias", "ProgramBiomarker", "ProgramIndication",
    "ProgramSponsorHistory", "ProgramTarget", "Project", "RawRecord",
    "SafetyFinding", "Scenario", "ScoreRecord", "Source", "Sponsor",
    "SponsorAlias", "Target", "TargetAlias", "TargetAssessment",
    "TargetPathway", "Trial", "User",
]
