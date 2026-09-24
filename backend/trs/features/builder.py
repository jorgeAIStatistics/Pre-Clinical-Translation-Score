"""Build a model-ready feature vector for one analysis unit.

An analysis unit is ``(development track, as-of date)``. The builder reads
evidence exclusively through an :class:`~trs.features.temporal.AsOfContext`, so
it structurally cannot see post-cutoff facts.

Three outputs, always together
------------------------------
``values``
    ``{feature_name: float | str | None}`` — raw, pre-encoding values.
``missingness``
    ``{feature_name: MissingnessReason}`` for every feature. Present features
    are recorded as ``PRESENT`` so the record is complete rather than implied.
``evidence_map``
    ``{feature_name: [evidence_id, ...]}`` — which rows produced each number.
    This is what makes "trace any figure back to its sources" a real capability
    rather than a slogan.

Aggregation philosophy
----------------------
Evidence items are narrow (one system, one endpoint, one effect). Turning many
into one number requires a choice, and the choice is stated per feature in the
registry's ``derivation`` field. Two rules hold throughout:

* **Absence of evidence is not evidence of absence.** A feature computed over
  an empty set is ``None``, not 0 — except where the registry explicitly
  declares ``ZERO_IS_ABSENCE`` because the sweep is exhaustive by construction
  (counts of things we would have found had they existed).
* **Only directional evidence contributes to concordance ratios.** Items with
  direction ``neutral``/``unknown`` are counted in volume features but excluded
  from numerator and denominator of supporting-fraction features, so that
  uninterpretable studies neither help nor hurt.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

from trs.db.models.core import Program, ProgramIndication
from trs.db.models.evidence import (
    EvidenceItem,
    PkPdBridge,
    ProgramBiomarker,
    SafetyFinding,
    TargetAssessment,
)
from trs.features.effects import standardise_effect
from trs.features.historical import HistoricalRateIndex, mechanism_key_for
from trs.features.registry import FEATURES, FEATURES_BY_NAME
from trs.features.spec import (
    MODALITY_GROUPS,
    TARGET_CLASS_GROUPS,
    MissingPolicy,
    ordinal,
)
from trs.features.temporal import AsOfContext, years_between
from trs.ontology import enums as E

#: Bump when the aggregation logic changes in a way that alters output for
#: unchanged inputs. Stored on every snapshot alongside the schema version.
BUILDER_VERSION = "fb-1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# Evidence-class groupings used by the aggregations
# ─────────────────────────────────────────────────────────────────────────────

EFFICACY_ENDPOINTS: frozenset[str] = frozenset({
    E.EndpointCategory.TUMOUR_GROWTH_INHIBITION,
    E.EndpointCategory.TUMOUR_REGRESSION,
    E.EndpointCategory.SURVIVAL,
    E.EndpointCategory.OBJECTIVE_RESPONSE,
    E.EndpointCategory.PROGRESSION_FREE_SURVIVAL,
    E.EndpointCategory.OVERALL_SURVIVAL,
    E.EndpointCategory.FUNCTIONAL_BEHAVIOURAL,
    E.EndpointCategory.HISTOPATHOLOGY,
})

#: Evidence generated in human material or human populations. Excludes
#: ``clinical_trial`` deliberately: a clinical readout on *this* program at or
#: before the decision point is possible for reformulation and repositioning
#: programs, but treating it as generic "human support" would conflate the
#: preclinical case with clinical fact. It is captured instead by
#: ``has_prior_human_clinical_on_pathway``, which is target-scoped.
HUMAN_DERIVED_CLASSES: frozenset[str] = frozenset({
    E.EvidenceClass.PRIMARY_HUMAN_CELL,
    E.EvidenceClass.ORGANOID,
    E.EvidenceClass.HUMAN_EX_VIVO,
    E.EvidenceClass.HUMAN_TISSUE_MOLECULAR,
    E.EvidenceClass.HUMAN_GENETIC,
    E.EvidenceClass.HUMAN_OBSERVATIONAL,
})

HIGH_FIDELITY_MODEL_TYPES: frozenset[str] = frozenset({
    E.ModelType.PATIENT_DERIVED_XENOGRAFT,
    E.ModelType.GENETICALLY_ENGINEERED,
    E.ModelType.ORTHOTOPIC,
    E.ModelType.SPONTANEOUS_DISEASE,
    E.ModelType.HUMANISED,
})

HIGH_FIDELITY_SUBTYPES: frozenset[str] = frozenset({
    E.OncologyModelSubtype.PDX_SUBCUTANEOUS,
    E.OncologyModelSubtype.PDX_ORTHOTOPIC,
    E.OncologyModelSubtype.GEMM_AUTOCHTHONOUS,
    E.OncologyModelSubtype.CELL_LINE_DERIVED_XENOGRAFT_ORTHO,
    E.OncologyModelSubtype.SYNGENEIC_ORTHOTOPIC,
    E.OncologyModelSubtype.HUMANISED_IMMUNE_MOUSE,
    E.OncologyModelSubtype.METASTASIS_MODEL,
})

RODENT_SPECIES: frozenset[str] = frozenset({
    E.Species.MOUSE, E.Species.RAT, E.Species.ZEBRAFISH,
})

IMMUNE_COMPETENT_STATES: frozenset[str] = frozenset({
    E.ImmuneCompetence.FULLY_IMMUNOCOMPETENT,
    E.ImmuneCompetence.HUMANISED_IMMUNE_SYSTEM,
    E.ImmuneCompetence.PARTIALLY_RECONSTITUTED,
})

RANDOMISED_STATES: frozenset[str] = frozenset({
    E.RandomizationLevel.RANDOMISED, E.RandomizationLevel.RANDOMISED_CONCEALED,
})

BLINDED_STATES: frozenset[str] = frozenset({
    E.BlindingLevel.FULLY_BLINDED,
    E.BlindingLevel.OUTCOME_ASSESSOR_BLINDED,
    E.BlindingLevel.PARTIALLY_BLINDED,
})

#: Values that mean "we have no assessment", never "the lowest level".
UNASSESSED = frozenset({"unknown", "not_assessed", "not_reported", ""})

M = E.MissingnessReason


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(slots=True)
class BuiltFeatures:
    track_id: str
    as_of: dt.date
    schema_version: str
    builder_version: str
    values: dict[str, Any]
    missingness: dict[str, str]
    evidence_map: dict[str, list[str]]
    contributing_evidence_ids: list[str]
    n_evidence_items: int
    contains_synthetic: bool

    def vector_hash(self) -> str:
        payload = json.dumps(
            {k: self.values[k] for k in sorted(self.values)},
            sort_keys=True, default=str, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class _Accumulator:
    """Collects values, missingness reasons and evidence attribution."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.missing: dict[str, str] = {}
        self.evidence: dict[str, list[str]] = {}

    def set(
        self,
        name: str,
        value: Any,
        *,
        reason: str = M.NOT_REPORTED,
        sources: Iterable[Any] | None = None,
    ) -> None:
        spec = FEATURES_BY_NAME[name]
        if value is None and spec.missing_policy is MissingPolicy.ZERO_IS_ABSENCE:
            # The registry asserts the sweep is exhaustive for this feature, so
            # "we found none" is the observation 0, not an absence of data.
            value = 0.0
        self.values[name] = value
        self.missing[name] = M.PRESENT if value is not None else reason
        if sources:
            ids = [getattr(s, "id", None) for s in sources]
            self.evidence[name] = [i for i in ids if i]


# ─────────────────────────────────────────────────────────────────────────────
# Small aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assessed(value: str | None) -> str | None:
    """Return the value, or ``None`` if it encodes 'no assessment'."""
    if value is None or str(value) in UNASSESSED:
        return None
    return str(value)


def _fraction(numerator_items: Sequence[Any], denominator_items: Sequence[Any]) -> float | None:
    if not denominator_items:
        return None
    return len(numerator_items) / len(denominator_items)


def _safe_max(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _safe_mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _safe_median(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _log2_ratio(value: float | None) -> float | None:
    import math
    if value is None or value <= 0 or not math.isfinite(value):
        return None
    return math.log2(value)


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    ctx: AsOfContext,
    track: ProgramIndication,
    *,
    history: HistoricalRateIndex | None = None,
    schema_version: str | None = None,
) -> BuiltFeatures:
    """Assemble the declared feature vector for ``track`` as of ``ctx.as_of``."""
    from trs.features.spec import FEATURE_SCHEMA_VERSION

    schema_version = schema_version or FEATURE_SCHEMA_VERSION
    session = ctx.session
    program: Program = session.get(Program, track.program_id)
    acc = _Accumulator()

    all_evidence = ctx.evidence_for_track(track.id)
    bridge = ctx.latest_pkpd_bridge(track.id)
    assessment = ctx.latest_target_assessment(program.primary_target_id, track.indication_id)
    biomarkers = ctx.biomarkers_for_track(track.id)
    safety = ctx.safety_findings(track.id, program.id)

    # Partitions used repeatedly below.
    in_vivo = [e for e in all_evidence if e.evidence_class == E.EvidenceClass.ANIMAL_IN_VIVO]
    in_vivo_eff = [e for e in in_vivo if e.endpoint_category in EFFICACY_ENDPOINTS]
    iv_supporting = [e for e in in_vivo_eff if e.direction == E.EvidenceDirection.SUPPORTS]
    iv_refuting = [e for e in in_vivo_eff if e.direction == E.EvidenceDirection.REFUTES]
    iv_directional = iv_supporting + iv_refuting

    human_ev = [e for e in all_evidence if e.evidence_class in HUMAN_DERIVED_CLASSES]
    human_sup = [e for e in human_ev if e.direction == E.EvidenceDirection.SUPPORTS]
    human_ref = [e for e in human_ev if e.direction == E.EvidenceDirection.REFUTES]

    supporting_all = [e for e in all_evidence if e.direction == E.EvidenceDirection.SUPPORTS]

    _program_context(acc, program, track, assessment)
    _animal_model(acc, in_vivo_eff, iv_supporting, iv_refuting, iv_directional)
    _cross_species(acc, in_vivo_eff, iv_supporting, iv_refuting)
    _pkpd(acc, bridge)
    _human_support(acc, human_ev, human_sup, human_ref, assessment)
    _target_validation(acc, program, assessment)
    _disease_relevance(acc, track, iv_supporting)
    _reproducibility(acc, supporting_all, in_vivo_eff)
    _safety(acc, safety)
    _biomarkers(acc, biomarkers)
    _historical(acc, program, track, ctx.as_of, history)
    _evidence_volume(acc, all_evidence, ctx.as_of)

    # Every declared feature must be present in the output, even if unset, so
    # a downstream consumer never has to distinguish "absent key" from "None".
    for spec in FEATURES:
        if spec.name not in acc.values:
            acc.values[spec.name] = None
            acc.missing[spec.name] = M.NOT_EXTRACTED

    contributing = sorted({eid for ids in acc.evidence.values() for eid in ids})
    contains_synth = (
        any(e.is_synthetic for e in all_evidence)
        or bool(program.is_synthetic)
        or bool(track.is_synthetic)
        or (bridge is not None and bool(bridge.is_synthetic))
        or (assessment is not None and bool(assessment.is_synthetic))
        or any(b.is_synthetic for b in biomarkers)
        or any(s.is_synthetic for s in safety)
    )

    return BuiltFeatures(
        track_id=track.id,
        as_of=ctx.as_of,
        schema_version=schema_version,
        builder_version=BUILDER_VERSION,
        values=acc.values,
        missingness=acc.missing,
        evidence_map=acc.evidence,
        contributing_evidence_ids=contributing,
        n_evidence_items=len(all_evidence),
        contains_synthetic=contains_synth,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Domain builders
# ─────────────────────────────────────────────────────────────────────────────

def _program_context(
    acc: _Accumulator,
    program: Program,
    track: ProgramIndication,
    assessment: TargetAssessment | None,
) -> None:
    acc.set("modality_group", MODALITY_GROUPS.get(program.modality, "unknown"))
    target_class = getattr(program.target, "target_class", None) if program.target else None
    acc.set("target_class_group", TARGET_CLASS_GROUPS.get(target_class or "unknown", "unknown"))

    if assessment is not None and assessment.prior_approved_drugs_on_target is not None:
        acc.set(
            "is_first_in_class",
            1.0 if assessment.prior_approved_drugs_on_target == 0 else 0.0,
            sources=[assessment],
        )
    else:
        acc.set("is_first_in_class", None, reason=M.NOT_REPORTED)

    acc.set("is_combination_regimen", 1.0 if track.combination_partner else 0.0)


def _animal_model(
    acc: _Accumulator,
    in_vivo_eff: list[EvidenceItem],
    iv_supporting: list[EvidenceItem],
    iv_refuting: list[EvidenceItem],
    iv_directional: list[EvidenceItem],
) -> None:
    acc.set("animal_n_invivo_studies", float(len(in_vivo_eff)), sources=in_vivo_eff)
    acc.set("animal_n_refuting_studies", float(len(iv_refuting)), sources=iv_refuting)

    acc.set(
        "animal_positive_fraction",
        _fraction(iv_supporting, iv_directional),
        reason=M.NOT_APPLICABLE if not in_vivo_eff else M.NOT_REPORTED,
        sources=iv_directional,
    )

    cv = [ordinal("qualitative_strength", _assessed(e.construct_validity)) for e in iv_supporting]
    acc.set("animal_best_construct_validity", _safe_max(cv), sources=iv_supporting)

    ds = [ordinal("qualitative_strength", _assessed(e.disease_similarity)) for e in iv_supporting]
    acc.set("animal_mean_disease_similarity", _safe_mean(ds), sources=iv_supporting)

    high_fidelity = [
        e for e in iv_supporting
        if e.model_type in HIGH_FIDELITY_MODEL_TYPES or e.model_subtype in HIGH_FIDELITY_SUBTYPES
    ]
    acc.set(
        "animal_uses_high_fidelity_model",
        1.0 if high_fidelity else 0.0,
        sources=high_fidelity or iv_supporting,
    )

    with_immune = [e for e in iv_supporting if _assessed(e.immune_competence)]
    competent = [e for e in with_immune if e.immune_competence in IMMUNE_COMPETENT_STATES]
    acc.set(
        "animal_immunocompetent_fraction",
        _fraction(competent, with_immune),
        sources=with_immune,
    )

    effects = [
        standardise_effect(e.effect_metric, e.effect_value,
                           favours_treatment=e.effect_favours_treatment)
        for e in iv_supporting
    ]
    with_effect = [e for e, v in zip(iv_supporting, effects, strict=True) if v is not None]
    acc.set("animal_median_effect_z", _safe_median(effects), sources=with_effect)
    acc.set("animal_max_effect_z", _safe_max(effects), sources=with_effect)

    ns = [float(e.n_per_arm) for e in iv_supporting if e.n_per_arm]
    acc.set(
        "animal_median_n_per_arm",
        statistics.median(ns) if ns else None,
        sources=[e for e in iv_supporting if e.n_per_arm],
    )


def _cross_species(
    acc: _Accumulator,
    in_vivo_eff: list[EvidenceItem],
    iv_supporting: list[EvidenceItem],
    iv_refuting: list[EvidenceItem],
) -> None:
    sup_species = {e.species for e in iv_supporting if _assessed(e.species)}
    ref_species = {e.species for e in iv_refuting if _assessed(e.species)}
    tested_species = sup_species | ref_species

    acc.set("species_count_supporting", float(len(sup_species)), sources=iv_supporting)
    acc.set(
        "species_concordance_fraction",
        (len(sup_species) / len(tested_species)) if tested_species else None,
        reason=M.NOT_APPLICABLE if not in_vivo_eff else M.NOT_REPORTED,
        sources=iv_supporting + iv_refuting,
    )
    nonrodent = [e for e in iv_supporting if _assessed(e.species) and e.species not in RODENT_SPECIES]
    acc.set("has_nonrodent_efficacy", 1.0 if nonrodent else 0.0, sources=nonrodent)

    model_types = {(e.model_type, e.model_subtype) for e in iv_supporting if _assessed(e.model_type)}
    acc.set("n_distinct_model_types", float(len(model_types)), sources=iv_supporting)

    discordant = bool(sup_species and ref_species and not (sup_species & ref_species))
    acc.set("has_contradictory_species", 1.0 if discordant else 0.0, sources=iv_refuting)


def _pkpd(acc: _Accumulator, bridge: PkPdBridge | None) -> None:
    if bridge is None:
        for name in (
            "exposure_margin_log2", "therapeutic_index_log2",
            "exposure_bridge_quality_ord", "target_engagement_ord",
            "exposure_response_established", "pk_scaling_species_count",
        ):
            acc.set(name, None, reason=M.NOT_REPORTED)
        return

    src = [bridge]
    acc.set("exposure_margin_log2", _log2_ratio(bridge.exposure_margin_ratio), sources=src)
    acc.set("therapeutic_index_log2", _log2_ratio(bridge.therapeutic_index_ratio), sources=src)
    acc.set(
        "exposure_bridge_quality_ord",
        ordinal("exposure_bridge", _assessed(bridge.exposure_bridge_quality)),
        sources=src,
    )
    acc.set(
        "target_engagement_ord",
        ordinal("target_engagement", _assessed(bridge.target_engagement_evidence)),
        sources=src,
    )
    acc.set(
        "exposure_response_established",
        None if bridge.exposure_response_established is None
        else float(bridge.exposure_response_established),
        sources=src,
    )
    acc.set(
        "pk_scaling_species_count",
        None if bridge.pk_scaling_species_count is None
        else float(bridge.pk_scaling_species_count),
        sources=src,
    )


def _human_support(
    acc: _Accumulator,
    human_ev: list[EvidenceItem],
    human_sup: list[EvidenceItem],
    human_ref: list[EvidenceItem],
    assessment: TargetAssessment | None,
) -> None:
    acc.set("human_evidence_count", float(len(human_ev)), sources=human_ev)
    acc.set(
        "human_evidence_supporting_fraction",
        _fraction(human_sup, human_sup + human_ref),
        reason=M.NOT_APPLICABLE if not human_ev else M.NOT_REPORTED,
        sources=human_sup + human_ref,
    )

    tissue = [
        e for e in human_sup
        if e.evidence_class in (
            E.EvidenceClass.HUMAN_TISSUE_MOLECULAR, E.EvidenceClass.HUMAN_EX_VIVO,
        )
    ]
    acc.set("has_patient_tissue_evidence", 1.0 if tissue else 0.0, sources=tissue)

    organoid = [
        e for e in human_sup
        if e.evidence_class == E.EvidenceClass.ORGANOID and e.species == E.Species.HUMAN
    ]
    acc.set("has_organoid_evidence", 1.0 if organoid else 0.0, sources=organoid)

    if assessment is not None and assessment.prior_clinical_programs_on_target is not None:
        acc.set(
            "has_prior_human_clinical_on_pathway",
            1.0 if assessment.prior_clinical_programs_on_target > 0 else 0.0,
            sources=[assessment],
        )
    else:
        acc.set("has_prior_human_clinical_on_pathway", None, reason=M.NOT_REPORTED)


def _target_validation(
    acc: _Accumulator, program: Program, assessment: TargetAssessment | None
) -> None:
    if assessment is None:
        for name in (
            "genetic_evidence_ord", "target_association_score", "causal_support_ord",
            "pathway_redundancy_ord", "human_tissue_expression_ord",
        ):
            acc.set(name, None, reason=M.NOT_REPORTED)
    else:
        src = [assessment]
        acc.set(
            "genetic_evidence_ord",
            ordinal("genetic_evidence", _assessed(assessment.genetic_evidence_tier)),
            sources=src,
        )
        acc.set("target_association_score", assessment.association_score, sources=src)
        acc.set(
            "causal_support_ord",
            ordinal("causal_support", _assessed(assessment.causal_support)),
            sources=src,
        )
        acc.set(
            "pathway_redundancy_ord",
            ordinal("pathway_redundancy", _assessed(assessment.pathway_redundancy)),
            sources=src,
        )
        acc.set(
            "human_tissue_expression_ord",
            ordinal(
                "qualitative_strength",
                _assessed(assessment.human_tissue_expression_evidence),
            ),
            sources=src,
        )

    identity = getattr(program.target, "ortholog_identity_mouse", None) if program.target else None
    acc.set("target_ortholog_identity_mouse", identity)


def _disease_relevance(
    acc: _Accumulator, track: ProgramIndication, iv_supporting: list[EvidenceItem]
) -> None:
    selected = track.biomarker_selection in (
        E.BiomarkerSelectionStrategy.REQUIRED_GENOMIC_ALTERATION,
        E.BiomarkerSelectionStrategy.REQUIRED_PROTEIN_EXPRESSION,
    )
    acc.set(
        "biomarker_selected_population",
        None if not _assessed(track.biomarker_selection) else float(selected),
    )
    acc.set("disease_stage_ord", ordinal("disease_stage", _assessed(track.disease_stage)))
    acc.set("unmet_need_ord", ordinal("unmet_need", _assessed(track.unmet_need)))

    matched = [e for e in iv_supporting if e.indication_id == track.indication_id]
    with_ind = [e for e in iv_supporting if e.indication_id]
    acc.set(
        "model_indication_match",
        _fraction(matched, with_ind),
        reason=M.NOT_APPLICABLE if not iv_supporting else M.NOT_REPORTED,
        sources=with_ind,
    )


def _reproducibility(
    acc: _Accumulator, supporting_all: list[EvidenceItem], in_vivo_eff: list[EvidenceItem]
) -> None:
    rep = [ordinal("replication", _assessed(e.replication_scope)) for e in supporting_all]
    acc.set("replication_scope_ord", _safe_max(rep), sources=supporting_all)

    labs = {e.lab_identifier for e in supporting_all if e.lab_identifier}
    with_lab = [e for e in supporting_all if e.lab_identifier]
    acc.set(
        "n_independent_labs",
        float(len(labs)) if with_lab else None,
        sources=with_lab,
    )
    acc.set(
        "single_lab_dependence",
        (1.0 if len(labs) <= 1 else 0.0) if (with_lab and len(supporting_all) >= 2) else None,
        sources=with_lab,
    )

    iv_with_rand = [e for e in in_vivo_eff if _assessed(e.randomization)]
    acc.set(
        "randomised_fraction",
        _fraction([e for e in iv_with_rand if e.randomization in RANDOMISED_STATES], iv_with_rand),
        sources=iv_with_rand,
    )
    iv_with_blind = [e for e in in_vivo_eff if _assessed(e.blinding)]
    acc.set(
        "blinded_fraction",
        _fraction([e for e in iv_with_blind if e.blinding in BLINDED_STATES], iv_with_blind),
        sources=iv_with_blind,
    )

    with_prespec = [
        e for e in supporting_all
        if e.prospective_protocol is not None or e.preregistered is not None
    ]
    prespec = [e for e in with_prespec if e.prospective_protocol or e.preregistered]
    acc.set("prespecified_fraction", _fraction(prespec, with_prespec), sources=with_prespec)

    sel = [ordinal("selective_reporting", _assessed(e.selective_reporting_risk))
           for e in supporting_all]
    acc.set("selective_reporting_ord", _safe_max(sel), sources=supporting_all)

    with_coi = [e for e in supporting_all if _assessed(e.conflict_of_interest)]
    affil = [e for e in with_coi if e.conflict_of_interest == E.ConflictOfInterest.SPONSOR_AFFILIATED]
    acc.set("sponsor_affiliated_fraction", _fraction(affil, with_coi), sources=with_coi)

    with_pub = [e for e in supporting_all if _assessed(e.publication_status)]
    peer = [e for e in with_pub if e.publication_status == E.PublicationStatus.PEER_REVIEWED]
    acc.set("peer_reviewed_fraction", _fraction(peer, with_pub), sources=with_pub)


def _safety(acc: _Accumulator, safety: list[SafetyFinding]) -> None:
    acc.set("n_safety_findings", float(len(safety)), sources=safety)

    sev = [ordinal("severity", _assessed(s.severity)) for s in safety]
    acc.set("max_safety_severity_ord", _safe_max(sev), sources=safety)

    irreversible = [
        s for s in safety
        if s.on_target and s.reversible is False
        and (ordinal("severity", _assessed(s.severity)) or 0.0) >= 4.0
    ]
    acc.set("has_irreversible_on_target_finding", 1.0 if irreversible else 0.0, sources=irreversible)

    with_monitor = [s for s in safety if s.monitorable_in_human is not None]
    acc.set(
        "safety_monitorable_fraction",
        _fraction([s for s in with_monitor if s.monitorable_in_human], with_monitor),
        reason=M.NOT_APPLICABLE if not safety else M.NOT_REPORTED,
        sources=with_monitor,
    )


def _biomarkers(acc: _Accumulator, biomarkers: list[ProgramBiomarker]) -> None:
    pd_markers = [
        b for b in biomarkers
        if b.biomarker_type in (
            E.BiomarkerType.PHARMACODYNAMIC, E.BiomarkerType.TARGET_ENGAGEMENT,
        )
    ]
    readiness = [ordinal("biomarker_readiness", _assessed(b.readiness)) for b in pd_markers]
    acc.set(
        "biomarker_readiness_ord",
        _safe_max(readiness),
        reason=M.NOT_APPLICABLE if not biomarkers else M.NOT_REPORTED,
        sources=pd_markers,
    )

    predictive = [
        b for b in biomarkers
        if b.biomarker_type == E.BiomarkerType.PREDICTIVE_SELECTION
        and (ordinal("biomarker_readiness", _assessed(b.readiness)) or 0.0) >= 3.0
    ]
    acc.set("has_predictive_selection_biomarker", 1.0 if predictive else 0.0, sources=predictive)

    conc = [
        ordinal("qualitative_strength", _assessed(b.cross_species_concordance))
        for b in biomarkers
    ]
    acc.set("biomarker_species_concordance_ord", _safe_max(conc), sources=biomarkers)


def _historical(
    acc: _Accumulator,
    program: Program,
    track: ProgramIndication,
    as_of: dt.date,
    history: HistoricalRateIndex | None,
) -> None:
    if history is None:
        for name in (
            "target_prior_failure_rate", "target_prior_n_programs",
            "mechanism_class_prior_failure_rate", "indication_prior_failure_rate",
        ):
            acc.set(name, None, reason=M.NOT_EXTRACTED)
        return

    rate, n = history.target_rate(program.primary_target_id, as_of, program.id)
    acc.set("target_prior_failure_rate", rate)
    acc.set("target_prior_n_programs", float(n))

    mech_rate, _ = history.mechanism_rate(mechanism_key_for(program), as_of, program.id)
    acc.set("mechanism_class_prior_failure_rate", mech_rate)

    ind_rate, _ = history.indication_rate(track.indication_id, as_of, program.id)
    acc.set("indication_prior_failure_rate", ind_rate)


def _evidence_volume(acc: _Accumulator, all_evidence: list[EvidenceItem], as_of: dt.date) -> None:
    acc.set("evidence_total_count", float(len(all_evidence)), sources=all_evidence)
    if not all_evidence:
        acc.set("evidence_span_years", None, reason=M.NOT_APPLICABLE)
        acc.set("evidence_recency_years", None, reason=M.NOT_APPLICABLE)
        return
    dates = [e.available_at for e in all_evidence]
    acc.set("evidence_span_years", years_between(min(dates), max(dates)), sources=all_evidence)
    acc.set("evidence_recency_years", years_between(max(dates), as_of), sources=all_evidence)
