"""Temporal-integrity tests: the feature vector at T depends only on facts knowable at T.

Why this module exists
----------------------
Every performance number this platform reports — AUROC, Brier score, calibration
slope, the decision curve, the usefulness verdict — is conditional on one
property holding: for a development track with cutoff ``T``, the feature vector
is a function of facts available at ``T`` and of nothing else. If a single
post-``T`` fact leaks in, every one of those numbers is optimistic by an unknown
amount and the product is worse than useless, because it is confidently wrong.

That property cannot be established by inspection. It is established here, by
constructing the leak and demonstrating that it does not land: build a snapshot
at ``T``, insert evidence dated after ``T`` that is designed to move every
feature it can touch, rebuild at the same ``T``, and require bit-identical
output. A companion test then proves the inserted evidence really is capable of
moving the vector, so the invariance assertion can never pass vacuously.

All data here is synthetic (``seeded_session`` runs the demonstration
generator). Nothing in this module measures translational biology; it measures
the code's temporal discipline, which is the only thing it claims to measure.
Everything runs offline against an in-memory database.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

import numpy as np
import pytest
from sqlalchemy import select

from trs.cohort.build import AS_OF_RULES, _as_of_for, build_cohort
from trs.cohort.outcomes import PRIMARY_ENDPOINT
from trs.db.models.core import Biomarker, Indication, Program, ProgramIndication, Target
from trs.db.models.evidence import (
    EvidenceItem,
    PkPdBridge,
    ProgramBiomarker,
    SafetyFinding,
    TargetAssessment,
)
from trs.features.builder import build_features
from trs.features.historical import (
    HistoricalRateIndex,
    ResolvedOutcome,
    shrunk_rate,
)
from trs.features.temporal import AsOfContext, TemporalIntegrityError
from trs.ontology import enums as E
from trs.pipeline.features import build_history_index, materialise_snapshot
from trs.validation.splitters import (
    GROUP_SCOPE_KEYS,
    ConnectedComponentGrouper,
    GroupScope,
    build_grouping_keys,
    leakage_audit,
    nested_temporal_folds,
    temporal_split,
)

DAY = dt.timedelta(days=1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _busiest_track(session) -> tuple[ProgramIndication, Program]:
    """The seeded track with the most evidence, ties broken by id.

    Deterministic given the fixture's seed, and picking the busiest track means
    the invariance test exercises the largest number of populated features.
    """
    counts: Counter[str] = Counter(
        e.program_indication_id
        for e in session.scalars(select(EvidenceItem))
        if e.program_indication_id
    )
    assert counts, "seeded universe has no track-level evidence"
    track_id = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    track = session.get(ProgramIndication, track_id)
    return track, session.get(Program, track.program_id)


def _insert_future_evidence(session, track: ProgramIndication, program: Program,
                            cutoff: dt.date) -> None:
    """Insert post-cutoff facts chosen to move as many features as possible.

    Every row is dated strictly after ``cutoff``: supporting and refuting in
    vivo studies in a new species with a high-fidelity model and pristine
    methodology, human-derived evidence, a PK/PD bridge with an enormous
    exposure margin, an indication-specific target assessment at the strongest
    genetic tier, biomarkers at the highest readiness, severe irreversible
    safety findings, and a resolved outcome on the same target belonging to
    another sponsor. If any of these reaches a snapshot built at ``cutoff``,
    the as-of layer is broken.
    """
    after = cutoff + DAY
    common = dict(
        organization_id=None,
        program_id=program.id,
        program_indication_id=track.id,
        target_id=program.primary_target_id,
        available_at=after,
        provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
        is_synthetic=True,
        synthetic_note="[SYNTHETIC] post-cutoff leakage probe",
    )

    for i in range(3):
        session.add(EvidenceItem(
            **common,
            indication_id=track.indication_id,
            evidence_class=str(E.EvidenceClass.ANIMAL_IN_VIVO),
            direction=str(E.EvidenceDirection.SUPPORTS),
            title=f"[SYNTHETIC] post-cutoff in vivo probe {i + 1}",
            species=str(E.Species.NON_HUMAN_PRIMATE),
            biological_system=str(E.BiologicalSystem.IN_VIVO_WHOLE_ANIMAL),
            model_type=str(E.ModelType.PATIENT_DERIVED_XENOGRAFT),
            model_subtype=str(E.OncologyModelSubtype.PDX_ORTHOTOPIC),
            immune_competence=str(E.ImmuneCompetence.HUMANISED_IMMUNE_SYSTEM),
            construct_validity="high",
            disease_similarity="high",
            endpoint_category=str(E.EndpointCategory.TUMOUR_GROWTH_INHIBITION),
            effect_metric=str(E.EffectMetric.PERCENT_TGI),
            effect_value=99.0,
            effect_favours_treatment=True,
            n_per_arm=200,
            n_arms=4,
            randomization=str(E.RandomizationLevel.RANDOMISED_CONCEALED),
            blinding=str(E.BlindingLevel.FULLY_BLINDED),
            replication_scope=str(E.ReplicationScope.INDEPENDENT_LABS_MULTIPLE),
            independent_replications=6,
            lab_identifier=f"POST-CUTOFF-LAB-{i + 1}",
            selective_reporting_risk=str(E.SelectiveReportingRisk.LOW),
            conflict_of_interest=str(E.ConflictOfInterest.ACADEMIC_ONLY),
            publication_status=str(E.PublicationStatus.PEER_REVIEWED),
            prospective_protocol=True,
            preregistered=True,
        ))

    # A refuting study too: refutation moves a different set of features
    # (refuting counts, species discordance) than support does.
    session.add(EvidenceItem(
        **common,
        indication_id=track.indication_id,
        evidence_class=str(E.EvidenceClass.ANIMAL_IN_VIVO),
        direction=str(E.EvidenceDirection.REFUTES),
        title="[SYNTHETIC] post-cutoff refuting in vivo probe",
        species=str(E.Species.DOG),
        biological_system=str(E.BiologicalSystem.IN_VIVO_WHOLE_ANIMAL),
        model_type=str(E.ModelType.SPONTANEOUS_DISEASE),
        model_subtype=str(E.OncologyModelSubtype.METASTASIS_MODEL),
        endpoint_category=str(E.EndpointCategory.SURVIVAL),
        effect_metric=str(E.EffectMetric.HAZARD_RATIO),
        effect_value=1.4,
        effect_favours_treatment=False,
        n_per_arm=40,
    ))

    for cls in (E.EvidenceClass.HUMAN_TISSUE_MOLECULAR, E.EvidenceClass.ORGANOID,
                E.EvidenceClass.HUMAN_GENETIC):
        session.add(EvidenceItem(
            **common,
            indication_id=track.indication_id,
            evidence_class=str(cls),
            direction=str(E.EvidenceDirection.SUPPORTS),
            title=f"[SYNTHETIC] post-cutoff human evidence — {cls}",
            species=str(E.Species.HUMAN),
            biological_system=str(E.BiologicalSystem.EX_VIVO_TISSUE),
            construct_validity="high",
            disease_similarity="high",
            endpoint_category=str(E.EndpointCategory.BIOMARKER_MODULATION),
            effect_metric=str(E.EffectMetric.FOLD_CHANGE),
            effect_value=9.0,
            effect_favours_treatment=True,
            publication_status=str(E.PublicationStatus.PEER_REVIEWED),
        ))

    # Program-level evidence (no indication): a separate retrieval path in
    # AsOfContext.evidence_for_track, and therefore a separate way to leak.
    session.add(EvidenceItem(
        organization_id=None,
        program_id=program.id,
        program_indication_id=None,
        target_id=program.primary_target_id,
        available_at=after,
        evidence_class=str(E.EvidenceClass.HUMAN_GENETIC),
        direction=str(E.EvidenceDirection.SUPPORTS),
        title="[SYNTHETIC] post-cutoff program-level probe",
        species=str(E.Species.HUMAN),
        publication_status=str(E.PublicationStatus.PEER_REVIEWED),
        provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
        is_synthetic=True,
    ))

    session.add(PkPdBridge(
        program_indication_id=track.id,
        available_at=after,
        preclinical_efficacious_auc=10.0,
        projected_human_auc=40960.0,
        exposure_unit="ng·h/mL",
        preclinical_species=str(E.Species.MOUSE),
        exposure_margin_ratio=4096.0,          # log2 → 12.0
        therapeutic_index_ratio=1024.0,        # log2 → 10.0
        exposure_bridge_quality=str(E.ExposureBridgeQuality.HUMAN_EXPOSURE_DEMONSTRATED),
        target_engagement_evidence=str(E.TargetEngagementEvidence.HUMAN_DIRECT),
        target_engagement_percent=99.0,
        exposure_response_established=True,
        pk_scaling_species_count=4,
        is_synthetic=True,
    ))

    if program.primary_target_id:
        session.add(TargetAssessment(
            target_id=program.primary_target_id,
            indication_id=track.indication_id,
            available_at=after,
            genetic_evidence_tier=str(E.GeneticEvidenceTier.MENDELIAN_CAUSAL),
            association_score=0.99,
            causal_support=str(E.CausalSupport.INTERVENTIONAL_HUMAN),
            pathway_redundancy=str(E.PathwayRedundancy.NO_KNOWN_REDUNDANCY),
            human_tissue_expression_evidence="high",
            prior_clinical_programs_on_target=42,
            prior_approved_drugs_on_target=7,
            prior_ph2_failures_on_target=3,
            is_synthetic=True,
        ))

    for kind in (E.BiomarkerType.PHARMACODYNAMIC, E.BiomarkerType.PREDICTIVE_SELECTION):
        session.add(ProgramBiomarker(
            program_indication_id=track.id,
            available_at=after,
            biomarker_type=str(kind),
            readiness=str(E.BiomarkerReadiness.CLINICALLY_QUALIFIED),
            cross_species_concordance="high",
            assay_validated_in_human=True,
            intended_use_in_trial="[SYNTHETIC] post-cutoff probe",
            is_synthetic=True,
        ))

    for organ in ("hepatic", "cardiovascular"):
        session.add(SafetyFinding(
            program_indication_id=track.id,
            program_id=program.id,
            target_id=program.primary_target_id,
            available_at=after,
            species=str(E.Species.NON_HUMAN_PRIMATE),
            organ_system=organ,
            finding="[SYNTHETIC] post-cutoff toxicology probe",
            severity="life_threatening",
            reversible=False,
            monitorable_in_human=False,
            on_target=True,
            is_synthetic=True,
        ))

    # A competitor program on the same target whose outcome resolves after the
    # cutoff — the leak the HistoricalRateIndex date filter exists to stop.
    rival = Program(
        name="[SYNTHETIC] post-cutoff rival program",
        primary_target_id=program.primary_target_id,
        modality=program.modality,
        is_synthetic=True,
    )
    session.add(rival)
    session.flush()
    session.add(ProgramIndication(
        program_id=rival.id,
        indication_id=track.indication_id,
        outcome_label=str(E.OutcomeLabel.PHASE_2_EFFICACY_FAILURE),
        outcome_date=after,
        highest_phase_reached=str(E.DevelopmentPhase.PHASE_2),
        ind_or_cta_on=cutoff - dt.timedelta(days=900),
        first_in_human_on=cutoff - dt.timedelta(days=700),
        is_synthetic=True,
    ))
    session.flush()


def _minimal_track(
    session, *, symbol: str = "SYNTHGENE1"
) -> tuple[ProgramIndication, Program, Target, Indication]:
    """A bare target/indication/program/track with no evidence attached.

    Used where the seeded universe's richness would obscure the property under
    test: these tests assert on exact row identity, so they need to own every
    row in scope.
    """
    target = Target(symbol=symbol, target_class=str(E.TargetClass.KINASE),
                    ortholog_identity_mouse=0.9, is_synthetic=True)
    indication = Indication(name=f"[SYNTHETIC] indication for {symbol}",
                            therapeutic_area=str(E.TherapeuticArea.ONCOLOGY),
                            is_synthetic=True)
    session.add_all([target, indication])
    session.flush()

    program = Program(name=f"[SYNTHETIC] program on {symbol}",
                      primary_target_id=target.id,
                      modality=str(E.Modality.SMALL_MOLECULE),
                      is_synthetic=True)
    session.add(program)
    session.flush()

    track = ProgramIndication(
        program_id=program.id,
        indication_id=indication.id,
        disease_stage=str(E.DiseaseStage.METASTATIC),
        unmet_need=str(E.UnmetNeed.LIMITED_EFFICACY_SOC),
        biomarker_selection=str(E.BiomarkerSelectionStrategy.ALL_COMERS),
        outcome_label=str(E.OutcomeLabel.PHASE_2_EFFICACY_FAILURE),
        is_synthetic=True,
    )
    session.add(track)
    session.flush()
    return track, program, target, indication


# ─────────────────────────────────────────────────────────────────────────────
# 1. The headline invariant
# ─────────────────────────────────────────────────────────────────────────────

def test_future_evidence_cannot_change_a_snapshot_built_at_the_cutoff(seeded_session):
    """If this fails, every reported metric is optimistic by an unknown amount.

    A feature vector built as of T would be reading facts that did not exist at
    T, so the held-out AUROC, Brier score, calibration curve and the
    "not_distinguishable_from_base_rate" usefulness verdict would all be
    measuring the model's access to the future rather than its ability to read
    preclinical evidence. Historical scores would also silently change whenever
    new evidence arrived, making any past recommendation unreproducible.
    """
    session, _ = seeded_session
    track, program = _busiest_track(session)
    cutoff = track.ind_or_cta_on
    assert cutoff is not None

    before = build_features(
        AsOfContext(session=session, as_of=cutoff, strict=True),
        track,
        history=build_history_index(session),
    )

    _insert_future_evidence(session, track, program, cutoff)

    after = build_features(
        AsOfContext(session=session, as_of=cutoff, strict=True),
        track,
        history=build_history_index(session),
    )

    assert after.values == before.values
    assert after.vector_hash() == before.vector_hash()
    assert after.missingness == before.missingness
    assert after.evidence_map == before.evidence_map
    assert after.contributing_evidence_ids == before.contributing_evidence_ids
    assert after.n_evidence_items == before.n_evidence_items

    # Nothing dated after the cutoff may appear in the provenance trail.
    contributing = set(after.contributing_evidence_ids)
    leaked = [
        e.id for e in session.scalars(
            select(EvidenceItem).where(EvidenceItem.available_at > cutoff)
        )
        if e.id in contributing
    ]
    assert leaked == []


def test_the_post_cutoff_probe_really_does_move_features(seeded_session):
    """If this fails, the invariance test above is vacuous and proves nothing.

    A leakage test that inserts evidence the builder ignores for unrelated
    reasons — wrong evidence class, wrong track, unparseable effect metric —
    would pass forever while the as-of filter rotted. This test asserts the
    same probe rows move the vector hard once the cutoff is moved past them,
    which is what makes the invariance assertion meaningful.
    """
    session, _ = seeded_session
    track, program = _busiest_track(session)
    cutoff = track.ind_or_cta_on
    later = cutoff + dt.timedelta(days=2000)

    at_cutoff = build_features(AsOfContext(session=session, as_of=cutoff), track)
    _insert_future_evidence(session, track, program, cutoff)
    at_later = build_features(AsOfContext(session=session, as_of=later), track)

    assert at_later.vector_hash() != at_cutoff.vector_hash()
    assert at_later.n_evidence_items > at_cutoff.n_evidence_items

    # Values taken directly from the probe rows, so these are exact.
    assert at_later.values["exposure_margin_log2"] == pytest.approx(12.0)
    assert at_later.values["therapeutic_index_log2"] == pytest.approx(10.0)
    assert at_later.values["genetic_evidence_ord"] == pytest.approx(5.0)
    assert at_later.values["causal_support_ord"] == pytest.approx(4.0)
    assert at_later.values["target_association_score"] == pytest.approx(0.99)
    assert at_later.values["exposure_bridge_quality_ord"] == pytest.approx(4.0)
    assert at_later.values["target_engagement_ord"] == pytest.approx(6.0)
    assert at_later.values["biomarker_readiness_ord"] == pytest.approx(5.0)
    assert at_later.values["has_predictive_selection_biomarker"] == 1.0
    assert at_later.values["max_safety_severity_ord"] == pytest.approx(5.0)
    assert at_later.values["has_irreversible_on_target_finding"] == 1.0
    assert at_later.values["has_nonrodent_efficacy"] == 1.0
    assert at_later.values["animal_uses_high_fidelity_model"] == 1.0
    assert at_later.values["is_first_in_class"] == 0.0        # 7 prior approvals
    assert at_later.values["has_prior_human_clinical_on_pathway"] == 1.0

    # Counts must strictly increase relative to the cutoff snapshot.
    for name in ("animal_n_invivo_studies", "animal_n_refuting_studies",
                 "human_evidence_count", "evidence_total_count", "n_safety_findings"):
        assert at_later.values[name] > (at_cutoff.values[name] or 0.0), name


def test_persisted_snapshot_is_stable_when_rebuilt_after_new_evidence(seeded_session):
    """If this fails, a score shown to a user in 2019 changes when 2021 data lands.

    Feature snapshots are the object a historical prediction is reconstructed
    from. Rebuilding one for the same (track, as-of, schema version) must
    reproduce the stored vector exactly, or the audit trail behind every
    delivered score is fiction.
    """
    session, _ = seeded_session
    track, program = _busiest_track(session)
    cutoff = track.ind_or_cta_on

    first = materialise_snapshot(session, track.id, cutoff,
                                 history=build_history_index(session))
    original_hash = first.vector_hash
    original_features = dict(first.features)
    assert original_hash

    _insert_future_evidence(session, track, program, cutoff)

    rebuilt = materialise_snapshot(session, track.id, cutoff,
                                   history=build_history_index(session),
                                   overwrite=True)
    assert rebuilt.vector_hash == original_hash
    assert rebuilt.features == original_features
    assert rebuilt.n_evidence_items == first.n_evidence_items


# ─────────────────────────────────────────────────────────────────────────────
# 2. The strict guard
# ─────────────────────────────────────────────────────────────────────────────

def test_strict_context_raises_on_a_row_from_after_the_cutoff(seeded_session):
    """If this fails, a future WHERE-clause bug leaks silently instead of loudly.

    AsOfContext.strict is defence in depth: it re-checks the rows a query
    returned rather than trusting the SQL. Without it, a mis-specified filter —
    a wrong column, a dropped predicate, a join that duplicates rows — produces
    a quietly contaminated feature vector rather than a crash, and the
    contamination is then baked into a trained model artifact.
    """
    session, _ = seeded_session
    cutoff = dt.date(2015, 1, 1)
    strict = AsOfContext(session=session, as_of=cutoff, strict=True)
    lenient = AsOfContext(session=session, as_of=cutoff, strict=False)

    # The guard is on unless a caller deliberately turns it off. A default flip
    # would disarm it everywhere at once, silently.
    assert AsOfContext(session=session, as_of=cutoff).strict is True

    future_row = EvidenceItem(
        evidence_class=str(E.EvidenceClass.ANIMAL_IN_VIVO),
        direction=str(E.EvidenceDirection.SUPPORTS),
        available_at=cutoff + DAY,
        title="[SYNTHETIC] one day too late",
    )

    with pytest.raises(TemporalIntegrityError) as excinfo:
        strict._verify([future_row])
    message = str(excinfo.value)
    assert "available_at" in message
    assert cutoff.isoformat() in message

    # Exactly on the cutoff is legal; one day before obviously so.
    on_cutoff = EvidenceItem(
        evidence_class=str(E.EvidenceClass.ANIMAL_IN_VIVO),
        available_at=cutoff,
    )
    assert strict._verify([on_cutoff]) == [on_cutoff]

    # strict=False is an explicit opt-out, not an accident: it must not raise.
    assert lenient._verify([future_row]) == [future_row]

    # And the realistic failure mode: a query that forgot its date predicate.
    unfiltered = list(session.scalars(
        select(EvidenceItem).where(EvidenceItem.available_at > cutoff).limit(5)
    ))
    assert unfiltered, "seeded universe has no evidence after 2015-01-01"
    with pytest.raises(TemporalIntegrityError):
        strict._verify(unfiltered)


def test_guard_applies_the_cutoff_predicate_to_an_arbitrary_select(seeded_session):
    """If this fails, call sites that build their own queries lose the cutoff.

    AsOfContext.guard is the escape hatch for queries the accessors do not
    cover. It must apply the same at-or-before semantics, or the escape hatch
    becomes the leak.
    """
    session, _ = seeded_session
    cutoff = dt.date(2015, 1, 1)
    ctx = AsOfContext(session=session, as_of=cutoff)

    guarded = ctx.guard(select(EvidenceItem), EvidenceItem.available_at)
    rows = list(session.scalars(guarded))
    assert rows, "expected some evidence at or before the cutoff"
    assert max(r.available_at for r in rows) <= cutoff

    unguarded_count = len(list(session.scalars(select(EvidenceItem))))
    assert len(rows) < unguarded_count, "guard removed nothing — predicate not applied"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Latest-at-or-before, never latest-overall
# ─────────────────────────────────────────────────────────────────────────────

def test_latest_pkpd_bridge_is_the_newest_at_or_before_the_cutoff(db_session):
    """If this fails, a 2014 decision is scored with a 2019 exposure argument.

    The PK/PD bridge is the single most improvable artifact in a program: the
    margin that looked marginal before the IND often looks comfortable after
    human PK is in hand. Reading the newest bridge overall imports precisely
    the knowledge the decision did not have, and inflates the model's apparent
    ability to predict PK/PD-driven failure.
    """
    track, program, _target, _ind = _minimal_track(db_session, symbol="SYNTHGENE_PKPD")

    def bridge(available_at: dt.date, margin: float, row_id: str | None = None) -> PkPdBridge:
        row = PkPdBridge(
            program_indication_id=track.id,
            available_at=available_at,
            exposure_margin_ratio=margin,
            exposure_bridge_quality=str(E.ExposureBridgeQuality.ALLOMETRIC_SIMPLE),
            is_synthetic=True,
        )
        if row_id:
            row.id = row_id
        db_session.add(row)
        return row

    oldest = bridge(dt.date(2012, 3, 1), 0.5)
    middle = bridge(dt.date(2014, 6, 1), 2.0)
    newest = bridge(dt.date(2019, 9, 1), 64.0)   # after every cutoff tested
    db_session.flush()

    cutoff = dt.date(2015, 1, 1)
    chosen = AsOfContext(session=db_session, as_of=cutoff).latest_pkpd_bridge(track.id)
    assert chosen is not None
    assert chosen.id == middle.id
    assert chosen.id != newest.id
    assert chosen.exposure_margin_ratio == 2.0

    # Move the cutoff back: the older bridge becomes the right answer.
    earlier = AsOfContext(session=db_session, as_of=dt.date(2013, 1, 1))
    assert earlier.latest_pkpd_bridge(track.id).id == oldest.id

    # Before any bridge exists there is no bridge — not the earliest one.
    assert AsOfContext(session=db_session, as_of=dt.date(2011, 1, 1)).latest_pkpd_bridge(
        track.id
    ) is None

    # A bridge dated exactly on the cutoff counts.
    on_cutoff = bridge(cutoff, 8.0)
    db_session.flush()
    assert AsOfContext(session=db_session, as_of=cutoff).latest_pkpd_bridge(
        track.id
    ).id == on_cutoff.id

    # Two bridges on the same date must resolve deterministically (id desc),
    # so that re-scoring the same decision reproduces the same number.
    bridge(cutoff, 16.0, row_id="Z" * 26)
    bridge(cutoff, 32.0, row_id="A" * 26)
    db_session.flush()
    repeated = {
        AsOfContext(session=db_session, as_of=cutoff).latest_pkpd_bridge(track.id).id
        for _ in range(3)
    }
    assert repeated == {"Z" * 26}


def test_latest_target_assessment_respects_the_cutoff_and_prefers_the_indication(db_session):
    """If this fails, 2019 target biology is used to justify a 2013 program.

    Target-validation knowledge accumulates: a target with no human genetic
    support in 2013 may be Mendelian-causal by 2019. Using the newest
    assessment would make `genetic_evidence_ord` — one of the most influential
    features in the model — a partial readout of what was learned *after* the
    program succeeded or failed.
    """
    track, program, target, indication = _minimal_track(db_session, symbol="SYNTHGENE_TA")
    other_indication = Indication(name="[SYNTHETIC] unrelated indication",
                                  therapeutic_area=str(E.TherapeuticArea.ONCOLOGY),
                                  is_synthetic=True)
    db_session.add(other_indication)
    db_session.flush()

    def assessment(available_at, tier, indication_id=None) -> TargetAssessment:
        row = TargetAssessment(
            target_id=target.id,
            indication_id=indication_id,
            available_at=available_at,
            genetic_evidence_tier=str(tier),
            is_synthetic=True,
        )
        db_session.add(row)
        return row

    old_target_level = assessment(dt.date(2010, 6, 30), E.GeneticEvidenceTier.GWAS_LOCUS_ONLY)
    mid_target_level = assessment(dt.date(2013, 6, 30), E.GeneticEvidenceTier.GWAS_FINE_MAPPED)
    future_target_level = assessment(dt.date(2019, 6, 30), E.GeneticEvidenceTier.MENDELIAN_CAUSAL)
    ind_specific = assessment(dt.date(2011, 6, 30), E.GeneticEvidenceTier.RARE_CODING_BURDEN,
                              indication_id=indication.id)
    future_ind_specific = assessment(dt.date(2020, 6, 30), E.GeneticEvidenceTier.MENDELIAN_CAUSAL,
                                     indication_id=indication.id)
    db_session.flush()

    cutoff = dt.date(2015, 1, 1)
    ctx = AsOfContext(session=db_session, as_of=cutoff)

    # Indication-specific wins even though a newer target-level row exists…
    chosen = ctx.latest_target_assessment(target.id, indication.id)
    assert chosen.id == ind_specific.id
    assert chosen.id not in {future_ind_specific.id, future_target_level.id}

    # …but the post-cutoff indication-specific row is never the answer.
    assert ctx.latest_target_assessment(target.id, indication.id).available_at <= cutoff

    # With no indication-specific row in scope, fall back to target level, and
    # to the newest target-level row at or before the cutoff.
    assert ctx.latest_target_assessment(target.id, other_indication.id).id == mid_target_level.id
    assert ctx.latest_target_assessment(target.id, None).id == mid_target_level.id
    assert AsOfContext(session=db_session, as_of=dt.date(2012, 1, 1)).latest_target_assessment(
        target.id, other_indication.id
    ).id == old_target_level.id

    # Before anything was known, nothing is returned.
    assert AsOfContext(session=db_session, as_of=dt.date(2009, 1, 1)).latest_target_assessment(
        target.id, indication.id
    ) is None
    # And a program with no target has no assessment rather than someone else's.
    assert ctx.latest_target_assessment(None, indication.id) is None


def test_biomarker_state_is_the_state_as_of_the_cutoff(db_session):
    """If this fails, biomarker readiness is reported at its eventual maturity.

    `biomarkers_for_track` keeps the latest state per biomarker. The latest
    state must be the latest *at or before* the cutoff: an assay that reached
    clinical qualification in 2020 was merely conceptual at a 2014 IND, and
    crediting the program with the 2020 state manufactures readiness the
    decision never had.
    """
    track, _program, _target, _ind = _minimal_track(db_session, symbol="SYNTHGENE_BM")
    cutoff = dt.date(2015, 1, 1)

    marker = Biomarker(name="[SYNTHETIC] PD marker",
                       biomarker_type=str(E.BiomarkerType.PHARMACODYNAMIC),
                       is_synthetic=True)
    db_session.add(marker)
    db_session.flush()

    for available_at, readiness in (
        (dt.date(2012, 1, 1), E.BiomarkerReadiness.CONCEPTUAL),
        (dt.date(2014, 1, 1), E.BiomarkerReadiness.PRECLINICAL_ONLY),
        (dt.date(2018, 1, 1), E.BiomarkerReadiness.CLINICALLY_QUALIFIED),
    ):
        db_session.add(ProgramBiomarker(
            program_indication_id=track.id,
            biomarker_id=marker.id,
            available_at=available_at,
            biomarker_type=str(E.BiomarkerType.PHARMACODYNAMIC),
            readiness=str(readiness),
            is_synthetic=True,
        ))
    db_session.flush()

    rows = AsOfContext(session=db_session, as_of=cutoff).biomarkers_for_track(track.id)
    assert len(rows) == 1, "one biomarker should collapse to one current state"
    assert rows[0].readiness == str(E.BiomarkerReadiness.PRECLINICAL_ONLY)

    later = AsOfContext(session=db_session, as_of=dt.date(2019, 1, 1)).biomarkers_for_track(
        track.id
    )
    assert later[0].readiness == str(E.BiomarkerReadiness.CLINICALLY_QUALIFIED)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Historical rates: no future outcomes, no self-inclusion
# ─────────────────────────────────────────────────────────────────────────────

def test_historical_rate_index_excludes_future_and_focal_outcomes():
    """If this fails, the label is recoverable from the features and AUROC is a lie.

    `target_prior_failure_rate` is among the strongest features available. Two
    leaks destroy it. Counting outcomes resolved after the cutoff tells a 2014
    decision how 2019 turned out. Counting the focal program's *own* outcome
    puts the answer in the question: the model learns to read the label back
    out of the feature, and the validation report shows discrimination that
    does not exist at serving time, when the outcome is unknown by definition.
    """
    cutoff = dt.date(2015, 1, 1)
    focal = "PROG-FOCAL"

    def outcome(program_id, target_id, indication_id, mech, date, failed):
        return ResolvedOutcome(
            track_id=f"TRACK-{program_id}-{indication_id}",
            program_id=program_id,
            target_id=target_id,
            indication_id=indication_id,
            mechanism_key=mech,
            outcome_date=date,
            failed=failed,
        )

    outcomes = [
        # Same target as the focal program, all resolved before the cutoff.
        outcome("PROG-A", "TGT-1", "IND-1", "kinase|small_molecule", dt.date(2010, 5, 1), True),
        outcome("PROG-B", "TGT-1", "IND-1", "kinase|small_molecule", dt.date(2011, 5, 1), True),
        outcome("PROG-C", "TGT-1", "IND-2", "kinase|small_molecule", dt.date(2012, 5, 1), False),
        # The focal program's own outcome, resolved BEFORE the cutoff, so only
        # self-exclusion can remove it.
        outcome(focal, "TGT-1", "IND-1", "kinase|small_molecule", dt.date(2013, 5, 1), True),
        # A sibling track of the focal program: same program id, so it must go too.
        outcome(focal, "TGT-1", "IND-2", "kinase|small_molecule", dt.date(2013, 8, 1), True),
        # Resolved AFTER the cutoff: only the date filter can remove it.
        outcome("PROG-D", "TGT-1", "IND-1", "kinase|small_molecule", dt.date(2019, 5, 1), False),
        # Unrelated target, to make the global base rate non-degenerate.
        outcome("PROG-E", "TGT-2", "IND-3", "gpcr|antibody", dt.date(2009, 1, 1), False),
        outcome("PROG-F", "TGT-2", "IND-3", "gpcr|antibody", dt.date(2010, 1, 1), True),
    ]
    idx = HistoricalRateIndex(outcomes)

    base, n_base = idx.base_rate(cutoff, focal)
    assert n_base == 5          # 8 total − 2 focal rows − 1 post-cutoff row
    assert base == pytest.approx(3 / 5)

    rate, n = idx.target_rate("TGT-1", cutoff, focal)
    assert n == 3               # PROG-A, PROG-B, PROG-C only
    assert rate == pytest.approx(shrunk_rate(2, 3, base))

    # Self-exclusion must MATTER: including the focal program changes both the
    # count and the rate, so a regression that dropped it would fail here.
    rate_with_self, n_with_self = idx.target_rate("TGT-1", cutoff, None)
    assert n_with_self == 5
    assert rate_with_self != pytest.approx(rate)
    assert rate_with_self > rate     # focal's own failures inflate the rate

    # The date filter must MATTER too.
    late_rate, late_n = idx.target_rate("TGT-1", dt.date(2020, 1, 1), focal)
    assert late_n == 4
    assert late_rate != pytest.approx(rate)

    # Nothing before the earliest outcome — and no silent fallback to a
    # group rate computed from zero observations.
    early_rate, early_n = idx.target_rate("TGT-1", dt.date(2008, 1, 1), focal)
    assert early_n == 0
    assert early_rate == pytest.approx(0.5)    # the uninformed base rate

    # Indication and mechanism rates obey exactly the same two rules.
    ind_rate, ind_n = idx.indication_rate("IND-1", cutoff, focal)
    assert ind_n == 2                           # PROG-A, PROG-B
    assert idx.indication_rate("IND-1", cutoff, None)[1] == 3
    assert ind_rate == pytest.approx(shrunk_rate(2, 2, base))

    mech_n = idx.mechanism_rate("kinase|small_molecule", cutoff, focal)[1]
    assert mech_n == 3
    assert idx.mechanism_rate("kinase|small_molecule", cutoff, None)[1] == 5

    # An unknown group is not an error and is not zero — it is the base rate.
    unknown_rate, unknown_n = idx.target_rate("TGT-NEVER-SEEN", cutoff, focal)
    assert unknown_n == 0
    assert unknown_rate == pytest.approx(shrunk_rate(0, 0, base))


def test_historical_rates_from_the_database_exclude_future_and_focal_outcomes(seeded_session):
    """If this fails, the leak is in the real query path rather than the index.

    The hand-built index above proves the arithmetic. This proves the object the
    pipeline actually constructs from the database obeys the same two rules, by
    re-deriving the expected counts with an independent query.
    """
    session, _ = seeded_session
    index = build_history_index(session)
    labelled = frozenset(PRIMARY_ENDPOINT.positive_outcomes | PRIMARY_ENDPOINT.negative_outcomes)

    # Group resolved, labelled tracks by target so a case with real priors can
    # be chosen deterministically.
    by_target: dict[str, list[tuple[str, dt.date, str]]] = defaultdict(list)
    for track, program in session.execute(
        select(ProgramIndication, Program).join(Program, Program.id == ProgramIndication.program_id)
    ):
        if track.outcome_date is None or track.outcome_label not in labelled:
            continue
        if program.primary_target_id:
            by_target[program.primary_target_id].append(
                (track.program_id, track.outcome_date, track.outcome_label)
            )

    # A target where the focal program's own outcome falls at or before the
    # cutoff, so self-exclusion is actually load-bearing.
    case = None
    for target_id in sorted(by_target):
        rows = by_target[target_id]
        if len({p for p, _, _ in rows}) < 2:
            continue
        focal_program = min(rows, key=lambda r: (r[1], r[0]))[0]
        cutoff = max(d for _, d, _ in rows)
        case = (target_id, focal_program, cutoff, rows)
        break
    assert case is not None, "seeded universe has no target with two resolved programs"
    target_id, focal_program, cutoff, rows = case

    expected = [
        r for r in rows if r[1] <= cutoff and r[0] != focal_program
    ]
    rate, n = index.target_rate(target_id, cutoff, focal_program)
    assert n == len(expected)

    expected_failed = sum(
        1 for r in expected if r[2] in PRIMARY_ENDPOINT.positive_outcomes
    )
    base, _ = index.base_rate(cutoff, focal_program)
    assert rate == pytest.approx(shrunk_rate(expected_failed, len(expected), base))

    # Self-exclusion removes at least the focal program's own resolved track.
    _, n_with_self = index.target_rate(target_id, cutoff, None)
    assert n_with_self > n

    # A cutoff before every outcome on this target admits nothing.
    earliest = min(d for _, d, _ in rows)
    assert index.target_rate(target_id, earliest - DAY, focal_program)[1] == 0


def test_builder_never_feeds_a_programs_own_outcome_into_its_features(seeded_session):
    """If this fails, the features contain the answer and the model is circular.

    build_features passes the focal program id to every historical query. This
    checks that at the end-to-end level: the historical features computed for a
    track must equal the rates computed with that program explicitly excluded,
    even when the cutoff is late enough for its own outcome to have resolved.
    """
    session, _ = seeded_session
    index = build_history_index(session)

    track = next(
        t for t in session.scalars(
            select(ProgramIndication).order_by(ProgramIndication.id)
        )
        if t.outcome_date is not None
        and t.outcome_label in PRIMARY_ENDPOINT.positive_outcomes
    )
    program = session.get(Program, track.program_id)
    cutoff = track.outcome_date + dt.timedelta(days=30)   # deliberately after resolution

    built = build_features(
        AsOfContext(session=session, as_of=cutoff), track, history=index
    )
    expected_rate, expected_n = index.target_rate(
        program.primary_target_id, cutoff, program.id
    )
    assert built.values["target_prior_failure_rate"] == pytest.approx(expected_rate)
    assert built.values["target_prior_n_programs"] == pytest.approx(float(expected_n))

    self_included_rate, self_included_n = index.target_rate(
        program.primary_target_id, cutoff, None
    )
    assert self_included_n > expected_n, "focal outcome was not in scope; test is vacuous"
    assert built.values["target_prior_failure_rate"] != pytest.approx(self_included_rate)


# ─────────────────────────────────────────────────────────────────────────────
# 5. The boundary: on the cutoff is inside it
# ─────────────────────────────────────────────────────────────────────────────

def test_evidence_dated_exactly_on_the_cutoff_is_included(db_session):
    """If this fails, the pivotal IND-enabling package is thrown away.

    The as-of rule is `available_at <= T`, not `< T`. Studies are routinely
    dated on the day of the filing they support, so an off-by-one to strict
    inequality would silently discard the most decision-relevant evidence,
    depress every completeness score, and bias the model toward programs whose
    paperwork happened to be filed slowly.
    """
    track, program, target, indication = _minimal_track(db_session, symbol="SYNTHGENE_EDGE")
    cutoff = dt.date(2016, 6, 15)

    def evidence(available_at: dt.date, label: str) -> EvidenceItem:
        row = EvidenceItem(
            program_id=program.id,
            program_indication_id=track.id,
            target_id=target.id,
            indication_id=indication.id,
            evidence_class=str(E.EvidenceClass.ANIMAL_IN_VIVO),
            direction=str(E.EvidenceDirection.SUPPORTS),
            endpoint_category=str(E.EndpointCategory.TUMOUR_GROWTH_INHIBITION),
            effect_metric=str(E.EffectMetric.PERCENT_TGI),
            effect_value=60.0,
            effect_favours_treatment=True,
            species=str(E.Species.MOUSE),
            model_type=str(E.ModelType.XENOGRAFT),
            construct_validity="moderate",
            disease_similarity="moderate",
            available_at=available_at,
            title=f"[SYNTHETIC] {label}",
            is_synthetic=True,
        )
        db_session.add(row)
        return row

    before = evidence(cutoff - DAY, "day before")
    on = evidence(cutoff, "on the cutoff")
    after = evidence(cutoff + DAY, "day after")

    db_session.add(PkPdBridge(program_indication_id=track.id, available_at=cutoff,
                              exposure_margin_ratio=4.0, is_synthetic=True))
    db_session.add(TargetAssessment(
        target_id=target.id, indication_id=None, available_at=cutoff,
        genetic_evidence_tier=str(E.GeneticEvidenceTier.GWAS_FINE_MAPPED),
        is_synthetic=True,
    ))
    db_session.add(ProgramBiomarker(program_indication_id=track.id, available_at=cutoff,
                                    biomarker_type=str(E.BiomarkerType.PHARMACODYNAMIC),
                                    readiness=str(E.BiomarkerReadiness.FIT_FOR_PURPOSE_HUMAN),
                                    is_synthetic=True))
    db_session.add(SafetyFinding(program_indication_id=track.id, program_id=program.id,
                                 available_at=cutoff, severity="moderate", is_synthetic=True))
    db_session.flush()

    ctx = AsOfContext(session=db_session, as_of=cutoff, strict=True)

    returned = [e.id for e in ctx.evidence_for_track(track.id)]
    assert returned == [before.id, on.id]        # ascending by available_at
    assert after.id not in returned

    assert ctx.latest_pkpd_bridge(track.id) is not None
    assert ctx.latest_pkpd_bridge(track.id).available_at == cutoff
    assert ctx.latest_target_assessment(target.id, indication.id).available_at == cutoff
    assert [b.available_at for b in ctx.biomarkers_for_track(track.id)] == [cutoff]
    assert [s.available_at for s in ctx.safety_findings(track.id, program.id)] == [cutoff]

    built = build_features(ctx, track)
    assert built.n_evidence_items == 2
    assert built.values["evidence_total_count"] == 2.0
    assert on.id in built.contributing_evidence_ids
    assert after.id not in built.contributing_evidence_ids
    assert built.values["exposure_margin_log2"] == pytest.approx(2.0)
    assert built.values["genetic_evidence_ord"] == pytest.approx(3.0)
    assert built.values["biomarker_readiness_ord"] == pytest.approx(3.0)
    assert built.values["n_safety_findings"] == 1.0
    # Zero elapsed time between the newest admissible evidence and the cutoff.
    assert built.values["evidence_recency_years"] == pytest.approx(0.0)

    # One day earlier, the on-cutoff rows are all correctly invisible.
    day_before = AsOfContext(session=db_session, as_of=cutoff - DAY, strict=True)
    assert [e.id for e in day_before.evidence_for_track(track.id)] == [before.id]
    assert day_before.latest_pkpd_bridge(track.id) is None
    assert day_before.latest_target_assessment(target.id, indication.id) is None
    assert day_before.biomarkers_for_track(track.id) == []
    assert day_before.safety_findings(track.id, program.id) == []


def test_program_level_evidence_is_filtered_by_the_cutoff_too(db_session):
    """If this fails, target- and chemistry-level facts leak in through a side door.

    Evidence attached to the parent program rather than to the track is fetched
    by a second OR-branch in the same query. A filter that covered only the
    track branch would pass a naive test while leaking every program-level fact
    — which is where target biology and safety findings usually live.
    """
    track, program, target, _ind = _minimal_track(db_session, symbol="SYNTHGENE_PROGLEVEL")
    cutoff = dt.date(2016, 1, 1)

    def program_evidence(available_at: dt.date) -> EvidenceItem:
        row = EvidenceItem(
            program_id=program.id,
            program_indication_id=None,
            target_id=target.id,
            evidence_class=str(E.EvidenceClass.HUMAN_GENETIC),
            direction=str(E.EvidenceDirection.SUPPORTS),
            species=str(E.Species.HUMAN),
            available_at=available_at,
            title="[SYNTHETIC] program-level fact",
            is_synthetic=True,
        )
        db_session.add(row)
        return row

    old = program_evidence(cutoff - dt.timedelta(days=100))
    new = program_evidence(cutoff + dt.timedelta(days=100))
    db_session.add(SafetyFinding(program_indication_id=None, program_id=program.id,
                                 available_at=cutoff + dt.timedelta(days=100),
                                 severity="life_threatening", on_target=True, reversible=False,
                                 is_synthetic=True))
    db_session.flush()

    ctx = AsOfContext(session=db_session, as_of=cutoff, strict=True)
    ids = [e.id for e in ctx.evidence_for_track(track.id)]
    assert ids == [old.id]
    assert new.id not in ids
    assert ctx.safety_findings(track.id, program.id) == []

    built = build_features(ctx, track)
    assert built.values["human_evidence_count"] == 1.0
    assert built.values["n_safety_findings"] == 0.0
    assert built.values["has_irreversible_on_target_finding"] == 0.0

    # Opting out of program-level evidence is honoured, and does not resurrect
    # the post-cutoff row either.
    assert ctx.evidence_for_track(track.id, include_program_level=False) == []


# ─────────────────────────────────────────────────────────────────────────────
# 6. The cohort's as-of rule never falls forward
# ─────────────────────────────────────────────────────────────────────────────

def test_as_of_rule_falls_back_to_first_in_human_never_forward_to_phase_two():
    """If this fails, Phase I results are fed into a "preclinical" feature vector.

    Falling back from the IND date to first-in-human is safe: it is an earlier
    or equal date, so it admits strictly less information. Falling forward to
    Phase II start is not: everything learned in Phase I — human PK, human
    tolerability, early efficacy signals — would become "preclinical evidence",
    and the model would appear to predict Phase II outcomes from facts that are
    really Phase I readouts.
    """
    ind = dt.date(2014, 3, 1)
    fih = dt.date(2014, 11, 1)
    ph2 = dt.date(2016, 5, 1)

    complete = ProgramIndication(program_id="P", indication_id="I",
                                 ind_or_cta_on=ind, first_in_human_on=fih,
                                 phase_2_start_on=ph2)
    assert _as_of_for(complete, "ind_or_cta_on") == ind

    no_ind = ProgramIndication(program_id="P", indication_id="I",
                               ind_or_cta_on=None, first_in_human_on=fih,
                               phase_2_start_on=ph2)
    assert _as_of_for(no_ind, "ind_or_cta_on") == fih
    assert _as_of_for(no_ind, "ind_or_cta_on") != ph2

    # No IND, no first-in-human: there is NO cutoff, even though a Phase II
    # start date is sitting right there.
    only_ph2 = ProgramIndication(program_id="P", indication_id="I",
                                 ind_or_cta_on=None, first_in_human_on=None,
                                 phase_2_start_on=ph2)
    assert _as_of_for(only_ph2, "ind_or_cta_on") is None

    # The other rules are literal: they never substitute a different milestone.
    no_fih = ProgramIndication(program_id="P", indication_id="I",
                               ind_or_cta_on=ind, first_in_human_on=None,
                               phase_2_start_on=ph2)
    assert _as_of_for(no_fih, "first_in_human_on") is None
    assert _as_of_for(no_fih, "phase_2_start_on") == ph2
    assert _as_of_for(complete, "phase_2_start_on") == ph2

    assert set(AS_OF_RULES) == {"ind_or_cta_on", "first_in_human_on", "phase_2_start_on"}


def test_cohort_builder_uses_first_in_human_and_excludes_tracks_with_no_cutoff(
    seeded_session, today
):
    """If this fails, the cohort silently dates some rows from after the decision.

    The exclusion waterfall shown on the Methods page depends on `no_as_of_date`
    being a real, counted exclusion rather than a track quietly rescued with a
    later milestone. A single rescued track carries Phase I knowledge into the
    training set and contaminates the fitted coefficients for every other row.
    """
    session, _ = seeded_session

    baseline = build_cohort(session, data_cutoff=today, persist=False)
    included_ids = [e.program_indication_id for e in baseline.entries]
    assert len(included_ids) >= 2

    fallback_track = session.get(ProgramIndication, sorted(included_ids)[0])
    dropped_track = session.get(ProgramIndication, sorted(included_ids)[1])

    # Case 1: no IND date, but a first-in-human date and a *later* Phase II date.
    fallback_track.ind_or_cta_on = None
    assert fallback_track.first_in_human_on is not None
    fallback_track.phase_2_start_on = fallback_track.first_in_human_on + dt.timedelta(days=400)

    # Case 2: no IND and no first-in-human, only a Phase II start date.
    dropped_track.ind_or_cta_on = None
    dropped_track.first_in_human_on = None
    dropped_track.phase_2_start_on = dt.date(2018, 1, 1)
    session.flush()

    rebuilt = build_cohort(session, data_cutoff=today, persist=False)
    entries = {e.program_indication_id: e for e in rebuilt.entries}

    assert fallback_track.id in entries
    assert entries[fallback_track.id].as_of_date == fallback_track.first_in_human_on
    assert entries[fallback_track.id].as_of_date != fallback_track.phase_2_start_on

    assert dropped_track.id not in entries
    assert rebuilt.exclusions["no_as_of_date"] >= 1
    assert rebuilt.exclusions["no_as_of_date"] > baseline.exclusions["no_as_of_date"]

    # Every retained cutoff is at or before the milestone it claims to be.
    for entry in rebuilt.entries:
        track = session.get(ProgramIndication, entry.program_indication_id)
        assert entry.as_of_date in {track.ind_or_cta_on, track.first_in_human_on}
        if track.phase_2_start_on is not None:
            assert entry.as_of_date < track.phase_2_start_on

    with pytest.raises(ValueError, match="Unknown as_of_rule"):
        build_cohort(session, as_of_rule="approval_on", persist=False)


# ─────────────────────────────────────────────────────────────────────────────
# 7. The temporal split keeps connected components whole
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_split_never_puts_one_component_in_both_train_and_test(seeded_session, today):
    """If this fails, the same asset trains the model that is then tested on it.

    One program developed in three indications is one bet, not three
    observations. If part of a connected component sits in training while the
    rest is scored in test, the held-out AUROC measures recognition of the
    program rather than assessment of its evidence — and the honest headline
    number (0.630 on the synthetic demonstration data, CI crossing the
    base rate) would be replaced by a flattering one that does not survive
    deployment.
    """
    session, _ = seeded_session
    cohort = build_cohort(session, data_cutoff=today, persist=False)
    entries = cohort.entries
    assert len(entries) >= 20

    keys = build_grouping_keys(entries)
    components = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.IDENTITY]).fit(keys)
    dates = [e.as_of_date for e in entries]

    ordinals = sorted(d.toordinal() for d in dates)
    train_end = dt.date.fromordinal(ordinals[int(0.6 * len(ordinals))])
    calibration_end = dt.date.fromordinal(ordinals[int(0.75 * len(ordinals))])

    split = temporal_split(dates, components, train_end=train_end,
                           calibration_end=calibration_end)
    sizes = split.sizes()
    assert sizes["train"] > 0 and sizes["test"] > 0

    train = set(split.train.tolist())
    calib = set(split.calibration.tolist())
    test = set(split.test.tolist())
    assert train | calib | test == set(range(len(entries)))
    assert not (train & calib) and not (train & test) and not (calib & test)

    train_comps = set(components[split.train].tolist())
    calib_comps = set(components[split.calibration].tolist())
    test_comps = set(components[split.test].tolist())
    assert train_comps & test_comps == set()
    assert train_comps & calib_comps == set()
    assert calib_comps & test_comps == set()

    # Every component lands in exactly one partition.
    partition_of = {}
    for name, idx in (("train", split.train), ("calibration", split.calibration),
                      ("test", split.test)):
        for i in idx.tolist():
            partition_of.setdefault(int(components[i]), name)
    for i in range(len(entries)):
        assert partition_of[int(components[i])] == (
            "train" if i in train else "calibration" if i in calib else "test"
        )

    audit = leakage_audit(components, split.train, split.test, keys)
    assert audit["clean"] is True
    assert audit["component"] == 0
    assert audit["program"] == 0
    assert audit["compound"] == 0
    assert leakage_audit(components, split.train, split.calibration, keys)["clean"] is True

    # Training as-of dates never reach past the boundary the split claims.
    assert max(dates[i] for i in train) <= train_end


def test_straddling_component_is_assigned_whole_to_the_later_partition():
    """If this fails, a component split across the boundary goes unnoticed.

    The seeded cohort may or may not contain a component that straddles a given
    date boundary, so the invariant is also exercised on a case constructed to
    straddle. The conservative direction matters: the whole component must move
    to the later partition, dropping rows from *training*, never pulling
    future-resolved siblings back into it.
    """
    train_end = dt.date(2015, 1, 1)
    calibration_end = dt.date(2016, 1, 1)
    dates = [
        dt.date(2012, 1, 1),   # component 0 — early
        dt.date(2018, 1, 1),   # component 0 — late: straddles
        dt.date(2013, 1, 1),   # component 1 — wholly in train
        dt.date(2019, 1, 1),   # component 2 — wholly in test
        dt.date(2015, 6, 1),   # component 3 — wholly in calibration
    ]
    components = np.asarray([0, 0, 1, 2, 3])

    split = temporal_split(dates, components, train_end=train_end,
                           calibration_end=calibration_end)

    assert split.train.tolist() == [2]
    assert split.calibration.tolist() == [4]
    assert sorted(split.test.tolist()) == [0, 1, 3]
    assert split.dropped["component_straddled_boundary"] == 1
    assert set(components[split.train].tolist()) & set(components[split.test].tolist()) == set()

    keys = [{"program": f"P{c}"} for c in components]
    assert leakage_audit(components, split.train, split.test, keys)["clean"] is True

    # Negative control: the naive date-only split of the same data IS dirty,
    # which is what the audit exists to catch.
    naive_train = np.asarray([0, 2])
    naive_test = np.asarray([1, 3])
    dirty = leakage_audit(components, naive_train, naive_test, keys)
    assert dirty["clean"] is False
    assert dirty["component"] == 1
    assert dirty["program"] == 1


def test_nested_temporal_folds_never_train_on_the_future(seeded_session, today):
    """If this fails, hyperparameters are tuned on the answers.

    Model selection is where leakage hides most comfortably: the penalty
    strength and the model family are chosen on these folds, so a fold that
    trains on rows dated after its own test window lets the future pick the
    model. The reported selection — L2 logistic at the chosen penalty, with
    gbdt and hierarchical_logistic disqualified on calibration slope — is only
    meaningful if every fold predicts strictly forward.
    """
    session, _ = seeded_session
    cohort = build_cohort(session, data_cutoff=today, persist=False)
    entries = cohort.entries
    keys = build_grouping_keys(entries)
    components = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.IDENTITY]).fit(keys)
    dates = [e.as_of_date for e in entries]

    folds = nested_temporal_folds(dates, components, n_folds=4)
    assert folds, "expected at least one expanding-window fold on the seeded cohort"

    for i, (train_idx, test_idx) in enumerate(folds):
        assert train_idx.size and test_idx.size
        latest_train = max(dates[j] for j in train_idx)
        earliest_test = min(dates[j] for j in test_idx)
        assert latest_train < earliest_test, f"fold {i} trains on or after its test window"

        train_comps = set(components[train_idx].tolist())
        test_comps = set(components[test_idx].tolist())
        assert train_comps & test_comps == set(), f"fold {i} shares a component"
        assert leakage_audit(components, train_idx, test_idx, keys)["clean"] is True

        # Expanding window: training never overlaps the test rows themselves.
        assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()
