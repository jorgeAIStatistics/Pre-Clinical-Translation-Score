"""Demonstration data generator.

READ THIS BEFORE INTERPRETING ANY OUTPUT OF THIS MODULE
--------------------------------------------------------
Everything generated here is synthetic. It exists so the platform can be run,
tested and demonstrated end to end before a real evidence base is assembled. It
is **not** a dataset about drug development and no coefficient fitted to it
says anything about real translational biology.

The data-generating process, in full
-------------------------------------
The simulation is written down rather than tuned by eye, so a reader can tell
exactly what a model trained on it is recovering.

**1. Latent quality.** Each development track gets a vector of latent
translational properties, drawn independently:

    genetic_support      ~ Categorical over evidence tiers, target-class-dependent
    exposure_margin      ~ LogNormal, modality-dependent
    model_fidelity       ~ Beta(2, 2)
    human_support        ~ Beta(1.5, 2.5)
    replication_breadth  ~ Poisson-ish count of independent laboratories
    pathway_redundancy   ~ Categorical, target-class-dependent
    safety_burden        ~ Gamma

**2. True failure log-odds.** A linear function of those latents, with
coefficients in :data:`TRUE_COEFFICIENTS`. This is the *only* place the outcome
is determined; nothing downstream peeks at it.

**3. Evidence as a noisy measurement.** Evidence items are then generated as
noisy, incomplete observations of the latents — not as the latents themselves.
A model fitted to the evidence therefore has to recover signal through
measurement error, which is why the achievable AUROC on this data sits in the
0.70–0.80 range rather than at 0.99. A demonstration dataset where the model
scores 0.97 would misrepresent what the product can do.

**4. Missingness is partly informative (MNAR).** Programs with poor exposure
margins disclose PK/PD detail less often; programs with weak replication
disclose lab identity less often. This is the single most realistic feature of
the simulation and it is what makes the missing-indicator columns earn their
place. It also means naive complete-case analysis on this data is biased, which
is the correct lesson.

**5. Competing outcomes.** A configurable share of tracks are discontinued for
safety, PK/PD, or commercial reasons. Commercial discontinuations are *not*
driven by the latent quality at all, which is exactly why the cohort builder
must exclude them: a model that learned to predict them would be predicting
corporate behaviour.

**6. Temporal structure.** Each track has an IND date between 2006 and 2020.
Evidence is dated before it, with a deliberate minority of items dated *after*
the IND date. Those post-cutoff items exist so the leakage test suite has
something real to catch: if the as-of filter ever breaks, they will move the
features and the test will fail.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from trs.db.models.core import (
    Compound,
    CompoundAlias,
    Indication,
    Program,
    ProgramAlias,
    ProgramIndication,
    ProgramSponsorHistory,
    Sponsor,
    Target,
    TargetAlias,
)
from trs.db.models.evidence import (
    EvidenceItem,
    PkPdBridge,
    ProgramBiomarker,
    SafetyFinding,
    Source,
    TargetAssessment,
)
from trs.db.models.tenancy import Organization, Project
from trs.ontology import enums as E
from trs.resolution.normalise import (
    normalise_compound,
    normalise_sponsor,
    normalise_target,
)
from trs.seed.vocabulary import (
    INDICATIONS,
    LABS,
    MODALITY_BY_TARGET_CLASS,
    SPONSORS,
    SYNTHETIC_NOTE,
    TARGET_INDICATION_AFFINITY,
    TARGETS,
)

# ─────────────────────────────────────────────────────────────────────────────
# The data-generating process
# ─────────────────────────────────────────────────────────────────────────────

#: Coefficients on the latent variables for the true failure log-odds.
#: Signs encode the simulation's premises; magnitudes are chosen so that no
#: single latent dominates.
#:
#: The intercept is set so the marginal Phase II efficacy-failure rate lands
#: near 0.60 given the *drawn means* of the latents — it is not a free
#: parameter. Summing coefficient × latent mean over the draws in
#: :meth:`DemoDataGenerator._draw_latents` gives roughly −2.08, so an intercept
#: of 2.45 centres the true log-odds near +0.37, i.e. p ≈ 0.59. That target is
#: chosen to sit in the range usually quoted for Phase II efficacy attrition in
#: oncology; it is a property of the simulation, not an estimate of it.
#: :func:`verify_base_rate` checks the realised rate after generation.
TRUE_COEFFICIENTS: dict[str, float] = {
    "intercept": 2.45,
    "genetic_support": -0.34,          # 0–5 tier scale
    "exposure_margin_log2": -0.30,     # log2 achievable/efficacious
    "model_fidelity": -1.10,           # 0–1
    "human_support": -1.35,            # 0–1
    "replication_labs": -0.22,         # count
    "pathway_redundancy": 0.42,        # 0–3
    "target_engagement": -0.21,        # 0–6 scale
    "biomarker_selected": -0.55,       # 0/1
    "safety_burden": 0.18,             # severity-weighted count
    "is_first_in_class": 0.30,         # 0/1
}

#: Additive modality effects on the true failure log-odds. Purely a property of
#: the simulation; they are *not* an assertion about real modality attrition.
MODALITY_EFFECT: dict[str, float] = {
    E.Modality.SMALL_MOLECULE: 0.00,
    E.Modality.MONOCLONAL_ANTIBODY: -0.18,
    E.Modality.BISPECIFIC_ANTIBODY: 0.12,
    E.Modality.ANTIBODY_DRUG_CONJUGATE: -0.05,
    E.Modality.PROTEIN_DEGRADER: 0.22,
    E.Modality.CELL_THERAPY_CAR_T: 0.05,
    E.Modality.RADIOPHARMACEUTICAL: 0.10,
    E.Modality.OLIGONUCLEOTIDE_ASO: 0.35,
    E.Modality.PEPTIDE: 0.28,
    E.Modality.THERAPEUTIC_PROTEIN: 0.08,
}

#: Probability of each terminal state, conditional on the drawn efficacy
#: outcome. Commercial discontinuation is independent of latent quality.
P_SAFETY_FAILURE = 0.11
P_PKPD_FAILURE = 0.05
P_COMMERCIAL = 0.13
P_STILL_ACTIVE_RECENT = 0.30   # applied only to tracks with recent IND dates

GENETIC_TIERS = (
    E.GeneticEvidenceTier.NO_HUMAN_GENETIC_SIGNAL,
    E.GeneticEvidenceTier.EXPRESSION_QTL_ONLY,
    E.GeneticEvidenceTier.GWAS_LOCUS_ONLY,
    E.GeneticEvidenceTier.GWAS_FINE_MAPPED,
    E.GeneticEvidenceTier.SOMATIC_DRIVER,
    E.GeneticEvidenceTier.RARE_CODING_BURDEN,
    E.GeneticEvidenceTier.MENDELIAN_CAUSAL,
)
GENETIC_TIER_VALUE = {t: v for t, v in zip(GENETIC_TIERS, (0.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0), strict=True)}


@dataclass(slots=True)
class Latents:
    """The unobserved truth for one development track."""

    genetic_tier: str
    genetic_value: float
    exposure_margin_log2: float
    therapeutic_index_log2: float
    model_fidelity: float
    human_support: float
    replication_labs: int
    pathway_redundancy: int
    target_engagement: float
    biomarker_selected: bool
    safety_burden: float
    is_first_in_class: bool
    modality: str

    def true_failure_logit(self) -> float:
        c = TRUE_COEFFICIENTS
        return (
            c["intercept"]
            + c["genetic_support"] * self.genetic_value
            + c["exposure_margin_log2"] * self.exposure_margin_log2
            + c["model_fidelity"] * self.model_fidelity
            + c["human_support"] * self.human_support
            + c["replication_labs"] * self.replication_labs
            + c["pathway_redundancy"] * self.pathway_redundancy
            + c["target_engagement"] * self.target_engagement
            + c["biomarker_selected"] * float(self.biomarker_selected)
            + c["safety_burden"] * self.safety_burden
            + c["is_first_in_class"] * float(self.is_first_in_class)
            + MODALITY_EFFECT.get(self.modality, 0.0)
        )


@dataclass
class SeedResult:
    organization: Organization
    project: Project
    n_targets: int = 0
    n_indications: int = 0
    n_sponsors: int = 0
    n_programs: int = 0
    n_tracks: int = 0
    n_evidence: int = 0
    n_sources: int = 0
    n_bridges: int = 0
    n_assessments: int = 0
    n_biomarkers: int = 0
    n_safety: int = 0
    outcome_counts: dict[str, int] = field(default_factory=dict)
    #: {track_id: true failure probability} from the simulation. Diagnostic only.
    ground_truth: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k not in ("organization", "project", "ground_truth")
        }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _date_between(rng: np.random.Generator, start: dt.date, end: dt.date) -> dt.date:
    span = max((end - start).days, 1)
    return start + dt.timedelta(days=int(rng.integers(0, span)))





# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class DemoDataGenerator:
    """Builds the full demonstration universe in one transaction."""

    def __init__(self, session: Session, *, seed: int = 20240917) -> None:
        self.session = session
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.result: SeedResult | None = None
        self._targets: dict[str, Target] = {}
        #: Per-target observable truth, shared between the assessment writer and
        #: the outcome model so the two cannot drift apart.
        self._target_truth: dict[str, dict[str, Any]] = {}
        #: {track_id: true failure probability}. Never written to the database —
        #: a real deployment has no ground-truth column — but returned so the
        #: oracle benchmark can be computed.
        self.ground_truth: dict[str, float] = {}
        self._indications: dict[str, Indication] = {}
        self._sponsors: list[Sponsor] = []
        self._sources: list[Source] = []

    # ---- orchestration -----------------------------------------------------

    def run(self, *, n_programs: int = 200) -> SeedResult:
        org, project = self._create_tenant()
        self.result = SeedResult(organization=org, project=project)

        self._create_targets()
        self._create_indications()
        self._create_sponsors()
        self._create_sources(n=90)
        self._create_target_assessments()
        self._create_programs(project, n_programs)

        self.session.flush()
        self.result.ground_truth = dict(self.ground_truth)
        return self.result

    # ---- tenancy -----------------------------------------------------------

    def _create_tenant(self) -> tuple[Organization, Project]:
        org = Organization(
            name="Demonstration Organisation",
            slug="demo",
            settings={"synthetic": True, "note": SYNTHETIC_NOTE},
        )
        self.session.add(org)
        self.session.flush()
        project = Project(
            organization_id=org.id,
            name="Oncology Translational Portfolio (Demo)",
            slug="oncology-demo",
            description=(
                "Demonstration portfolio of simulated oncology programs. "
                + SYNTHETIC_NOTE
            ),
            is_private=False,
        )
        self.session.add(project)
        self.session.flush()
        return org, project

    # ---- reference entities ------------------------------------------------

    def _create_targets(self) -> None:
        for symbol, name, tclass, identity in TARGETS:
            t = Target(
                symbol=symbol, name=name, target_class=str(tclass),
                ortholog_identity_mouse=identity,
                is_synthetic=False,  # gene symbols are real public vocabulary
                description=(
                    "Real gene symbol used as controlled vocabulary. Program-level "
                    "records referencing it in this deployment are synthetic."
                ),
                external_refs={"vocabulary": "HGNC symbol"},
            )
            self.session.add(t)
            self.session.flush()
            self.session.add(TargetAlias(
                target_id=t.id, alias=symbol, alias_normalised=normalise_target(symbol),
                alias_type="symbol", match_method=E.EntityMatchMethod.EXACT_IDENTIFIER,
                match_confidence=1.0, is_primary=True,
            ))
            self._targets[symbol] = t
        self.result.n_targets = len(self._targets)

    def _create_indications(self) -> None:
        for name, area, onto, subtype in INDICATIONS:
            ind = Indication(
                name=name, therapeutic_area=str(area), ontology_id=onto,
                ontology_source="EFO", molecular_subtype=subtype, is_synthetic=False,
            )
            self.session.add(ind)
            self.session.flush()
            self._indications[name] = ind
        self.result.n_indications = len(self._indications)

    def _create_sponsors(self) -> None:
        for name, stype in SPONSORS:
            s = Sponsor(
                name=name, name_normalised=normalise_sponsor(name), sponsor_type=stype,
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            )
            self.session.add(s)
            self.session.flush()
            self._sponsors.append(s)
        self.result.n_sponsors = len(self._sponsors)

    def _create_sources(self, n: int) -> None:
        """Synthetic citations, explicitly typed as such.

        Source rows carry ``source_type = synthetic_demonstration`` and no PMID
        or DOI. Fabricating a plausible-looking PMID would be indistinguishable
        from a real citation once it reached the interface, which is precisely
        the failure mode this platform must not have.
        """
        for i in range(n):
            year = int(self.rng.integers(2002, 2022))
            s = Source(
                source_type=E.SourceType.SYNTHETIC_DEMONSTRATION,
                title=f"[SYNTHETIC RECORD {i + 1:03d}] Simulated preclinical study report",
                container_title="Synthetic Demonstration Corpus",
                published_on=dt.date(year, int(self.rng.integers(1, 13)), 1),
                publication_status=str(
                    self.rng.choice(
                        [E.PublicationStatus.PEER_REVIEWED, E.PublicationStatus.PREPRINT,
                         E.PublicationStatus.CONFERENCE_ABSTRACT,
                         E.PublicationStatus.COMPANY_DISCLOSURE],
                        p=[0.62, 0.14, 0.16, 0.08],
                    )
                ),
                url=None, pmid=None, doi=None,
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
                abstract=(
                    "This record does not correspond to any real publication. It "
                    "exists so the demonstration evidence base has a provenance "
                    "anchor with the same shape a real source would have."
                ),
            )
            self.session.add(s)
            self._sources.append(s)
        self.session.flush()
        self.result.n_sources = len(self._sources)

    def _create_target_assessments(self) -> None:
        """Time-varying target assessments, so target knowledge accumulates.

        These rows are the *observable* form of the target-level latents. The
        latent genetic tier and pathway redundancy used to generate outcomes are
        read back from here by :meth:`_draw_latents` rather than drawn again
        independently. Drawing them twice would make two of the largest terms in
        the true failure log-odds unobservable in principle — the model would be
        asked to recover signal that appears in no feature, and the
        demonstration would understate the pipeline for reasons that have
        nothing to do with the pipeline.
        """
        for symbol, target in self._targets.items():
            base_tier_idx = int(self.rng.integers(0, len(GENETIC_TIERS)))
            n_versions = int(self.rng.integers(2, 5))
            years = sorted(self.rng.choice(range(2004, 2022), size=n_versions, replace=False))
            approved = 0
            prior_programs = 0
            prior_failures = 0
            for k, year in enumerate(years):
                # Knowledge improves: tiers can strengthen over time, not weaken.
                tier_idx = min(len(GENETIC_TIERS) - 1, base_tier_idx + (1 if k >= 2 else 0))
                prior_programs += int(self.rng.integers(0, 4))
                prior_failures += int(self.rng.integers(0, max(1, prior_programs // 2 + 1)))
                if k >= 2 and self.rng.random() < 0.25:
                    approved += 1
                if symbol not in self._target_truth:
                    self._target_truth[symbol] = {
                        "redundancy": str(self.rng.choice([
                            E.PathwayRedundancy.NO_KNOWN_REDUNDANCY,
                            E.PathwayRedundancy.LIMITED_REDUNDANCY,
                            E.PathwayRedundancy.SUBSTANTIAL_REDUNDANCY,
                            E.PathwayRedundancy.DOCUMENTED_RESISTANCE_BYPASS,
                        ], p=[0.18, 0.38, 0.30, 0.14])),
                    }
                # The assessed tier and redundancy ARE this target's truth; the
                # outcome model reads them back so the features can recover them.
                self._target_truth[symbol].update({
                    "genetic_tier": str(GENETIC_TIERS[tier_idx]),
                    "approved": approved,
                    "prior_programs": prior_programs,
                })
                self.session.add(TargetAssessment(
                    target_id=target.id,
                    indication_id=None,
                    available_at=dt.date(int(year), 6, 30),
                    genetic_evidence_tier=str(GENETIC_TIERS[tier_idx]),
                    association_score=float(np.clip(self.rng.beta(2.2, 2.0), 0.02, 0.99)),
                    causal_support=str(self.rng.choice([
                        E.CausalSupport.CORRELATIONAL_ONLY, E.CausalSupport.ANIMAL_INTERVENTIONAL,
                        E.CausalSupport.MENDELIAN_RANDOMISATION, E.CausalSupport.HUMAN_GENETIC_CAUSAL,
                        E.CausalSupport.INTERVENTIONAL_HUMAN,
                    ], p=[0.30, 0.34, 0.12, 0.16, 0.08])),
                    pathway_redundancy=self._target_truth[symbol]["redundancy"],
                    human_tissue_expression_evidence=str(self.rng.choice(
                        ["low", "moderate", "high"], p=[0.2, 0.45, 0.35]
                    )),
                    prior_clinical_programs_on_target=prior_programs,
                    prior_approved_drugs_on_target=approved,
                    prior_ph2_failures_on_target=min(prior_failures, prior_programs),
                    provenance_method=E.ProvenanceMethod.SYNTHETIC_GENERATOR,
                    is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
                ))
                self.result.n_assessments += 1

    # ---- programs ----------------------------------------------------------

    def _draw_latents(self, target: Target, modality: str, ind_name: str) -> Latents:
        rng = self.rng
        # Target-level properties are read from the target's assessment history
        # rather than re-drawn, so the features can observe them. See
        # _create_target_assessments for why this matters.
        truth = self._target_truth.get(target.symbol, {})
        tier = truth.get(
            "genetic_tier",
            str(rng.choice(GENETIC_TIERS, p=[0.16, 0.10, 0.16, 0.18, 0.20, 0.12, 0.08])),
        )
        redundancy_map = {
            str(E.PathwayRedundancy.NO_KNOWN_REDUNDANCY): 0,
            str(E.PathwayRedundancy.LIMITED_REDUNDANCY): 1,
            str(E.PathwayRedundancy.SUBSTANTIAL_REDUNDANCY): 2,
            str(E.PathwayRedundancy.DOCUMENTED_RESISTANCE_BYPASS): 3,
        }
        redundancy = redundancy_map.get(truth.get("redundancy", ""), int(rng.integers(0, 4)))
        first_in_class = truth.get("approved", 0) == 0
        # Exposure margin: biologics generally achieve their target exposure more
        # predictably in this simulation than small molecules do.
        loc = 0.9 if modality in (
            E.Modality.MONOCLONAL_ANTIBODY, E.Modality.BISPECIFIC_ANTIBODY,
            E.Modality.THERAPEUTIC_PROTEIN,
        ) else 0.15
        return Latents(
            genetic_tier=tier,
            genetic_value=GENETIC_TIER_VALUE[tier],
            exposure_margin_log2=float(rng.normal(loc, 1.6)),
            therapeutic_index_log2=float(rng.normal(2.0, 1.8)),
            model_fidelity=float(rng.beta(2.0, 2.0)),
            human_support=float(rng.beta(1.5, 2.5)),
            replication_labs=int(rng.poisson(1.1)),
            pathway_redundancy=redundancy,
            target_engagement=float(rng.choice([0, 1, 2, 3, 5, 6], p=[0.08, 0.14, 0.26, 0.32, 0.14, 0.06])),
            biomarker_selected=bool("mutant" in ind_name.lower() or "+" in ind_name
                                    or rng.random() < 0.28),
            safety_burden=float(rng.gamma(1.6, 0.9)),
            is_first_in_class=first_in_class,
            modality=modality,
        )

    def _create_programs(self, project: Project, n_programs: int) -> None:
        rng = self.rng
        symbols = [s for s, *_ in TARGETS]
        data_cutoff = dt.date(2026, 1, 1)

        for i in range(n_programs):
            symbol = str(rng.choice(symbols))
            target = self._targets[symbol]
            modalities = MODALITY_BY_TARGET_CLASS.get(
                target.target_class, (E.Modality.SMALL_MOLECULE,)
            )
            modality = str(rng.choice(modalities))
            sponsor = self._sponsors[int(rng.integers(0, len(self._sponsors)))]

            code = f"DEMO-{symbol}-{i + 1:03d}"
            compound = Compound(
                preferred_name=code, modality=modality,
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            )
            self.session.add(compound)
            self.session.flush()
            self.session.add(CompoundAlias(
                compound_id=compound.id, alias=code, alias_normalised=normalise_compound(code),
                alias_type="development_code",
                match_method=E.EntityMatchMethod.EXACT_IDENTIFIER, is_primary=True,
            ))

            first_disclosure = _date_between(rng, dt.date(2003, 1, 1), dt.date(2018, 6, 1))
            program = Program(
                organization_id=self.result.organization.id,
                project_id=project.id,
                name=code, code=code,
                primary_compound_id=compound.id,
                primary_target_id=target.id,
                originator_sponsor_id=sponsor.id,
                current_sponsor_id=sponsor.id,
                modality=modality,
                mechanism_of_action=_mechanism_text(symbol, modality),
                first_disclosure_on=first_disclosure,
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            )
            self.session.add(program)
            self.session.flush()

            # Alias graph: a discovery code and, sometimes, a later INN-style
            # name under a second sponsor. This is what the entity-resolution
            # and grouped-splitting logic must survive.
            for alias, atype in (
                (code, "development_code"),
                (f"{code.replace('DEMO-', 'DM')}", "internal_code"),
            ):
                self.session.add(ProgramAlias(
                    program_id=program.id, alias=alias,
                    alias_normalised=normalise_compound(alias), alias_type=atype,
                    match_method=E.EntityMatchMethod.EXACT_IDENTIFIER,
                    sponsor_id=sponsor.id, available_at=first_disclosure,
                    is_primary=(atype == "development_code"),
                ))
            self.session.add(ProgramSponsorHistory(
                program_id=program.id, sponsor_id=sponsor.id,
                role="originator", start_on=first_disclosure,
            ))
            if rng.random() < 0.18:
                new_sponsor = self._sponsors[int(rng.integers(0, len(self._sponsors)))]
                if new_sponsor.id != sponsor.id:
                    handover = first_disclosure + dt.timedelta(days=int(rng.integers(500, 2200)))
                    program.current_sponsor_id = new_sponsor.id
                    self.session.add(ProgramSponsorHistory(
                        program_id=program.id, sponsor_id=new_sponsor.id,
                        role="developer", start_on=handover,
                    ))
                    inn = f"{symbol.lower()}{'mab' if 'antibody' in modality else 'nib'}-{i + 1:03d}"
                    self.session.add(ProgramAlias(
                        program_id=program.id, alias=inn,
                        alias_normalised=normalise_compound(inn), alias_type="inn_style",
                        match_method=E.EntityMatchMethod.REGISTERED_ALIAS,
                        match_confidence=0.9, sponsor_id=new_sponsor.id,
                        available_at=handover,
                    ))

            # One to three indications per program.
            preferred = TARGET_INDICATION_AFFINITY.get(symbol)
            pool = [n for n in (preferred or ()) if n in self._indications] or list(self._indications)
            n_tracks = min(len(pool), int(rng.choice([1, 1, 2, 3], p=[0.52, 0.18, 0.22, 0.08])))
            chosen = list(rng.choice(pool, size=n_tracks, replace=False))

            for ind_name in chosen:
                self._create_track(
                    program, target, self._indications[str(ind_name)], modality,
                    first_disclosure, data_cutoff,
                )
            self.result.n_programs += 1

    def _create_track(
        self,
        program: Program,
        target: Target,
        indication: Indication,
        modality: str,
        first_disclosure: dt.date,
        data_cutoff: dt.date,
    ) -> None:
        rng = self.rng
        latents = self._draw_latents(target, modality, indication.name)

        ind_date = first_disclosure + dt.timedelta(days=int(rng.integers(700, 2600)))
        if ind_date > dt.date(2021, 12, 31):
            ind_date = _date_between(rng, dt.date(2014, 1, 1), dt.date(2021, 12, 31))
        fih_date = ind_date + dt.timedelta(days=int(rng.integers(60, 400)))

        p_fail = _sigmoid(latents.true_failure_logit())
        outcome, reason, outcome_date, highest_phase = self._draw_outcome(
            p_fail, ind_date, data_cutoff
        )
        milestones = self._milestones(highest_phase, ind_date, fih_date, outcome_date)

        track = ProgramIndication(
            program_id=program.id,
            indication_id=indication.id,
            disease_stage=str(rng.choice([
                E.DiseaseStage.METASTATIC, E.DiseaseStage.RELAPSED_REFRACTORY,
                E.DiseaseStage.LOCALLY_ADVANCED, E.DiseaseStage.ADJUVANT,
            ], p=[0.46, 0.34, 0.14, 0.06])),
            line_of_therapy=str(rng.choice(
                ["first_line", "second_line", "third_line_plus"], p=[0.3, 0.38, 0.32]
            )),
            biomarker_selection=str(
                E.BiomarkerSelectionStrategy.REQUIRED_GENOMIC_ALTERATION
                if latents.biomarker_selected and indication.molecular_subtype
                else (
                    E.BiomarkerSelectionStrategy.ENRICHED_NOT_REQUIRED
                    if latents.biomarker_selected
                    else E.BiomarkerSelectionStrategy.ALL_COMERS
                )
            ),
            unmet_need=str(rng.choice([
                E.UnmetNeed.NO_APPROVED_THERAPY, E.UnmetNeed.LIMITED_EFFICACY_SOC,
                E.UnmetNeed.EFFECTIVE_SOC_TOLERABILITY_GAP, E.UnmetNeed.CROWDED_EFFECTIVE_SOC,
            ], p=[0.18, 0.44, 0.24, 0.14])),
            combination_partner=(
                str(rng.choice(["platinum doublet", "anti-PD-1", "endocrine therapy"]))
                if rng.random() < 0.22 else None
            ),
            preclinical_start_on=first_disclosure,
            ind_or_cta_on=ind_date,
            first_in_human_on=fih_date,
            phase_2_start_on=milestones["phase_2_start_on"],
            phase_3_start_on=milestones["phase_3_start_on"],
            approval_on=milestones["approval_on"],
            highest_phase_reached=highest_phase,
            outcome_label=outcome,
            outcome_date=outcome_date,
            discontinuation_reason=reason,
            outcome_confidence=str(rng.choice(
                [E.ConfidenceLevel.HIGH, E.ConfidenceLevel.MODERATE, E.ConfidenceLevel.LOW],
                p=[0.55, 0.33, 0.12],
            )),
            status_verified_on=data_cutoff,
            is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
        )
        self.session.add(track)
        self.session.flush()
        self.ground_truth[track.id] = p_fail
        self.result.n_tracks += 1
        self.result.outcome_counts[outcome] = self.result.outcome_counts.get(outcome, 0) + 1

        self._create_evidence(track, program, target, indication, latents, ind_date)
        self._create_pkpd(track, latents, ind_date)
        self._create_biomarkers(track, latents, ind_date)
        self._create_safety(track, program, latents, ind_date)

    def _milestones(
        self,
        highest_phase: str,
        ind_date: dt.date,
        fih_date: dt.date,
        outcome_date: dt.date | None,
    ) -> dict[str, dt.date | None]:
        """Derive the milestone chain implied by the highest phase reached.

        The dates are generated rather than drawn independently so the chain is
        always internally consistent: a program that reached Phase III has a
        Phase II start, and every milestone falls strictly after the one before
        it and no later than the outcome that ended the program. An
        inconsistent chain would make the alternative as-of rules
        (`first_in_human_on`, `phase_2_start_on`) produce cutoffs that precede
        the evidence they are supposed to admit.
        """
        rng = self.rng
        order = [
            E.DevelopmentPhase.PHASE_1, E.DevelopmentPhase.PHASE_1_2,
            E.DevelopmentPhase.PHASE_2, E.DevelopmentPhase.PHASE_2_3,
            E.DevelopmentPhase.PHASE_3, E.DevelopmentPhase.FILED,
            E.DevelopmentPhase.APPROVED,
        ]
        rank = {str(p): i for i, p in enumerate(order)}
        reached = rank.get(str(highest_phase), -1)
        ceiling = outcome_date or (fih_date + dt.timedelta(days=4000))

        out: dict[str, dt.date | None] = {
            "phase_2_start_on": None, "phase_3_start_on": None, "approval_on": None,
        }

        def _after(prev: dt.date, lo: int, hi: int) -> dt.date | None:
            candidate = prev + dt.timedelta(days=int(rng.integers(lo, hi)))
            # A milestone that would postdate the program's own ending never
            # happened; returning None is correct, not a gap to be filled.
            return candidate if candidate < ceiling else None

        if reached >= rank[str(E.DevelopmentPhase.PHASE_2)]:
            out["phase_2_start_on"] = _after(fih_date, 240, 900)

        if reached >= rank[str(E.DevelopmentPhase.PHASE_3)] and out["phase_2_start_on"]:
            out["phase_3_start_on"] = _after(out["phase_2_start_on"], 400, 1300)

        if reached >= rank[str(E.DevelopmentPhase.APPROVED)]:
            # Approval IS the outcome date for an approved program; using a
            # separately drawn date would let the two disagree.
            out["approval_on"] = outcome_date
            if out["phase_3_start_on"] is None and out["phase_2_start_on"]:
                out["phase_3_start_on"] = _after(out["phase_2_start_on"], 400, 1100)
        return out

    def _draw_outcome(
        self, p_fail: float, ind_date: dt.date, data_cutoff: dt.date
    ) -> tuple[str, str, dt.date | None, str]:
        rng = self.rng
        years_observed = (data_cutoff - ind_date).days / 365.25

        # Recent programs may simply not have resolved yet.
        if years_observed < 5 and rng.random() < P_STILL_ACTIVE_RECENT:
            return (
                str(E.OutcomeLabel.ACTIVE_NO_OUTCOME_YET),
                str(E.DiscontinuationReason.NOT_DISCONTINUED),
                None,
                str(rng.choice([E.DevelopmentPhase.PHASE_1, E.DevelopmentPhase.PHASE_2])),
            )

        u = rng.random()
        if u < P_COMMERCIAL:
            # Independent of latent quality by construction — this is the case
            # the cohort builder must exclude.
            return (
                str(E.OutcomeLabel.DISCONTINUED_THROUGH_PHASE_2),
                str(rng.choice([
                    E.DiscontinuationReason.COMMERCIAL_STRATEGIC,
                    E.DiscontinuationReason.PORTFOLIO_REPRIORITISATION,
                    E.DiscontinuationReason.SPONSOR_INSOLVENCY,
                ], p=[0.5, 0.35, 0.15])),
                ind_date + dt.timedelta(days=int(rng.integers(500, 2200))),
                str(rng.choice([E.DevelopmentPhase.PHASE_1, E.DevelopmentPhase.PHASE_2])),
            )
        if u < P_COMMERCIAL + P_SAFETY_FAILURE:
            return (
                str(E.OutcomeLabel.PHASE_1_SAFETY_FAILURE),
                str(E.DiscontinuationReason.SAFETY_TOXICITY),
                ind_date + dt.timedelta(days=int(rng.integers(300, 1300))),
                str(E.DevelopmentPhase.PHASE_1),
            )
        if u < P_COMMERCIAL + P_SAFETY_FAILURE + P_PKPD_FAILURE:
            return (
                str(E.OutcomeLabel.PHASE_1_PKPD_FAILURE),
                str(E.DiscontinuationReason.PK_PD_INADEQUATE),
                ind_date + dt.timedelta(days=int(rng.integers(300, 1200))),
                str(E.DevelopmentPhase.PHASE_1),
            )

        if rng.random() < p_fail:
            return (
                str(E.OutcomeLabel.PHASE_2_EFFICACY_FAILURE),
                str(E.DiscontinuationReason.EFFICACY_INSUFFICIENT),
                ind_date + dt.timedelta(days=int(rng.integers(900, 2600))),
                str(E.DevelopmentPhase.PHASE_2),
            )

        # Progressed past Phase II; a minority reach approval.
        if rng.random() < 0.34 and years_observed > 8:
            return (
                str(E.OutcomeLabel.REGULATORY_APPROVAL),
                str(E.DiscontinuationReason.NOT_DISCONTINUED),
                ind_date + dt.timedelta(days=int(rng.integers(2200, 3800))),
                str(E.DevelopmentPhase.APPROVED),
            )
        return (
            str(E.OutcomeLabel.PROGRESSED_PAST_PHASE_2),
            str(E.DiscontinuationReason.NOT_DISCONTINUED),
            ind_date + dt.timedelta(days=int(rng.integers(1100, 2600))),
            str(E.DevelopmentPhase.PHASE_3),
        )

    # ---- evidence ----------------------------------------------------------

    def _source(self) -> Source:
        return self._sources[int(self.rng.integers(0, len(self._sources)))]

    def _create_evidence(
        self,
        track: ProgramIndication,
        program: Program,
        target: Target,
        indication: Indication,
        latents: Latents,
        ind_date: dt.date,
    ) -> None:
        rng = self.rng
        # Evidence volume correlates weakly with quality (better programs tend
        # to attract more work) but is dominated by noise, as in reality.
        n_invivo = max(1, int(rng.poisson(1.5 + 3.0 * latents.model_fidelity)))
        n_human = int(rng.poisson(0.6 + 3.2 * latents.human_support))
        n_invitro = int(rng.poisson(2.0))

        lab_pool = list(rng.choice(
            LABS, size=max(1, latents.replication_labs + 1), replace=False
        ))
        species_pool = [E.Species.MOUSE]
        if rng.random() < 0.25 + 0.3 * latents.model_fidelity:
            species_pool.append(E.Species.RAT)
        if rng.random() < 0.06 + 0.18 * latents.model_fidelity:
            species_pool.append(E.Species.NON_HUMAN_PRIMATE)
        if rng.random() < 0.05:
            species_pool.append(E.Species.DOG)

        for k in range(n_invivo):
            # A minority of evidence is dated AFTER the IND date on purpose:
            # the as-of filter must exclude it, and the leakage tests check that.
            post_cutoff = rng.random() < 0.15
            if post_cutoff:
                available = ind_date + dt.timedelta(days=int(rng.integers(30, 1500)))
            else:
                available = ind_date - dt.timedelta(days=int(rng.integers(30, 2200)))

            # P(supporting) tracks model fidelity — this is the measurement model.
            p_support = 0.35 + 0.55 * latents.model_fidelity
            supports = rng.random() < p_support
            species = str(rng.choice(species_pool))
            fidelity_draw = rng.random()
            if fidelity_draw < latents.model_fidelity:
                model_type = str(rng.choice([
                    E.ModelType.PATIENT_DERIVED_XENOGRAFT, E.ModelType.GENETICALLY_ENGINEERED,
                    E.ModelType.ORTHOTOPIC,
                ]))
                subtype = str(rng.choice([
                    E.OncologyModelSubtype.PDX_ORTHOTOPIC,
                    E.OncologyModelSubtype.GEMM_AUTOCHTHONOUS,
                    E.OncologyModelSubtype.PDX_SUBCUTANEOUS,
                ]))
                construct = str(rng.choice(["high", "moderate"], p=[0.62, 0.38]))
            else:
                model_type = str(rng.choice([E.ModelType.XENOGRAFT, E.ModelType.SYNGENEIC]))
                subtype = str(rng.choice([
                    E.OncologyModelSubtype.CELL_LINE_DERIVED_XENOGRAFT_SUBQ,
                    E.OncologyModelSubtype.SYNGENEIC_SUBCUTANEOUS,
                ]))
                construct = str(rng.choice(["moderate", "low"], p=[0.45, 0.55]))

            tgi = float(np.clip(rng.normal(20 + 70 * latents.model_fidelity, 18), -40, 99))
            if not supports:
                tgi = float(np.clip(rng.normal(8, 14), -40, 45))

            # MNAR: low-replication programs report lab identity less often.
            report_lab = rng.random() < (0.45 + 0.2 * min(latents.replication_labs, 3))

            self.session.add(EvidenceItem(
                organization_id=self.result.organization.id,
                program_id=program.id, program_indication_id=track.id,
                target_id=target.id, indication_id=(
                    indication.id if rng.random() < 0.55 + 0.35 * latents.model_fidelity else None
                ),
                evidence_class=str(E.EvidenceClass.ANIMAL_IN_VIVO),
                direction=str(E.EvidenceDirection.SUPPORTS if supports else E.EvidenceDirection.REFUTES),
                title=f"[SYNTHETIC] In vivo efficacy study {k + 1} — {indication.name}",
                summary=(
                    f"Simulated {model_type.replace('_', ' ')} study in {species.replace('_', ' ')}; "
                    f"{'supporting' if supports else 'non-supporting'} result."
                ),
                species=species,
                biological_system=str(E.BiologicalSystem.IN_VIVO_WHOLE_ANIMAL),
                model_type=model_type, model_subtype=subtype,
                model_identifier=f"MODEL-{int(rng.integers(100, 999))}",
                immune_competence=str(
                    E.ImmuneCompetence.IMMUNODEFICIENT
                    if model_type == E.ModelType.PATIENT_DERIVED_XENOGRAFT
                    else rng.choice([
                        E.ImmuneCompetence.FULLY_IMMUNOCOMPETENT,
                        E.ImmuneCompetence.IMMUNODEFICIENT,
                        E.ImmuneCompetence.HUMANISED_IMMUNE_SYSTEM,
                    ], p=[0.45, 0.42, 0.13])
                ),
                construct_validity=construct,
                construct_validity_assessor="synthetic-generator-rubric-v1",
                disease_similarity=str(rng.choice(["high", "moderate", "low"],
                                                  p=[0.25, 0.5, 0.25])),
                endpoint_category=str(E.EndpointCategory.TUMOUR_GROWTH_INHIBITION),
                endpoint_description="Tumour growth inhibition versus vehicle control",
                effect_metric=str(E.EffectMetric.PERCENT_TGI),
                effect_value=tgi,
                effect_favours_treatment=True,
                n_per_arm=int(rng.integers(5, 15)),
                n_arms=int(rng.integers(2, 5)),
                study_duration_days=int(rng.integers(14, 90)),
                dose_value=float(np.round(rng.uniform(1, 100), 1)),
                dose_unit="mg/kg",
                dose_schedule=str(rng.choice(["QD", "BID", "Q3D", "QW"])),
                route=str(rng.choice(["PO", "IP", "IV"])),
                randomization=str(rng.choice([
                    E.RandomizationLevel.RANDOMISED, E.RandomizationLevel.NOT_RANDOMISED,
                    E.RandomizationLevel.NOT_REPORTED,
                ], p=[0.42, 0.18, 0.40])),
                blinding=str(rng.choice([
                    E.BlindingLevel.OUTCOME_ASSESSOR_BLINDED, E.BlindingLevel.NOT_BLINDED,
                    E.BlindingLevel.NOT_REPORTED,
                ], p=[0.24, 0.20, 0.56])),
                replication_scope=str(
                    E.ReplicationScope.INDEPENDENT_LABS_MULTIPLE if latents.replication_labs >= 2
                    else (E.ReplicationScope.INDEPENDENT_LAB_SINGLE if latents.replication_labs == 1
                          else E.ReplicationScope.SINGLE_STUDY)
                ),
                independent_replications=latents.replication_labs,
                lab_identifier=str(rng.choice(lab_pool)) if report_lab else None,
                selective_reporting_risk=str(rng.choice(
                    ["low", "moderate", "high", "not_assessed"], p=[0.2, 0.3, 0.14, 0.36]
                )),
                conflict_of_interest=str(rng.choice([
                    E.ConflictOfInterest.SPONSOR_AFFILIATED, E.ConflictOfInterest.ACADEMIC_ONLY,
                    E.ConflictOfInterest.NOT_REPORTED,
                ], p=[0.44, 0.23, 0.33])),
                publication_status=str(rng.choice([
                    E.PublicationStatus.PEER_REVIEWED, E.PublicationStatus.CONFERENCE_ABSTRACT,
                    E.PublicationStatus.INTERNAL_UNPUBLISHED,
                ], p=[0.58, 0.2, 0.22])),
                prospective_protocol=bool(rng.random() < 0.25) if rng.random() < 0.6 else None,
                preregistered=bool(rng.random() < 0.08) if rng.random() < 0.5 else None,
                sample_size_justified=bool(rng.random() < 0.22) if rng.random() < 0.55 else None,
                source_id=self._source().id,
                available_at=available,
                observed_on=available - dt.timedelta(days=int(rng.integers(60, 400))),
                extraction_date=dt.date(2026, 1, 15),
                provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
                extraction_confidence=str(E.ConfidenceLevel.NOT_ASSESSED),
                source_snippet="[synthetic record — no source text exists]",
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            ))
            self.result.n_evidence += 1

        human_classes = (
            E.EvidenceClass.HUMAN_TISSUE_MOLECULAR, E.EvidenceClass.ORGANOID,
            E.EvidenceClass.PRIMARY_HUMAN_CELL, E.EvidenceClass.HUMAN_EX_VIVO,
            E.EvidenceClass.HUMAN_GENETIC,
        )
        for k in range(n_human):
            available = ind_date - dt.timedelta(days=int(rng.integers(60, 2400)))
            supports = rng.random() < (0.4 + 0.5 * latents.human_support)
            cls = str(rng.choice(human_classes, p=[0.3, 0.2, 0.22, 0.13, 0.15]))
            self.session.add(EvidenceItem(
                organization_id=self.result.organization.id,
                program_id=program.id, program_indication_id=track.id,
                target_id=target.id, indication_id=indication.id,
                evidence_class=cls,
                direction=str(E.EvidenceDirection.SUPPORTS if supports else E.EvidenceDirection.REFUTES),
                title=f"[SYNTHETIC] Human-derived evidence {k + 1} — {cls.replace('_', ' ')}",
                species=str(E.Species.HUMAN),
                biological_system=str(
                    E.BiologicalSystem.ORGANOID if cls == E.EvidenceClass.ORGANOID
                    else E.BiologicalSystem.EX_VIVO_TISSUE
                ),
                model_type=str(E.ModelType.NOT_APPLICABLE),
                model_subtype=str(E.OncologyModelSubtype.NOT_APPLICABLE),
                immune_competence=str(E.ImmuneCompetence.NOT_APPLICABLE),
                construct_validity=str(rng.choice(["high", "moderate"], p=[0.5, 0.5])),
                disease_similarity="high",
                endpoint_category=str(rng.choice([
                    E.EndpointCategory.CELL_VIABILITY, E.EndpointCategory.BIOMARKER_MODULATION,
                    E.EndpointCategory.TARGET_ENGAGEMENT,
                ])),
                effect_metric=str(E.EffectMetric.FOLD_CHANGE),
                effect_value=float(np.clip(rng.lognormal(0.55 if supports else -0.1, 0.5), 0.2, 20)),
                effect_favours_treatment=supports,
                n_per_arm=int(rng.integers(3, 30)),
                replication_scope=str(rng.choice([
                    E.ReplicationScope.SINGLE_STUDY, E.ReplicationScope.INDEPENDENT_LAB_SINGLE,
                    E.ReplicationScope.NOT_REPORTED,
                ], p=[0.5, 0.25, 0.25])),
                lab_identifier=str(rng.choice(lab_pool)) if rng.random() < 0.6 else None,
                conflict_of_interest=str(rng.choice([
                    E.ConflictOfInterest.ACADEMIC_ONLY, E.ConflictOfInterest.SPONSOR_AFFILIATED,
                    E.ConflictOfInterest.NOT_REPORTED,
                ], p=[0.46, 0.28, 0.26])),
                publication_status=str(rng.choice([
                    E.PublicationStatus.PEER_REVIEWED, E.PublicationStatus.PREPRINT,
                ], p=[0.82, 0.18])),
                source_id=self._source().id,
                available_at=available,
                extraction_date=dt.date(2026, 1, 15),
                provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            ))
            self.result.n_evidence += 1

        for k in range(n_invitro):
            available = ind_date - dt.timedelta(days=int(rng.integers(120, 2600)))
            self.session.add(EvidenceItem(
                organization_id=self.result.organization.id,
                program_id=program.id, program_indication_id=track.id,
                target_id=target.id,
                evidence_class=str(rng.choice([
                    E.EvidenceClass.BIOCHEMICAL, E.EvidenceClass.CELL_LINE,
                ], p=[0.45, 0.55])),
                direction=str(E.EvidenceDirection.SUPPORTS),
                title=f"[SYNTHETIC] In vitro characterisation {k + 1}",
                species=str(E.Species.HUMAN),
                biological_system=str(E.BiologicalSystem.IMMORTALISED_CELL_LINE),
                model_type=str(E.ModelType.NOT_APPLICABLE),
                model_subtype=str(E.OncologyModelSubtype.NOT_APPLICABLE),
                immune_competence=str(E.ImmuneCompetence.NOT_APPLICABLE),
                endpoint_category=str(E.EndpointCategory.ENZYME_INHIBITION),
                effect_metric=str(E.EffectMetric.IC50_NM),
                effect_value=float(np.clip(rng.lognormal(2.5, 1.4), 0.05, 5000)),
                lab_identifier=str(rng.choice(lab_pool)) if rng.random() < 0.7 else None,
                publication_status=str(E.PublicationStatus.PEER_REVIEWED),
                source_id=self._source().id,
                available_at=available,
                extraction_date=dt.date(2026, 1, 15),
                provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            ))
            self.result.n_evidence += 1

    def _create_pkpd(
        self, track: ProgramIndication, latents: Latents, ind_date: dt.date
    ) -> None:
        rng = self.rng
        # MNAR: a weak exposure bridge is disclosed less often. This is the most
        # realistic property of the simulation and the reason the missing
        # indicators carry signal.
        p_disclose = float(np.clip(0.42 + 0.14 * latents.exposure_margin_log2, 0.12, 0.94))
        if rng.random() > p_disclose:
            return

        n_versions = int(rng.choice([1, 2], p=[0.7, 0.3]))
        for v in range(n_versions):
            available = ind_date - dt.timedelta(days=int(rng.integers(30, 900 * (v + 1))))
            observed_margin = latents.exposure_margin_log2 + float(rng.normal(0, 0.55))
            self.session.add(PkPdBridge(
                program_indication_id=track.id,
                available_at=available,
                preclinical_efficacious_auc=float(np.round(rng.lognormal(6.0, 0.8), 1)),
                exposure_unit="ng·h/mL",
                preclinical_species=str(E.Species.MOUSE),
                projected_human_auc=float(np.round(rng.lognormal(6.4, 0.9), 1)),
                human_exposure_observed=False,
                exposure_margin_ratio=float(np.clip(2 ** observed_margin, 0.02, 400)),
                therapeutic_index_ratio=float(np.clip(
                    2 ** (latents.therapeutic_index_log2 + rng.normal(0, 0.6)), 0.05, 2000
                )),
                exposure_bridge_quality=str(rng.choice([
                    E.ExposureBridgeQuality.ALLOMETRIC_SIMPLE,
                    E.ExposureBridgeQuality.ALLOMETRIC_WITH_PK_MODEL,
                    E.ExposureBridgeQuality.ASSUMED_NOT_MODELLED,
                    E.ExposureBridgeQuality.EXPOSURE_GAP_IDENTIFIED,
                ], p=[0.36, 0.34, 0.18, 0.12])),
                target_engagement_evidence=str(_engagement_enum(latents.target_engagement)),
                target_engagement_percent=(
                    float(np.clip(rng.normal(55 + 6 * latents.target_engagement, 15), 5, 99))
                    if rng.random() < 0.6 else None
                ),
                exposure_response_established=bool(rng.random() < 0.3 + 0.1 * latents.target_engagement)
                if rng.random() < 0.75 else None,
                pk_scaling_species_count=int(rng.integers(1, 4)),
                pk_model_type=str(rng.choice(["allometric", "PBPK", "compartmental"])),
                provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            ))
            self.result.n_bridges += 1

    def _create_biomarkers(
        self, track: ProgramIndication, latents: Latents, ind_date: dt.date
    ) -> None:
        rng = self.rng
        if rng.random() > 0.35 + 0.35 * min(latents.target_engagement / 6.0, 1.0):
            return
        readiness_pool = [
            E.BiomarkerReadiness.PRECLINICAL_ONLY, E.BiomarkerReadiness.FIT_FOR_PURPOSE_HUMAN,
            E.BiomarkerReadiness.VALIDATED_ASSAY_HUMAN, E.BiomarkerReadiness.CONCEPTUAL,
        ]
        self.session.add(ProgramBiomarker(
            program_indication_id=track.id,
            available_at=ind_date - dt.timedelta(days=int(rng.integers(30, 1100))),
            biomarker_type=str(E.BiomarkerType.PHARMACODYNAMIC),
            readiness=str(rng.choice(readiness_pool, p=[0.34, 0.3, 0.2, 0.16])),
            cross_species_concordance=str(rng.choice(
                ["high", "moderate", "low"], p=[0.3, 0.45, 0.25]
            )),
            assay_validated_in_human=bool(rng.random() < 0.4),
            intended_use_in_trial="Dose selection in the first-in-human study",
            provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
            is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
        ))
        self.result.n_biomarkers += 1

        if latents.biomarker_selected and rng.random() < 0.7:
            self.session.add(ProgramBiomarker(
                program_indication_id=track.id,
                available_at=ind_date - dt.timedelta(days=int(rng.integers(30, 800))),
                biomarker_type=str(E.BiomarkerType.PREDICTIVE_SELECTION),
                readiness=str(rng.choice([
                    E.BiomarkerReadiness.VALIDATED_ASSAY_HUMAN,
                    E.BiomarkerReadiness.FIT_FOR_PURPOSE_HUMAN,
                    E.BiomarkerReadiness.CLINICALLY_QUALIFIED,
                ], p=[0.45, 0.4, 0.15])),
                cross_species_concordance=str(rng.choice(["high", "moderate"], p=[0.6, 0.4])),
                assay_validated_in_human=True,
                intended_use_in_trial="Patient selection",
                provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            ))
            self.result.n_biomarkers += 1

    def _create_safety(
        self,
        track: ProgramIndication,
        program: Program,
        latents: Latents,
        ind_date: dt.date,
    ) -> None:
        rng = self.rng
        n = int(rng.poisson(latents.safety_burden))
        for _ in range(min(n, 6)):
            severity = str(rng.choice(
                ["minimal", "mild", "moderate", "severe", "life_threatening"],
                p=[0.18, 0.28, 0.32, 0.17, 0.05],
            ))
            self.session.add(SafetyFinding(
                program_indication_id=track.id,
                program_id=program.id,
                available_at=ind_date - dt.timedelta(days=int(rng.integers(30, 800))),
                species=str(rng.choice([E.Species.RAT, E.Species.DOG, E.Species.NON_HUMAN_PRIMATE],
                                       p=[0.5, 0.35, 0.15])),
                organ_system=str(rng.choice([
                    "haematopoietic", "hepatic", "renal", "cardiovascular",
                    "gastrointestinal", "neurological", "dermatological",
                ])),
                finding="[SYNTHETIC] Simulated toxicology observation",
                severity=severity,
                reversible=bool(rng.random() < 0.65) if rng.random() < 0.8 else None,
                monitorable_in_human=bool(rng.random() < 0.7) if rng.random() < 0.75 else None,
                on_target=bool(rng.random() < 0.45),
                noael_dose=float(np.round(rng.lognormal(2.0, 1.0), 1)),
                noael_unit="mg/kg/day",
                is_known_class_effect=bool(rng.random() < 0.3),
                provenance_method=str(E.ProvenanceMethod.SYNTHETIC_GENERATOR),
                is_synthetic=True, synthetic_note=SYNTHETIC_NOTE,
            ))
            self.result.n_safety += 1


def _engagement_enum(value: float) -> str:
    ladder = [
        (0.5, E.TargetEngagementEvidence.NONE_DEMONSTRATED),
        (1.5, E.TargetEngagementEvidence.IN_VITRO_ONLY),
        (2.5, E.TargetEngagementEvidence.ANIMAL_INDIRECT_PD),
        (4.0, E.TargetEngagementEvidence.ANIMAL_DIRECT),
        (5.5, E.TargetEngagementEvidence.HUMAN_INDIRECT_PD),
    ]
    for threshold, label in ladder:
        if value < threshold:
            return str(label)
    return str(E.TargetEngagementEvidence.HUMAN_DIRECT)


def _mechanism_text(symbol: str, modality: str) -> str:
    verb = {
        E.Modality.SMALL_MOLECULE: "small-molecule inhibitor of",
        E.Modality.MONOCLONAL_ANTIBODY: "monoclonal antibody targeting",
        E.Modality.BISPECIFIC_ANTIBODY: "bispecific antibody engaging",
        E.Modality.ANTIBODY_DRUG_CONJUGATE: "antibody-drug conjugate directed at",
        E.Modality.PROTEIN_DEGRADER: "targeted protein degrader of",
        E.Modality.CELL_THERAPY_CAR_T: "autologous CAR-T cell therapy directed at",
        E.Modality.RADIOPHARMACEUTICAL: "radioligand therapy directed at",
        E.Modality.OLIGONUCLEOTIDE_ASO: "antisense oligonucleotide against",
        E.Modality.PEPTIDE: "peptide antagonist of",
        E.Modality.THERAPEUTIC_PROTEIN: "engineered protein modulating",
    }.get(modality, "agent modulating")
    return f"Simulated {verb} {symbol}."


def verify_base_rate(session: Session) -> dict[str, Any]:
    """Report the realised outcome mix after generation.

    The simulation's intercept is chosen analytically from the latent means,
    but the realised rate depends on the draw. This is called by the seeding
    CLI so a drifting generator is noticed immediately rather than silently
    producing a degenerate cohort.
    """
    from sqlalchemy import func, select

    rows = session.execute(
        select(ProgramIndication.outcome_label, func.count())
        .group_by(ProgramIndication.outcome_label)
    ).all()
    counts = {label: int(n) for label, n in rows}
    fail = counts.get(str(E.OutcomeLabel.PHASE_2_EFFICACY_FAILURE), 0)
    success = (
        counts.get(str(E.OutcomeLabel.PROGRESSED_PAST_PHASE_2), 0)
        + counts.get(str(E.OutcomeLabel.REGULATORY_APPROVAL), 0)
    )
    denom = fail + success
    return {
        "outcome_counts": counts,
        "primary_cohort_size": denom,
        "primary_failure_rate": (fail / denom) if denom else None,
        "target_failure_rate": 0.60,
        "within_tolerance": (abs(fail / denom - 0.60) < 0.12) if denom else False,
    }


def seed_database(session: Session, *, n_programs: int = 200, seed: int = 20240917) -> SeedResult:
    """Convenience entry point used by the CLI."""
    return DemoDataGenerator(session, seed=seed).run(n_programs=n_programs)
