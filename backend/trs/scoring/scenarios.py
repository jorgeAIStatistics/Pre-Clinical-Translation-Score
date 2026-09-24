"""Scenario analysis: what would change if new translational evidence arrived.

What a scenario is
------------------
A counterfactual evaluation of the *model*, under a hypothetical feature state.
"If an independent laboratory replicated the pivotal in vivo result, and
nothing else changed, this model would return X instead of Y."

What a scenario is not
----------------------
It is not a prediction that the experiment would succeed, and it is not a
statement about what the experiment would show. A scenario answers "how much
does the estimate depend on this piece of evidence?" — which is a question about
the model's sensitivity, not about biology. Every scenario response carries that
distinction in words, because the temptation to read it the other way is strong
and the consequence is a planning decision made on a fiction.

Scenarios never write evidence. Overrides are applied to a copy of the feature
vector at scoring time; the evidence base is untouched, so speculation cannot
contaminate the record or leak into a future retraining set.

Lever design
------------
A lever is a named, curated bundle of feature overrides representing one
realistic action a team could take. Users can also set raw feature overrides,
but levers are the primary interface: "add an independent replication" is a
decision a translational scientist can act on, whereas
``replication_scope_ord = 4.0`` is not.

Only features declared ``scenario_adjustable`` in the registry may be
overridden. That excludes things a team cannot change by doing work — the
historical failure rate on the target, the disease's unmet need — because
offering to vary them would invite conclusions the analysis cannot support.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any

from trs.features.registry import FEATURES_BY_NAME, scenario_adjustable_features
from trs.features.spec import DOMAIN_LABELS, FeatureDomain


class ScenarioValidationError(ValueError):
    """Raised when an override targets a feature that may not be varied."""


class LeverDirection(StrEnum):
    """How a lever's value combines with the program's current value.

    ``IMPROVEMENT``
        The lever represents evidence being *added*. It can only move a feature
        in the risk-reducing direction: applied to a program that already
        exceeds the lever's value, it does nothing.

        This matters. An early version applied lever values unconditionally,
        so "add an independent replication" applied to a program already
        replicated across several laboratories *lowered* its replication score
        and raised its estimated risk. The scenario then read as "commissioning
        this study would make the program worse", which is nonsense and would
        have been acted on.

    ``ADVERSE``
        The lever represents bad news arriving. It can only move a feature in
        the risk-increasing direction, by the mirror-image argument.

    ``ABSOLUTE``
        The lever asserts a state outright, replacing whatever is there. Used
        where the feature is a fact about the trial design rather than an
        accumulating body of evidence — enrolling a biomarker-selected
        population is a choice, not a discovery.
    """

    IMPROVEMENT = "improvement"
    ADVERSE = "adverse"
    ABSOLUTE = "absolute"


#: Per-feature orientation: True when a LARGER raw value reduces risk. Used to
#: decide which way an IMPROVEMENT lever is allowed to move a feature. Derived
#: from the registry's declared expected direction, so the two cannot disagree.
def _larger_is_better(feature_name: str) -> bool | None:
    spec = FEATURES_BY_NAME.get(feature_name)
    if spec is None:
        return None
    from trs.features.spec import Direction

    if spec.expected_direction is Direction.DECREASES_RISK:
        return True
    if spec.expected_direction is Direction.INCREASES_RISK:
        return False
    return None  # uncertain: treated as ABSOLUTE


@dataclasses.dataclass(frozen=True, slots=True)
class ScenarioLever:
    key: str
    label: str
    description: str
    #: What a user would actually have to do to realise this state.
    action: str
    domain: FeatureDomain
    #: {feature_name: value}. Values are in raw (pre-transform) units.
    overrides: dict[str, float]
    #: How values combine with the program's current state.
    direction: LeverDirection = LeverDirection.IMPROVEMENT
    #: Levers that only make sense applied together with this one.
    implies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "action": self.action,
            "domain": str(self.domain),
            "domain_label": DOMAIN_LABELS[self.domain],
            "direction": str(self.direction),
            "overrides": self.overrides,
            "implies": list(self.implies),
            "affected_features": [
                {"name": n, "label": FEATURES_BY_NAME[n].label}
                for n in self.overrides
                if n in FEATURES_BY_NAME
            ],
        }


#: The curated lever catalogue. Values are chosen to represent the *stated*
#: state, not an optimistic one: "independent replication added" sets the
#: replication scope to a single independent laboratory, not to multiple.
LEVERS: tuple[ScenarioLever, ...] = (
    ScenarioLever(
        key="independent_replication",
        label="Independent replication added",
        description=(
            "One laboratory unaffiliated with the sponsor reproduces the pivotal "
            "in vivo efficacy result."
        ),
        action="Commission a contract or academic laboratory to repeat the pivotal study.",
        domain=FeatureDomain.REPRODUCIBILITY,
        overrides={
            "replication_scope_ord": 2.5,   # INDEPENDENT_LAB_SINGLE
            "n_independent_labs": 2.0,
            "single_lab_dependence": 0.0,
        },
    ),
    ScenarioLever(
        key="multi_lab_replication",
        label="Replicated across multiple independent laboratories",
        description="Two or more unaffiliated laboratories reproduce the pivotal result.",
        action="Run a multi-site replication, or publish a protocol others adopt.",
        domain=FeatureDomain.REPRODUCIBILITY,
        overrides={
            "replication_scope_ord": 4.0,   # INDEPENDENT_LABS_MULTIPLE
            "n_independent_labs": 3.0,
            "single_lab_dependence": 0.0,
        },
    ),
    ScenarioLever(
        key="human_organoid_validation",
        label="Human organoid validation added",
        description=(
            "The mechanism is demonstrated in patient-derived organoids from the "
            "intended indication."
        ),
        action="Run a patient-derived organoid panel against the lead molecule.",
        domain=FeatureDomain.HUMAN_BIOLOGICAL_SUPPORT,
        overrides={
            "has_organoid_evidence": 1.0,
            "human_evidence_count": 4.0,
            "human_evidence_supporting_fraction": 0.8,
        },
    ),
    ScenarioLever(
        key="patient_tissue_evidence",
        label="Patient tissue evidence added",
        description="Target modulation is demonstrated in primary human tumour tissue.",
        action="Obtain and profile banked patient specimens from the target population.",
        domain=FeatureDomain.HUMAN_BIOLOGICAL_SUPPORT,
        overrides={"has_patient_tissue_evidence": 1.0, "human_evidence_count": 3.0},
    ),
    ScenarioLever(
        key="exposure_margin_demonstrated",
        label="Wider exposure margin demonstrated",
        description=(
            "A refined PK model shows achievable human exposure four-fold above "
            "the efficacious preclinical exposure."
        ),
        action="Run additional PK studies and build a PBPK model to support the projection.",
        domain=FeatureDomain.PKPD_TRANSLATION,
        overrides={
            "exposure_margin_log2": 2.0,        # 4× margin
            "exposure_bridge_quality_ord": 3.0,  # allometric with PK model
            "pk_scaling_species_count": 3.0,
        },
    ),
    ScenarioLever(
        key="human_target_engagement",
        label="Target engagement demonstrated in humans",
        description=(
            "A Phase I biomarker sub-study shows direct target engagement at a "
            "tolerated dose."
        ),
        action="Add a target-engagement endpoint to the first-in-human protocol.",
        domain=FeatureDomain.PKPD_TRANSLATION,
        overrides={
            "target_engagement_ord": 6.0,        # HUMAN_DIRECT
            "exposure_bridge_quality_ord": 4.0,  # human exposure demonstrated
        },
        implies=("pd_biomarker_ready",),
    ),
    ScenarioLever(
        key="pd_biomarker_ready",
        label="Validated PD biomarker available",
        description=(
            "A pharmacodynamic biomarker with a human-validated assay is ready "
            "for the first-in-human study."
        ),
        action="Qualify the PD assay in human matrix before first-in-human dosing.",
        domain=FeatureDomain.BIOMARKER_READINESS,
        overrides={
            "biomarker_readiness_ord": 4.0,          # VALIDATED_ASSAY_HUMAN
            "biomarker_species_concordance_ord": 3.0,
        },
    ),
    ScenarioLever(
        key="biomarker_selected_population",
        label="Enrol a biomarker-selected population",
        description=(
            "The first efficacy study requires a genomic alteration rather than "
            "enrolling all comers."
        ),
        action="Develop a companion diagnostic and restrict eligibility.",
        domain=FeatureDomain.DISEASE_MODEL_RELEVANCE,
        overrides={
            "biomarker_selected_population": 1.0,
            "has_predictive_selection_biomarker": 1.0,
        },
        direction=LeverDirection.ABSOLUTE,
    ),
    ScenarioLever(
        key="high_fidelity_model",
        label="High-fidelity model result added",
        description=(
            "Efficacy is demonstrated in a patient-derived xenograft or an "
            "autochthonous genetically engineered model."
        ),
        action="Run a PDX panel or a GEMM study in the intended indication.",
        domain=FeatureDomain.ANIMAL_MODEL_VALIDITY,
        overrides={
            "animal_uses_high_fidelity_model": 1.0,
            "animal_best_construct_validity": 3.0,
            "n_distinct_model_types": 3.0,
        },
    ),
    ScenarioLever(
        key="second_species_efficacy",
        label="Efficacy confirmed in a second species",
        description="The in vivo effect reproduces in a second, non-rodent species.",
        action="Repeat the efficacy study in a second species.",
        domain=FeatureDomain.CROSS_SPECIES_CONCORDANCE,
        overrides={
            "species_count_supporting": 2.0,
            "species_concordance_fraction": 1.0,
            "has_nonrodent_efficacy": 1.0,
        },
    ),
    ScenarioLever(
        key="human_genetic_support",
        label="Human genetic support established",
        description=(
            "A biobank-scale analysis establishes a directionally consistent "
            "human genetic association between the target and the disease."
        ),
        action=(
            "Commission a rare-variant burden analysis or fine-mapping study in "
            "a population biobank."
        ),
        domain=FeatureDomain.TARGET_VALIDATION,
        overrides={"genetic_evidence_ord": 4.0},   # RARE_CODING_BURDEN
    ),
    ScenarioLever(
        key="exposure_response_shown",
        label="Exposure–response relationship demonstrated",
        description=(
            "A monotone relationship between exposure and effect is established "
            "preclinically, supporting dose selection."
        ),
        action="Run a dose-fractionation study across the efficacious exposure range.",
        domain=FeatureDomain.PKPD_TRANSLATION,
        overrides={"exposure_response_established": 1.0},
    ),

    # ---- adverse scenarios. A sensitivity analysis that only permits good
    # news is a marketing tool, not an analysis. ----------------------------
    ScenarioLever(
        key="contradictory_study_appears",
        label="A contradictory animal study appears",
        description=(
            "An independent group publishes an in vivo study that fails to "
            "reproduce the efficacy result."
        ),
        action="(Adverse scenario — models the risk that replication fails.)",
        domain=FeatureDomain.ANIMAL_MODEL_VALIDITY,
        overrides={
            "animal_n_refuting_studies": 1.0,
            "animal_positive_fraction": 0.6,
            "species_concordance_fraction": 0.5,
        },
        direction=LeverDirection.ADVERSE,
    ),
    ScenarioLever(
        key="exposure_gap_identified",
        label="An exposure gap is identified",
        description=(
            "Human PK shows the efficacious preclinical exposure cannot be "
            "reached at a tolerated dose."
        ),
        action="(Adverse scenario — models the classic PK/PD translation failure.)",
        domain=FeatureDomain.PKPD_TRANSLATION,
        overrides={
            "exposure_margin_log2": -1.5,
            "exposure_bridge_quality_ord": 0.0,
            "therapeutic_index_log2": 0.0,
        },
        direction=LeverDirection.ADVERSE,
    ),
)

LEVERS_BY_KEY: dict[str, ScenarioLever] = {lv.key: lv for lv in LEVERS}


def validate_overrides(overrides: dict[str, Any]) -> dict[str, float]:
    """Reject overrides on features users must not be able to vary."""
    allowed = {f.name for f in scenario_adjustable_features()}
    clean: dict[str, float] = {}
    for name, value in overrides.items():
        spec = FEATURES_BY_NAME.get(name)
        if spec is None:
            raise ScenarioValidationError(f"Unknown feature {name!r}.")
        if name not in allowed:
            raise ScenarioValidationError(
                f"Feature {name!r} ({spec.label}) is not adjustable in scenario "
                "analysis. Only features a team could change by generating new "
                "evidence may be varied; historical class performance and "
                "fixed disease context may not."
            )
        if value is None:
            clean[name] = None
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ScenarioValidationError(
                f"Feature {name!r} requires a numeric value, got {value!r}."
            ) from exc
        if spec.valid_min is not None and numeric < spec.valid_min:
            raise ScenarioValidationError(
                f"{name}={numeric} is below the declared minimum {spec.valid_min}."
            )
        if spec.valid_max is not None and numeric > spec.valid_max:
            raise ScenarioValidationError(
                f"{name}={numeric} is above the declared maximum {spec.valid_max}."
            )
        clean[name] = numeric
    return clean


def resolve_levers(
    lever_keys: list[str], current: dict[str, Any] | None = None
) -> dict[str, float]:
    """Merge lever bundles into a single override map.

    When ``current`` (the program's present feature vector) is supplied, each
    lever's values are combined with it according to the lever's
    :class:`LeverDirection`, so an improvement lever can never make a program
    look worse and an adverse lever can never make one look better. Without
    ``current`` the raw values are returned, which is what the lever catalogue
    endpoint needs.

    Later levers win on conflict, and the resolved map is validated exactly
    like a hand-written override so a lever cannot bypass the adjustability
    rules.
    """
    merged: dict[str, float] = {}
    seen: set[str] = set()

    def combine(name: str, value: float, direction: LeverDirection) -> float:
        if current is None or direction is LeverDirection.ABSOLUTE:
            return value
        existing = current.get(name)
        if existing is None:
            return value
        better_is_larger = _larger_is_better(name)
        if better_is_larger is None:
            return value
        try:
            existing_f = float(existing)
        except (TypeError, ValueError):
            return value

        improving = direction is LeverDirection.IMPROVEMENT
        # An improvement moves the feature the way that reduces risk; an
        # adverse scenario moves it the other way. Either way the lever is
        # clamped so it cannot push past where the program already stands.
        move_up = better_is_larger if improving else not better_is_larger
        return max(existing_f, value) if move_up else min(existing_f, value)

    def apply(key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        lever = LEVERS_BY_KEY.get(key)
        if lever is None:
            raise ScenarioValidationError(
                f"Unknown scenario lever {key!r}. Available: {sorted(LEVERS_BY_KEY)}"
            )
        for implied in lever.implies:
            apply(implied)
        for name, value in lever.overrides.items():
            merged[name] = combine(name, value, lever.direction)

    for key in lever_keys:
        apply(key)
    return validate_overrides(merged)


def apply_overrides(
    feature_row: dict[str, Any], overrides: dict[str, float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a modified copy of the feature vector plus a change log.

    The original row is never mutated: the caller holds a persisted snapshot,
    and a scenario must not be able to alter the record it is hypothesising
    about.
    """
    modified = dict(feature_row)
    changes: list[dict[str, Any]] = []
    for name, new_value in overrides.items():
        old = feature_row.get(name)
        if old == new_value:
            continue
        modified[name] = new_value
        spec = FEATURES_BY_NAME[name]
        changes.append({
            "feature": name,
            "label": spec.label,
            "domain": str(spec.domain),
            "domain_label": DOMAIN_LABELS[spec.domain],
            "from_value": old,
            "to_value": new_value,
            "was_missing": old is None,
        })
    return modified, changes


def lever_catalogue() -> list[dict[str, Any]]:
    return [lv.to_dict() for lv in LEVERS]
