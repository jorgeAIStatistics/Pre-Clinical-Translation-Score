"""Feature engineering, cohort construction and entity handling.

These are the layers between raw evidence and the model matrix. A regression
here is silent: nothing crashes, the pipeline still trains, and the numbers it
produces are wrong in a way no downstream check would notice. So the
assertions below are on *values and invariants*, not on shapes.

Four properties are load-bearing and each has a test whose docstring says what
breaks in the product if it fails:

* Heterogeneous effect sizes land on one scale with a true zero (effects).
* Absence of evidence is encoded as missing, never as the worst level
  (ordinal encoding, completeness, builder aggregation).
* Everything fitted is fitted on the training fold only (transformer).
* A commercial discontinuation is not a scientific failure (cohort outcomes).

Everything runs offline against the in-memory database and the 60-program
seeded universe from ``tests/conftest.py``. The seeded data is synthetic; no
number derived from it is a claim about any real therapeutic program.
"""

from __future__ import annotations

import datetime as dt
import math
import re

import numpy as np
import pytest
from sqlalchemy import func, select

from trs.cohort.build import build_cohort
from trs.cohort.outcomes import (
    ENDPOINTS,
    NON_SCIENTIFIC_REASONS,
    PH1_SAFETY_FAILURE,
    PH2_ANY_SCIENTIFIC_FAILURE,
    PH2_EFFICACY_FAILURE,
    PRIMARY_ENDPOINT,
    label_track,
    resolve_composite_discontinuation,
)
from trs.db.models.core import Program, ProgramAlias, ProgramIndication
from trs.features.effects import effect_precision, standardise_effect
from trs.features.missingness import (
    CRITICAL_DOMAINS,
    GRADE_THRESHOLDS,
    CompletenessReport,
    compute_completeness,
    grade_evidence_confidence,
)
from trs.features.registry import FEATURES, FEATURES_BY_NAME, features_in_domain
from trs.features.spec import (
    GENETIC_EVIDENCE_ORDINAL,
    TARGET_ENGAGEMENT_ORDINAL,
    FeatureDomain,
    MissingPolicy,
    ordinal,
)
from trs.features.transform import INDICATOR_RATE_BAND, FeatureTransformer
from trs.ontology.enums import ConfidenceLevel as CL
from trs.ontology.enums import DiscontinuationReason as R
from trs.ontology.enums import EffectMetric as EM
from trs.ontology.enums import MissingnessReason as MR
from trs.ontology.enums import OutcomeLabel as O
from trs.resolution.normalise import normalise_compound
from trs.scoring.scenarios import (
    LEVERS,
    LEVERS_BY_KEY,
    ScenarioValidationError,
    apply_overrides,
    resolve_levers,
    validate_overrides,
)
from trs.validation.splitters import (
    GROUP_SCOPE_KEYS,
    ConnectedComponentGrouper,
    GroupScope,
    build_grouping_keys,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. Effect-size harmonisation
# ═════════════════════════════════════════════════════════════════════════════


class TestStandardiseEffect:
    """trs/features/effects.py — putting incomparable effect sizes on one scale."""

    def test_zero_tgi_is_exactly_the_null(self):
        """If the null is not exactly zero, every downstream "effect size"
        feature carries a constant offset: a program with no demonstrated
        effect scores as though it had one, and the median/max effect features
        stop meaning "how large were the animal effects"."""
        assert standardise_effect(EM.PERCENT_TGI, 0.0) == 0.0
        assert standardise_effect(EM.PERCENT_CHANGE, 0.0) == 0.0
        assert standardise_effect(EM.RESPONSE_RATE, 0.0) == 0.0
        # The log-ratio family has its null at a ratio of 1, not 0.
        assert standardise_effect(EM.HAZARD_RATIO, 1.0) == 0.0
        assert standardise_effect(EM.ODDS_RATIO, 1.0) == 0.0
        assert standardise_effect(EM.FOLD_CHANGE, 1.0) == 0.0

    def test_mapping_is_monotone_increasing_in_tgi(self):
        """The model reads this feature ordinally. If the mapping is not
        monotone, "stronger animal effect" and "larger feature value" come
        apart, and the fitted coefficient means nothing at all."""
        grid = [0.0, 5.0, 10.0, 25.0, 40.0, 50.0, 65.0, 75.0, 90.0, 99.0]
        values = [standardise_effect(EM.PERCENT_TGI, v) for v in grid]
        assert all(b > a for a, b in zip(values[:-1], values[1:], strict=True))

        # Documented anchors: the scale calibration claimed in the module
        # docstring (50% -> ~0.90, 90% -> ~2.99) must hold.
        assert standardise_effect(EM.PERCENT_TGI, 50.0) == pytest.approx(0.90, abs=0.01)
        assert standardise_effect(EM.PERCENT_TGI, 90.0) == pytest.approx(2.99, abs=0.01)

        # Above 100% the remaining-fraction floor saturates deliberately, so a
        # single complete response cannot dominate an aggregate.
        assert standardise_effect(EM.PERCENT_TGI, 120.0) == standardise_effect(
            EM.PERCENT_TGI, 100.0
        )
        assert standardise_effect(EM.PERCENT_TGI, 100.0) == pytest.approx(5.09, abs=0.01)

    def test_negative_tgi_gives_a_negative_value(self):
        """A treatment that accelerated tumour growth must not be scored as
        weak-but-positive evidence. If the sign is lost, refuting animal data
        pushes the estimate in the same direction as supporting data."""
        worse = standardise_effect(EM.PERCENT_TGI, -30.0)
        assert worse is not None and worse < 0.0
        assert worse == pytest.approx(-math.log(1.3) * 1.3, rel=1e-9)
        # ... and monotonicity holds across the null.
        assert worse < standardise_effect(EM.PERCENT_TGI, 0.0) < standardise_effect(
            EM.PERCENT_TGI, 30.0
        )

    @pytest.mark.parametrize("metric", [EM.IC50_NM, EM.EC50_NM, EM.KI_NM, EM.KD_NM])
    def test_potency_metrics_return_none(self, metric):
        """Potency is not efficacy. A 3 nM IC50 converted into an "effect size"
        would enter the animal-efficacy features as an enormous positive
        effect, and every potent-but-unproven molecule would score as though it
        had strong in vivo evidence. Potency enters via the PK/PD bridge."""
        assert standardise_effect(metric, 3.0) is None
        assert standardise_effect(metric, 3.0, favours_treatment=True) is None
        # Endpoint-specific absolute differences are not comparable either.
        assert standardise_effect(EM.ABSOLUTE_DIFFERENCE, 3.0) is None
        assert standardise_effect(EM.OTHER, 3.0) is None

    def test_favours_treatment_false_flips_sign_and_keeps_magnitude(self):
        """The curator's direction assertion overrides the metric default. If
        the magnitude were recomputed rather than preserved, a curator marking
        an endpoint's direction would silently change how large the effect is,
        not just which way it points."""
        default = standardise_effect(EM.PERCENT_TGI, 80.0)
        flipped = standardise_effect(EM.PERCENT_TGI, 80.0, favours_treatment=False)
        kept = standardise_effect(EM.PERCENT_TGI, 80.0, favours_treatment=True)
        assert flipped == pytest.approx(-default)
        assert abs(flipped) == pytest.approx(abs(default))
        assert kept == pytest.approx(default)

        # It is an assertion about direction, so it also rescues a value whose
        # metric-default orientation is wrong: magnitude preserved, sign taken.
        negative_input = standardise_effect(EM.PERCENT_TGI, -30.0)
        asserted_good = standardise_effect(EM.PERCENT_TGI, -30.0, favours_treatment=True)
        assert asserted_good == pytest.approx(abs(negative_input))

    def test_cap_is_respected(self):
        """Preclinical effect reporting has a long right tail. Without the cap
        one near-complete response in five animals dominates the median and
        maximum effect features for the whole program."""
        from trs.features.effects import _SMD_CAP

        assert standardise_effect(EM.STANDARDISED_MEAN_DIFFERENCE, 50.0) == _SMD_CAP
        assert standardise_effect(EM.STANDARDISED_MEAN_DIFFERENCE, -50.0) == -_SMD_CAP
        assert standardise_effect(EM.FOLD_CHANGE, 2.0**40) == _SMD_CAP
        for metric, value in (
            (EM.HAZARD_RATIO, 1e-9), (EM.ODDS_RATIO, 1e9), (EM.PERCENT_TGI, 100.0),
        ):
            out = standardise_effect(metric, value)
            assert out is not None and -_SMD_CAP <= out <= _SMD_CAP

    def test_unusable_inputs_return_none_and_never_zero(self):
        """None propagates as missing. Coercing an unusable input to 0.0 would
        report "no effect demonstrated" for evidence that was merely
        unreadable — absence of evidence presented as evidence of absence."""
        assert standardise_effect(None, 50.0) is None
        assert standardise_effect(EM.PERCENT_TGI, None) is None
        assert standardise_effect(EM.PERCENT_TGI, float("nan")) is None
        assert standardise_effect(EM.PERCENT_TGI, float("inf")) is None
        # Ratios must be strictly positive; a response rate must be a fraction.
        assert standardise_effect(EM.HAZARD_RATIO, 0.0) is None
        assert standardise_effect(EM.HAZARD_RATIO, -1.0) is None
        assert standardise_effect(EM.ODDS_RATIO, 0.0) is None
        assert standardise_effect(EM.RESPONSE_RATE, 1.5) is None
        assert standardise_effect(EM.RESPONSE_RATE, -0.1) is None

    def test_ratio_families_are_oriented_toward_treatment_benefit(self):
        """Hazard/risk ratios below 1 favour treatment; odds ratios and fold
        changes above 1 do. If an orientation flips, protective and harmful
        evidence are pooled with the wrong sign and the aggregate is worse than
        no feature at all."""
        assert standardise_effect(EM.HAZARD_RATIO, 0.5) > 0
        assert standardise_effect(EM.HAZARD_RATIO, 2.0) < 0
        assert standardise_effect(EM.RISK_RATIO, 0.5) > 0
        assert standardise_effect(EM.ODDS_RATIO, 2.0) > 0
        assert standardise_effect(EM.ODDS_RATIO, 0.5) < 0
        assert standardise_effect(EM.FOLD_CHANGE, 4.0) > 0
        # Hasselblad-Hedges: ln(OR) * sqrt(3)/pi.
        assert standardise_effect(EM.ODDS_RATIO, 2.0) == pytest.approx(
            math.log(2.0) * math.sqrt(3.0) / math.pi
        )

    def test_precision_weight_is_never_assumed(self):
        """effect_precision orders evidence. Defaulting an unreported precision
        to a number would rank undocumented studies alongside well-powered
        ones when the builder picks a representative effect."""
        assert effect_precision(None, None, None, None) is None
        assert effect_precision(None, None, 0.5, None) == pytest.approx(1 / 1.5)
        assert effect_precision(1.0, 2.0, None, None) == pytest.approx(0.5)
        # A reported standard error beats a CI, which beats group size.
        assert effect_precision(1.0, 5.0, 0.1, 100) == pytest.approx(1 / 1.1)
        assert effect_precision(None, None, None, 49) == pytest.approx(0.7)
        # Degenerate inputs do not fabricate precision.
        assert effect_precision(None, None, 0.0, None) is None
        assert effect_precision(3.0, 1.0, None, None) is None
        assert effect_precision(None, None, None, 0) is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. Ordinal encoding
# ═════════════════════════════════════════════════════════════════════════════


class TestOrdinalEncoding:
    """trs/features/spec.py — ordered categories to numbers, and what is not."""

    @pytest.mark.parametrize(
        ("map_name", "value"),
        [
            ("target_engagement", "unknown"),
            ("target_engagement", "not_assessed"),
            ("replication", "unknown"),
            ("replication", "not_reported"),
            ("qualitative_strength", "unknown"),
            ("genetic_evidence", "not_assessed"),
            ("biomarker_readiness", "unknown"),
            ("causal_support", "not_assessed"),
            ("severity", "not_assessed"),
            ("disease_stage", "unknown"),
            ("unmet_need", "not_assessed"),
        ],
    )
    def test_unknown_and_not_assessed_map_to_none_not_zero(self, map_name, value):
        """This is the "absence of evidence is not evidence of absence" rule at
        the encoding layer. If an undocumented field encoded as 0 — the worst
        level — the product would systematically penalise programs for thin
        documentation and reward sponsors who disclose nothing ambiguous. It
        would also be invisible: the estimate would simply be wrong, with no
        missing-data indicator to reveal it."""
        assert ordinal(map_name, value) is None
        assert ordinal(map_name, None) is None
        # An unrecognised value is missing too, not silently floored.
        assert ordinal(map_name, "something_never_declared") is None

    def test_declared_levels_encode_to_their_documented_values(self):
        """The ordinal maps are the model's spacing assumptions, written down.
        A changed value silently rescales a coefficient and every explanation
        built from it."""
        assert ordinal("qualitative_strength", "contested") == 0.0
        assert ordinal("qualitative_strength", "high") == 3.0
        assert ordinal("target_engagement", "human_direct") == 6.0
        assert ordinal("replication", "independent_labs_multiple") == 4.0
        assert ordinal("selective_reporting", "high") == 2.0

    def test_deliberate_gaps_in_the_scales_are_preserved(self):
        """Two gaps are wider than one step on purpose: any human genetic
        signal versus none, and human versus animal target engagement. Flatten
        them and the model loses the only qualitative distinctions the schema
        claims to encode."""
        no_signal = GENETIC_EVIDENCE_ORDINAL["no_human_genetic_signal"]
        eqtl = GENETIC_EVIDENCE_ORDINAL["expression_qtl_only"]
        gwas = GENETIC_EVIDENCE_ORDINAL["gwas_locus_only"]
        assert eqtl - no_signal > gwas - eqtl

        animal_direct = TARGET_ENGAGEMENT_ORDINAL["animal_direct"]
        human_indirect = TARGET_ENGAGEMENT_ORDINAL["human_indirect_pd"]
        assert human_indirect - animal_direct == 2.0

        # Every map is strictly ordered as declared.
        for name in ("qualitative_strength", "causal_support", "pathway_redundancy",
                     "exposure_bridge", "target_engagement", "biomarker_readiness",
                     "replication", "selective_reporting", "unmet_need", "severity"):
            from trs.features.spec import ORDINAL_MAPS

            values = list(ORDINAL_MAPS[name].values())
            assert values == sorted(values), name

    def test_encoded_values_stay_inside_the_registry_bounds(self):
        """The registry declares valid_min/valid_max per feature and scenario
        validation enforces them. An ordinal map that outgrew its declared
        bound would make legitimate observed values un-settable in a scenario."""
        pairs = [
            ("genetic_evidence", "genetic_evidence_ord"),
            ("causal_support", "causal_support_ord"),
            ("pathway_redundancy", "pathway_redundancy_ord"),
            ("exposure_bridge", "exposure_bridge_quality_ord"),
            ("target_engagement", "target_engagement_ord"),
            ("biomarker_readiness", "biomarker_readiness_ord"),
            ("replication", "replication_scope_ord"),
            ("selective_reporting", "selective_reporting_ord"),
            ("unmet_need", "unmet_need_ord"),
            ("severity", "max_safety_severity_ord"),
        ]
        from trs.features.spec import ORDINAL_MAPS

        for map_name, feature_name in pairs:
            spec = FEATURES_BY_NAME[feature_name]
            values = list(ORDINAL_MAPS[map_name].values())
            assert min(values) >= spec.valid_min, feature_name
            assert max(values) <= spec.valid_max, feature_name


# ═════════════════════════════════════════════════════════════════════════════
# 3. FeatureTransformer
# ═════════════════════════════════════════════════════════════════════════════

#: 100 rows so that a single missing value is a 1% missing rate — just below
#: the indicator band's lower edge, which is the boundary under test.
N_FIT_ROWS = 100


def _fit_rows(seed: int = 11) -> list[dict]:
    """A small fit set with four deliberately chosen missing rates."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(N_FIT_ROWS):
        rows.append({
            # present everywhere -> missing rate 0.00, below the band
            "animal_positive_fraction": float(rng.uniform(0.2, 1.0)),
            # missing once -> rate 0.01, still below the band
            "species_concordance_fraction": None if i == 0 else float(rng.uniform(0.0, 1.0)),
            # missing in half the rows -> rate 0.50, inside the band
            "exposure_margin_log2": None if i % 2 else float(rng.normal(1.0, 2.0)),
            # missing everywhere -> rate 1.00, above the band
            "therapeutic_index_log2": None,
            # rate 0.50 but ZERO_IS_ABSENCE, so no indicator is allowed
            "animal_n_invivo_studies": None if i % 2 else float(rng.integers(0, 9)),
            "modality_group": "antibody" if i % 3 else "small_molecule",
        })
    return rows


@pytest.fixture
def fit_rows() -> list[dict]:
    return _fit_rows()


@pytest.fixture
def fitted(fit_rows) -> FeatureTransformer:
    return FeatureTransformer().fit(fit_rows, random_state=1)


class TestFeatureTransformer:
    """trs/features/transform.py — encode, impute, standardise. Fit on train."""

    def test_fit_statistics_come_from_the_fit_set_only(self, fit_rows, fitted):
        """Every imputation constant, one-hot level and standardisation
        statistic is a leakage vector. Fitted on all rows, held-out performance
        is inflated and the reported AUROC is not the number a user would get
        in deployment — which is the only number this product exists to give."""
        state = fitted.state
        observed = [r["exposure_margin_log2"] for r in fit_rows
                    if r["exposure_margin_log2"] is not None]
        assert state.n_fit_rows == N_FIT_ROWS
        assert state.medians["exposure_margin_log2"] == pytest.approx(float(np.median(observed)))

        # An unseen row with that feature missing is filled with the FIT-set
        # median, not with anything derived from itself or its neighbours.
        col = fitted.columns.index("exposure_margin_log2")
        raw = fitted.transform_raw([{"animal_positive_fraction": 0.5}])
        assert raw[0, col] == pytest.approx(state.medians["exposure_margin_log2"])

        # Refitting with shifted rows moves the median: proof the statistic is
        # a property of the fit set and not a constant.
        shifted = fit_rows + [{"exposure_margin_log2": 500.0} for _ in range(N_FIT_ROWS)]
        refit = FeatureTransformer().fit(shifted, random_state=1)
        assert refit.state.medians["exposure_margin_log2"] > state.medians[
            "exposure_margin_log2"
        ] + 1.0

    def test_unseen_nominal_level_gets_no_column_and_falls_back_to_reference(self, fitted):
        """One-hot level sets are fitted too. If a level seen only at scoring
        time created a column, the design matrix would not match the trained
        coefficient vector and scoring would either crash or silently
        mis-align every coefficient."""
        assert "modality_group=cell_therapy" not in fitted.columns
        assert fitted.state.nominal_reference["modality_group"] == "small_molecule"
        assert fitted.state.nominal_levels["modality_group"] == ["antibody"]

        raw = fitted.transform_raw([{"modality_group": "cell_therapy"}])
        col = fitted.columns.index("modality_group=antibody")
        assert raw[0, col] == 0.0
        assert raw.shape[1] == len(fitted.columns)

    def test_transform_standardises_the_fit_set(self, fit_rows, fitted):
        """The explanation layer reads coefficients as effects per standard
        deviation and treats the origin as the cohort-average program. Both
        claims are false if the fit set is not centred and scaled, so every
        contribution shown to a user would be misattributed."""
        X = fitted.transform(fit_rows)
        assert X.shape == (N_FIT_ROWS, len(fitted.columns))
        assert np.abs(X.mean(axis=0)).max() < 1e-8

        raw_sd = fitted.transform_raw(fit_rows).std(axis=0)
        informative = raw_sd > 1e-8
        sd = X.std(axis=0)
        assert np.allclose(sd[informative], 1.0, atol=1e-8)
        # Degenerate columns are left alone rather than divided by ~0.
        assert np.allclose(sd[~informative], 0.0, atol=1e-12)

        varying = {c for c, keep in zip(fitted.columns, informative, strict=True) if keep}
        assert varying == {
            "animal_positive_fraction",
            "species_concordance_fraction",
            "exposure_margin_log2",
            "animal_n_invivo_studies",
            "exposure_margin_log2__missing",
            "modality_group=antibody",
        }
        # The cohort-average row really is the origin.
        assert np.allclose(fitted.reference_row_standardised(), 0.0)

    def test_missing_indicators_are_created_only_within_the_rate_band(self, fitted):
        """Outside the band the indicator is near-constant: it contributes
        nothing but absorbs penalty, and — worse — a 99%-missing feature's
        indicator becomes a near-duplicate of the intercept, destabilising the
        fit. The band decision is recorded so it stays auditable."""
        state = fitted.state
        assert INDICATOR_RATE_BAND == (0.02, 0.98)
        assert state.missing_rates["animal_positive_fraction"] == pytest.approx(0.0)
        assert state.missing_rates["species_concordance_fraction"] == pytest.approx(0.01)
        assert state.missing_rates["exposure_margin_log2"] == pytest.approx(0.5)
        assert state.missing_rates["therapeutic_index_log2"] == pytest.approx(1.0)

        assert state.indicators == ["exposure_margin_log2"]
        indicator_columns = [c for c in fitted.columns if c.endswith("__missing")]
        assert indicator_columns == ["exposure_margin_log2__missing"]
        for name in ("animal_positive_fraction", "species_concordance_fraction",
                     "therapeutic_index_log2"):
            assert f"{name}__missing" not in fitted.columns

    def test_zero_is_absence_features_never_get_an_indicator(self, fitted):
        """For these features "missing" means zero by construction, so an
        indicator would encode nothing and would be collinear with the value
        itself. animal_n_invivo_studies sits at a 50% missing rate here —
        squarely inside the band — so only the policy can be keeping it out."""
        spec = FEATURES_BY_NAME["animal_n_invivo_studies"]
        assert spec.missing_policy is MissingPolicy.ZERO_IS_ABSENCE
        rate = fitted.state.missing_rates["animal_n_invivo_studies"]
        assert INDICATOR_RATE_BAND[0] <= rate <= INDICATOR_RATE_BAND[1]
        assert "animal_n_invivo_studies__missing" not in fitted.columns

        declared_zero_is_absence = {
            f.name for f in FEATURES if f.missing_policy is MissingPolicy.ZERO_IS_ABSENCE
        }
        assert declared_zero_is_absence.isdisjoint(set(fitted.state.indicators))

    def test_multiple_imputation_varies_only_the_missing_features(self, fitted):
        """Rubin's between-imputation variance is the "what we do not know"
        term of the reported interval. If a present feature also wobbled across
        draws, that term would absorb noise the program's evidence does not
        have, and the interval would widen for reasons no user could trace."""
        row = {
            "animal_positive_fraction": 0.7,
            "species_concordance_fraction": 0.5,
            "animal_n_invivo_studies": 3.0,
            "exposure_margin_log2": None,          # the only missing feature
            "modality_group": "antibody",
        }
        draws = fitted.transform_with_imputations(row, 64, np.random.default_rng(3))
        assert draws.shape == (64, len(fitted.columns))

        varying = {
            fitted.columns[i] for i in range(draws.shape[1]) if draws[:, i].std() > 1e-9
        }
        assert varying == {"exposure_margin_log2"}

        present_col = fitted.columns.index("animal_positive_fraction")
        missing_col = fitted.columns.index("exposure_margin_log2")
        assert draws[:, present_col].std() < 1e-12
        assert len(set(np.round(draws[:, present_col], 10))) == 1
        assert draws[:, missing_col].std() > 0.1
        assert len(set(np.round(draws[:, missing_col], 9))) > 5

        # The indicator stays at "missing" in every draw: imputing a value does
        # not pretend the value was observed.
        ind_col = fitted.columns.index("exposure_margin_log2__missing")
        assert draws[:, ind_col].std() < 1e-12
        assert draws[:, ind_col][0] != fitted.transform([{**row, "exposure_margin_log2": 1.0}])[
            0, ind_col
        ]

    def test_imputation_draws_collapse_to_the_point_estimate_when_nothing_is_missing(
        self, fitted
    ):
        """A fully documented program must not be handed a spurious
        between-imputation interval."""
        complete = {
            name: 1.0 for name in fitted.state.numeric_features
        } | {"modality_group": "antibody"}
        draws = fitted.transform_with_imputations(complete, 8, np.random.default_rng(0))
        assert draws.shape == (8, len(fitted.columns))
        assert np.allclose(draws.std(axis=0), 0.0, atol=1e-12)
        assert np.allclose(draws[0], fitted.transform([complete])[0])

    def test_column_origin_maps_a_design_column_back_to_its_feature(self, fitted):
        """Explanations, sign audits and scenario diffs all address columns and
        must report them as features a scientist recognises. Break this and the
        UI shows "exposure_margin_log2__missing" as if it were an observation."""
        assert fitted.column_origin("animal_positive_fraction") == (
            "animal_positive_fraction", FEATURES_BY_NAME["animal_positive_fraction"].label,
        )
        base, label = fitted.column_origin("exposure_margin_log2__missing")
        assert base == "exposure_margin_log2"
        assert label.endswith("not reported")
        assert FEATURES_BY_NAME["exposure_margin_log2"].label in label

        base, label = fitted.column_origin("modality_group=antibody")
        assert base == "modality_group"
        assert label == "Modality class: antibody"

        # Every emitted column resolves to a declared feature.
        for column in fitted.columns:
            origin, _ = fitted.column_origin(column)
            assert origin in FEATURES_BY_NAME, column

    def test_use_before_fit_raises_rather_than_guessing(self):
        """An unfitted transformer has no medians and no columns. Silently
        producing an empty or zero matrix would score every program against a
        matrix the model was never trained on."""
        with pytest.raises(RuntimeError, match="before fit"):
            FeatureTransformer().transform([{"animal_positive_fraction": 0.5}])
        with pytest.raises(RuntimeError, match="before fit"):
            _ = FeatureTransformer().columns


# ═════════════════════════════════════════════════════════════════════════════
# 4-5. Completeness and the evidence-confidence grade
# ═════════════════════════════════════════════════════════════════════════════

GRADE_ORDER = [CL.VERY_LOW, CL.LOW, CL.MODERATE, CL.HIGH]


def _all_present() -> dict[str, str]:
    return {f.name: str(MR.PRESENT) for f in FEATURES}


def _report(overall: float, critical_gaps: tuple[str, ...] = ()) -> CompletenessReport:
    return CompletenessReport(
        overall=overall, by_domain={}, n_present=0, n_total=len(FEATURES),
        n_not_applicable=0, critical_gaps=critical_gaps, reason_counts={},
    )


def _threshold_grade(overall: float) -> CL:
    for threshold, level in GRADE_THRESHOLDS:
        if overall >= threshold:
            return level
    return CL.VERY_LOW


class TestCompleteness:
    """trs/features/missingness.py — what fraction of what matters is present."""

    def test_all_present_is_exactly_one(self):
        """"100% complete" is shown next to the probability. If a fully
        documented program cannot reach 1.0, the grade thresholds are
        unreachable and every estimate is under-qualified."""
        report = compute_completeness(_all_present())
        assert report.overall == 1.0
        assert report.n_present == len(FEATURES)
        assert report.n_total == len(FEATURES)
        assert report.critical_gaps == ()
        assert set(report.by_domain) == {str(d) for d in FeatureDomain}
        assert all(v == 1.0 for v in report.by_domain.values())
        assert report.reason_counts == {str(MR.PRESENT): len(FEATURES)}

    def test_zero_is_absence_features_are_excluded_from_the_denominator(self):
        """These features are present by construction — "no non-human-primate
        study exists" is a value, not a gap. Counting them would pin every
        score to a constant floor and make completeness uninformative, exactly
        where it is used to qualify the estimate."""
        zero_is_absence = [
            f.name for f in FEATURES if f.missing_policy is MissingPolicy.ZERO_IS_ABSENCE
        ]
        assert zero_is_absence, "fixture assumes at least one such feature"

        missingness = _all_present()
        for name in zero_is_absence:
            missingness[name] = str(MR.NOT_REPORTED)
        report = compute_completeness(missingness)

        assert report.overall == 1.0
        assert report.n_present == len(FEATURES) - len(zero_is_absence)
        assert report.reason_counts[str(MR.NOT_REPORTED)] == len(zero_is_absence)
        # Conversely, a penalised feature really does move the score.
        penalised = next(
            f for f in FEATURES if f.missing_policy is not MissingPolicy.ZERO_IS_ABSENCE
        )
        missingness[penalised.name] = str(MR.NOT_REPORTED)
        assert compute_completeness(missingness).overall < 1.0

    def test_not_applicable_does_not_count_against_a_program(self):
        """"Not applicable" and "not reported" are different facts. Conflating
        them marks every monotherapy incomplete for lacking combination data,
        and the grade then punishes programs for questions that do not arise."""
        applicable_specs = [
            f for f in features_in_domain(FeatureDomain.REPRODUCIBILITY)
            if f.missing_policy is not MissingPolicy.ZERO_IS_ABSENCE
        ][:4]
        assert applicable_specs

        missingness = _all_present()
        for spec in applicable_specs:
            missingness[spec.name] = str(MR.NOT_APPLICABLE)
        report = compute_completeness(missingness)

        assert report.overall == 1.0
        assert report.by_domain[str(FeatureDomain.REPRODUCIBILITY)] == 1.0
        assert report.n_not_applicable == len(applicable_specs)
        assert report.n_present == len(FEATURES) - len(applicable_specs)

        # NOT_REPORTED on the same features is penalised, which is the contrast
        # the whole distinction exists to make.
        not_reported = _all_present()
        for spec in applicable_specs:
            not_reported[spec.name] = str(MR.NOT_REPORTED)
        assert compute_completeness(not_reported).overall < 1.0

    def test_critical_gap_is_flagged_when_a_critical_domain_is_absent(self):
        """A program with no PK/PD bridge cannot be assessed with confidence
        however complete the rest of the record is. Without the callout, a
        high headline completeness would hide a hole in the one domain that
        most often decides translation."""
        missingness = _all_present()
        for spec in features_in_domain(FeatureDomain.PKPD_TRANSLATION):
            missingness[spec.name] = str(MR.NOT_REPORTED)
        report = compute_completeness(missingness)

        assert report.by_domain[str(FeatureDomain.PKPD_TRANSLATION)] == 0.0
        assert report.critical_gaps == (str(FeatureDomain.PKPD_TRANSLATION),)
        assert report.overall > 0.8  # the rest of the record is complete
        assert str(FeatureDomain.PKPD_TRANSLATION) in {str(d) for d in CRITICAL_DOMAINS}

        # A non-critical domain being empty is not a critical gap.
        other = _all_present()
        for spec in features_in_domain(FeatureDomain.EVIDENCE_VOLUME):
            other[spec.name] = str(MR.NOT_REPORTED)
        assert compute_completeness(other).critical_gaps == ()

        # The threshold is a real threshold, not a non-null check.
        half = _all_present()
        pkpd = [f for f in features_in_domain(FeatureDomain.PKPD_TRANSLATION)]
        for spec in pkpd[: len(pkpd) // 3]:
            half[spec.name] = str(MR.NOT_REPORTED)
        partial = compute_completeness(half)
        assert partial.by_domain[str(FeatureDomain.PKPD_TRANSLATION)] > 0.5
        assert partial.critical_gaps == ()

    def test_missing_features_default_to_not_extracted_rather_than_present(self):
        """A feature absent from the missingness map must not be assumed
        present. An empty map should score 0, not 1 — otherwise a builder bug
        that dropped the map would report perfect evidence for everything."""
        report = compute_completeness({})
        assert report.overall == 0.0
        assert report.n_present == 0
        assert report.reason_counts == {str(MR.NOT_EXTRACTED): len(FEATURES)}


class TestEvidenceConfidenceGrade:
    """trs/features/missingness.py — the qualifier shown beside the number."""

    @pytest.mark.parametrize("overall", [1.0, 0.9, 0.85, 0.7, 0.65, 0.5, 0.4, 0.2, 0.0])
    def test_grade_never_upgrades(self, overall):
        """The grade starts at weighted completeness and only ever falls. If
        any check could raise it, a narrow interval computed on four data
        points would advertise itself as strong evidence — but a narrow
        interval on thin evidence is a property of the model, not the
        evidence, and a user acting on it would be misled twice over."""
        baseline = _threshold_grade(overall)
        for n_items in (0, 1, 4, 5, 11, 12, 500):
            for width in (None, 0.01, 0.4, 0.9):
                for gaps in ((), (str(FeatureDomain.PKPD_TRANSLATION),)):
                    grade, reasons = grade_evidence_confidence(
                        _report(overall, gaps),
                        n_evidence_items=n_items,
                        interval_width=width,
                    )
                    assert GRADE_ORDER.index(grade) <= GRADE_ORDER.index(baseline), (
                        overall, n_items, width, gaps, grade,
                    )
                    assert reasons  # the grade always explains itself

    def test_favourable_inputs_leave_the_threshold_grade_untouched(self):
        """No gratuitous downgrade either: the grade must be a function of the
        evidence, so a complete package with a deep evidence base keeps the
        grade its completeness earned."""
        for overall in (1.0, 0.86, 0.7, 0.45, 0.1):
            grade, _ = grade_evidence_confidence(
                _report(overall), n_evidence_items=200, interval_width=0.05
            )
            assert grade == _threshold_grade(overall)

    def test_thin_evidence_and_critical_gaps_downgrade(self):
        """Each downgrade must actually bite, and must be stated in words the
        UI can show. A downgrade that is recorded but not applied is worse
        than none: the reason text would contradict the grade."""
        full = _report(1.0)
        assert grade_evidence_confidence(full, n_evidence_items=40)[0] == CL.HIGH

        thin, reasons = grade_evidence_confidence(full, n_evidence_items=2)
        assert thin == CL.MODERATE
        assert any("2 evidence item" in r for r in reasons)

        gapped, reasons = grade_evidence_confidence(
            _report(1.0, (str(FeatureDomain.PKPD_TRANSLATION),)), n_evidence_items=40
        )
        assert gapped == CL.MODERATE
        assert any("PK/PD translation" in r for r in reasons)

        both, _ = grade_evidence_confidence(
            _report(1.0, (str(FeatureDomain.PKPD_TRANSLATION),)), n_evidence_items=1
        )
        assert both == CL.LOW  # two independent downgrades compose

        _, reasons = grade_evidence_confidence(full, n_evidence_items=40, interval_width=0.5)
        assert any("Wide uncertainty interval" in r for r in reasons)

    def test_synthetic_contamination_forces_very_low_regardless_of_completeness(self):
        """The demonstration dataset is simulated. A perfectly complete
        synthetic program would otherwise be graded "high" and read as a
        scientific claim about a real asset — the single most damaging thing
        this product could do."""
        for overall in (1.0, 0.9, 0.5, 0.0):
            for gaps in ((), (str(FeatureDomain.PKPD_TRANSLATION),)):
                grade, reasons = grade_evidence_confidence(
                    _report(overall, gaps),
                    n_evidence_items=10_000,
                    interval_width=0.01,
                    contains_synthetic=True,
                )
                assert grade == CL.VERY_LOW
                assert "synthetic" in reasons[0].lower()
                assert "no scientific interpretation" in reasons[0].lower()


# ═════════════════════════════════════════════════════════════════════════════
# 6. Cohort outcomes
# ═════════════════════════════════════════════════════════════════════════════

AS_OF = dt.date(2015, 1, 1)
DATA_CUTOFF = dt.date(2026, 1, 1)


def _track(**overrides) -> ProgramIndication:
    """A transient development track. Never flushed: the endpoint logic reads
    plain attributes, so no database round-trip is needed to exercise it."""
    fields = {
        "program_id": "prog", "indication_id": "ind",
        "ind_or_cta_on": AS_OF,
        "first_in_human_on": AS_OF + dt.timedelta(days=120),
        "outcome_label": str(O.UNKNOWN),
        "discontinuation_reason": str(R.UNKNOWN),
        "outcome_date": None,
        "status_verified_on": DATA_CUTOFF,
    }
    fields.update(overrides)
    return ProgramIndication(**fields)


def _label(track, endpoint=PH2_EFFICACY_FAILURE, as_of=AS_OF) -> dict:
    return dict(label_track(track, endpoint, as_of=as_of, data_cutoff=DATA_CUTOFF))


class TestCohortOutcomes:
    """trs/cohort/outcomes.py — observed histories to labels, and exclusions."""

    @pytest.mark.parametrize("reason", sorted(NON_SCIENTIFIC_REASONS))
    def test_commercial_discontinuation_is_excluded_not_labelled_a_failure(self, reason):
        """This is the single most important scientific rule in the cohort
        layer. A program shelved for portfolio reasons says nothing about
        whether its biology would have worked. Label it a failure and the model
        learns to predict corporate behaviour — sponsor size, competitive
        crowding, cash position — while the product claims to predict
        translation. Every explanation it produced would then be a fiction."""
        stopped = _track(
            outcome_label=str(O.DISCONTINUED_THROUGH_PHASE_2), discontinuation_reason=reason
        )
        decision = _label(stopped)
        assert decision["label"] is None
        assert decision["excluded"] is True
        assert decision["exclusion_reason"] == f"non_scientific_discontinuation:{reason}"

        # Even when the recorded outcome *says* efficacy failure, a
        # non-scientific reason wins and the row leaves the cohort.
        mislabelled = _track(
            outcome_label=str(O.PHASE_2_EFFICACY_FAILURE), discontinuation_reason=reason
        )
        assert _label(mislabelled)["label"] is None
        assert _label(mislabelled)["excluded"] is True

        # ... and it is excluded from the composite endpoint too.
        assert _label(stopped, PH2_ANY_SCIENTIFIC_FAILURE)["excluded"] is True

    def test_safety_failure_is_a_competing_event_but_positive_for_the_composite(self):
        """A program stopped for toxicity never got the chance to demonstrate
        efficacy, so it is neither a positive nor a negative for the primary
        cause-specific endpoint. Counting it as a negative would deflate the
        estimated efficacy-failure rate; counting it as a positive would
        conflate two different failure modes. The composite endpoint exists so
        the unconditional question can still be asked."""
        toxic = _track(
            outcome_label=str(O.PHASE_2_SAFETY_FAILURE),
            discontinuation_reason=str(R.SAFETY_TOXICITY),
            outcome_date=dt.date(2018, 6, 1),
        )
        primary = _label(toxic)
        assert primary["label"] is None
        assert primary["excluded"] is True
        assert primary["exclusion_reason"] == f"outcome_excluded:{O.PHASE_2_SAFETY_FAILURE}"

        composite = _label(toxic, PH2_ANY_SCIENTIFIC_FAILURE)
        assert composite["label"] == 1
        assert composite["excluded"] is False

        # Phase I safety and PK/PD failures behave the same way.
        for outcome in (O.PHASE_1_SAFETY_FAILURE, O.PHASE_1_PKPD_FAILURE):
            track = _track(
                outcome_label=str(outcome),
                discontinuation_reason=str(R.SAFETY_TOXICITY),
                outcome_date=dt.date(2017, 1, 1),
            )
            assert _label(track)["excluded"] is True
            assert _label(track, PH2_ANY_SCIENTIFIC_FAILURE)["label"] == 1

    @pytest.mark.parametrize(
        ("reason", "endpoint", "expected_label", "expected_exclusion"),
        [
            (R.EFFICACY_INSUFFICIENT, PH2_EFFICACY_FAILURE, 1, None),
            (R.EFFICACY_INSUFFICIENT, PH2_ANY_SCIENTIFIC_FAILURE, 1, None),
            (R.SAFETY_TOXICITY, PH2_EFFICACY_FAILURE, None, "competing_event:safety_toxicity"),
            (R.PK_PD_INADEQUATE, PH2_EFFICACY_FAILURE, None, "competing_event:pk_pd_inadequate"),
            (R.SAFETY_TOXICITY, PH2_ANY_SCIENTIFIC_FAILURE, 1, None),
            (R.PK_PD_INADEQUATE, PH2_ANY_SCIENTIFIC_FAILURE, 1, None),
            (R.EFFICACY_INSUFFICIENT, PH1_SAFETY_FAILURE, 0, None),
            (
                R.COMMERCIAL_STRATEGIC, PH2_EFFICACY_FAILURE, None,
                "non_scientific_discontinuation:commercial_strategic",
            ),
            (
                R.UNDISCLOSED, PH2_EFFICACY_FAILURE, None,
                "undisclosed_discontinuation_reason:undisclosed",
            ),
            (
                R.REGULATORY_REQUIREMENT, PH2_EFFICACY_FAILURE, None,
                "undisclosed_discontinuation_reason:regulatory_requirement",
            ),
        ],
    )
    def test_composite_discontinuation_is_resolved_by_reason(
        self, reason, endpoint, expected_label, expected_exclusion
    ):
        """DISCONTINUED_THROUGH_PHASE_2 records "stopped at or before Phase II"
        without saying why; the reason field disambiguates it. Resolve it
        wrongly and the largest ambiguous bucket in the data is silently
        assigned to the wrong class — the positives would be contaminated with
        toxicity stops, or genuine efficacy failures would be thrown away."""
        track = _track(
            outcome_label=str(O.DISCONTINUED_THROUGH_PHASE_2),
            discontinuation_reason=str(reason),
            outcome_date=dt.date(2019, 3, 1),
        )
        decision = resolve_composite_discontinuation(track, endpoint)
        assert decision is not None
        assert decision["label"] == expected_label
        assert decision["excluded"] is (expected_label is None)
        assert decision.get("exclusion_reason") == expected_exclusion
        if expected_label is not None:
            assert decision["source_outcome"] == (
                f"{O.DISCONTINUED_THROUGH_PHASE_2}:{reason}"
            )
        # label_track must route through the resolver, not around it.
        assert dict(
            label_track(track, endpoint, as_of=AS_OF, data_cutoff=DATA_CUTOFF)
        ) == dict(decision)

    def test_resolver_defers_on_every_other_outcome(self):
        """The resolver must return None — not a label — for outcomes it does
        not own, or it would short-circuit the endpoint definitions and every
        unambiguous outcome would be classified twice, by different rules."""
        for outcome in O:
            if outcome is O.DISCONTINUED_THROUGH_PHASE_2:
                continue
            track = _track(
                outcome_label=str(outcome),
                discontinuation_reason=str(R.EFFICACY_INSUFFICIENT),
            )
            assert resolve_composite_discontinuation(track, PH2_EFFICACY_FAILURE) is None

    def test_insufficient_followup_excludes_rather_than_counting_as_a_negative(self):
        """A program that entered the clinic two years ago has not yet had the
        opportunity to fail. Calling it a success biases every estimated
        failure rate downward, and does so worst for the most recent cohort —
        precisely the programs a user is asking about."""
        recent = _track(
            outcome_label=str(O.PROGRESSED_PAST_PHASE_2),
            discontinuation_reason=str(R.NOT_DISCONTINUED),
            outcome_date=None,
            status_verified_on=AS_OF + dt.timedelta(days=700),
        )
        decision = _label(recent)
        assert decision["label"] is None
        assert decision["excluded"] is True
        assert decision["exclusion_reason"].startswith("insufficient_followup:")
        assert "<5.0y" in decision["exclusion_reason"]

        mature = _track(
            outcome_label=str(O.PROGRESSED_PAST_PHASE_2),
            discontinuation_reason=str(R.NOT_DISCONTINUED),
            outcome_date=None,
            status_verified_on=AS_OF + dt.timedelta(days=2600),
        )
        assert _label(mature)["label"] == 0

        # A resolved outcome needs no follow-up window: the event is observed.
        observed = _track(
            outcome_label=str(O.PROGRESSED_PAST_PHASE_2),
            discontinuation_reason=str(R.NOT_DISCONTINUED),
            outcome_date=AS_OF + dt.timedelta(days=400),
            status_verified_on=AS_OF + dt.timedelta(days=400),
        )
        assert _label(observed)["label"] == 0

        # The window is per endpoint: the Phase I endpoints need only 2 years,
        # so a track with 3.3 years of observation is censored for the primary
        # endpoint and usable for those.
        assert PH1_SAFETY_FAILURE.min_followup_years == 2.0
        middling = _track(
            outcome_label=str(O.PROGRESSED_PAST_PHASE_2),
            discontinuation_reason=str(R.NOT_DISCONTINUED),
            outcome_date=None,
            status_verified_on=AS_OF + dt.timedelta(days=1200),
        )
        assert _label(middling)["exclusion_reason"].startswith("insufficient_followup:")
        assert _label(middling, PH1_SAFETY_FAILURE)["label"] == 0
        assert _label(recent, PH1_SAFETY_FAILURE)["exclusion_reason"].startswith(
            "insufficient_followup:"
        )

    def test_a_program_that_never_dosed_a_human_is_excluded_from_clinical_endpoints(self):
        """The primary estimand is conditional on entering human development.
        Including never-dosed programs would answer a different question than
        the one the API response states."""
        never = _track(
            ind_or_cta_on=None, first_in_human_on=None,
            outcome_label=str(O.PHASE_2_EFFICACY_FAILURE),
            discontinuation_reason=str(R.EFFICACY_INSUFFICIENT),
        )
        decision = _label(never)
        assert decision["excluded"] is True
        assert decision["exclusion_reason"] == "never_entered_human_development"
        assert PH2_EFFICACY_FAILURE.requires_human_entry is True

    def test_reaching_phase_three_clears_the_phase_two_bar(self):
        """A Phase III failure is a *negative* for a Phase II efficacy
        endpoint: the program demonstrated enough efficacy to progress. Score
        it positive and the model is trained to call successes failures."""
        ph3 = _track(
            outcome_label=str(O.PHASE_3_FAILURE),
            discontinuation_reason=str(R.EFFICACY_INSUFFICIENT),
            outcome_date=dt.date(2021, 1, 1),
        )
        assert _label(ph3)["label"] == 0
        approved = _track(
            outcome_label=str(O.REGULATORY_APPROVAL),
            discontinuation_reason=str(R.NOT_DISCONTINUED),
            outcome_date=dt.date(2020, 1, 1),
        )
        assert _label(approved)["label"] == 0
        failed = _track(
            outcome_label=str(O.PHASE_2_EFFICACY_FAILURE),
            discontinuation_reason=str(R.EFFICACY_INSUFFICIENT),
            outcome_date=dt.date(2017, 5, 1),
        )
        assert _label(failed)["label"] == 1

    def test_every_endpoint_partitions_its_outcome_vocabulary(self):
        """Positive, negative and excluded sets must not overlap, and the
        primary endpoint must state its conditional estimand. An outcome in two
        sets would be classified by whichever branch ran first."""
        for name, endpoint in ENDPOINTS.items():
            assert not endpoint.positive_outcomes & endpoint.negative_outcomes, name
            assert not endpoint.positive_outcomes & endpoint.excluded_outcomes, name
            assert not endpoint.negative_outcomes & endpoint.excluded_outcomes, name
        assert PRIMARY_ENDPOINT.is_primary is True
        assert "conditional on not being withdrawn" in PRIMARY_ENDPOINT.estimand


# ═════════════════════════════════════════════════════════════════════════════
# 6b. build_cohort on the seeded universe
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildCohort:
    """trs/cohort/build.py — the attrition waterfall over the seeded data.

    All counts below come from the synthetic demonstration universe. They are
    tested as invariants (sums, bounds, nullity), never as headline figures.
    """

    def test_every_candidate_is_either_included_or_excluded_with_a_reason(
        self, seeded_session, today
    ):
        """The Methods page shows an attrition waterfall, not a final N. If the
        counts do not reconcile, rows are vanishing somewhere unrecorded and
        the published cohort size cannot be audited."""
        session, _ = seeded_session
        result = build_cohort(session, data_cutoff=today, persist=False)

        n_tracks = session.scalar(select(func.count()).select_from(ProgramIndication))
        assert result.n_candidates == n_tracks
        assert result.n_candidates == len(result.entries) + sum(result.exclusions.values())

        summary = result.summary
        assert summary["n_included"] + summary["n_excluded"] == summary["n_candidates"]
        assert summary["n_included"] == len(result.entries) > 0
        assert all(count > 0 for count in result.exclusions.values())
        assert all(reason for reason in result.exclusions)

        cv = result.cohort_version
        assert cv.n_entries == len(result.entries)
        assert cv.n_positive + cv.n_negative == cv.n_entries
        assert cv.n_excluded == sum(result.exclusions.values())
        assert cv.contains_synthetic is True  # the demo universe is simulated

    def test_every_included_entry_carries_a_label(self, seeded_session, today):
        """An entry with a null label would reach the training matrix as an
        unlabelled row. Downstream that is either a crash or, worse, a silent
        cast to 0 — thousands of fabricated successes."""
        session, _ = seeded_session
        result = build_cohort(session, data_cutoff=today, persist=False)

        for entry in result.entries:
            assert entry.label_primary in (0, 1)
            assert entry.exclusion_reason is None
            assert entry.label_source_outcome
            assert entry.as_of_date is not None
            assert entry.group_program_id
            # Secondary endpoints are evaluated for every entry; None there is
            # legitimate (excluded for that endpoint) but the key must exist.
            assert set(entry.labels_secondary) == set(ENDPOINTS) - {PRIMARY_ENDPOINT.name}
            for value in entry.labels_secondary.values():
                assert value in (0, 1, None)

    def test_no_entry_has_an_as_of_date_after_the_data_cutoff(self, seeded_session, today):
        """The as-of date is the temporal chokepoint: features are built from
        evidence available at that date. A cutoff that is not enforced means
        the model is trained on facts that did not exist at the decision point,
        and the held-out metrics become unachievable in deployment."""
        session, _ = seeded_session
        result = build_cohort(session, data_cutoff=today, persist=False)
        assert all(entry.as_of_date <= today for entry in result.entries)

        # Moving the cutoff earlier must shrink the cohort and re-enforce the
        # bound, rather than leaving the same rows in place.
        earlier = dt.date(2015, 1, 1)
        clipped = build_cohort(session, data_cutoff=earlier, persist=False)
        assert all(entry.as_of_date <= earlier for entry in clipped.entries)
        assert len(clipped.entries) < len(result.entries)
        assert clipped.exclusions["as_of_after_data_cutoff"] > 0
        assert clipped.n_candidates == result.n_candidates

    def test_as_of_rule_never_falls_forward_to_a_later_milestone(self, seeded_session, today):
        """Falling forward would admit Phase I results into a preclinical
        feature vector. An unknown rule must fail loudly rather than quietly
        selecting a permissive one."""
        session, _ = seeded_session
        with pytest.raises(ValueError, match="Unknown as_of_rule"):
            build_cohort(session, as_of_rule="phase_3_start_on", persist=False)

        result = build_cohort(session, data_cutoff=today, persist=False)
        tracks = {
            t.id: t for t in session.execute(select(ProgramIndication)).scalars().all()
        }
        for entry in result.entries:
            track = tracks[entry.program_indication_id]
            expected = track.ind_or_cta_on or track.first_in_human_on
            assert entry.as_of_date == expected
            if track.phase_2_start_on is not None:
                assert entry.as_of_date <= track.phase_2_start_on


# ═════════════════════════════════════════════════════════════════════════════
# 7. Scenario levers
# ═════════════════════════════════════════════════════════════════════════════


class TestScenarioLevers:
    """trs/scoring/scenarios.py — counterfactual evaluations of the model."""

    def test_validate_overrides_rejects_a_non_adjustable_feature(self):
        """Only features a team could change by generating evidence may be
        varied. Allowing the historical failure rate on a target to be edited
        would let a user manufacture a favourable number by asserting that
        history was different, and the resulting scenario would read as
        analysis."""
        with pytest.raises(ScenarioValidationError, match="not adjustable"):
            validate_overrides({"target_prior_failure_rate": 0.1})
        with pytest.raises(ScenarioValidationError, match="not adjustable"):
            validate_overrides({"disease_stage_ord": 1.0})
        with pytest.raises(ScenarioValidationError, match="Unknown feature"):
            validate_overrides({"not_a_feature": 1.0})
        with pytest.raises(ScenarioValidationError, match="numeric"):
            validate_overrides({"animal_positive_fraction": "high"})

        non_adjustable = [f.name for f in FEATURES if not f.scenario_adjustable]
        assert non_adjustable
        for name in non_adjustable:
            with pytest.raises(ScenarioValidationError):
                validate_overrides({name: 1.0})

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("animal_positive_fraction", 1.5),
            ("animal_positive_fraction", -0.2),
            ("exposure_margin_log2", 99.0),
            ("exposure_margin_log2", -99.0),
            ("target_engagement_ord", 7.0),
            ("genetic_evidence_ord", -1.0),
            ("species_concordance_fraction", 1.01),
        ],
    )
    def test_validate_overrides_rejects_out_of_range_values(self, name, value):
        """Bounds are the declared domain of each feature. An out-of-range
        override is extrapolation past anything the model saw in training, and
        the probability it returns would be an artefact of the link function."""
        with pytest.raises(ScenarioValidationError, match="declared (minimum|maximum)"):
            validate_overrides({name: value})

        # The boundary itself is legal, so the check is a bound and not a gap.
        spec = FEATURES_BY_NAME[name]
        edge = spec.valid_max if value > 0 else spec.valid_min
        assert validate_overrides({name: edge}) == {name: float(edge)}

    def test_improvement_lever_is_a_no_op_on_an_already_better_program(self):
        """This was a real bug. Applied unconditionally, "add an independent
        replication" *lowered* the replication score of a program already
        replicated in several laboratories and raised its estimated risk — so
        the scenario read as "commissioning this study would make the program
        worse". That is nonsense, and a team would have acted on it."""
        already_strong = {
            "replication_scope_ord": 4.0,   # independent labs, multiple
            "n_independent_labs": 5.0,
            "single_lab_dependence": 0.0,
        }
        resolved = resolve_levers(["independent_replication"], already_strong)
        assert resolved == already_strong

        # The lever still acts where there is room to act.
        weak = {
            "replication_scope_ord": 0.0, "n_independent_labs": 0.0,
            "single_lab_dependence": 1.0,
        }
        improved = resolve_levers(["independent_replication"], weak)
        assert improved["replication_scope_ord"] == 2.5
        assert improved["n_independent_labs"] == 2.0
        assert improved["single_lab_dependence"] == 0.0   # risk-increasing: moves down

        # Clamping is monotone per feature, not all-or-nothing.
        mixed = {"replication_scope_ord": 4.0, "n_independent_labs": 0.0}
        partial = resolve_levers(["independent_replication"], mixed)
        assert partial["replication_scope_ord"] == 4.0
        assert partial["n_independent_labs"] == 2.0

        # Without a current vector the catalogue values are returned verbatim.
        assert resolve_levers(["independent_replication"])["replication_scope_ord"] == 2.5

    def test_adverse_lever_moves_the_other_way_and_cannot_flatter(self):
        """A sensitivity analysis that only permits good news is a marketing
        tool. The mirror-image clamp is what makes the adverse levers an
        analysis: bad news can only make a program look worse."""
        healthy = {
            "animal_n_refuting_studies": 0.0,
            "animal_positive_fraction": 0.9,
            "species_concordance_fraction": 1.0,
        }
        worsened = resolve_levers(["contradictory_study_appears"], healthy)
        assert worsened["animal_n_refuting_studies"] == 1.0    # risk-increasing: up
        assert worsened["animal_positive_fraction"] == 0.6     # risk-reducing: down
        assert worsened["species_concordance_fraction"] == 0.5

        # Applied to a program that is already worse, it changes nothing.
        already_bad = {
            "animal_n_refuting_studies": 3.0,
            "animal_positive_fraction": 0.3,
            "species_concordance_fraction": 0.2,
        }
        assert resolve_levers(["contradictory_study_appears"], already_bad) == already_bad

        adverse_pkpd = resolve_levers(
            ["exposure_gap_identified"],
            {"exposure_margin_log2": 3.0, "exposure_bridge_quality_ord": 4.0,
             "therapeutic_index_log2": 5.0},
        )
        assert adverse_pkpd["exposure_margin_log2"] == -1.5
        assert adverse_pkpd["exposure_bridge_quality_ord"] == 0.0
        assert adverse_pkpd["therapeutic_index_log2"] == 0.0

    def test_implied_levers_resolve_transitively(self):
        """Demonstrating target engagement in humans entails having a
        human-ready PD assay. If the implication did not resolve, the scenario
        would describe a state that cannot exist, and its delta would be
        attributed to the wrong action."""
        lever = LEVERS_BY_KEY["human_target_engagement"]
        assert lever.implies == ("pd_biomarker_ready",)

        resolved = resolve_levers(["human_target_engagement"])
        assert resolved == {
            "target_engagement_ord": 6.0,
            "exposure_bridge_quality_ord": 4.0,
            "biomarker_readiness_ord": 4.0,
            "biomarker_species_concordance_ord": 3.0,
        }
        # The implied lever is applied first, so the explicit one wins on any
        # shared feature, and naming both changes nothing.
        assert resolve_levers(["human_target_engagement", "pd_biomarker_ready"]) == resolved
        assert resolve_levers(["pd_biomarker_ready", "human_target_engagement"]) == resolved

        with pytest.raises(ScenarioValidationError, match="Unknown scenario lever"):
            resolve_levers(["no_such_lever"])

    def test_absolute_lever_asserts_its_state_outright(self):
        """Enrolling a biomarker-selected population is a trial-design choice,
        not an accumulating body of evidence, so it replaces rather than
        clamps. Clamping it would make "what if we restricted eligibility?"
        unanswerable for an all-comers program."""
        from trs.scoring.scenarios import LeverDirection

        lever = LEVERS_BY_KEY["biomarker_selected_population"]
        assert lever.direction is LeverDirection.ABSOLUTE
        for current in ({"biomarker_selected_population": 0.0}, {}, None):
            resolved = resolve_levers(["biomarker_selected_population"], current)
            assert resolved["biomarker_selected_population"] == 1.0
            assert resolved["has_predictive_selection_biomarker"] == 1.0

    def test_every_catalogue_lever_resolves_and_validates(self):
        """A lever is validated exactly like a hand-written override, so it
        cannot smuggle in a non-adjustable feature. A catalogue entry that
        failed validation would raise at request time, in the UI, for a user
        who did nothing wrong."""
        assert LEVERS
        for lever in LEVERS:
            resolved = resolve_levers([lever.key])
            assert resolved, lever.key
            for name in lever.overrides:
                assert FEATURES_BY_NAME[name].scenario_adjustable, f"{lever.key}:{name}"
            # The same lever applied to a concrete program still validates.
            current = {name: 0.0 for name in lever.overrides}
            assert resolve_levers([lever.key], current) is not None
            payload = lever.to_dict()
            assert payload["key"] == lever.key
            assert payload["action"]

    def test_apply_overrides_never_mutates_the_persisted_row(self):
        """Overrides are applied to a copy at scoring time. If the snapshot
        were mutated, speculation would contaminate the evidence record and
        could leak into a future retraining set."""
        original = {"replication_scope_ord": 0.0, "n_independent_labs": None}
        snapshot = dict(original)
        modified, changes = apply_overrides(
            original, {"replication_scope_ord": 2.5, "n_independent_labs": 2.0}
        )
        assert original == snapshot
        assert modified["replication_scope_ord"] == 2.5
        assert {c["feature"] for c in changes} == {
            "replication_scope_ord", "n_independent_labs"
        }
        was_missing = {c["feature"]: c["was_missing"] for c in changes}
        assert was_missing == {"replication_scope_ord": False, "n_independent_labs": True}

        # A no-op override is not reported as a change.
        _, none = apply_overrides({"replication_scope_ord": 2.5},
                                  {"replication_scope_ord": 2.5})
        assert none == []


# ═════════════════════════════════════════════════════════════════════════════
# 8. Entity aliases and leakage-safe grouping
# ═════════════════════════════════════════════════════════════════════════════


class TestEntityAliasesAndGrouping:
    """One asset must be one unit, however many names it carries."""

    def test_two_aliases_of_one_program_resolve_to_the_same_program_id(self, seeded_session):
        """An asset renamed at a hand-over — development code, then INN — would
        otherwise resolve to two programs. The cohort would then contain the
        same biological bet twice, once in training and once in test, and the
        model would be scored on its ability to recognise a molecule it had
        already seen."""
        session, _ = seeded_session
        multi_alias = [
            program for program in session.execute(select(Program)).scalars().all()
            if len(program.aliases) >= 2
        ]
        assert multi_alias, "seeded universe is expected to carry alias graphs"

        for program in multi_alias[:10]:
            aliases = program.aliases
            normalised = {a.alias_normalised for a in aliases}
            assert len(normalised) == len(aliases)  # genuinely different strings

            resolved = set()
            for alias in aliases:
                # Look the alias up the way the resolver does: fold, then match.
                hits = session.execute(
                    select(ProgramAlias.program_id).where(
                        ProgramAlias.alias_normalised
                        == normalise_compound(alias.alias.upper())
                    )
                ).scalars().all()
                assert hits, alias.alias
                resolved.update(hits)
            assert resolved == {program.id}

    def test_alias_normalisation_folds_case_and_punctuation(self, seeded_session):
        """The normalised form is the blocking key for entity resolution. If a
        raw or half-folded string were stored, "DEMO-EGFR-001" and
        "demo egfr 001" would resolve to different assets and the deduplication
        this whole layer exists for would not happen.

        Program aliases are keyed with :func:`normalise_compound`, which removes
        separators entirely rather than collapsing them to spaces: a development
        code is one token however a source chose to punctuate it. Using
        :func:`normalise_general` here instead would key "ABT-199" as ``abt 199``
        and silently fail to match the same molecule filed as ``ABT199``.
        """
        session, _ = seeded_session
        aliases = session.execute(select(ProgramAlias)).scalars().all()
        assert aliases

        for alias in aliases:
            # The stored key is what the resolver would compute for this name.
            assert alias.alias_normalised == normalise_compound(alias.alias)
            # ...and it satisfies the compound-key contract: lowercase, bare
            # alphanumerics, no separators of any kind left to vary on.
            assert alias.alias_normalised == alias.alias_normalised.lower()
            assert not re.search(r"[^a-z0-9]", alias.alias_normalised)
            assert alias.match_method

    def test_punctuation_variants_of_one_code_share_a_blocking_key(self):
        """The property the blocking key exists for, asserted directly rather
        than by re-deriving the implementation: every way a source might
        punctuate one development code has to collapse to one key, and two
        genuinely different codes must not."""
        variants = ["DEMO-EGFR-001", "demo egfr 001", "DEMO_EGFR_001",
                    "demoEGFR001", "  Demo-EGFR-001  ", "DEMO\u2013EGFR\u2013001"]
        keys = {normalise_compound(v) for v in variants}
        assert keys == {"demoegfr001"}, keys

        # The asymmetry that matters: a false merge is worse than a missed one.
        assert normalise_compound("ABT-199") != normalise_compound("ABT-1990")
        assert normalise_compound("DEMO-EGFR-001") != normalise_compound("DEMO-EGFR-002")

    def test_the_same_assets_tracks_land_in_one_component(self, seeded_session, today):
        """Components, not rows, are assigned to folds. If one program's three
        indications scattered across train and test, the held-out metric would
        be measuring memorisation of the asset rather than assessment of its
        evidence — the exact failure the splitter exists to prevent."""
        session, _ = seeded_session
        result = build_cohort(session, data_cutoff=today, persist=False)
        entries = result.entries

        compound_by_track = dict(
            session.execute(
                select(ProgramIndication.id, Program.primary_compound_id).join(
                    Program, Program.id == ProgramIndication.program_id
                )
            ).all()
        )
        keys = build_grouping_keys(
            entries, [compound_by_track.get(e.program_indication_id) for e in entries]
        )
        assert len(keys) == len(entries)
        assert all(set(k) == {"program", "compound", "target", "sponsor", "mechanism"}
                   for k in keys)

        grouper = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.IDENTITY])
        components = grouper.fit(keys)
        assert components.shape == (len(entries),)

        by_program: dict[str, set[int]] = {}
        for entry, component in zip(entries, components, strict=True):
            by_program.setdefault(entry.group_program_id, set()).add(int(component))
        assert all(len(comps) == 1 for comps in by_program.values())
        assert any(
            sum(1 for e in entries if e.group_program_id == pid) > 1 for pid in by_program
        ), "expected at least one multi-indication program to make this meaningful"

        summary = grouper.summary(components)
        assert summary["n_rows"] == len(entries)
        assert summary["n_components"] <= len(by_program)
        assert summary["largest_component"] >= 2

    def test_one_asset_under_two_program_ids_still_forms_one_component(self):
        """Grouping on the program id alone misses the case the entity layer
        exists for: one molecule carried under two development codes by two
        sponsors. The shared compound must close the chain, or that asset
        appears twice in the cohort and can straddle the split."""
        keys = [
            {"program": "P1", "compound": "C1", "target": "T1", "sponsor": "S1"},
            {"program": "P2", "compound": "C1", "target": "T1", "sponsor": "S2"},
            {"program": "P3", "compound": "C2", "target": "T1", "sponsor": "S3"},
            {"program": "P4", "compound": None, "target": "T9", "sponsor": "S4"},
        ]
        identity = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.IDENTITY])
        components = identity.fit(keys)
        assert components[0] == components[1]          # linked through the compound
        assert components[2] != components[0]          # a different asset, same target
        assert components[3] not in (components[0], components[2])
        assert len(set(components.tolist())) == 3

        # A row with no keys at all is its own component, never merged with the
        # other key-less rows.
        keyless = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.IDENTITY]).fit(
            [{"program": None, "compound": None}, {"program": None, "compound": None}]
        )
        assert keyless[0] != keyless[1]

        # The strict scope additionally merges target siblings; identity, which
        # the headline split uses, deliberately does not.
        strict = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.STRICT])
        strict_components = strict.fit(keys)
        assert strict_components[0] == strict_components[2]
        assert strict_components[3] != strict_components[0]
        assert len(set(strict_components.tolist())) == 2
