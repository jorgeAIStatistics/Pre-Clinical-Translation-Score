# Model Card — `ptrs-pef-004`

**Preclinical Translational Risk Scorer**, primary endpoint `ph2_efficacy_failure`.

> **Read this first.** This model was fitted to a **synthetic demonstration dataset**
> produced by `backend/trs/seed/generator.py`. Its coefficients, its metrics and
> every prediction it makes describe a documented simulation. They say nothing
> about real targets, real compounds or real therapeutic programs. The model has
> **no regulatory standing** and **no clinical validation**. On its own held-out
> data it is **not distinguishable from quoting the cohort base rate**
> (§[8](#8-caveats-and-recommendations)).

Companion document: [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) — where data comes
from, how it is tracked, and exactly what is real versus simulated here.

---

## Table of contents

1. [Model details](#1-model-details)
2. [Intended use](#2-intended-use)
3. [Out-of-scope uses](#3-out-of-scope-uses)
4. [Factors](#4-factors)
5. [Metrics](#5-metrics)
6. [Training data](#6-training-data--synthetic)
7. [Quantitative analyses](#7-quantitative-analyses)
8. [Caveats and recommendations](#8-caveats-and-recommendations)
9. [Ethical considerations](#9-ethical-considerations)
10. [How to reproduce every number here](#10-how-to-reproduce-every-number-here)

---

## 1. Model details

### 1.1 Identity

| Field | Value | Source of truth |
|---|---|---|
| Model version | `ptrs-pef-004` | `model_artifact.version` |
| Active in registry | `true` | `model_artifact.is_active` |
| Package version | `trs 0.1.0` | `trs/__init__.py` |
| Model family | `l2_logistic` | `trs/modeling/logistic.py` |
| Endpoint | `ph2_efficacy_failure` | `trs/cohort/outcomes.py` |
| Feature schema version | `fs-1.0.0` | `trs/features/spec.py` |
| Cohort version id | `01M39QGBG4S645QRPJRRVHH3C1` | `cohort_version.id` |
| Cohort name / version | `oncology_ph2_efficacy` / `v1` | `cohort_version` |
| Cohort dataset hash | `92cd561dfbbe222fa1cc8ff741058cbdec2cde226fc2af53a8b3072ebe331c1e` | `cohort_version.dataset_hash` (SHA-256 over the cohort definition plus sorted `(track, as_of, label)` members) |
| Training-matrix hash | `2aee684d46c1a413ee09ea30374c29011fae68e94277f5f7142dbb95dbff150b` | `model_artifact.dataset_hash` (SHA-256 over columns, `X` rounded to 1e-10, and `y`) |
| Cohort built at | `2026-09-24T12:51:03.811594+00:00` | `cohort_version.built_at` |
| Trained at | `2026-10-13T23:10:43.416456+00:00` | `model_artifact.trained_at` |
| `trained_on_synthetic` | `true` | `model_artifact.trained_on_synthetic` |
| **Status** | **`demo_only`** | `model_artifact.status`, derived by `trs/pipeline/status.py` against criteria `pc-1.0.0` |
| May this artifact serve a score? | Only for synthetic evidence, in an explicitly selected demonstration context | `SERVING_PERMITTED` + `check_eligibility()` |
| Split protocol | `temporal_outer_maturity` | `model_artifact.protocol` |
| Split manifest hash | `165cb164f07e…` | `metrics.protocol.split_manifest.manifest_hash` |
| Served prediction functional | `calibrated_plug_in` | `model_artifact.prediction_functional`, contract `pcx-1.0.0` |
| Artifact payload format | `2` | format `1` artifacts still load, with denying defaults |
| Git commit | `4e55da097a9b4035da94c25d86cdfafcf3a3c9aa` | `model_artifact.git_commit` — the commit the training run was made from. Earlier artifacts recorded `null` because the working tree had no commits yet, which the field reported honestly rather than guessing |

Two hashes exist because they answer different questions. The cohort hash
answers "is this the same set of labelled units?"; the training-matrix hash
answers "is this the same numeric matrix?" A feature-builder change moves the
second without moving the first, which is exactly the drift you want to catch.

### 1.2 Numerical environment

Recorded at training time by `capture_library_versions()`
(`trs/pipeline/artifacts.py`), because a refit under a different BLAS or a
different scikit-learn is not bit-identical:

```
python 3.14.5   numpy 2.5.3   scipy 1.18.1
pandas 3.0.6    sklearn 1.9.1  statsmodels 0.15.0
```

### 1.3 Architecture and fit

L2-penalised logistic regression with a Laplace (Gaussian) posterior over the
coefficients, fitted by hand-written IRLS. The choice is deliberate and its cost
is stated in §[7.5](#75-model-family-selection): a linear model gives an **exact
additive attribution** (driver contributions are terms of the linear predictor
and sum precisely to the reported log-odds) and an **analytic posterior** that
the uncertainty decomposition depends on. A gradient-boosted model would give
neither, and on this data it did not earn the trade.

| Property | Value |
|---|---|
| Design-matrix columns | 109 |
| — numeric / ordinal features | 61 |
| — missing indicators | 36 |
| — one-hot columns (2 nominal features) | 12 |
| Development rows | 187 (118 positive) |
| L2 penalty | 64.0 |
| `max_iter` / `tol` | 500 / 1e-8 |
| `class_weight_balanced` | `false` |
| Converged | `true`, in 4 IRLS iterations |
| Hessian condition number | 14.54 |
| Effective degrees of freedom | 110.0 |
| Max posterior SD (any coefficient) | 0.1035 |
| Intercept (standardised design) | 0.4240 |

Nominal features are one-hot encoded with an explicit reference level:
`modality_group` (reference `small_molecule`, 5 further levels) and
`target_class_group` (reference `kinase`, 7 further levels). Ordered categories
are encoded as integers instead, which asserts equal spacing between adjacent
levels — `trs/features/spec.py` names that assumption and defends it as a
variance/bias trade that is right at n in the hundreds, not as a fact.

### 1.4 Calibration

| Property | Value |
|---|---|
| Method | logistic recalibration |
| Design | cross-validated out-of-fold |
| Folds | 5, grouped by connected component, over the training partition only |
| Rows used | 187 out-of-fold predictions (grouped, full coverage) |
| Fitted slope | 1.0360 |
| Fitted intercept | -0.0336 |
| Parameter uncertainty propagated | yes |

The recorded selection reason: identity scored the lowest cross-validated log
loss (0.6402) with logistic next (0.6498), and logistic was preferred under a
slope-correction tie-break. The artifact records its own known bias — out-of-fold
predictions come from models trained on a fraction of the data and are
marginally less sharp than the final model, so the calibrator corrects for a
sharpness the deployed model does not have. That is a real, acknowledged
conservatism, not a neutral choice.

### 1.5 Sibling artifacts

Two further artifacts were trained on the identical matrix and registered
inactive, so the comparison in §[7.5](#75-model-family-selection) is auditable
rather than asserted: `ptrs-pef-004-alt-hierarchical_logistic` and
`ptrs-pef-004-alt-gbdt`.

---

## 2. Intended use

**Primary intended use.** To support a human portfolio or translational-science
discussion about a preclinical program entering human development, by making
explicit (a) what evidence exists at a stated decision date, (b) what is
missing, and (c) how a fitted model weighs the two.

**The estimand, stated precisely.** The probability that a program entering human
development fails to demonstrate sufficient efficacy to progress successfully
beyond Phase II, **conditional on not being withdrawn first for safety,
pharmacokinetic, manufacturing or commercial reasons**. This is a cause-specific
estimand. It is not an unconditional probability of programme failure, and it is
not comparable to headline attrition statistics that pool all causes.

**Primary intended users.** Translational scientists, portfolio reviewers and
biostatisticians who will interrogate the decomposition, the completeness report
and the validation record — not consume the number alone.

**Intended decision point.** The as-of rule for this cohort is `ind_or_cta_on`:
the regulatory filing that authorises first human dosing. Features are built
from evidence with `available_at <= as_of` and nothing later
(`trs/features/temporal.py`).

**What every estimate is required to carry.** No interface in this platform
returns a bare probability. Each score arrives with an uncertainty interval, a
three-source decomposition of that interval (parameter / imputation /
calibration), an evidence-completeness score, an evidence-confidence grade, the
exact additive driver attribution, and the model version that produced it.

---

## 3. Out-of-scope uses

These are not cautions; they are exclusions.

1. **Any real-world decision whatsoever, in this deployment.** The model is
   fitted to simulated outcomes. A number it produces about a real program is a
   statement about a random number generator.
2. **Regulatory submission, or any claim of regulatory standing.** There is
   none. No component has been reviewed by any regulator.
3. **Clinical decision-making, patient-level risk, or anything touching an
   individual patient.** The unit of analysis is a development track, never a
   person.
4. **Killing or funding a program on the number alone.** See
   §[9.1](#91-the-risk-of-a-program-killed-on-a-number-the-model-cannot-support).
   On this data the model's probabilities are not statistically distinguishable
   from the cohort base rate.
5. **Unconditional attrition forecasting.** The endpoint excludes safety, PK/PD
   and commercial withdrawals as competing events. Using it as "probability the
   program dies" overstates efficacy risk by construction.
6. **Ranking programs across therapeutic areas.** The cohort is oncology-only
   (`oncology_ph2_efficacy`). Nothing here has been shown to transfer.
7. **Inference about a specific target or modality.** Coefficients index
   simulated relationships. §[7.2](#72-coefficient-sign-audit) shows 16 of 56
   checked coefficients already carry a sign opposite to the pre-registered
   expectation.
8. **Scoring a program whose evidence postdates the as-of date.** The temporal
   chokepoint will refuse the rows; bypassing it invalidates every number in
   §[5](#5-metrics).
9. **Automated or unattended scoring feeding another system.** Every output is
   flagged `is_demonstration` precisely so a downstream consumer cannot treat it
   as validated.

---

## 4. Factors

Factors the card reports performance across, and where each comes from.

| Factor | Levels present in the test partition | Definition |
|---|---|---|
| `modality_group` | `small_molecule` (ref), `antibody`, `adc`, `peptide_protein`, `nucleic_acid`, `cell_therapy` | Nominal, one-hot. Ordinal coding was explicitly rejected here: the spacing assumption is not defensible across modalities. |
| `target_class_group` | `kinase` (ref), `enzyme_other`, `epigenetic`, `transcription_factor`, `surface_antigen`, `immune`, `ppi`, `other` | Nominal, one-hot, same reasoning. |
| `era` | `2012–2016`, `2016–2021` | As-of date band. |
| `evidence_completeness` | `low` / `mid` / `high` | Tertiles of the weighted completeness score from `trs/features/missingness.py`. |

**Disease context** is present as a cohort-level constraint rather than a
reported factor: the cohort is a single therapeutic area, built from 26
indication names, and the demonstration data is too thin to report per-indication
performance without misleading. The `disease_model_relevance` and
`disease_stage_ord` features carry disease context into the model; their
contribution is quantified by ablation in §[7.1](#71-domain-ablation).

**Environmental / equipment factors** in the usual model-card sense do not apply.
The nearest analogue — the *disclosure environment* a program sits in — is a real
confounder and is treated as one; see
§[9.3](#93-evidence-volume-is-confounded-with-disclosure-behaviour).

---

## 5. Metrics

### 5.1 Why these metrics

The selection criterion is **test log loss**, a proper scoring rule, with AUROC
reported but explicitly *not* used to choose (`selection_record.gate.note`). The
reasoning is in the code and is the right reasoning: AUROC is invariant to
monotone distortions of probability, and a distorted probability is precisely the
failure this product most needs to avoid, because its output *is* a probability.

For the same reason the headline usefulness statistic is **scaled Brier** (Brier
skill against the cohort base rate), not AUROC.

### 5.2 Evaluation design

| Property | Value |
|---|---|
| Split protocol | **`temporal_outer_maturity`** (`trs/validation/protocol.py`) |
| May these numbers be called deployment performance? | **No** — `permits_deployment_claim = false` |
| Protocol rejected first | `strict_temporal`, with the measured reason recorded |
| Split manifest hash | `165cb164f07e18e7efc37ae6f033ba9def25edecab7db65318a0bf6f13ab72f4` |
| Assessment boundary | 2017-09-04 (76th percentile of admissible as-of dates) |
| Fitting origin = label cutoff | 2017-09-04 |
| Corpus data cutoff | 2026-10-13 |
| Grouping | Connected components, `identity` scope (keys: program, compound) |
| Components / rows | 465 components from 552 rows; 390 singletons; largest component 3; median size 1 |
| Development / evaluation | 187 / 127 |
| Rows with an out-of-fold selection prediction | 122 |
| Purged — outcome unresolved at the data cutoff | 24 |
| Purged — component straddled the assessment boundary | 12 |
| Purged — label not knowable at the fitting origin | 202 |
| Evaluation positives / prevalence | 70 / 0.5512 |
| Bootstrap | 500 resamples, **unit = connected component**, 95% level |
| Leakage audit | `clean: true` — 0 shared components, 0 shared programs, 0 shared compounds |

Five things are worth stating plainly.

- **The evaluation partition is read exactly once**, after the family,
  hyperparameters, coefficients and calibrator are all fixed. Hyperparameter
  selection, family selection and calibration consume development rows only, via
  forward-chaining folds in which each fold fits its own feature transformer.
  A regression test permutes the evaluation labels and requires the manifest
  hash, chosen family, L2 penalty, encoder state, coefficients, intercept and
  calibrator parameters to be bit-identical — and separately requires the
  evaluation log loss to *change*, so it cannot pass vacuously.

  The predecessor did not have this property. It passed the test labels to
  `choose_family` and then reported the winner on those same rows, which made
  every headline figure a selection-set figure.
- **Label maturity is enforced, and it is expensive.** A development row's
  outcome must have been knowable at the fitting origin. That purges 202 rows —
  the single largest exclusion in the pipeline, and the reason development is
  187 rows rather than ~400. The median outcome lag on this cohort is 5.3 years.
  It also introduces a selection bias of its own, quantified in
  §[8](#8-caveats-and-recommendations).
- **No random split is reported anywhere.** The artifact says why: on this problem
  a random split would be optimistic by an unquantifiable margin.
- **Confidence intervals are computed over 465 independent units, not 552 rows.**
  This widens every interval below, correctly.
- **Informational overlap remains and is reported, not hidden:** targets,
  sponsors and mechanism keys appear on both sides of the split. Target overlap
  is addressed directly in §[7.4](#74-target-grouped-sensitivity).

### 5.3 Evaluation-partition performance (127 rows, 70 positive)

| Metric | Value | 95% CI (component bootstrap, 500 resamples) |
|---|---|---|
| AUROC | **0.622** | [0.515, 0.718] |
| AUPRC | 0.673 | [0.566, 0.792] |
| Brier | **0.238** | [0.213, 0.265] |
| Log loss | **0.731** | [0.637, 0.825] |
| **Scaled Brier (skill vs base rate)** | **+0.035** | **[-0.065, +0.112]** |
| Calibration slope | 0.851 | [0.165, 1.615] |
| Calibration intercept | -0.229 | [-0.538, +0.127] |
| ECE (10 quantile bins) | 0.098 | — |
| MCE | 0.220 | — |

**The scaled-Brier interval includes zero.** That single fact governs how every
other number on this page should be read.

A calibration slope of 0.851 means the model's probabilities are **over-dispersed**:
it is more confident than the data support, and the correct reading of an extreme
score is that it should be shrunk toward the base rate.

### 5.4 The oracle ceiling

Because the simulation knows each track's true failure probability, the maximum
attainable performance is computable: score every track by the probability the
generator actually used. Nothing can beat it.

| Scorer | AUROC | Brier | Log loss |
|---|---|---|---|
| **Oracle** (true simulated probability) | **0.782** | 0.185 | **0.549** |
| Cohort base rate (no covariates) | 0.500 by construction | 0.242 | 0.678 |
| Fitted model, locked evaluation | 0.622 | 0.263 | 0.731 |

Mean true failure probability 0.5954, SD 0.2077.

**An important caveat about this comparison, stated because it is easy to get
wrong.** The oracle row is computed over all **552** included cohort entries
(`trs oracle`), whereas the model row is computed over the **127**-row evaluation
test partition. The split assignment is computed in memory during training and is
not persisted to `cohort_entry.split` (§[10.2](#102-known-defects-found-while-writing-this-card)),
so a like-for-like oracle on the same 127 rows cannot be recovered without a
retrain. The two rows are therefore **indicative, not a matched comparison.**

With that caveat: the ceiling is far below 1.0 *by construction*, because outcomes
are Bernoulli draws from the true probability. The gap between 0.622 and 0.782 is
the combined cost of measurement noise in the evidence, non-disclosure, finite
sample and model misspecification. A demonstration dataset on which a model scored
0.97 would be advertising a capability the product does not have.

### 5.5 Calibration bins

Ten quantile bins of ~15 test rows each. Observed rates carry their own intervals
because a 15-row bin cannot resolve much.

| Bin | Mean predicted | Observed | n | Observed 95% CI |
|---|---|---|---|---|
| 0 | 0.524 | 0.385 | 13 | [0.177, 0.645] |
| 1 | 0.614 | 0.385 | 13 | [0.177, 0.645] |
| 2 | 0.645 | 0.417 | 12 | [0.193, 0.680] |
| 3 | 0.682 | 0.538 | 13 | [0.291, 0.768] |
| 4 | 0.713 | 0.692 | 13 | [0.424, 0.873] |
| 5 | 0.737 | 0.583 | 12 | [0.320, 0.807] |
| 6 | 0.767 | 0.538 | 13 | [0.291, 0.768] |
| 7 | 0.801 | 0.417 | 12 | [0.193, 0.680] |
| 8 | 0.829 | 0.846 | 13 | [0.578, 0.957] |
| 9 | 0.874 | 0.692 | 13 | [0.424, 0.873] |

**Nine of the ten bins sit below their predicted value**, several of them far
below. That is the calibration intercept of −0.778 made visible: the model's
level is set from a development era whose failure rate is 0.631, and it is
applied to an evaluation era whose rate is 0.551. The slope of 0.851 reports
the remaining over-dispersion on top of that shift.

Note also that the predicted range is compressed into [0.52, 0.92]. A model
with little signal predicts near its training base rate for everyone; the bins
are narrow because the predictions are.

### 5.6 Operating points

| Threshold | Sensitivity | Specificity | PPV | NPV | Flagged | TP / FP / TN / FN |
|---|---|---|---|---|---|---|
| 0.30 | 0.988 | 0.015 | 0.558 | 0.500 | 98.7% | 82 / 65 / 1 / 1 |
| 0.50 | 0.831 | 0.318 | 0.605 | 0.600 | 76.5% | 69 / 45 / 21 / 14 |
| 0.70 | 0.325 | 0.803 | 0.675 | 0.486 | 26.8% | 27 / 13 / 53 / 56 |

At a 0.30 threshold the model flags essentially everything; specificity 0.015
means it is doing almost no work. That is the confusion-matrix view of the same
verdict.

### 5.7 Decision curve

Net benefit of acting on the model's flag versus flagging every program
(`treat_all`) or none (`treat_none`), at evaluation prevalence 0.551.

- **Thresholds 0.05–0.20:** model net benefit is identical to `treat_all`
  (0.534, 0.508, 0.479, 0.446). The model adds nothing a blanket policy does not.
- **Thresholds 0.25–0.60:** model net benefit exceeds `treat_all` — e.g. at 0.50,
  0.161 versus 0.114; at 0.60, 0.101 versus -0.107.
- **Thresholds ≥ 0.65:** model net benefit goes negative (-0.012 at 0.65, -0.022
  at 0.70, -0.027 at 0.80), i.e. worse than flagging nothing.

So the only region where this model is not dominated by a trivial policy is a
mid-range indifference threshold, and even there the advantage is not
distinguishable from noise given §5.3.

### 5.8 Subgroup performance

Levels with fewer than 15 programs, or only one outcome class, are **not
reported** — the metric would not be interpretable. That suppression rule is
applied by the report generator, not by hand.

**By modality group**

| Level | n | pos | AUROC [95% CI] | Brier | Cal. slope | Scaled Brier |
|---|---|---|---|---|---|---|
| `small_molecule` | 88 | 48 | 0.689 [0.590, 0.789] | 0.226 | 1.126 | +0.090 |
| `antibody` | 31 | 18 | 0.671 [0.443, 0.860] | 0.229 | 0.833 | +0.060 |
| `adc` | 13 | 8 | not reported | | | |
| `nucleic_acid` | 6 | 3 | not reported | | | |
| `peptide_protein` | 6 | 3 | not reported | | | |
| `cell_therapy` | 5 | 3 | not reported | | | |

**By target class group**

| Level | n | pos | AUROC [95% CI] | Brier | Cal. slope | Scaled Brier |
|---|---|---|---|---|---|---|
| `immune` | 23 | 13 | 0.785 [0.572, 0.953] | 0.201 | 2.734 | +0.183 |
| `kinase` | 60 | 31 | 0.703 [0.578, 0.820] | 0.226 | 1.336 | +0.094 |
| `surface_antigen` | 34 | 19 | **0.537** [0.341, 0.762] | 0.270 | **-0.074** | **-0.093** |
| `ppi` | 12 | 8 | not reported | | | |
| `transcription_factor` | 10 | 6 | not reported | | | |
| `epigenetic` | 7 | 5 | not reported | | | |
| `other` | 2 | 0 | not reported | | | |
| `enzyme_other` | 1 | 1 | not reported | | | |

`surface_antigen` is the subgroup to look at hardest: AUROC 0.537 (interval spans
0.5), negative skill, and a **negative calibration slope** — on those 34 programs
the model's ordering is, if anything, inverted. On 34 rows that is well within
noise, which is the point: this card cannot tell you whether that subgroup is
broken or merely small.

**By era**

| Level | n | pos | AUROC [95% CI] | Brier | Cal. slope | Scaled Brier |
|---|---|---|---|---|---|---|
| `2016–2021` | 143 | 80 | 0.617 [0.538, 0.704] | 0.241 | 0.662 | +0.022 |
| `2012–2016` | 6 | 3 | not reported | | | |

**By evidence completeness**

| Level | n | pos | AUROC [95% CI] | Brier | Cal. slope | Scaled Brier |
|---|---|---|---|---|---|---|
| `low_completeness` | 38 | 26 | 0.715 [0.552, 0.889] | 0.188 | 1.910 | +0.128 |
| `mid_completeness` | 55 | 31 | 0.582 [0.417, 0.755] | 0.247 | 0.424 | -0.006 |
| `high_completeness` | 56 | 26 | **0.564** [0.405, 0.700] | 0.263 | 0.279 | **-0.057** |

This ordering is **backwards** — the model does worst on the programs with the
best-documented evidence. Two readings are available and this card cannot
distinguish them: either the completeness signal itself is doing the work (in the
simulation, missingness is informative by construction, see
§[6.4](#64-missingness-is-partly-informative-mnar)), or the intervals are simply
too wide for the ordering to mean anything. Both readings argue against trusting
the point estimates.

---

## 6. Training data — SYNTHETIC

### 6.1 What was generated

`trs seed --programs 600 --seed 20240917` (`trs/seed/generator.py`). Realised
contents of the database this model was trained from:

| Table | Rows |
|---|---|
| `program` | 600 |
| `program_indication` (development tracks) | 765 |
| `evidence_item` | 5,273 |
| `source` | 90 |
| `compound` | 600 |
| `target` | 42 |
| `indication` | 26 |
| `sponsor` | 25 |
| `pkpd_bridge` | 474 |
| `program_biomarker` | 506 |
| `safety_finding` | 1,091 |
| `target_assessment` | 123 |
| `raw_record` | **0** |
| `evidence_field_provenance` | **0** |
| `trial` | **0** |

Every `evidence_item` carries `provenance_method = synthetic_generator`; every
`source` carries `source_type = synthetic_demonstration`, with `pmid`, `doi` and
`url` all `NULL`. Fabricating a plausible PMID would be indistinguishable from a
real citation once it reached the interface, which is exactly the failure this
platform must not have.

The zeros in that table are not a defect of the generator — they are the shape of
a simulated corpus. They are discussed in [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

### 6.2 The data-generating process

The DGP is **written down rather than tuned by eye**, so a reader can tell exactly
what a model trained on it is recovering.

**Step 1 — latent translational properties**, drawn per track:

| Latent | Distribution |
|---|---|
| `genetic_tier` | Categorical over 7 evidence tiers, read back from the target's assessment history |
| `exposure_margin_log2` | `Normal(loc, 1.6)`, `loc = 0.9` for antibody / bispecific / therapeutic protein, else `0.15` |
| `therapeutic_index_log2` | `Normal(2.0, 1.8)` |
| `model_fidelity` | `Beta(2, 2)` |
| `human_support` | `Beta(1.5, 2.5)` |
| `replication_labs` | `Poisson(1.1)` |
| `pathway_redundancy` | Categorical 0–3, read back from target assessment |
| `target_engagement` | Categorical over {0,1,2,3,5,6}, p = [0.08, 0.14, 0.26, 0.32, 0.14, 0.06] |
| `biomarker_selected` | 1 if the indication name implies selection, else Bernoulli(0.28) |
| `safety_burden` | `Gamma(1.6, 0.9)` |
| `is_first_in_class` | 1 when no approved agent exists for the target |

Target-level latents (`genetic_tier`, `pathway_redundancy`) are **read back from
the generated `target_assessment` rows rather than drawn a second time**. Drawing
them twice would make two of the largest terms in the true log-odds unobservable
in principle, and the demonstration would understate the pipeline for reasons
that have nothing to do with the pipeline.

**Step 2 — the true failure log-odds.** `TRUE_COEFFICIENTS`, verbatim:

| Term | Coefficient | Scale |
|---|---|---|
| `intercept` | **+2.45** | — |
| `genetic_support` | -0.34 | 0–5 tier scale |
| `exposure_margin_log2` | -0.30 | log2 achievable/efficacious |
| `model_fidelity` | -1.10 | 0–1 |
| `human_support` | **-1.35** | 0–1 |
| `replication_labs` | -0.22 | count |
| `pathway_redundancy` | +0.42 | 0–3 |
| `target_engagement` | -0.21 | 0–6 |
| `biomarker_selected` | -0.55 | 0/1 |
| `safety_burden` | +0.18 | severity-weighted count |
| `is_first_in_class` | +0.30 | 0/1 |

Plus an additive `MODALITY_EFFECT`: `small_molecule` 0.00, `monoclonal_antibody`
-0.18, `bispecific_antibody` +0.12, `antibody_drug_conjugate` -0.05,
`protein_degrader` +0.22, `cell_therapy_car_t` +0.05, `radiopharmaceutical` +0.10,
`oligonucleotide_aso` +0.35, `peptide` +0.28, `therapeutic_protein` +0.08.

These magnitudes are chosen so that no single latent dominates. **They are not an
assertion about real translational biology or real modality attrition** — the
generator says so explicitly, and so does this card.

### 6.3 The analytically-derived intercept

The intercept is **not a free parameter**. It is set so that the marginal Phase II
efficacy-failure rate lands near 0.60 given the drawn means of the latents. The
generator documents the arithmetic: summing coefficient × latent mean gives
roughly -2.08, so an intercept of +2.45 centres the true log-odds near +0.37,
i.e. p ≈ 0.59. The 0.60 target is chosen to sit in the range usually quoted for
oncology Phase II efficacy attrition; **it is a property of the simulation, not an
estimate of it**.

Re-derived from the realised draw (`artifacts/demo_ground_truth.json`, 765 tracks):

| Quantity | Generator's stated approximation | Realised |
|---|---|---|
| Σ coefficient × latent mean | ≈ -2.08 | **-1.985** |
| Mean true log-odds | ≈ +0.37 | **+0.465** |
| Mean true probability | ≈ 0.59 | **0.5945** (median 0.6117, SD of logit 1.027) |

`verify_base_rate()` is run by the seeding CLI so a drifting generator is noticed
immediately. On this database it reports a realised primary failure rate of
**0.5867** against a target of 0.60 — within the ±0.12 tolerance. Realised outcome
mix across all 765 tracks:

```
phase_2_efficacy_failure    335     progressed_past_phase_2      167
phase_1_safety_failure       72     regulatory_approval           69
discontinued_through_phase_2 86     phase_1_pkpd_failure          39
active_no_outcome_yet        17
```

### 6.4 Missingness is partly informative (MNAR)

This is the most realistic property of the simulation and the reason the 36
missing-indicator columns earn their place in the design matrix. Two explicit
mechanisms:

- **PK/PD disclosure depends on the thing being disclosed.** A PK/PD bridge row is
  written only with probability
  `clip(0.42 + 0.14 × exposure_margin_log2, 0.12, 0.94)`. A weak exposure bridge
  is disclosed less often, so *absence of the row is itself evidence about the
  exposure margin*.
- **Lab identity depends on replication breadth.** `lab_identifier` is populated
  with probability `0.45 + 0.20 × min(replication_labs, 3)`. Programs replicated
  in few labs report lab identity less often, which is how single-lab clusters
  masquerade as independent replication in real corpora.

The consequence is stated in the generator and is the correct lesson: **naive
complete-case analysis on this data is biased.** The feature layer therefore
distinguishes `not_reported` (the study exists and is silent — informative about
quality) from `not_applicable` (the question does not arise — not informative)
and from `not_yet_available` (exists but postdates the cutoff and was withheld to
preserve temporal integrity).

### 6.5 Competing outcomes

Conditional on a track resolving, the generator draws: commercial /
strategic discontinuation with probability **0.13**, Phase I safety failure
**0.11**, Phase I PK/PD failure **0.05**; tracks with under 5 years of follow-up
are left `active_no_outcome_yet` with probability **0.30**.

**Commercial discontinuation is independent of latent quality by construction.**
That is exactly why the cohort builder must exclude it: a model that learned to
predict it would be predicting corporate behaviour, not biology. The realised
exclusion waterfall (765 candidates → 552 included, 213 excluded) is:

| Exclusion reason | n |
|---|---|
| `outcome_excluded:phase_1_safety_failure` | 72 |
| `non_scientific_discontinuation:commercial_strategic` | 39 |
| `outcome_excluded:phase_1_pkpd_failure` | 39 |
| `non_scientific_discontinuation:portfolio_reprioritisation` | 27 |
| `non_scientific_discontinuation:sponsor_insolvency` | 20 |
| `outcome_excluded:active_no_outcome_yet` | 17 |

Resulting cohort: **552 entries, 325 positive (58.9%), 236 negative.**

### 6.6 Deliberate post-cutoff evidence

15% of in vivo evidence items are dated **after** the track's IND date on purpose.
They exist so the temporal filter has something real to catch: if the as-of filter
ever breaks, those items move the features. (§[10.2](#102-known-defects-found-while-writing-this-card)
notes that the test file the code points to for this does not currently exist.)

### 6.7 Real vocabulary, simulated everything else

Gene symbols, target classes, indication names and therapeutic areas are **real
public vocabulary** — `KRAS` is a real gene. Every sponsor, program, compound,
experiment, effect size, source record and outcome is **simulated**. Sponsors are
named after mountain ranges and rivers so they cannot be mistaken for real
companies; every program code is prefixed `DEMO-`. The rationale and the mitigations
are set out in [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

---

## 7. Quantitative analyses

### 7.1 Domain ablation

Each of the 12 feature domains is removed in turn and the model refitted. A domain
whose removal costs nothing is not contributing to the estimate, however prominent
it is in the interface. Baseline: AUROC 0.6298, log loss 0.6708. Positive
`Δ log loss` = removal made the model worse.

| Domain | Features removed | AUROC without | Δ AUROC | Δ log loss |
|---|---|---|---|---|
| PK/PD translation | 6 | 0.597 | -0.032 | **+0.0152** |
| Reproducibility & study quality | 9 | 0.615 | -0.015 | **+0.0118** |
| Target validation | 6 | 0.584 | **-0.046** | **+0.0107** |
| Animal-model validity | 10 | 0.635 | +0.005 | +0.0012 |
| Safety translation | 4 | 0.631 | +0.001 | +0.0006 |
| Program context | 4 | 0.633 | +0.003 | +0.0001 |
| Evidence volume & recency | 3 | 0.631 | +0.001 | -0.0002 |
| Human biological support | 5 | 0.632 | +0.002 | -0.0023 |
| Cross-species concordance | 5 | 0.642 | +0.012 | -0.0024 |
| Historical class performance | 4 | 0.631 | +0.001 | -0.0027 |
| Biomarker readiness | 3 | 0.631 | +0.001 | -0.0051 |
| Disease-model relevance | 4 | **0.655** | +0.025 | **-0.0129** |

Only three domains carry their weight: PK/PD translation, reproducibility and
target validation, whose removal costs 0.011–0.015 log loss. Three more
(animal-model validity, safety translation, program context) sit within 0.0012 of
neutral. **Removing any of the remaining six *improves* log loss** — most sharply
disease-model relevance, whose removal improves AUROC by 0.025 and log loss by
0.0129. On this data those four features are noise the penalty has not fully
suppressed.

This is a finding worth naming rather than burying: a scorecard interface that
gives all 12 domains equal visual weight would misrepresent what the model is
actually using.

### 7.2 Coefficient sign audit

`expected_direction` records a pre-registered belief about each feature's
association with failure. It is used **only** to flag discordant coefficients for
human review; it never constrains fitting. A model that silently enforces the
analyst's priors cannot surprise the analyst, and surprising the analyst is the
point.

| Quantity | Value |
|---|---|
| Coefficients checked | 56 |
| Sign-discordant | **16** |
| Discordant with a credible interval excluding zero | **0** |

Discordant features: `animal_n_invivo_studies`, `animal_mean_disease_similarity`,
`animal_immunocompetent_fraction`, `animal_max_effect_z`,
`species_concordance_fraction`, `has_nonrodent_efficacy`,
`has_contradictory_species`, `exposure_response_established`,
`pk_scaling_species_count`, `has_prior_human_clinical_on_pathway`,
`target_association_score`, `randomised_fraction`, `prespecified_fraction`,
`sponsor_affiliated_fraction`, `max_safety_severity_ord`,
`biomarker_species_concordance_ord`.

Zero of the 21 are credible. Under an L2 penalty of 64 on 187 rows, coefficients
are shrunk hard and sign is close to arbitrary for weak features. The honest
reading is not "the model discovered that randomisation increases risk" — it is
**"do not read individual coefficients from this fit."**

### 7.3 Interval agreement

The analytic Laplace interval is checked against a component-level bootstrap
refit, on the log-odds scale, over 30 probe rows and 40 refits.

| Quantity | Value |
|---|---|
| Median width ratio (analytic / bootstrap) | **1.885** |
| p10 / p90 | 1.546 / 2.441 |
| Verdict | **`analytic_intervals_too_wide`** |

Acceptable is 0.75–1.25. The published intervals are roughly **1.5× to 2.4× wider**
than a bootstrap refit suggests. Being too wide is the safe direction — a
conservative interval overstates ignorance rather than hiding it — but it is still
a mis-specification, and users comparing two programs' intervals should know the
overlap is exaggerated.

### 7.4 Target-grouped sensitivity

The headline split enforces component integrity on program and compound, but
39 targets appear on both sides. This analysis rebuilds components from program,
compound **and** target, so no program sharing a target with a test program can
appear in training. It is 5-fold cross-sectional cross-validation over all 552
rows — **not temporal** — so it is not directly comparable to §5.3.

| Metric | Target-grouped CV | Headline temporal split |
|---|---|---|
| AUROC | **0.642** | 0.622 |
| Brier | 0.237 | 0.238 |
| Log loss | 0.654 | 0.731 |
| Scaled Brier | +0.021 | +0.035 |
| Calibration slope | 1.066 | 0.851 |

AUROC falls by 0.029 and skill roughly halves when target sharing is forbidden.
Given the interval on the headline AUROC ([0.537, 0.720]) this drop is not
statistically resolvable, but its direction is what you would expect if part of
the apparent signal is target-level memorisation.

### 7.5 Model family selection

Selection criterion: **test log loss**. Gate: minimum relative improvement 2%,
plus a paired component-level bootstrap at 95% that must exclude zero, plus a
calibration-slope band of [0.80, 1.25]. A non-linear challenger additionally pays
a 5% **attribution surcharge** (it cannot give exact additive attribution) and a
3% **posterior surcharge** (it cannot give an analytic posterior). Those
surcharges are a stated product judgement, not a statistical one, and they should
be read as such.

| Family | Log loss | AUROC | Brier | Scaled Brier | Cal. slope | Verdict |
|---|---|---|---|---|---|---|
| `l2_logistic` | 0.6135 | 0.6516 | — | — | 0.837 | **adopted** (incumbent baseline) |
| `gbdt` | **0.6298** | 0.5781 | — | — | **0.711** | disqualified — slope outside [0.80, 1.25] |
| `hierarchical_logistic` | 0.6714 | 0.6322 | 0.2386 | +0.0329 | **0.776** | disqualified — slope outside [0.80, 1.25] |

All three passed the loose sanity band [0.15, 6.0]. Note that **GBDT had the best
log loss and the best skill** and was rejected on calibration slope alone. That is
a defensible rule for a product whose output is a probability, but it is a rule,
and a reader is entitled to disagree with it. The incumbent's own development slope of 0.837
is *also* outside the [0.80, 1.25] band — the band disqualifies challengers, it
does not demote the baseline. That asymmetry is worth knowing.

### 7.6 Hyperparameter selection

Expanding-window folds inside the training partition only: each fold trains on
everything before a cut date and evaluates on the next slice, mirroring
deployment. Four folds.

| L2 penalty | Mean log loss |
|---|---|
| 0.25 | 1.2058 |
| 1.0 | 0.9298 |
| 4.0 | 0.7683 |
| 16.0 | 0.6769 |
| 32.0 | 0.6521 |
| 64.0 | 0.6401 |
| **64.0** | **0.6274** |
| 256.0 | 0.6445 |
| 512.0 | 0.6525 |
| 1024.0 | 0.6597 |
| 2048.0 | 0.6649 |

The optimum is interior to the grid (`optimum_at_grid_boundary: false`), which is
the check that matters — a boundary optimum would mean the grid was wrong. The
curve is very flat between 64 and 256, and the selected penalty is large relative
to 187 rows and 109 columns. That is the model telling you there is not much
signal to fit.

---

## 8. Caveats and recommendations

### 8.1 The usefulness verdict — read this before any other number

> **Verdict: `not_distinguishable_from_base_rate`.**
>
> "The model is not distinguishable from quoting the cohort base rate: Brier skill
> -6.5% (95% CI -21.5% to +5.9%), interval includes zero. Treat individual
> estimates as weakly informative at best."
> — `metrics.usefulness.headline`, generated by the validation pipeline, not by
> a human writing documentation.

The reference is a no-covariate model that predicts the cohort base rate for
every program. On the held-out partition, this model does not beat it by a margin
the data can resolve.

The artifact also records *why this is the threshold question*: discrimination
metrics are invariant to monotone distortions of probability, so a model can rank
programs correctly and still produce probabilities less accurate than the base
rate. Because this product's output **is** a probability, skill against the base
rate is the test it has to pass. It does not pass it here.

### 8.2 Plain statements

- **Trained on simulated data.** Every coefficient and every metric on this page
  describes `trs/seed/generator.py`, not drug development.
- **No regulatory standing.** No component of this platform has been submitted
  to, reviewed by, or cleared by any regulator.
- **No clinical validation.** No prospective study, no external dataset, no
  real-world outcome has ever been compared against a prediction from this model.
- **No real-world performance is known.** The numbers in §5 are an upper bound on
  nothing real; they characterise a pipeline running on a simulation.

### 8.3 The limitations the artifact records for itself

Reproduced verbatim from `model_artifact.limitations`:

1. This model was trained on synthetic demonstration data. Its coefficients,
   metrics and predictions describe a simulated data-generating process and carry
   NO scientific interpretation about real therapeutic programs.
2. Validation rests on 465 independent connected components, not 552 rows.
   Confidence intervals are computed at the component level for this reason.
3. 12 rows were moved to a later partition because their connected component
   straddled a temporal boundary; they did not contribute to training.
4. Historical class-performance features are computed as of the cutoff and
   exclude the focal program, but other programs in the same cross-validation
   fold do contribute to them. This reflects what an analyst genuinely knew at
   the decision point; its effect is quantified by the historical-class-performance
   row of the domain ablation.
5. Publicly disclosed programs are not a random sample of all programs. Assets
   abandoned before an IND are systematically under-represented, so estimates
   conditional on entering human development are better supported than
   unconditional attrition estimates.

### 8.4 Recommendations

**If you are evaluating this platform:** judge the pipeline, not the model. The
validation machinery — component-level bootstrap, temporal split with component
integrity, oracle ceiling, domain ablation, sign audit, interval-agreement check,
and a usefulness verdict that reports its own failure — is the artifact worth
assessing. The fitted model is a demonstration that the machinery runs end to end.

**Before any real use, in order:**

1. Replace the synthetic corpus with a real, provenance-tracked evidence base
   (see [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)) and re-run the whole pipeline.
   Restore ClinicalTrials.gov egress first; outcome labels realistically depend on it.
2. Re-derive every number in §5. **None of them transfers.**
3. Require the usefulness verdict to clear zero on real held-out data before any
   score is shown to a decision-maker. Ship the verdict next to the score, not in
   an appendix.
4. Fix the interval mis-specification in §7.3, or publish bootstrap intervals
   instead of analytic ones.
5. Re-examine the three domains that carry the model and the eight that do not
   (§7.1) before committing interface real estate to a 12-row scorecard.
6. Write the temporal-leakage test suite the code claims exists
   (§[10.2](#102-known-defects-found-while-writing-this-card)). Every performance
   number depends on an invariant that is currently unasserted by any test.

---

## 9. Ethical considerations

### 9.1 The risk of a program killed on a number the model cannot support

This is the sharpest foreseeable harm and it deserves to be stated without
hedging. A translational-risk score is consumed in a setting where somebody wants
a defensible reason to stop spending. A number between 0 and 1, attached to a
scientific-looking decomposition, is an unusually convenient object for that
purpose: it converts a contested judgement into an apparently settled one, and it
relocates accountability from a person to a model.

On this deployment the score is **not statistically distinguishable from the
cohort base rate** (§8.1). A program stopped on the strength of a 0.78 rather
than a 0.62 would have been stopped on noise. The program does not get a control
arm; nobody observes the counterfactual; and the decision looks well-founded in
retrospect precisely because no one can check it.

Design choices that push against this, all verifiable in the code:

- The usefulness verdict is computed by the pipeline and stored on the artifact.
  It is not optional prose that a deployer can quietly drop.
- No API endpoint returns a bare probability. Interval, uncertainty decomposition,
  completeness score, confidence grade, driver attribution and model version travel
  with every score.
- `grade_evidence_confidence()` only ever **downgrades**. Nothing upgrades a grade:
  a thin evidence base cannot be rescued by a narrow interval, because a narrow
  interval on thin evidence is a property of the model, not of the evidence.
- Any estimate touching synthetic data is forced to `very_low` confidence
  regardless of completeness, and flagged `is_demonstration` end to end.

None of that prevents misuse. A determined reader can screenshot the number. The
mitigation that actually matters is organisational: **do not let this score be the
stated reason for a kill decision**, and keep the decision owner a person.

### 9.2 Survivorship bias in publicly disclosed programs

The cohort is conditioned on entering human development, and — in a real
deployment — assembled from public disclosure. Both filters bite:

- **Assets abandoned before an IND are systematically absent.** The estimand is
  therefore explicitly conditional on human entry. It is not, and must not be
  read as, an unconditional probability of programme failure.
- **Public disclosure is itself selected.** Programs that fail quietly are
  under-represented relative to programs that publish, and failure is less likely
  to be written up than success. A model fitted to what was published learns, in
  part, what gets published.

The platform's response is to declare the conditioning in the estimand rather
than to correct for it, and to record the bias in `model_artifact.limitations`
(item 5). That is honest but it is **not a correction**: a reweighting scheme
would need a sampling model for disclosure that nobody has.

Note that the simulation *understates* this problem. The generator draws every
track's evidence from the same process regardless of outcome, so simulated
disclosure is not outcome-dependent in the way real disclosure is. **Real data will
be worse than these metrics suggest, not better.**

### 9.3 Evidence volume is confounded with disclosure behaviour

`evidence_total_count` is in the model, and its own registry entry says what is
wrong with it:

> "Size of the evidence base at the cutoff. Confounded with disclosure behaviour —
> a well-published academic program and a quiet industrial one are not equally
> documented for reasons unrelated to their biology. Interpreted with that caveat."

A large-pharma program with a strict publication policy and a
university-spinout program with an active publication record will differ on this
feature for reasons that have nothing to do with translational risk. Anything the
model learns from it is partly a model of **sponsor communications strategy**.

Three partial mitigations, and their limits:

- The feature's `expected_direction` is `uncertain`, so it is not flagged as
  discordant whichever way it points — an honest refusal to pretend a prior exists.
- `sponsor_affiliated_fraction` and conflict-of-interest fields let disclosure
  context enter as its own signal rather than contaminating volume silently.
- The **evidence-volume domain ablation is neutral** (Δ log loss -0.0002, Δ AUROC
  +0.001), so on this data the confound is not doing much damage — because it is
  not doing much of anything.

The limit is that none of this *removes* the confound. On real data, where
disclosure behaviour varies far more than in the simulation, this feature is a
plausible route for sponsor identity to leak into a score under a scientific label.
It should be re-ablated on every real refit and dropped if it does not earn its place.

### 9.4 Real vocabulary attached to simulated outcomes

The demonstration pairs real gene symbols and real indication names with
simulated failures. A row stating that a program against a real target failed in
Phase II is a statement about a random number generator — but it does not *look*
like one, and a screenshot carries no context. The reasons for accepting that
risk, and the four propagating mitigations, are in
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

---

## 10. How to reproduce every number here

### 10.1 Commands

All paths absolute; there is no activated virtualenv.

```bash
# Full demonstration pipeline: init-db -> seed -> build-cohort -> train
cd backend && \
  ../.venv/bin/python -m trs.cli pipeline \
  --programs 600 --seed 20240917

# Oracle ceiling (§5.4)
cd backend && \
  ../.venv/bin/python -m trs.cli oracle

# Registered artifacts
cd backend && \
  ../.venv/bin/python -m trs.cli models

# Test suite
cd backend && \
  ../.venv/bin/python -m pytest tests -q
```

`trs oracle`, `trs models` and `pytest tests` were run against the current
database while writing this card; the numbers in §5.4 and §1.1 are their output.
The pipeline command was **not** re-run — doing so would regenerate the dataset
and invalidate the hashes quoted here.

### 10.2 Known defects found while writing this card

Recorded here because a model card that hides what its author noticed is not
doing its job. **None of these has been changed** — this document does not modify code.

1. **The temporal-leakage test suite does not exist.**
   `trs/features/temporal.py:14` states that
   `tests/statistical/test_temporal_leakage.py` "exercises the invariant directly".
   The only test file in the repository is `backend/tests/unit/test_connectors.py`
   (220 tests, all passing). The single most load-bearing invariant in the platform
   — the one whose violation "silently invalidates every reported performance
   number" — is currently asserted by no test.
2. **`cohort_entry.split` and `split_fold` are never written.** The column
   docstring says "Assigned by `trs.validation.splitters`", but no code path
   assigns them; all 552 rows have `split = NULL`. The split is computed in memory
   in `trs/pipeline/train.py` and discarded. The practical cost appears in §5.4:
   an oracle restricted to the test partition cannot be computed after the fact.
3. **`SplitAssignment.dropped` is a misleading name.** In
   `trs/validation/splitters.py:202` it counts rows whose partition *changed*
   (`final != provisional`), not rows removed. Under the current protocol
   187 + 127 + 238 purged = 552, and the purge reasons are itemised, so nothing is
   dropped. The artifact's own limitation text gets this right ("moved to a later
   partition"); the key name does not.
4. **`ProvenanceMethod.MODEL_GENERATED_UNVERIFIED` is not enforced anywhere.** The
   enum docstring says the validation layer "rejects [it] from any model-ready
   dataset". A repository-wide search finds the constant only in its own
   definition — no filter in the cohort builder, the feature builder or the
   pipeline excludes it. See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) §5.
5. **15 orphan feature snapshots.** `feature_snapshot` holds 586 rows for 586
   distinct tracks while the cohort has 552 entries; all 552 have a snapshot and
   15 snapshots belong to tracks outside it. Benign residue from an earlier run,
   but it means a snapshot count is not a cohort count.

---

*Generated against the live repository and database on 2026-09-24. Every figure in
this card was read from `backend/artifacts/ptrs-pef-004.json`,
`backend/artifacts/demo_ground_truth.json`, `backend/trs.db`, or recomputed by
running the commands in §10.1. No figure was transcribed from memory or from any
prior document.*
