# Product Specification

**Preclinical Translational Risk Scorer — what the product does, for whom, and what it refuses to do**

This document defines the product. It states the question the platform answers, the users who
bring that question, the unit of analysis, the endpoints and their estimands, the structure of the
assessment it returns, and — at equal length — the claims it is built not to make.

Its companions are [ARCHITECTURE.md](ARCHITECTURE.md) (how the system is built),
[MODELING_PLAN.md](MODELING_PLAN.md) (which estimator and why),
[VALIDATION_PLAN.md](VALIDATION_PLAN.md) (how it is evaluated),
[DATA_DICTIONARY.md](DATA_DICTIONARY.md) (the physical schema and feature schema) and
[MODEL_CARD.md](MODEL_CARD.md) (the registered model's scope of use). [README.md](README.md) is the
entry point.

Every claim below is checkable against a file in this repository. Where a number appears, the file
or command that produces it is named.

> ## ⚠ Demonstration data
>
> Every program, every piece of evidence and every outcome currently in `backend/trs.db` was
> produced by `backend/trs/seed/generator.py`. **The dataset is synthetic.** Gene symbols, target
> classes and indication names are drawn from real public vocabulary; all program-level facts and
> all outcomes are simulated. Numbers quoted in this document describe a documented simulation and
> carry no scientific interpretation about real therapeutic programs. The platform has no
> regulatory standing and no clinical validation.

---

## Contents

1. [The question](#1-the-question)
2. [Users, and the decision each is making](#2-users-and-the-decision-each-is-making)
3. [The principle chain](#3-the-principle-chain)
4. [The unit of analysis](#4-the-unit-of-analysis)
5. [Endpoints and estimands](#5-endpoints-and-estimands)
6. [The translational scorecard](#6-the-translational-scorecard)
7. [What ships with every estimate](#7-what-ships-with-every-estimate)
8. [Scenario analysis](#8-scenario-analysis)
9. [Comparators](#9-comparators)
10. [Non-goals and out-of-scope claims](#10-non-goals-and-out-of-scope-claims)
11. [Where the product actually stands](#11-where-the-product-actually-stands)

---

## 1. The question

A therapeutic program sits in front of a decision: commit the capital and the years required to
take it into humans, or do not. The evidence supporting that commitment is a heterogeneous pile —
animal efficacy studies of varying construct validity, a PK/PD argument of varying strength,
genetic association of varying causal character, a safety package, a handful of human-tissue
experiments, and a literature whose completeness is a function of what the sponsor chose to
publish.

The platform answers one question about that pile:

> **Given only what is knowable at a stated date, how likely is it that this program fails to show
> enough efficacy in humans to progress beyond Phase II — and which parts of the evidence are
> driving that estimate?**

Two words in that sentence carry most of the weight.

**"Knowable"** — the estimate is a function of a stated cutoff date and of nothing after it. This
is enforced structurally, in `trs/features/temporal.py`, not by convention at each call site.

**"Which parts"** — a bare probability is not the product. The product is a calibrated probability
*plus* an exact decomposition of it into feature-level contributions that sum, to floating point,
to the log-odds on screen (`trs/scoring/explanation.py`).

### What it is not

It is not a discovery tool, not a target-prioritisation engine, not a literature search product,
and not an oracle. It does not tell a team what to work on. It tells a team what its own evidence
currently supports, how strongly, and where the weak joints are.

---

## 2. Users, and the decision each is making

The product has one output surface and six distinct readings of it. The design consequence is
that every estimate ships with enough structure that each reader can find the part they need
without the others' parts being hidden from them.

### Biotech CSO / Head of Research

*Decision:* which of five programs gets the IND-enabling budget.

*What they use:* the portfolio view — every program's estimate computed as of its own decision
date, sorted, with interval width and evidence-confidence grade beside the point estimate.

*What they must not do with it:* rank on the point estimate alone. On the demonstration data the
mean 95% interval spans tens of percentage points; a 0.55 and a 0.62 are not distinguishable. The
interface therefore shows the interval as a bar, not as a footnote.

### Translational medicine

*Decision:* which experiment to commission next.

*What they use:* the scorecard domains and the scenario levers. "Independent replication added"
and "target engagement demonstrated in humans" are levers precisely because they name work a team
can commission, and the scenario reports how much the *estimate* moves — see
[§8](#8-scenario-analysis) for why that is not the same as reporting how much the *risk* moves.

### Biostatistics / quantitative science

*Decision:* whether to let the number into a governance process at all.

*What they use:* the Methods & validation surface, which is generated from the registered model
artifact rather than written by hand. It leads with the unflattering material: the Brier skill
against the base rate and whether its interval excludes zero, the attrition waterfall, the
effective sample size (connected components, not rows), the domain ablation, the coefficient sign
audit, and the analytic-versus-bootstrap interval agreement check.

This is the user the system is designed around. `trs/validation/report.py` states the reason
plainly: the report "is written to be read by someone looking for a reason not to trust the
model."

### Portfolio / governance committee

*Decision:* stop, continue, or restructure.

*What they use:* the program detail surface, with the estimand printed verbatim above the number
and the exclusion logic stated. A committee reading a cause-specific estimate as if it were
unconditional will systematically under-estimate total attrition; the conditional clause is
therefore not in a tooltip.

### Business development / licensing

*Decision:* what an in-licensing candidate's evidence package is actually worth.

*What they use:* the evidence explorer and the per-program evidence matrix, including what is
*missing*. A completeness figure computed over weighted features — "80% of the evidence that
actually moves this estimate is present", not "80% of table columns are non-null"
(`trs/features/missingness.py`) — is the diligence-relevant quantity.

### Investors / diligence

*Decision:* whether a sponsor's translational narrative is supported by its own package.

*What they use:* provenance. Every figure traces back through feature → evidence items → source →
raw connector payload (`trs/pipeline/artifacts.py` documents the chain). A number that cannot be
traced is a number that should not be relied on.

---

## 3. The principle chain

Six stages. Each has one job, one module, and one property it is responsible for preserving.

```
Evidence  →  Features  →  Model  →  Calibrated Risk  →  Explanation  →  Decision Context
```

### Evidence

`trs/connectors/`, `trs/db/models/evidence.py`

Structured claims extracted from named sources. The verbatim payload is stored first, as an
immutable `RawRecord` keyed by content hash, before anything interprets it — so any downstream
value can be re-derived and any extraction bug diagnosed against exactly the bytes that produced
it. Every evidence-bearing row carries `available_at`: the date from which a diligent analyst
could have known the fact. That is deliberately not `published_on`; a 2015 paper reporting a 2011
experiment became knowable in 2015, and a registry field populated in 2018 is a 2018 fact whatever
the trial's start date says.

*Property preserved:* traceability and temporal availability.

### Features

`trs/features/` — `spec.py`, `registry.py`, `builder.py`, `effects.py`, `historical.py`,
`missingness.py`, `transform.py`, `temporal.py`

63 declared features across 12 domains under feature schema `fs-1.0.0`. The schema is a contract:
a feature not declared in the registry cannot enter a model matrix, and a model can only score a
snapshot built under the schema version it was trained on.

Three commitments are worth naming because each has a cost:

- **Ordinal encodings are assumptions, written down.** Mapping an ordered category to an integer
  asserts equal spacing between levels. At cohort sizes of hundreds, that assumption buys a large
  variance reduction over one-hot coding. It is a trade, and each ordinal map records the scale it
  encodes. Where the spacing assumption is not defensible — modality, target class — the feature is
  nominal.
- **Direction hypotheses are recorded but never enforced.** `expected_direction` flags
  sign-discordant coefficients for human review; it never constrains fitting. A model that silently
  enforces the analyst's priors cannot surprise the analyst.
- **Missingness is modelled, not defaulted.** "Not reported" and "not applicable" are different
  facts and encode differently. A feature computed over an empty evidence set is `None`, not 0 —
  except where the registry declares `ZERO_IS_ABSENCE` because the sweep is exhaustive by
  construction.

Heterogeneous effect sizes (78% tumour-growth inhibition here, HR 0.42 there, 3.1-fold change
elsewhere) are put on one scale by `effects.py` using two documented monotone families. This is a
comparability device so that "how large were the animal effects" can be a feature at all. It is
explicitly **not** a meta-analytic pooled estimate — the inputs lack the variance information a
real meta-analysis requires — and the resulting feature is therefore interpreted ordinally.

*Property preserved:* the feature vector is a function only of facts knowable at the cutoff.

### Model

`trs/modeling/`, `trs/pipeline/`

Three candidate families are fitted and compared: L2 logistic regression with a Laplace posterior,
hierarchical logistic regression with partial pooling on mechanism groups, and a grouped-bootstrap
gradient-boosted ensemble. The linear family is the incumbent, and the gate in
`trs/pipeline/select.py` requires a challenger to beat it on test log loss by a material margin,
survive a paired component-level bootstrap, and remain calibrated.

Log loss rather than AUROC is the selection criterion. The artifact records the reason verbatim:
AUROC "is invariant to monotone distortions of probability, which is the failure this product most
needs to avoid."

*Property preserved:* the deployed family can produce a defensible posterior and an exact additive
attribution. A tree ensemble can do neither, and `GradientBoostedRisk.attribution()` returns
`None` rather than a Shapley approximation presented beside an exact decomposition.

### Calibrated risk

`trs/modeling/calibration.py`

The product's output is a probability, so calibration is mandatory rather than polish. Three rules
are enforced: never fit and assess calibration on the same rows; choose the calibration method
out-of-sample too, by cross-validation within the calibration data; and propagate calibration
parameter uncertainty when the method admits it (logistic recalibration does, isotonic does not,
and when isotonic is selected the scoring layer reports that the calibration component is
excluded).

*Property preserved:* "70%" is intended to mean seventy percent, and the evidence for or against
that is reported rather than assumed.

### Explanation

`trs/scoring/explanation.py`

The deployed family is linear on the log-odds scale, so with calibration intercept `a` and slope
`b`:

```
logit(p) = a + b·(β₀ + Σⱼ βⱼ zⱼ)
```

Each feature's contribution is exactly `b·βⱼ·zⱼ`, and `explain()` **asserts** that the
contributions sum to the reported log-odds rather than assuming it. Because the design matrix is
standardised on the training mean, `zⱼ = 0` is the cohort-average program, so each contribution
reads as "how this program differs from the average program, in log-odds".

The distinction this enforces is explanation versus rationalisation. A language model shown a
score can produce fluent reasons for it; those reasons are generated *from the score*, not from the
mechanism that produced it, and they sound equally convincing when the score is wrong. Nothing in
this platform works that way.

*Property preserved:* every driver shown is a term in the model's own arithmetic.

### Decision context

`trs/scoring/comparators.py`, `trs/scoring/scenarios.py`, `trs/validation/report.py`

The estimate is placed next to historical comparators (as of the same cutoff), counterfactual
scenarios, alternative-family predictions reported as specification sensitivity, and the model's
own validation record — including the verdict on whether it beats the base rate at all.

*Property preserved:* the number is never presented alone.

---

## 4. The unit of analysis

```
Program  ×  Indication  ×  Cutoff Date
```

**Program** (`trs/db/models/core.py::Program`) is one asset pursued against one mechanism. It
persists across renaming, re-sponsoring and re-indication, and it is the grouping key for
leakage-resistant cross-validation.

**Program × Indication** is a *development track* (`ProgramIndication`). The same molecule
pursued in three indications is three tracks, because the translational argument differs in each:
the disease model, the unmet need, the biomarker strategy and the competitive prior are all
indication-specific.

**× Cutoff date** completes the unit. The cutoff is deliberately not stored on the track, because
the same track is legitimately scored at many cutoffs; it lives on the `CohortEntry` and the
`FeatureSnapshot` instead.

### Why the cutoff is load-bearing

Three reasons, in increasing order of consequence.

**1. It defines the decision the score supports.** The default as-of rule, `ind_or_cta_on`, is
described in `trs/cohort/build.py` as "the regulatory filing that authorises first human dosing —
the point at which the go/no-go decision this product supports is actually made." Two alternatives
are available, `first_in_human_on` and `phase_2_start_on`, and they answer different questions. A
score computed at Phase II start has seen Phase I data; it is not a preclinical assessment.

**2. It is the only thing standing between the platform and circularity.** Without it, "how often
has this target failed before?" silently counts the program's own outcome, and the label becomes
perfectly recoverable from the features. `trs/features/historical.py` exists as an isolated module
for exactly this reason, and it unconditionally excludes both post-cutoff outcomes and the focal
program's own siblings.

**3. A violation invalidates every downstream number.** If post-cutoff evidence leaks in, every
validation metric the platform reports is optimistic by an unknown amount. This is why the
constraint is structural rather than procedural: call sites cannot obtain evidence any other way
than through an `AsOfContext`, and `trs/features/builder.py` takes a context, not a database
session. `tests/statistical/test_temporal_leakage.py` exercises the invariant directly by building
a snapshot at `T`, inserting post-`T` evidence designed to move every feature, rebuilding, and
asserting bit-identical output.

### The cost of this design

Scoring the same track at many cutoffs multiplies the feature-building work, and the three-way
temporal split (train / calibrate / test) costs sample size that a random split would not. On the
demonstration cohort, insisting on a forward temporal split with whole connected components moved
12 rows out of the partition their date alone would have put them in. That is a real cost, paid
deliberately: `trs/validation/splitters.py` records that a random split on this problem "would be
optimistic by an unquantifiable margin."

---

## 5. Endpoints and estimands

Six endpoints are declared in `trs/cohort/outcomes.py`. Each states explicitly which observed
outcomes count as positive, which count as negative, and which are **excluded** — and exclusions
are counted and surfaced, because a cohort that discards 40% of its candidates is telling you
something.

The central decision this file encodes: "did the program fail?" is not one question. A program can
stop because the biology did not work, because the molecule was toxic, because exposure could not
be achieved, because the sponsor ran out of money, or because a portfolio committee preferred a
different asset. Collapsing those into one label produces a model that predicts *corporate
behaviour* while claiming to predict *biology*.

### Primary — `ph2_efficacy_failure`

Estimand, verbatim from `PH2_EFFICACY_FAILURE.estimand`:

> Probability that a program entering human development fails to demonstrate sufficient efficacy to
> progress successfully beyond Phase II, conditional on not being withdrawn first for safety,
> pharmacokinetic, manufacturing or commercial reasons.

This is a **cause-specific** estimand. A program discontinued for toxicity never got the chance to
demonstrate efficacy, so it is neither a positive nor a negative example for the efficacy question
and is removed as a competing event. The conditional clause is printed verbatim in the API
response and on the portfolio and program surfaces, because a reader who takes the number as
unconditional will under-estimate total attrition.

- *Positive:* `phase_2_efficacy_failure`
- *Negative:* `progressed_past_phase_2`, `phase_3_failure`, `regulatory_approval` — reaching Phase
  III means the Phase II bar was cleared, whatever happened afterwards
- *Excluded:* pre-human failure, Phase I safety and PK/PD failures, Phase II safety failure,
  unresolved and censored tracks, unknown
- *Minimum follow-up before an unresolved track may count as a negative:* 5.0 years

### Secondary endpoints

| Name | Estimand (verbatim) | Min. follow-up |
|---|---|---|
| `ph2_any_scientific_failure` | Probability that a program entering human development stops for a scientific reason — insufficient efficacy, unacceptable safety, or inadequate pharmacokinetics — at or before Phase II. Commercial and strategic discontinuations are excluded. | 5.0 y |
| `ph1_safety_failure` | Probability that a program entering human development stops in Phase I for safety or tolerability reasons. | 2.0 y |
| `ph1_pkpd_failure` | Probability that a program entering human development stops in Phase I because the required exposure or target engagement could not be achieved. | 2.0 y |
| `preclinical_attrition` | Probability that a preclinical program is abandoned before any human is dosed, for a scientific reason. | 3.0 y |
| `eventual_approval` | Probability that a program entering human development is eventually approved by a major regulator in this indication. | 8.0 y |

`ph2_any_scientific_failure` is the composite that widens the *cause* — efficacy, safety or PK/PD
rather than efficacy alone. It has fewer exclusions than the primary and therefore a larger cohort,
but it is **still conditional**: it continues to exclude commercial and strategic discontinuations,
because those are not statements about the biology. No endpoint defined here is unconditional, and
none can be while sponsors stop programmes for reasons a model of translation should not be asked
to predict.

`preclinical_attrition` carries a standing warning in its own `notes` field: it is severely
affected by survivorship bias, because programs that die before an IND are frequently never
disclosed. Estimates conditional on entering human development are better supported than
unconditional attrition estimates, and this asymmetry is a property of the public record, not of
the estimator.

### Censoring

A program that entered the clinic recently has not yet had the opportunity to fail. Labelling it
"did not fail" would bias every rate downward, so each endpoint declares `min_followup_years` and a
track with an unresolved outcome and less follow-up than that is excluded as censored rather than
counted as a success.

### The composite outcome

`discontinued_through_phase_2` is deliberately ambiguous in the ontology — it records "stopped
somewhere at or before Phase II" without asserting why. `resolve_composite_discontinuation()`
disambiguates it from the discontinuation reason, in one place: efficacy-insufficient counts as
positive for the efficacy endpoints, safety and PK/PD are excluded as competing events, and
non-scientific reasons are excluded outright.

---

## 6. The translational scorecard

The 12 feature domains in `trs/features/spec.py` are the rows of the scorecard presented to users.
Labels below are `DOMAIN_LABELS`; feature counts are from `features_in_domain()` in
`trs/features/registry.py` (63 features in total).

| Domain | Features | What it asks |
|---|---:|---|
| Program context | 4 | Modality and target class, entered explicitly because translational performance is not uniform across them. |
| Animal-model validity | 10 | Were the animal experiments constructed to interrogate the *human* disease mechanism, and how strong and precise were the effects? |
| Cross-species concordance | 5 | Does the effect reproduce across species and systems, or rest on a single model? |
| PK/PD translation | 6 | Is the exposure associated with preclinical efficacy plausibly achievable and tolerable in humans, and has target engagement been shown? |
| Human biological support | 5 | Evidence generated in human material or human populations: tissue, primary cells, organoids, genetics, prior clinical experience. |
| Target validation | 6 | Strength and causal character of the target–disease link, and how easily compensatory biology bypasses the target. |
| Disease-model relevance | 4 | Fit between the intended trial population and the biology the preclinical package interrogates. |
| Reproducibility & study quality | 9 | Randomisation, blinding, independent replication, pre-registration, reporting integrity. |
| Safety translation | 4 | Preclinical safety findings that constrain the dose, and therefore the exposure, achievable in humans. |
| Biomarker readiness | 3 | Is a PD or patient-selection biomarker ready to deploy in first-in-human studies? |
| Historical class performance | 4 | Outcomes of earlier programs sharing the target, mechanism class or indication — counted strictly as of the cutoff. |
| Evidence volume & recency | 3 | How much evidence exists and how current it is. Confounded with publication and disclosure behaviour. |

Two domains carry standing caveats in the source:

- **Historical class performance** is the most predictive domain available and the easiest way to
  destroy a model's validity. Fold contamination through this channel is *deliberately not*
  prevented, because at the historical decision point the analyst genuinely did know other
  companies' resolved outcomes; removing them would make the training distribution differ from the
  deployment distribution. The consequence — grouped folds are not fully independent through this
  channel — is disclosed, and the validation suite reports metrics with the domain ablated so its
  contribution is visible.
- **Evidence volume & recency** is confounded with publication and disclosure behaviour. A program
  with more papers is not necessarily better supported; it may simply have a sponsor with a
  publication strategy.

The scorecard is a presentation of features, not a scoring rubric. Domains do not have fixed
weights: the model's coefficients determine the contribution, and the domain ablation in the
validation report shows which domains actually earn their place.

---

## 7. What ships with every estimate

No endpoint in this platform returns a bare probability. Every score carries:

| Component | Source | Why |
|---|---|---|
| Calibrated probability | `trs/modeling/calibration.py` | The product's output. |
| Uncertainty interval | `trs/modeling/uncertainty.py` | A point estimate without one invites false precision. |
| Interval decomposition | `predict_with_uncertainty()` | Parameter, imputation and calibration components, reported by ablation on the log-odds scale. |
| Specification sensitivity | `ScoringService._alternative_family_predictions` | Disagreement between model families — reported *beside* the interval, never folded into it. |
| Evidence completeness | `trs/features/missingness.py` | Weighted by how much each feature moves the estimate. |
| Evidence-confidence grade | `grade_evidence_confidence()` | A qualifier on the probability, which it never modifies. |
| Driver attribution | `trs/scoring/explanation.py` | Exact terms of the model's linear predictor. |
| Model version & feature schema | `trs/pipeline/artifacts.py` | Reproducibility. |
| `is_demonstration` flag | `ScoringService` | Set when the model, the evidence, or both include synthetic data. |

Three of these deserve their rationale stated.

**Why the interval is decomposed on the log-odds scale.** The logistic link compresses probability
differences near 0 and 1, so the same amount of underlying ignorance produces a much narrower
probability-scale band at 5% than at 50%. Ablation widths measured in probability units are
therefore not comparable between programs, nor even between components for the same program.
`variance_share` normalises squared component standard deviations to sum to one, and is documented
as an approximation — the sources interact, so the total is reported separately and never
reconstructed from the shares.

**Why imputation is one of the three sources.** Each of `M` imputations fills a program's gaps with
an independent draw from the training distribution, so the spread across imputations is exactly the
cost of not knowing. A complete evidence package contributes zero here; a sparse one contributes a
lot. This is what makes the interval widen for the right reasons rather than uniformly.

**Why the confidence grade is separate from the probability.** A calibrated 70% built on a complete
package and a calibrated 70% built on four data points are the same number and not the same claim.
The interval does widen for the second, but interval width alone under-communicates. The grade
qualifies the probability; it never modifies it.

If the deployed family cannot produce a posterior, the scoring service **raises** rather than
returning a fabricated range.

---

## 8. Scenario analysis

`trs/scoring/scenarios.py`

A scenario is a counterfactual evaluation of the *model*, under a hypothetical feature state:
"if an independent laboratory replicated the pivotal in vivo result, and nothing else changed, this
model would return X instead of Y."

### What it is not

It is **not** a prediction that the experiment would succeed, and **not** a statement about what
the experiment would show. A scenario answers "how much does this estimate depend on this piece of
evidence?" — a question about the model's sensitivity, not about biology. Every scenario response
carries that distinction in words, because the temptation to read it the other way is strong and
the consequence is a planning decision made on a fiction.

### Design rules, and the bug that produced one of them

**Scenarios never write evidence.** Overrides are applied to a copy of the feature vector at
scoring time. The evidence base is untouched, so speculation cannot contaminate the record or leak
into a future retraining set.

**Levers are directional.** A lever is a curated bundle of feature overrides representing one
realistic action, and it carries a `LeverDirection`. `IMPROVEMENT` levers can only move a feature
in the risk-reducing direction. The source records why: an early version applied lever values
unconditionally, so "add an independent replication" applied to a program already replicated across
several laboratories *lowered* its replication score and raised its estimated risk. The scenario
then read as "commissioning this study would make the program worse", which is nonsense and would
have been acted on. `ADVERSE` levers are the mirror image; `ABSOLUTE` levers assert a state
outright and are used where the feature is a design choice rather than an accumulating body of
evidence — enrolling a biomarker-selected population is a choice, not a discovery.

**Only some features may be varied.** 26 of the 63 features are declared `scenario_adjustable`.
The exclusions are deliberate: a team cannot change the historical failure rate on its target or
the disease's unmet need by doing work, and offering to vary them would invite conclusions the
analysis cannot support. A lever cannot bypass this — resolved lever bundles are validated exactly
like a hand-written override.

The 14 levers currently exposed at `GET /api/v1/scenarios/levers` cover independent and multi-lab
replication, human organoid and patient-tissue evidence, exposure margin and exposure–response,
human target engagement, PD-biomarker readiness, biomarker-selected enrolment, high-fidelity model
results, second-species efficacy, human genetic support, and two adverse levers — a contradictory
animal study appearing, and an exposure gap being identified.

---

## 9. Comparators

`trs/scoring/comparators.py`

A retrieval over structured facets — target, pathway, mechanism class, modality, indication,
disease stage, biomarker strategy, preclinical model construction — returning earlier programs that
resemble the focal one, with their observed outcomes and the reason each is considered comparable.

### The causality caveat

It is **not** evidence that the focal program will share their fate. Similarity on a handful of
recorded facets is not exchangeability: two KRAS inhibitors can differ in exactly the ways that
decide the outcome and agree on everything this system records. Comparators are presented as
context for judgement, never as a prediction, and the API response carries that statement rather
than leaving it to be inferred.

### The temporal rule

Only programs whose outcome was already known at the focal program's cutoff are returned. A 2020
failure is not a comparator for a 2014 decision; showing it would let a user read the score as if
it had been informed by information that did not exist. This mirrors the constraint on the model's
own features.

---

## 10. Non-goals and out-of-scope claims

Stated as flatly as possible, because a decision-support product's failure mode is being used for
the decision next door.

**Not a regulatory instrument.** No submission, no qualification, no regulatory standing. The
footer of every page says so.

**Not clinically validated.** There is no prospective validation. There is no retrospective
validation on real programs either — the registered model was trained and evaluated entirely on
synthetic data.

**Not a safety or toxicology predictor.** The primary endpoint deliberately *excludes* safety
discontinuations as competing events. `ph1_safety_failure` exists as a declared endpoint but no
model is registered against it in this repository.

**Not a substitute for the preclinical package.** The platform reads a package; it does not
replace one. A program with no evidence gets a wide interval and a low confidence grade, not a
verdict.

**Not a causal model.** Feature contributions are associations estimated in a small observational
cohort. A negative coefficient on a scenario-adjustable feature does not license "do this and the
risk falls". The sign audit exists precisely because some coefficients will contradict their
stated priors, and that is information, not something to be silently corrected.

**Not a target-discovery or indication-selection engine.** The unit of analysis presupposes a
program and an indication already chosen.

**Not a valuation model.** There is no NPV, no risk-adjusted revenue, no cost of capital. Turning
a probability into a value requires assumptions this platform does not hold.

**Not a ranking service.** Programs can be sorted by estimate, but the intervals overlap heavily
and the interface shows that. See [§11](#11-where-the-product-actually-stands).

**Not an LLM-narrated product.** A language model may re-word the structured explanation output;
it never generates the drivers, their directions or their magnitudes. Those come from
`explanation.py` and are asserted to reconstruct the score.

**Not a general literature tool.** Connectors exist to populate declared features, not to answer
free-text questions.

---

## 11. Where the product actually stands

The platform's own verdict on the registered demonstration model, read from
`backend/artifacts/ptrs-pef-007.json` (`metrics.usefulness`):

> **`not_distinguishable_from_base_rate`**
>
> "The model is not distinguishable from quoting the cohort base rate: Brier skill -6.5% (95% CI
> -21.5% to +5.9%), interval includes zero. Treat individual estimates as weakly informative at
> best."

The artifact also records *why* this is the threshold question rather than a footnote:

> "Discrimination metrics are invariant to monotone distortions of probability. A model can rank
> programs correctly and still produce probabilities less accurate than the base rate. Because this
> product's output is a probability, skill against the base rate is the threshold question."

This is the product working as designed. A platform that reports a favourable AUROC and hides a
Brier skill interval that includes zero is not a better platform; it is a less honest one. The
scepticism is load-bearing: the same machinery that produces this verdict on synthetic data is what
would produce a defensible verdict on real data.

What follows from it, for each user in [§2](#2-users-and-the-decision-each-is-making), is that on
the current demonstration deployment **no individual estimate should be acted on**. The surfaces
that remain useful today are structural rather than predictive: the exclusion waterfall, the
completeness assessment, the evidence matrix, the provenance chain, and the scenario sensitivity —
each of which reports something the platform actually knows.

For the full evaluation record, its design and its limitations, see
[VALIDATION_PLAN.md](VALIDATION_PLAN.md) and [MODEL_CARD.md](MODEL_CARD.md).
