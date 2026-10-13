# Refinement audit

Baseline commit: `b0a6ca38113559c4d6e6967a1a8ce7cf557f3c80` (`main`), working tree clean.
Audit branch: `refinement/scientific-integrity`.
Environment: macOS 23.5.0, CPython 3.14 in `.venv`, Node/npm via `make install`.

Every finding below was reproduced against this checkout. Where a claim in the
repository's own documentation is contradicted by the code, the document is
quoted so the gap is on the record. Findings are ordered by severity, and each
carries the regression test that must exist before it can be called fixed.

---

## 1. Commands actually run, and what they returned

| Command | Exit | Result |
|---|---|---|
| `make install` | 0 | Editable backend install + `npm install` both succeed |
| `cd backend && ../.venv/bin/python -m pytest -q` | 0 | **255 passed** in 14.8 s |
| `make lint` | 0 | `ruff check trs tests` → "All checks passed!", then `npm run typecheck` → clean |
| `cd frontend && npm run lint` | **1** | `next lint` → `Invalid project directory provided, no such directory: .../frontend/lint` |
| `cd frontend && npm run build` | 0 | Compiled in 1.83 s; 5 routes + `/_not-found`, all `ƒ` (dynamic) |
| `make facts` | 0 | Full fact sheet; figures quoted throughout this document |
| `python -c "...ScoringService.score_track(...)"` | 0 | Scored a track: `p = 0.6949953625617773` |
| `python -c "...app.openapi()..."` | 0 | 30 operations under `/api/v1` |

Environment limitations, distinguished from application failures: no network
access was used for any connector call, so live-connector behaviour is
unverified in this pass and the connector findings below are read from code and
offline fixtures only. No PostgreSQL instance was available, so migrations were
exercised against SQLite only.

---

## 2. Confirmed defects

### D1 — Selection and final evaluation share one partition (critical)

`backend/trs/pipeline/train.py:386-401`. Candidates are each scored on `te`,
then:

```python
selection = choose_family(
    candidate_results, y[te], {f: v[2] for f, v in fitted.items()}, components[te],
    random_state=random_state,
)
...
final_metrics = evaluate(y[te], p_test, components=components[te], ...)
```

`choose_family` receives the outer-test labels and every candidate's test
predictions, runs a paired component bootstrap on them, applies the calibration
band and the margin rules — and the winner is then reported on the same rows.
`te` is a model-selection partition. Every headline figure in the fact sheet
(AUROC 0.6383, log loss 0.6825, Brier skill 0.0220, calibration slope 1.0791)
is a selection-set figure presented as held-out performance.

The module docstring at `train.py:16-18` states the opposite:

> **Evaluate once, on test.** No iterating against the test partition. When a
> candidate loses, it loses.

That is true of the *final* `evaluate` call in isolation and false of the
procedure. Three families competed on `te`; the gbdt candidate was disqualified
by a calibration slope measured on `te`; the hierarchical candidate lost by a
margin measured on `te`.

**Failure mechanism.** Selection on a 147-row partition with 465 independent
components, using a criterion (log loss) evaluated on the same rows that then
report the criterion. The optimism is not bounded by the margin rule, because
the margin rule is itself computed on the contaminated partition.

**Fix.** A declared development/evaluation protocol: a locked evaluation
partition that no selection step may read, with all tuning, family choice and
calibration confined to development data. Record the legacy numbers as
exploratory.

**Regression test.** Under a fixed split manifest, permuting the evaluation
labels must leave the chosen family, hyperparameters, coefficients and
calibrator bit-identical.

### D2 — Preprocessing is fitted outside the inner folds (high)

`train.py:292-298`:

```python
transformer = FeatureTransformer().fit(
    [rows[i] for i in split.train], random_state=random_state
)
X = transformer.transform(rows)
```

`tune_l2`, `_tune_group_precision` and `cross_validated_calibration` all consume
this single `X`. Imputation medians, standardisation constants and categorical
vocabularies are therefore estimated from the whole outer-training partition,
including each inner fold's own held-out rows. There is no outer-test leakage —
the transformer never sees `split.test` — but every inner-fold number
(the selected L2 penalty, the group precision, the pooled out-of-fold
predictions the calibrator is fitted on) is optimistic.

**Fix.** Fold-local pipelines: the transformer, any learned prior and any
feature selection fitted inside each fold on that fold's permitted training rows.

**Regression test.** A fold-local transformer must produce different imputation
constants per fold on data with fold-varying medians, and the inner-fold
predictions must change accordingly.

### D3 — Label maturity is not enforced at the fitting origin (high)

The temporal split orders rows by `CohortEntry.as_of_date` (`train.py:215-240`,
`_split_fractions`, `temporal_split`). `ProgramIndication.outcome_date` exists
and is used by `features/historical.py`, `cohort/build.py`,
`scoring/comparators.py` and `cohort/outcomes.py` — but **never** by the
splitters or by `train()`:

```
$ grep -rn 'outcome_date' trs/ | grep -v 'models/core.py'
trs/seed/generator.py, trs/features/registry.py, trs/features/historical.py,
trs/cohort/outcomes.py, trs/cohort/build.py, trs/scoring/comparators.py,
trs/db/migrations/..., trs/api/schemas.py, trs/api/v1/programs.py
```

No hit in `trs/validation/splitters.py` or `trs/pipeline/train.py`.

**Failure mechanism.** A program assessed in 2008 whose outcome resolved in 2019
sits in the training partition of a split whose boundary is 2012. The model is
described as reflecting what was knowable at a historical origin; its labels
were not knowable then. Feature availability is enforced rigorously by
`AsOfContext`; label availability is not enforced at all.

**Fix.** Carry a label-availability date per row and require, for a declared
fitting origin, that `label_available_on <= origin`. Record assessment date,
fitting origin and label cutoff as three separate fields.

**Regression test.** A row whose `outcome_date` falls after the fitting origin
must be excluded from training at that origin, and the exclusion must be counted.

### D4 — The served quantity is not the evaluated quantity (high)

`backend/trs/modeling/uncertainty.py:169-170`:

```python
probs = sigmoid(cal_draws).ravel()
point = float(np.mean(probs))
```

That is a Monte-Carlo mean over the product of imputation draws, parameter draws
and calibration-parameter draws. `scoring/service.py:260` and `:293` serve it as
`probability`. Training and validation, by contrast, use

```python
test_logit = _predict(model, X[te], mech[te])
p_test = calibrator.transform(test_logit)
```

— a plug-in point prediction at posterior-mean parameters on median-imputed
features. By Jensen's inequality these differ systematically, not just by Monte
Carlo noise, and the difference grows with imputation variance, i.e. precisely
for the programs with the least evidence.

**Fix.** Declare one primary prediction functional, evaluate that functional,
and name the secondary quantity explicitly wherever it is exposed.

**Regression test.** For a fixed program, the probability returned by the API
must equal the functional the metrics were computed from, to a declared
tolerance.

### D5 — The attribution reconstruction check is self-referential (medium)

`scoring/explanation.py:163-164` and `:230-232`:

```python
total_logit = baseline_logit + float(scaled.sum())
...
reconstructed = baseline_logit + sum(d.contribution for d in drivers) + other
reconstruction_error = abs(reconstructed - total_logit)
```

Both sides are built from the same `scaled` vector, so the assertion verifies
that a partition of a sum sums to the sum. It cannot detect that the explanation
targets the wrong quantity — and it does: it reconstructs the plug-in logit,
while the product serves the draw-mean of D4. The module docstring claims the
decomposition reconstructs "the log-odds of the number on screen".

**Fix.** Compare the reconstruction against the independently computed served
quantity, and record `explained_quantity`, `prediction_method` and
`calibration_scale` on the payload.

**Regression test.** Mutating the served functional without updating the
explainer must fail the reconstruction check.

### D6 — The skill reference is the evaluation set's own prevalence (medium)

`validation/metrics.py:67-74`:

```python
base = float(np.mean(y))
bs_base = float(np.mean((base - y) ** 2))
return float(1.0 - brier_score(y, p) / bs_base)
```

`base` is computed from the evaluation labels, so the reference is the best
possible constant *on the evaluation set* — not a model anyone could have
deployed. Measured on this checkout: cohort positive rate **0.5888**, evaluation
partition prevalence **0.5170**.

Note the direction, because it matters for honesty: the current choice is the
*harder* bar. A deployable intercept-only model fitted on development data would
predict ≈0.589 against a partition whose prevalence is 0.517 and would therefore
score worse, making the model's apparent skill *larger*. Correcting this
reference will move Brier skill up. That is a change of reference, not an
improvement in the model, and must be reported as such.

**Fix.** Report both: the development-fitted intercept-only reference (the
deployable baseline) and the evaluation-prevalence normalisation, labelled
distinctly. Compute the paired component bootstrap of the difference against the
deployable one.

**Regression test.** With a development set at prevalence p1 and an evaluation
set at p2 ≠ p1, the two references must differ and both must be reported.

### D7 — Activation is automatic and model status is binary (medium)

`train()` registers the chosen artifact and activates it in the same call.
`trained_on_synthetic=True` is recorded on the artifact and surfaced in API
responses, but it gates nothing: the scoring route returns a probability for any
track regardless. There are no `demo_only` / `insufficient_data` /
`evaluation_failed` / `research_only` / `approved_for_defined_research_use`
states, and no concept of "registered but not serving".

The usefulness verdict is already computed and already says
`not_distinguishable_from_base_rate` — it is displayed and ignored.

**Fix.** An explicit status enum with a default-deny posture, enforced in shared
backend policy consumed by the API, the UI and exports; nullable scores with
machine-readable refusal reasons; training able to register without activating.

**Regression test.** A synthetic-trained model asked to score real (non-synthetic)
evidence must refuse with a machine-readable reason, and unknown status must not
behave as approved.

### D8 — Snapshot identity omits the evidence revision, and overwrite mutates (medium)

`pipeline/features.py:50-57` keys snapshots on
`(program_indication_id, as_of_date, feature_schema_version)`. With
`overwrite=True` the existing row's `features`, `missingness`, `vector_hash`,
`built_at` and `contributing_evidence_ids` are rewritten in place. Any
`Score` row already referencing that snapshot now points at different numbers,
and the module docstring's claim that the constraint "makes accidental
divergence impossible to miss" is the opposite of what overwrite does.

**Fix.** An immutable evidence revision/manifest identity in the snapshot key, so
corrected evidence produces a *new* snapshot at the same scientific cutoff;
refuse to overwrite a snapshot referenced by a score.

**Regression test.** Adding evidence and rebuilding must create a second
snapshot; the original score must remain reproducible from the first.

### D9 — The historical-rate index ignores the requested endpoint (medium)

`pipeline/features.py:31-40` defaults to `PRIMARY_ENDPOINT`, and
`materialise_cohort_snapshots` takes no endpoint argument at all, so it always
calls `build_history_index(session)` with the primary endpoint's positive and
negative outcome sets. Training any of the other five declared endpoints
silently uses primary-endpoint historical labels as predictors.

**Fix.** Thread the endpoint through snapshot materialisation, and make the
endpoint part of snapshot identity, or declare the history features explicitly
endpoint-independent.

**Regression test.** Snapshots built for two endpoints with different
`positive_outcomes` must not be interchangeable.

### D10 — Calibration folds are random-grouped, not forward-chaining (medium-low)

`pipeline/calibrate.py` uses `grouped_kfold`. The reasoning in its docstring is
sound as far as leakage goes, and the alternative it replaced was measurably
worse. But a grouped out-of-fold prediction is not a forward-in-time prediction,
and the artifact records `design: cross_validated_out_of_fold` with no such
caveat. On this checkout `n_calibration = 0` and all 405 training rows go through
this path, so it is not an edge case — it is the only calibration path in use.

**Fix.** State the limitation in the artifact; offer forward-chaining OOF
calibration and report which was used.

### D11 — `total` is a page count presented as a portfolio count (medium)

**This entry is a correction.** The first version of this audit claimed the
portfolio page computed its means over 60 rows while displaying the full cohort
count of 552 — "two denominators, one panel". That was wrong, and the wrong
version was committed. `backend/trs/api/v1/portfolio.py:153` returned
`total=len(rows)`, so with `limit: 60` the field reported **60**: the two
denominators agreed. Verified by counting the filtered set directly (765
matching tracks) against what the endpoint would report (60).

The actual defect is worse than the one I described. The field was *named*
`total`, which every consumer reads as the portfolio size; it was the page
size. There was no `offset` parameter and no pagination control, so **705 of
765 programmes were unreachable through the interface, with nothing to indicate
they existed.** And `frontend/app/page.tsx` introduced the honesty problem the
API had merely enabled, describing its 60-row view as:

> Translational-risk estimates for every program in the portfolio

**Fix.** Replace `total` with `n_returned` and `total_matching` (counted from
the same filtered statement, so it cannot drift from the rows), add
`offset`/`has_more` and a server-computed `page_summary` whose own payload
declares `scope: "this_page_only"` and names the portfolio-level denominator it
is not describing. Add addressable server-rendered pagination to the page, and
state the visible row range beside the summaries.

**Regression test.** Pagination and summary must agree about their denominator
and scope; consecutive pages must not overlap; `total_matching` must not move
with the page.

### D12 — The declared frontend lint script does not run (low)

`frontend/package.json` declares `"lint": "next lint"`. Next.js 16 removed that
subcommand, so the string is parsed as a project directory:

```
$ npm run lint
> next lint
Invalid project directory provided, no such directory: .../frontend/lint
exit 1
```

No ESLint or Biome config exists in `frontend/`. `make lint` does not call it —
it runs `npm run typecheck`, and its own help text says so ("Ruff on the
backend, TypeScript on the frontend"), so the substitution is disclosed but the
frontend is not linted at all.

**Fix.** Install a real linter with a config, or delete the dead script. Do not
leave a declared script that exits 1.

### D13 — Personal absolute paths in shipped documentation (low)

`git grep -c '/Users/jorgerodriguez'`: README.md 12, DATA_PROVENANCE.md 10,
MODEL_CARD.md 8, VALIDATION_PLAN.md 5, MODELING_PLAN.md 4, DATA_DICTIONARY.md 2
— 41 occurrences. A fresh clone cannot run the documented commands verbatim.

### D14 — Documentation figures are stale (low)

The eight root documents quote the pre-`_milestones()` training run (cohort 571,
AUROC 0.630, log loss 0.671, oracle 0.737). The live values are cohort 552,
AUROC 0.6383, log loss 0.6825, oracle 0.7820. `make facts` is the authority.

### D15 — `outcomes.py` calls the composite endpoint "unconditional" (low)

`cohort/outcomes.py:31` — "exists precisely so the unconditional scientific
attrition question can also be asked". The endpoint's own `estimand` field is
correct and says commercial and strategic discontinuations are excluded. The
docstring overclaims; the field does not.

---

## 3. Working capabilities confirmed by inspection

These are real and must survive the refactor:

- **`AsOfContext`** (`features/temporal.py`) — single chokepoint, applies
  `available_at <= as_of`, and with `strict=True` re-verifies returned rows and
  raises `TemporalIntegrityError`. Feature-side temporal integrity is genuinely
  enforced, which is what makes D3 a label-side gap rather than a general one.
- **Connected-component grouping** with `GroupScope.IDENTITY` / `STRICT`, and a
  recorded effective sample size (465 components for 552 rows, 390 singletons).
- **The usefulness gate** — already computes and records
  `not_distinguishable_from_base_rate` with a component bootstrap interval that
  includes zero. The platform's own verdict on itself is correct and published.
- **The oracle ceiling** (AUROC 0.7820) — an honest upper bound for the
  synthetic generator, which prevents reading 0.638 as a failure of fitting.
- **Grid-boundary detection** in `tune_l2`, **coefficient sign audit** (56
  checked, 19 discordant, 0 credibly discordant), **domain ablation**,
  **exclusion accounting** (213 exclusions itemised by reason).
- **Artifact integrity** — `load_bundle` recomputes SHA-256 and refuses a
  mismatch.
- **Tenant scoping** present in `api/v1/portfolio.py`, `programs.py`, `scores.py`
  via `principal.organization_id`.
- **Synthetic labelling** — `trained_on_synthetic` on artifacts, `is_synthetic`
  on snapshots, prominent README warning. The labelling is thorough; D7 is that
  it does not *enforce* anything.
- 255 backend tests pass; frontend typechecks and builds; 30 API operations.

---

## 4. Claims in the repository not supported by its code

| Claim | Location | Status |
|---|---|---|
| "Evaluate once, on test. No iterating against the test partition." | `train.py:16-18` | Contradicted by D1 |
| Encoder comment "fit on TRAIN only" implying fold-safety | `train.py:292` | True of outer test, misleading for inner folds (D2) |
| Attribution sums "to the log-odds of the number on screen" | `explanation.py:8-12`, README | False while D4 stands |
| Snapshot constraint "makes accidental divergence impossible to miss" | `pipeline/features.py:1-8` | Contradicted by `overwrite=True` (D8) |
| "the unconditional scientific attrition question" | `outcomes.py:31` | Overclaim (D15) |
| `"lint": "next lint"` | `frontend/package.json` | Exits 1 (D12) |

---

## 5. Not defects, recorded to prevent mis-fixing

- **Analytic intervals 1.88× wider than bootstrap** (`interval_agreement`
  verdict `analytic_intervals_too_wide`). This is a disagreement, not a proof
  that the analytic interval is wrong. Bootstrap intervals that refit frozen
  components estimate a different quantity. No shrinkage will be applied to make
  a README sentence come true.
- **AUROC 0.638 below the oracle's 0.782.** Expected: the oracle knows the
  simulation's own probabilities. A finite sample can even beat a
  population-optimal oracle by luck. Not a defect.
- **`backend/artifacts/ptrs-pef-001.json` is present** (520,835 bytes) and loads
  with a matching SHA-256. An earlier pass of this audit reported it missing;
  that was a truncated `ls`, not a defect.
- **The portfolio's summary denominators did agree with each other.** The first
  committed version of D11 said they did not. See the correction in D11: the
  real defect was a misnamed field and 705 unreachable programmes, not
  mismatched arithmetic.
