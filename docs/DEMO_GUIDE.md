# Demo guide

How to show this system in ten minutes without saying anything untrue.

The platform's own verdict on itself is that its model does not beat quoting a
base rate. That is not a caveat to work around in a demonstration — it is the
most interesting thing about it, and the parts worth showing are the parts that
make such a verdict unavoidable rather than optional.

Everything below runs against the synthetic demonstration dataset. No command
here touches the network.

---

## 0. Start it

```bash
make install     # once
make demo        # migrate, seed, build cohort, train. ~4 minutes.
make api         # terminal 1
make web         # terminal 2 → http://localhost:3000
```

`make facts` prints every checkable number in the system. If a figure in any
document disagrees with that output, the document is wrong.

---

## 1. Explain the project in four sentences

> For one programme, in one indication, assessed as of one historical decision
> date, the system assembles the preclinical evidence that existed at that
> date, reduces it to a declared 63-feature schema, and returns a calibrated
> probability that the programme fails to show sufficient Phase II efficacy.
>
> Every number is traceable to the evidence rows that produced it, and the
> driver attribution sums exactly to the log-odds of the number on screen.
>
> The model is fitted under a named split protocol that is recorded in the
> artifact, and the system refuses to serve a score when its own status does
> not permit it.
>
> On this demonstration dataset the model does not beat a base rate, the system
> says so on its own front page, and the status machinery makes that
> consequential rather than decorative.

Resist the urge to call it "a risk model with an AUROC of 0.62". The number
without its protocol is not a claim anyone can check.

---

## 2. Show that it refuses

This is the shortest path to the point.

```bash
# No demonstration context declared.
curl -s localhost:8000/api/v1/programs/<TRACK_ID>/score | jq '{probability, eligibility}'
```

```json
{
  "probability": null,
  "eligibility": {
    "permitted": false,
    "model_status": "demo_only",
    "refusal_code": "demo_context_not_selected",
    "reason": "This is a demonstration model on demonstration evidence. …"
  }
}
```

`probability` is **absent**, not zero and not a wide interval. Then show the
interface rendering the same state: set `NEXT_PUBLIC_DEMO_CONTEXT=false` in
`frontend/.env.local`, restart the dev server, and open any programme. The page
shows "No risk estimate for this programme", the refusal code and the model
status — and then shows the evidence, the domain matrix, the PK/PD bridge, the
target assessment and the provenance chain exactly as before.

That is the claim worth making: **declining to score is not declining to be
useful.** Delete `.env.local` to go back.

The rule that does the real work is stricter than this one. A synthetic-trained
model asked to score *real* evidence is refused regardless of any flag, because
a simulation's opinion about a real programme is not a meaningful quantity at
any confidence.

---

## 3. Inspect a source

From any programme page, the evidence table lists each item with its class,
species, endpoint and availability date. Pick one and follow it:

```bash
curl -s localhost:8000/api/v1/provenance/evidence/<EVIDENCE_ID> | jq
```

You get the source record, its identifier, the extraction method, the review
status, the availability date, and the raw payload hash. The point to make:
**`available_at` is the universal knowability date**, and every feature query
goes through `AsOfContext`, which applies `available_at <= as_of` and — with
`strict=True` — re-checks the rows the query returned rather than trusting the
SQL.

Then show the gap this does *not* close:

> Feature availability was always enforced to the day. Label availability was
> not enforced at all until this revision. A programme assessed in 2008 whose
> outcome resolved in 2019 was training a model described as sitting at a 2012
> origin.

---

## 4. Examine a limitation the system enforces on itself

Open **Methods & validation**, or:

```bash
curl -s localhost:8000/api/v1/model/validation | jq '{protocol: .protocol.split_manifest.protocol,
  permits_deployment_claim: .protocol.split_manifest.permits_deployment_claim,
  rejected: [.protocol.protocol_trace[] | select(.rejected_because) | .protocol],
  status: .status.status, usefulness: .usefulness.verdict}'
```

```json
{
  "protocol": "temporal_outer_maturity",
  "permits_deployment_claim": false,
  "rejected": ["strict_temporal"],
  "status": "demo_only",
  "usefulness": "not_distinguishable_from_base_rate"
}
```

Three things to point at:

1. **A protocol was rejected, and the reason is recorded.** `strict_temporal`
   passes the row-count checks and then yields zero usable folds, because its
   inner folds must also respect label maturity and at a 2009 fold origin only
   five outcomes had resolved. That is a fact about the data. The trace carries
   the measurement, not a shrug.
2. **`permits_deployment_claim` is `false`.** The accepted protocol may not
   describe its numbers as deployment performance, and this flag — not a
   sentence in a document — is what the interface reads.
3. **Two reference models are reported.** `scaled_brier` normalises against the
   evaluation set's own prevalence, which is the best constant *in hindsight*;
   the deployable reference is an intercept-only model fitted on development
   labels, which is what could actually have been shipped. The model loses to
   both.

If you have time for one more, show the calibration scale audit. The chosen
pool's audit is at `.calibration_audit` (verdict `consistent`), and the pool it
beat is recorded beside it:

```bash
curl -s localhost:8000/api/v1/model/validation \
  | jq '{chosen: .calibration_selection.design,
         rejected: .calibration_selection.rejected_designs}'
```

The forward-chaining out-of-fold pool was rejected with verdict
`scale_mismatch`: its base rate was 0.664 against the development partition's
0.631, and the fold models' mean logit sat 0.33 below the final model's, so the
recalibration intercept would have been fitted to lift flat logits onto the
wrong level. That is not hypothetical — it happened, and it cost 0.19 nats of
evaluation log loss before the audit existed.

---

## 5. Reproduce an assessment

Every score carries the identifiers needed to recreate it:

```bash
curl -s "localhost:8000/api/v1/programs/<TRACK_ID>/score?demo_context=true" \
  | jq '{model_version, feature_snapshot_id, as_of_date,
         functional: .prediction_contract.functional,
         evaluated: .prediction_contract.evaluated_functional,
         recon_error: .explanation.served_reconstruction_error}'
```

Then check the two properties that make the number auditable:

**The served quantity is the evaluated quantity.** `functional` and
`evaluated_functional` are both `calibrated_plug_in`. Before this revision the
API returned a Monte-Carlo draw-mean while every metric in the artifact
described the plug-in value — two different quantities under one name,
differing most for the programmes with the least evidence.

**The attribution reconstructs what is displayed.**
`served_reconstruction_error` is 0 to machine precision, and
`external_check_ran` is `true`. The older internal check compared a partition of
a sum against that sum, both built from the same terms; it could not notice that
the decomposition targeted a quantity the interface never showed.

**The snapshot is immutable.** Correcting an evidence item produces a *new*
snapshot at the same scientific cutoff with a new `evidence_revision`, and marks
the old one superseded without altering its values — so the score you just
reproduced stays reproducible.

---

## 6. What not to say

- Do not call any number here a prediction about a real programme. Every
  programme, every evidence item and every outcome in this dataset is
  synthetic.
- Do not present AUROC as evidence that the model is useful. It is invariant to
  monotone distortions of probability, which is exactly the failure this
  product would suffer from; the Brier skill against a deployable reference is
  the question that decides usefulness, and it is negative.
- Do not describe `approved_for_defined_research_use` as validation. It is a
  deployment-policy decision about a defined research use. Nothing in this
  codebase can confer clinical validation or regulatory qualification, and no
  artifact trained on synthetic data can reach that status at all.
- Do not describe the scenario panel as a forecast. It recomputes the model
  under a hypothetical evidence state. It does not predict that the experiment
  would succeed, or that commissioning the work would lower real risk.

---

## 7. If someone asks "so does it work?"

The honest answer, and the one the artifact gives:

> As an evidence-review instrument, yes — and that part does not depend on the
> model. As a risk estimator, not on this data: it does not beat an
> intercept-only model fitted on the same development partition, and the paired
> component bootstrap on the difference does not exclude zero.
>
> Whether the method works on real evidence is untested. No real-programme
> predictive validation exists, prospectively or retrospectively.
