# Data Provenance

**Preclinical Translational Risk Scorer** — where every value comes from, how it
is tracked, and what is real versus simulated in this deployment.

> **This deployment runs on synthetic demonstration data.** Gene symbols,
> indication names, the ontology, the connector implementations and the recorded
> API fixtures are real. Every sponsor, program, compound, experiment, effect
> size, source record and development outcome is simulated. §[8](#8-what-is-real-and-what-is-synthetic-in-this-deployment)
> is the definitive list. Companion document: [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Table of contents

1. [The provenance chain](#1-the-provenance-chain)
2. [The immutable raw landing zone](#2-the-immutable-raw-landing-zone)
3. [Sources and entity resolution](#3-sources-and-entity-resolution)
4. [Evidence items and per-field provenance](#4-evidence-items-and-per-field-provenance)
5. [ProvenanceMethod and the model-generated rule](#5-provenancemethod-and-the-model-generated-rule)
6. [Temporal integrity: `available_at`](#6-temporal-integrity-available_at)
7. [The connector catalogue](#7-the-connector-catalogue)
8. [What is real and what is synthetic in this deployment](#8-what-is-real-and-what-is-synthetic-in-this-deployment)
9. [Fixture labelling convention](#9-fixture-labelling-convention)
10. [Reproducing and inspecting provenance](#10-reproducing-and-inspecting-provenance)
11. [Gaps between the documented design and the running code](#11-gaps-between-the-documented-design-and-the-running-code)

---

## 1. The provenance chain

Every number the platform shows is reachable from the bytes that produced it.
The chain, with real table and column names:

```
raw_record ──► source ──► evidence_item ──► feature_snapshot ──► score_record
     │            │             │                   │                  │
     │            │             └─► evidence_field_provenance          │
     │            │                                                    │
     └────────────┴──────────── content_hash, connector, fetched_at ────┘
```

| Step | Table | Link forward | Link back |
|---|---|---|---|
| Verbatim API payload | `raw_record` | — | — |
| Citable object | `source` | — | `source.raw_record_id → raw_record.id` |
| Structured claim | `evidence_item` | — | `evidence_item.source_id → source.id` |
| Per-field origin | `evidence_field_provenance` | — | `.evidence_item_id → evidence_item.id`, `.source_id → source.id` |
| Model-ready vector | `feature_snapshot` | `contributing_evidence_ids`, `feature_evidence_map` | `program_indication_id → program_indication.id` |
| Persisted prediction | `score_record` | — | `feature_snapshot_id → feature_snapshot.id`, `model_artifact_id → model_artifact.id` |

Two columns make the middle of that chain a capability rather than a claim:

- **`feature_snapshot.contributing_evidence_ids`** — every evidence row that fed
  the vector, stored as `{"ids": [...]}`.
- **`feature_snapshot.feature_evidence_map`** — `{feature_name: [evidence_id, …]}`,
  so a *single feature* can be traced to the exact rows behind it, not just the
  vector as a whole.

And two make it verifiable rather than merely recorded:

- **`feature_snapshot.vector_hash`** — SHA-256 over the sorted feature dict.
  Silent drift in the feature builder changes it.
- **`feature_snapshot`'s uniqueness constraint** on
  `(program_indication_id, as_of_date, feature_schema_version)`. Snapshots are
  persisted rather than recomputed on demand for three reasons: they are what a
  historical prediction is reconstructed from; recomputation after an evidence
  update would silently change an old score; and the constraint makes accidental
  divergence impossible to miss.

The chain is exposed over HTTP by `trs/api/v1/provenance.py`:

```
GET /api/v1/provenance/sources/{source_id}        source + its raw_record
GET /api/v1/provenance/evidence/{evidence_id}     evidence + source + field provenance
GET /api/v1/provenance/scores/{score_id}/lineage  score → snapshot → evidence → source
GET /api/v1/provenance/features/{feature_name}    the feature's declared derivation
GET /api/v1/provenance/connectors                 live health of every connector
GET /api/v1/provenance/connectors/catalogue       names, versions, licences
GET /api/v1/provenance/ingestion-runs             connector_run history
```

### 1.1 Supporting tables in the same chain

| Table | Role |
|---|---|
| `trial` | Registry-specific fields kept off `source` so registry semantics stay explicit — `overall_status`, `why_stopped`, `first_posted_on`, `last_update_on`, `enrollment`, `has_posted_results`. |
| `target_assessment`, `pkpd_bridge`, `program_biomarker`, `safety_finding` | Domain assessment tables, each carrying its own `available_at` and `provenance_method`. |
| `cohort_version`, `cohort_entry` | The frozen labelled dataset. `cohort_version.dataset_hash` is a SHA-256 over the definition plus sorted `(track, as_of, label)` members. |
| `model_artifact` | `dataset_hash`, `feature_schema_version`, `cohort_version_id`, `library_versions`, `git_commit`, `limitations`. |
| `connector_run` | One row per ingestion operation: connector, version, parameters, status, timings. |
| `audit_event` | Tenancy-level audit trail. |

---

## 2. The immutable raw landing zone

`raw_record` is the verbatim bytes a connector received, before anything
interprets them. It is **append-only**: never rewritten, never deleted. The
reason is diagnostic rather than archival — any downstream value can be
re-derived, and any extraction bug can be diagnosed against exactly the payload
that produced it, rather than against a re-fetch that may have changed.

```python
class RawRecord(Base, IdMixin):
    __tablename__ = "raw_record"
    __table_args__ = (
        UniqueConstraint("content_hash"),
        Index("ix_raw_record_connector_fetched", "connector", "fetched_at"),
    )
    connector, connector_version, endpoint, request_params
    http_status, media_type
    content_hash        # SHA-256 of the response body — the dedup and integrity key
    content_bytes       # verbatim bytes, when the payload is not JSON
    content_json        # parsed JSON, when it is
    fetched_at
    from_cache          # True when replayed from the on-disk cache, not the network
```

### 2.1 Content-hash deduplication

`Connector.persist_raw()` (`trs/connectors/base.py`) computes
`hashlib.sha256(payload.content).hexdigest()`, looks for an existing row with
that hash, and **returns the existing row unchanged** if one is found. A
`UniqueConstraint("content_hash")` enforces the same property at the database
level, so a race cannot produce a duplicate.

The consequence worth understanding: two fetches of an unchanged upstream record
collapse to one `raw_record`. A fetch that *does* differ — by one byte — creates a
new row, and both are retained. The landing zone is therefore a **change log of
upstream state**, which is what makes "what did this record say when we read it
in March?" an answerable question.

`content_bytes` and `content_json` are mutually exclusive: JSON payloads are
parsed and stored structured, everything else is stored as bytes. Nothing is
normalised on the way in.

### 2.2 `fetch` → `normalise` is a hard boundary

The connector contract (`trs/connectors/base.py`) fixes the same pipeline shape
for every source:

```
fetch ──► RawRecord ──► normalise ──► domain objects ──► entity resolution ──► evidence
```

- **`fetch` is the only method permitted to touch the network**, and it always
  writes the `RawRecord` before anything interprets it.
- **`normalise` is a pure function** from a raw payload to `NormalisedRecord`
  objects. It performs no I/O. Being pure, it is testable against stored fixtures
  with no network — which is how the connector test suite runs
  (`backend/tests/unit/test_connectors.py`, 220 tests, all passing as of
  2026-09-24).

`NormalisedRecord.available_at` is **mandatory**. A record whose availability
date cannot be established is not admissible, because it cannot be placed in time
and therefore cannot be used without risking leakage.

### 2.3 Reachability is a first-class state

`Connector.health()` returns one of `ok`, `degraded`, `unreachable`,
`not_configured`, `disabled`. A connector that cannot reach its source reports
`unreachable` and, if a cached payload exists, serves it while marking every
derived record `from_cache`. **It never fabricates, and it never silently returns
an empty result that a caller could mistake for "no evidence found."** That
distinction is the difference between a missing-data feature and a false one.

### 2.4 The on-disk response cache

`trs/connectors/cache.py`. Entries are content-addressed by
`SHA-256(method, URL, params, body)` and stored as a metadata JSON file beside
the raw body, so a cached payload can be inspected with ordinary tools. Default
root `./.cache` (`TRS_CONNECTOR_CACHE_DIR`), default TTL 168 hours
(`TRS_CONNECTOR_CACHE_TTL_HOURS`).

The cache is a reproducibility mechanism, not only a politeness one: a pipeline
re-run produces byte-identical raw payloads, which is a precondition for
reproducing a historical prediction.

---

## 3. Sources and entity resolution

`source` is the citable object — an article, a registry entry, a label. It
carries every external identifier the platform can join on (`pmid`, `pmcid`,
`doi`, `nct_id`, `openalex_id`, `chembl_id`, `other_identifier`, `url`), the
publication status, a retraction flag with its note, the licence, and both
`published_on` (when the source was published) and `retrieved_at` /
`retrieved_by_connector` (when *we* read it). Those two are not the same question
and the schema refuses to conflate them.

### 3.1 Aliases carry their own provenance

Every alias table (`target_alias`, `compound_alias`, `sponsor_alias`,
`indication_alias`, `program_alias`) shares `AliasMixin`:

| Column | Purpose |
|---|---|
| `alias`, `alias_normalised` | Raw and case/punctuation-folded forms |
| `alias_type` | e.g. `symbol`, `development_code`, `synonym` |
| `match_method` | `EntityMatchMethod` — `exact_identifier`, `structure_inchikey`, `fuzzy_string`, `human_asserted`, … |
| `match_confidence` | 0–1 |
| `source_id` | Where the alias came from |
| `available_at` | From when the alias was publicly knowable |

An alias has an `available_at` because a name can become public later than the
thing it names. A compound resolved by a development code disclosed in 2019
should not be joinable in a 2016 feature vector.

### 3.2 `entity_resolution_decision`

The schema provides an audit row for every automated or human entity-linking
decision: `entity_type`, `canonical_id`, `candidate_key`, `method`, `score`,
`accepted`, `decided_at`, `decided_by`, plus a JSON `evidence` blob. The intent
is that nothing is merged without a row here, so a spurious merge can always be
traced and reversed.

**In the running code this table is not yet written.** See
§[11](#11-gaps-between-the-documented-design-and-the-running-code).

---

## 4. Evidence items and per-field provenance

### 4.1 One claim per row

`evidence_item` is narrow on purpose: one system, one endpoint, one effect. A
paper reporting three models and two endpoints yields six rows. Aggregation into
features is the feature layer's job, not the extractor's, **so the aggregation
rule is versioned and testable** rather than baked into a curator's judgement.

Provenance-bearing columns on every row:

| Column | Meaning |
|---|---|
| `source_id` | The citable object this claim came from |
| `available_at` | The temporal-integrity field — earliest date this fact was knowable |
| `observed_on` | When the underlying study was conducted/reported, if distinct |
| `extraction_date` | When we extracted it |
| `provenance_method` | `ProvenanceMethod` — how the value got here |
| `extractor_version` | Which extractor produced it |
| `extraction_confidence` | `ConfidenceLevel` — confidence in the *extraction*, not the science |
| `source_snippet` | Verbatim text the value was read from |
| `source_locator` | Where in the source (page, table, figure) |
| `field_missingness` | `{field_name: MissingnessReason}` for every empty field |

`field_missingness` is load-bearing and is discussed in
[`MODEL_CARD.md`](MODEL_CARD.md) §6.4. `not_reported` (the study exists and is
silent) is informative about study quality; `not_applicable` (the question does
not arise) is not; `not_yet_available` means the information exists but postdates
the as-of cutoff and was withheld to preserve temporal integrity. Collapsing
those three into a single NULL destroys signal and hides a leak at the same time.

### 4.2 `evidence_field_provenance`

One row per field whose origin differs from the item's own. Populated when a
single evidence item is assembled from more than one source, or when one field
was human-verified while the others were not.

```python
class EvidenceFieldProvenance(Base, IdMixin):
    __tablename__ = "evidence_field_provenance"
    __table_args__ = (UniqueConstraint("evidence_item_id", "field_name"),)

    evidence_item_id   # FK, CASCADE
    field_name
    value_repr         # the value as recorded
    source_id          # the source for THIS field, which may differ from the item's
    method             # ProvenanceMethod, per field
    confidence         # ConfidenceLevel, per field
    snippet            # verbatim text this particular field was read from
    extracted_at
    extracted_by
```

The `UniqueConstraint(evidence_item_id, field_name)` means a field has exactly
one recorded origin — there is no ambiguity about which of two rows is the
authoritative provenance for `effect_value`.

The design cost, stated: this is a sparse side-table, so a consumer must join it
and fall back to the item-level provenance when no row exists. Absence of a row
means "this field's origin is the item's origin", not "unknown".

---

## 5. `ProvenanceMethod` and the model-generated rule

`trs/ontology/enums.py`. Nine values:

| Value | Meaning |
|---|---|
| `api_structured` | Read from a structured API field. The strongest provenance available. |
| `rule_based_parse` | Deterministic parse of a structured document (e.g. MEDLINE XML). |
| `regex_extraction` | Pattern extraction from text. |
| `llm_assisted_extraction` | A language model **located or normalised** a value that a human or deterministic rule then confirmed. |
| `human_curation` | A curator entered it. |
| `human_verified_llm` | LLM proposal, human confirmation. |
| `derived_computation` | Computed from other stored values. |
| `synthetic_generator` | Produced by `trs/seed/generator.py`. Demonstration data. |
| `model_generated_unverified` | A value originating from a language model's unsupported generation. |

### 5.1 The rule

The enum's own docstring states the policy:

> `LLM_ASSISTED_EXTRACTION` is permitted only for locating and normalising values
> that a human or a deterministic rule then confirms; the resulting record
> retains the verbatim source snippet. No value may originate from a language
> model's unsupported generation — see `ProvenanceMethod.MODEL_GENERATED_UNVERIFIED`,
> which the validation layer rejects from any model-ready dataset.

The distinction is the right one. An LLM that finds "TGI 68%" in a results
paragraph and normalises it is doing information retrieval, and the retained
`source_snippet` lets a human check it. An LLM that *produces* "TGI 68%" because
the number is plausible has manufactured evidence, and no amount of downstream
statistics can undo that. The enum exists so the second case is nameable, and so
that a record in that state is visibly different from one in the first.

### 5.2 Enforcement status — stated honestly

**The rule is a declared policy, not an enforced invariant in the current code.**
A repository-wide search for `MODEL_GENERATED_UNVERIFIED` / `model_generated_unverified`
finds the constant only in its own definition and in the docstring that describes
it (`trs/ontology/enums.py` lines 605 and 617). No filter in `trs/cohort/build.py`,
`trs/features/builder.py` or `trs/pipeline/features.py` excludes rows carrying it.

The practical exposure is currently zero — all 5,273 `evidence_item` rows in this
database carry `provenance_method = synthetic_generator`, and no ingestion path
ever writes the forbidden value. But the guard described in the docstring does
not exist, and anyone connecting a real extraction pipeline must add it. The
natural place is the evidence query in `trs/features/temporal.py`, because that
is the single chokepoint every model-ready read already passes through.

---

## 6. Temporal integrity: `available_at`

Provenance and temporal integrity are the same problem here. A value with a
correct source and a wrong date is worse than a missing value, because it looks
sound.

### 6.1 The chokepoint

For a track whose decision point is date `T`, the feature vector must be a
function only of facts knowable at `T`. Rather than trusting every call site to
remember a `WHERE available_at <= T` clause, **all evidence retrieval goes through
`AsOfContext`** (`trs/features/temporal.py`). Call sites cannot obtain evidence
rows any other way: `trs.features.builder` takes a context, not a session.

`AsOfContext` also runs in `strict` mode by default, re-checking the rows it
returns rather than trusting the SQL to be right, and raising
`TemporalIntegrityError` on a violation. A redundant check on a query you already
wrote is cheap; a missing filter is not.

### 6.2 `available_at`, not `published_on`

They differ and the difference matters. A 2015 paper reporting a 2011 experiment
became *knowable* in 2015. A trial registered in 2013 whose "why stopped" field
was populated in 2018 yields **two facts with two different availability dates**.
`available_at` is always "when a diligent analyst could first have found this" —
never the date of the underlying event.

### 6.3 The first-day rule for partial dates

Sources report mixed precision: ClinicalTrials.gov gives `2015-03`, PubMed gives
`2015 Mar` or a bare `1975`, Crossref gives `[[1998, 2]]`. A partial date names a
*period*, not an instant, and is resolved to the **first day** of that period.
`2015 Mar` → `2015-03-01`; `1975` → `1975-01-01`.

The rationale: `available_at` answers "from when could a diligent analyst have
known this?", and the earliest instant consistent with the source is the earliest
answer the source supports. Resolving to the last day of the period would claim
the fact was unknowable during weeks when it may well have been public.

**The residual risk is stated in the code rather than hidden.** The rule is
anti-conservative in the other direction: for a bare year it can place a fact up
to eleven months earlier than it was truly knowable. The exposure is bounded by
date precision, not by the rule, so the mitigation is to prefer precise date
fields where a source offers several — which every connector does, and documents
where it does not.

PubMed seasons follow the same logic: a range `Mar-Apr` resolves to its first
month, a season to the first month of the northern-hemisphere season, and
`Winter` — which straddles a year boundary — to January.

### 6.4 Sources whose records cannot be back-dated

Two connectors emit records dated at **retrieval time**, never earlier, and every
such record carries a fixed `TEMPORAL_CAVEAT` warning plus
`provenance["temporal_caveat"] = True`.

**Open Targets.** Association scores are recomputed from scratch at every data
release, from *all* evidence known at that release, using that release's scoring
model and disease ontology. A score is a statement about today; there is no API
field that says what it would have been in 2014, and scores move substantially
between releases as new GWAS, ClinVar submissions and trial records land.

The specific danger is mechanical: `target_assessment` rows are selected by
"latest `available_at` at or before the cutoff", so a back-dated Open Targets
score would be **silently chosen into historical feature vectors** — and a 2024
score for a target-disease pair partly *reflects* the trials that succeeded or
failed on it. That is the outcome the platform is trying to predict. Dating at
retrieval means these rows simply do not exist for historical cutoffs, which is
the correct answer even though it costs coverage.

The connector notes for future work that the API exposes
`Target.associationTimeSeries`, a per-year association score — and rejects it for
now on the correct grounds: it is still a retrospective reconstruction (current
evidence re-binned by publication year, with the current scoring model and
ontology), so it is a better approximation but **not** a historical snapshot, and
adopting it would need its own validation and its own version bump.

**ChEMBL.** A molecule record has no "as of" date. It carries `first_approval`
and `usan_year`, and dating from those is tempting — but the same record carries
`max_phase`, `withdrawn_flag` and mechanism annotations, all revised release by
release. Dating at `first_approval` would assert that today's `max_phase` was
knowable on the day of approval, which is exactly backwards: **`max_phase` is the
strongest possible outcome leak for a model predicting whether a program
advances.** So `available_at` is the retrieval date, and `first_approval` is
surfaced separately as `first_approval_on` — a dated fact *inside* an undated
record, and the two must not be conflated.

### 6.5 The ClinicalTrials.gov `why_stopped` trap

This is the sharpest leak available in any source the platform reads, and it is
easy to write by accident.

**The mechanism.** A ClinicalTrials.gov study record is a *mutable document with
one URL*. The API returns its current state; it does not return the state as of
any earlier date. The record mixes two kinds of field:

- **Registration-time fields** — registry id, titles, phase, study type,
  allocation, masking, planned enrolment, conditions, interventions, primary
  outcomes, lead sponsor, planned start. Present when first posted, rarely
  revised in substance.
- **Current-state fields** — `overallStatus`, `whyStopped`, actual enrolment,
  actual completion dates, `hasResults`. Written later, sometimes years later.

**The trap.** A trial registered in 2013 and terminated for futility in 2018 has a
*single* API record carrying both. Attaching "terminated: interim analysis showed
no efficacy benefit" to `available_at = first_posted_on` would hand a 2014 feature
vector the answer to the question the platform exists to ask.

**The fix: the stop reason becomes knowable at `last_update_on`, not
`first_posted_on`.** The connector emits **two** records per study:

| Record kind | `available_at` | Carries |
|---|---|---|
| `trial` | `first_posted_on` — the day the registry record became public | Registration-time fields |
| `trial_status_update` | `last_update_on` — the day the registry *posted* the update, and the first day the stop reason was knowable | `why_stopped`, current `overall_status`, `has_posted_results` |

The `trial` payload still carries the current-state fields, because the `trial`
table has columns for them and the loader must populate the row. They are listed
explicitly in `provenance["current_state_fields"]` (the constant
`CURRENT_STATE_FIELDS`: `overall_status`, `why_stopped`, `enrollment`,
`enrollment_type`, `primary_completion_on`, `completion_on`, `last_update_on`,
`has_posted_results`) alongside `provenance["current_state_available_at"]`, and
the record carries a warning naming them.

**The loader must not let those columns reach a feature vector whose cutoff
precedes `last_update_on`. Nothing in the connector can enforce that; flagging it
is all a connector can do.** That sentence is in the module docstring and is
repeated here because it is a live obligation on anyone extending the loader.

Two further guards in the same connector:

- `studyFirstSubmitDate` is the sponsor's submission, which is **not public** and
  is typically days to weeks earlier than the post date. It is used only as a
  fallback for `available_at`, and only with a warning. A record with neither a
  post date nor a submit date is not emitted at all.
- When `last_update_on` precedes the study's first public date, the connector
  emits a warning that the registry dates are internally inconsistent and the
  record's `available_at` is suspect, rather than silently picking one.

---

## 7. The connector catalogue

Six connectors, all registered via `@register_connector` and reached through
`get_connector(name)` so a source can be swapped or retired without touching call
sites. Each is version-stamped (`connector_version` is written onto every
`raw_record`), and the version is bumped when `normalise` changes output for
unchanged input.

### 7.1 Live status as measured from this machine

Measured 2026-09-24 by running:

```bash
cd backend && \
  TRS_ALLOW_LIVE_NETWORK=true \
  ../.venv/bin/python -m trs.cli connectors
```

| Connector | Status reported | Latency | Detail (verbatim from the probe) |
|---|---|---|---|
| `pubmed` | `degraded` | 376 ms | esearch responded; no `TRS_NCBI_API_KEY` configured, throttled to 3 requests/second |
| `openalex` | `degraded` | 445 ms | works index reachable (327,970,096 works) but no `TRS_OPENALEX_MAILTO` configured, so requests use the shared anonymous pool and will be throttled before polite-pool traffic |
| `crossref` | `degraded` | 1,595 ms | works index reachable (186,912,544 works) but no `TRS_CROSSREF_MAILTO` configured, so requests use the anonymous pool with no availability guarantee |
| `opentargets` | **`ok`** | 345 ms | apiVersion 26.9.0, dataVersion 26.09 — scores are current-release only and are not back-datable |
| `chembl` | **`ok`** | 17,782 ms | direct resource fetch and name search both responding |
| `ctgov` | **`unreachable`** | 319 ms | HTTP 403 from `https://clinicaltrials.gov/api/v2/version` |

**Reading these honestly:**

- `degraded` for PubMed, OpenAlex and Crossref means **reachable and usable but
  un-identified**. All three sources ask callers to identify themselves; none is
  configured here. The connectors report the state rather than pretending it is
  fine, because under load an anonymous caller is the first to be throttled.
- **ChEMBL's 17.8-second probe latency is the number to notice**, not the `ok`.
  The probe exercises two paths separately, and on this network the search
  endpoints answer but slowly — re-probed directly with `curl`, the same four
  endpoints returned HTTP 200 in 1.7 s (`/molecule/{id}.json`), 8.6 s
  (`/molecule/search.json`), 7.0 s (`/mechanism.json`) and 5.8 s
  (`/target/search.json`). ChEMBL's search/filter endpoints have previously been
  observed returning HTTP 500 from EBI for extended periods; the connector is
  built for that, and `ChemblConnector.health()` reports `degraded` — not
  `unreachable` — when direct fetches by ChEMBL id work and search does not,
  because in that state every workflow that already holds a ChEMBL id is still
  fully served. **Treat ChEMBL search as intermittent.**
- **ClinicalTrials.gov is blocked at this network's edge, not broken.** Every
  `/api/v2` path — including `/version` — returns HTTP 403. The connector reports
  `unreachable` with the observed status and a note about egress, rather than
  pretending the source is merely slow. `normalise` is fully exercised offline
  against fixtures, so the connector will work unchanged the moment egress is
  fixed. **Until then there is no trial ingestion**, which for a platform whose
  outcome labels realistically come from registry status is the single most
  consequential gap in this deployment.

### 7.2 Terms of use, per source

Reproduced verbatim from each connector's `license_note`
(`trs connectors --list`).

| Connector | Terms |
|---|---|
| **PubMed** | PubMed/MEDLINE citation records are produced by the U.S. National Library of Medicine and are free of copyright restrictions; abstracts remain the copyright of their publishers. <https://www.nlm.nih.gov/databases/download/terms_and_conditions.html> |
| **OpenAlex** | OpenAlex data are released into the public domain under CC0 1.0. Abstracts are served as an inverted index and remain subject to the original publisher's rights. <https://docs.openalex.org/> |
| **Crossref** | Crossref metadata are made available under the Crossref REST API terms and may be used freely with attribution; per-work licence URLs in the payload govern the underlying content, not the metadata. <https://www.crossref.org/documentation/retrieve-metadata/rest-api/> |
| **Open Targets** | Open Targets Platform data are released under CC0 1.0; individual contributing datasources retain their own terms (ChEMBL CC BY-SA, Orphanet CC BY 4.0, and others). <https://platform-docs.opentargets.org/licence> |
| **ChEMBL** | ChEMBL data are released by EMBL-EBI under CC BY-SA 3.0; **derived works must carry the same licence and attribute ChEMBL.** <https://chembl.gitbook.io/chembl-interface-documentation/about> |
| **ClinicalTrials.gov** | ClinicalTrials.gov records are US Government works in the public domain; the NLM asks that reuse not imply endorsement. <https://clinicaltrials.gov/about-site/terms-conditions> |

Two of these have teeth and should be read before any commercial deployment:
**ChEMBL's CC BY-SA 3.0 is share-alike** — a derived dataset inherits the licence;
and **Open Targets aggregates sources with their own terms**, so CC0 on the
platform does not blanket everything inside it. Abstract text from PubMed and
OpenAlex remains publisher copyright in both cases, which is why the platform
stores structured claims rather than redistributing abstracts.

### 7.3 Rate-limit policy

Enforced per instance by `Connector._throttle()`, which sleeps to maintain
`min_request_interval` between requests. Retries are bounded
(`TRS_CONNECTOR_MAX_RETRIES`, default 3) with exponential backoff capped at 8
seconds, and HTTP 429/500/502/503/504 are treated as retryable.

| Connector | `min_request_interval` | Basis |
|---|---|---|
| `pubmed` | 0.34 s (0.11 s with an API key) | NCBI publishes 3 req/s without a key and 10 req/s with one, and **enforces it** — this connector was developed against a live 429. The interval is chosen per instance from settings, because the limit is a property of the runtime credentials, not of the class. |
| `openalex` | 0.11 s | OpenAlex asks for ≤10 req/s and 100k/day on the polite pool. |
| `crossref` | 0.20 s | Crossref publishes no hard anonymous limit but asks for restraint; polite-pool guidance is ~50 req/s, which the platform does not need. |
| `opentargets` | 0.35 s | No published limit; **self-imposed restraint on a free public API.** |
| `chembl` | 0.35 s | EBI asks for courteous use of the public REST service. |
| `ctgov` | 0.35 s | No published limit; the API documentation asks for reasonable use. |

Only PubMed and OpenAlex publish a numeric limit. The other four intervals are
self-imposed restraint on sources that merely ask for reasonable use. That is a
deliberate choice: an unmetered public API funded by research budgets is a
commons, and the cost of being slower than necessary is lower than the cost of
being the caller that gets an IP range blocked.

Identification is configured through `TRS_NCBI_EMAIL` / `TRS_NCBI_API_KEY` /
`TRS_NCBI_TOOL`, `TRS_OPENALEX_MAILTO` and `TRS_CROSSREF_MAILTO`
(see `.env.example`). None is set in this deployment, which is why three
connectors report `degraded`.

### 7.4 Source-specific provenance decisions worth knowing

These are places where the obvious implementation is wrong, and the connector
documents why it does something else.

**PubMed — dating.** A single article carries up to five dates that routinely
disagree by months. `available_at` is the **minimum** over every date denoting
public availability: `<History>` entries with `PubStatus` in
`entrez`/`pubmed`/`epublish`/`aheadofprint`/`ecollection`, the
`<ArticleDate DateType="Electronic">`, and the journal issue `<PubDate>`.
`received`, `accepted`, `revised` and `submitted` are **deliberately excluded** —
they are earlier, and using them would be the classic leak, because a manuscript
under review is not public and no analyst could have acted on it. The issue date
participates in the minimum rather than serving as a last-resort fallback,
because for back-filled older records the Entrez date is the date NLM digitised
the citation, which can be decades after the issue reached library shelves.

**Crossref — retraction detection.** Crossref models a retraction as a directed
edge between two DOIs: the *notice* carries `update-to`, the *retracted work*
carries `updated-by`. Reading `update-to` alone flags the wrong object — it marks
the notice (a sound editorial document) as retracted and leaves the discredited
paper clean. Checked against the live API on the canonical case
(DOI `10.1016/S0140-6736(97)11096-0`), the retracted article returns
`update-to: null` and an `updated-by` entry of type `retraction` dated
2010-02-06; a pipeline keyed on `update-to` would have missed it entirely. This
connector sets `is_retracted` from `updated-by` (and from
`relation.is-retracted-by`), and reports `update-to` separately as
`is_retraction_notice`.

**OpenAlex — field selection is part of the contract.** A full work is ~35 KB,
most of it `referenced_works` and topic scaffolding the platform does not read.
Every request pins an explicit `select` list. Two benefits: raw payloads stay
small enough to store verbatim forever, and **schema drift becomes visible** — a
field that disappears is a field we asked for and did not get, rather than one we
silently stopped noticing.

**Open Targets — genetic tier from datasources, never from the score.** The
aggregate `genetic_association` score cannot answer *what kind* of human genetic
evidence exists: a 0.8 from ClinVar pathogenic variants in a Mendelian disorder
and a 0.8 from a common-variant GWAS locus are different claims about the world
and imply different translational risk. The tier is derived from **which
datasources contributed** (strongest match wins: `eva`/`clingen`/`gene2phenotype`/
`genomics_england`/`orphanet`/`uniprot_*` → `mendelian_causal`; `gene_burden` →
`rare_coding_burden`; `gwas_credible_sets`/`ot_genetics_portal` →
`gwas_fine_mapped`; `gwas_catalog` → `gwas_locus_only`;
`cancer_gene_census`/`intogen`/`eva_somatic` → `somatic_driver`), and the score is
carried separately as `association_score`.

**ChEMBL — modality mapping refuses to guess.** `molecule_type` maps to the
platform `Modality` enum only where unambiguous. ChEMBL's `Antibody` covers
monoclonals, bispecifics and ADCs without distinguishing them, and
`Oligonucleotide` covers antisense and siRNA alike; the platform ontology has a
specific member for each but none for the generic parent. Rather than pick the
modal case and be wrong for the rest, those map to `Modality.OTHER` with a
warning and the verbatim `molecule_type` retained, so a curator or a
mechanism-aware pass can refine them.

---

## 8. What is real and what is synthetic in this deployment

This section is the one to read before quoting anything from a screenshot.

### 8.1 Real

| What | Detail |
|---|---|
| **Gene symbols and target names** | 42 real oncology targets — `KRAS`, `EGFR`, `ERBB2`, `BRAF`, `PIK3CA`, `AKT1`, `MTOR`, `CDK4/6/7/9`, `PARP1`, `ATR`, `WEE1`, `MYC`, `TP53`, `PDCD1`, `CD274`, … — stored with `is_synthetic = False` and described in the row itself as "Real gene symbol used as controlled vocabulary." 36 are HGNC symbols; 6 (`MEK1`, `TROP2`, `BCMA`, `HIF2A`, `SHP2`, `STING`) are common protein aliases rather than HGNC symbols — see §[11](#11-gaps-between-the-documented-design-and-the-running-code). |
| **Target classes** | Real classifications (kinase, GTPase, transcription factor, epigenetic regulator, …). |
| **Mouse ortholog identities** | Approximate published values, rounded. Used only to give the demonstration data plausible spread on that feature. |
| **Indication names and therapeutic areas** | 26 real disease names with EFO ontology ids, `ontology_source = "EFO"`, `is_synthetic = False`. |
| **The ontology** | All 38 controlled vocabularies in `trs/ontology/enums.py` — evidence classes, model types, genetic evidence tiers, discontinuation reasons, provenance methods, missingness reasons, and the rest. |
| **The connector implementations** | All six are real, working adapters against the real public APIs, with real rate-limit handling, real dating logic and real licence notes. Four reach their live source from this machine. |
| **The recorded API fixtures** | 16 of the 19 fixtures in `backend/tests/fixtures/connectors/` are genuine recorded responses from PubMed, OpenAlex, Crossref, ChEMBL and Open Targets. See §[9](#9-fixture-labelling-convention). |
| **The validation machinery** | Component-level bootstrap, temporal splitting, oracle ceiling, domain ablation, sign audit, interval-agreement check — all real code doing real statistics on simulated inputs. |

### 8.2 Synthetic

| What | Count in this database | Marking |
|---|---|---|
| Sponsors | 25 | `is_synthetic = True`; named after mountain ranges and rivers so they cannot be confused with real companies |
| Programs | 600 | `is_synthetic = True`; every program code prefixed `DEMO-` |
| Compounds | 600 | `is_synthetic = True` |
| Development tracks (`program_indication`) | 765 | `is_synthetic = True` |
| Evidence items | 5,273 | `is_synthetic = True`, `provenance_method = synthetic_generator` |
| Sources | 90 | `is_synthetic = True`, `source_type = synthetic_demonstration`, titled `[SYNTHETIC RECORD nnn]`, **`pmid` / `doi` / `url` all NULL** |
| PK/PD bridges | 474 | `is_synthetic = True` |
| Program biomarkers | 506 | `is_synthetic = True` |
| Safety findings | 1,091 | `is_synthetic = True` |
| Target assessments | 123 | `is_synthetic = True` |
| **Every effect size, dose, exposure, n-per-arm, blinding status and endpoint value** | — | drawn from the documented simulation |
| **Every development outcome** | — | a Bernoulli draw from a simulated probability |

**No synthetic source carries a PMID, a DOI or a URL.** The generator's docstring
gives the reason and it is the right one: fabricating a plausible-looking PMID
would be indistinguishable from a real citation once it reached the interface,
which is precisely the failure mode this platform must not have. Each synthetic
source's abstract field says so in words: "This record does not correspond to any
real publication."

### 8.3 Why real vocabulary was used with simulated outcomes

The decision is deliberate, and `trs/seed/vocabulary.py` states the reasoning:

> Using real vocabulary is deliberate: an ontology populated with invented gene
> names could not be swapped for a live Open Targets or ChEMBL feed, and the
> point of the demonstration data is that replacing it is a data problem, not a
> rewrite.

Concretely: `KRAS` in this database resolves against the real Open Targets and
ChEMBL APIs today. A demonstration built on `GENE-0001` would prove nothing about
whether the ingestion path, the alias matching or the genetic-tier mapping
actually work, and swapping it for real data later would be a rewrite rather than
a load.

### 8.4 The risk this creates, stated plainly

The same file states the cost without hedging:

> **The consequence, stated plainly**: this dataset pairs real target and disease
> names with simulated outcomes. A row saying a `DEMO-` program against a real
> target failed in Phase II is a statement about a random number generator, not
> about that target.

A screenshot of that row carries no context. The risk is real and is not fully
removable while real vocabulary is in use.

### 8.5 The four mitigations, and what each actually does

**1. `is_synthetic` propagates, structurally.** `SyntheticFlagMixin` puts
`is_synthetic` (indexed) and `synthetic_note` on every table that can hold
demonstration content. The flag then propagates by explicit OR at each stage:

| Stage | Rule | Code |
|---|---|---|
| Feature snapshot | synthetic if **any** contributing evidence item, the program, the track, the PK/PD bridge, the target assessment, any biomarker or any safety finding is synthetic | `trs/features/builder.py` |
| Cohort entry | `track.is_synthetic or program.is_synthetic` | `trs/cohort/build.py` |
| Cohort version | `contains_synthetic = any(entry.is_synthetic …)` | `trs/cohort/build.py` |
| Model artifact | `trained_on_synthetic = any(snapshot.is_synthetic …) or cohort.contains_synthetic` | `trs/pipeline/train.py` |
| Score record | `is_demonstration = snapshot.is_synthetic or bundle.trained_on_synthetic` | `trs/scoring/service.py` |
| API health | `is_demonstration_deployment = artifact.trained_on_synthetic` | `trs/api/main.py` |

It is a one-way ratchet: any synthetic input anywhere in a score's lineage marks
the score. On the active artifact, `trained_on_synthetic = true`.

**2. Confidence is forced to `very_low`.** `grade_evidence_confidence()`
(`trs/features/missingness.py`) computes a grade from weighted completeness and
then **only ever downgrades** — nothing upgrades it, because a narrow interval on
thin evidence is a property of the model, not of the evidence. When
`contains_synthetic` is true the grade is set to `ConfidenceLevel.VERY_LOW`
outright, regardless of how complete the evidence looks, and this reason is
**inserted at the top** of the reason list:

> "Estimate depends on synthetic demonstration data and carries no scientific
> interpretation."

**3. A non-dismissible UI banner.** `DemoBanner`
(`frontend/components/primitives/index.tsx`) is rendered in the root layout
(`frontend/app/layout.tsx`) whenever the API's health endpoint reports
`is_demonstration_deployment`. It appears above the header on **every** page. It
has `role="note"`, no close control, no dismissal state and no `localStorage`
key — there is no code path that hides it. Its text:

> **Demonstration data.** Programs, evidence and outcomes on this deployment are
> synthetic. Estimates describe a documented simulation and carry no scientific
> interpretation.

**4. The evidence record says so in its own words.** Every synthetic row carries
`synthetic_note`:

> DEMO / SYNTHETIC DATA — NOT FOR SCIENTIFIC INTERPRETATION. Generated by
> `trs.seed.generator` from a documented simulation; target and indication names
> are real vocabulary, all program-level facts and outcomes are simulated.

So the disclosure survives a database export, an API response or a CSV dump —
not only the rendered page.

**What none of this prevents.** A screenshot. A number copied into a slide. A
model artifact lifted out of `artifacts/` and scored elsewhere. The mitigations
make the synthetic status *impossible to miss in the product* and *impossible to
strip from the data*; they cannot follow a number out of the building.

### 8.6 Provenance tables that are empty here — and why

| Table | Rows | Why |
|---|---|---|
| `raw_record` | 0 | The generator writes evidence directly; no connector ran against this database. There are no upstream bytes to land. |
| `evidence_field_provenance` | 0 | Synthetic items have a single uniform origin, so no field's provenance differs from its item's. |
| `trial` | 0 | Registry ingestion requires ClinicalTrials.gov, which is unreachable from this network (§7.1). |
| `connector_run` | 0 | No ingestion run has been executed against this database. |
| `score_record` | 0 | No score has been persisted yet; the pipeline stops at a registered model artifact. |
| `entity_resolution_decision` | 0 | Not written by any code path (§11). |

These zeros are the correct shape for a simulated corpus, not a defect — but they
mean **the raw-landing-zone half of the chain in §1 is exercised by the connector
test suite and by live probes, not by this database.** Anyone evaluating the
provenance story should read `trs/connectors/base.py` and run the connector
tests, rather than querying `raw_record` and finding it empty.

---

## 9. Fixture labelling convention

Every fixture in `backend/tests/fixtures/connectors/` carries a `_fixture_note`
key stating its origin. There are exactly two forms, and the distinction is
enforced by convention in the file itself rather than by a filename or a
directory.

**Real recordings** — 16 files:

```
"RECORDED LIVE FROM <url> ON <date>"
```

for example:

```
RECORDED LIVE FROM https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL4535757.json ON 2026-09-24
RECORDED LIVE FROM https://api.crossref.org/works/10.1016/S0140-6736(97)11096-0 ON 2026-09-24
RECORDED LIVE FROM https://api.platform.opentargets.org/api/v4/graphql ON 2026-09-24 (HTTP POST)
RECORDED LIVE FROM https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?...&id=33138715,9500320,1&retmode=xml ON 2026-09-24 (<ReferenceList> elements removed to keep the fixture small)
```

The note includes the full request URL, so the recording can be re-made and
diffed. Where the payload was trimmed, the note says what was removed and why —
the PubMed fixtures record that `<ReferenceList>` elements were stripped for size.

**Hand-constructed** — 3 files, all ClinicalTrials.gov, because the source is
unreachable from this network and no live recording could be made:

```
HAND-CONSTRUCTED FIXTURE — shape follows the documented ClinicalTrials.gov API v2
schema; field VALUES are illustrative and are NOT real trial data.
```

`ctgov_studies_search.json`, `ctgov_study_terminated.json` and
`ctgov_version.json` carry this label. The wording distinguishes the two things a
reader needs to separate: the **shape** is faithful to the documented v2 schema,
so `normalise` is genuinely exercised; the **values** are invented, so no NCT id,
sponsor or stop reason in those files refers to a real trial.

Every fixture is loadable and every note is machine-readable, so "which of our
tests run against real payloads?" is a query, not an archaeology exercise.

---

## 10. Reproducing and inspecting provenance

All commands use the absolute interpreter path; there is no activated
virtualenv. Every command below was run while writing this document.

```bash
# Connector catalogue — names, versions, homepages, licence notes
cd backend && \
  ../.venv/bin/python -m trs.cli connectors --list

# Live health probe of every connector (§7.1)
cd backend && \
  TRS_ALLOW_LIVE_NETWORK=true \
  ../.venv/bin/python -m trs.cli connectors

# Connector test suite — normalise() against stored fixtures, no network
cd backend && \
  ../.venv/bin/python -m pytest tests -q

# Regenerate the synthetic corpus (destructive: changes every hash)
cd backend && \
  ../.venv/bin/python -m trs.cli seed \
  --programs 600 --seed 20240917
```

At runtime the chain is walkable over HTTP through the endpoints listed in
§[1](#1-the-provenance-chain) — `/api/v1/provenance/scores/{id}/lineage` is the
one that returns score → snapshot → evidence → source in a single response.

---

## 11. Gaps between the documented design and the running code

Recorded here because a provenance document that overstates its own guarantees is
worse than none. **Nothing in this section has been changed** — this document does
not modify code. Items 1, 3, 4 and 6 also appear, from the modelling side, in
[`MODEL_CARD.md`](MODEL_CARD.md) §10.2.

1. **`MODEL_GENERATED_UNVERIFIED` is not enforced.** Documented as rejected by
   the validation layer; no code performs the rejection. §[5.2](#52-enforcement-status--stated-honestly).
2. **`entity_resolution_decision` is never written.** Its docstring says it is
   written by `trs.resolution`; `backend/trs/resolution/` is an **empty
   directory**, and the model class is referenced only by the package `__init__`
   export list. The audit trail described in §3.2 does not currently exist at
   runtime. Alias-level provenance (`match_method`, `match_confidence`,
   `available_at`) *is* written and is real.
3. **The temporal-leakage test suite does not exist.** `trs/features/temporal.py`
   line 14 states that `tests/statistical/test_temporal_leakage.py` "exercises the
   invariant directly: it builds a snapshot at `T`, inserts post-`T` evidence
   designed to move every feature, rebuilds, and asserts bit-identical output."
   The only test file in the repository is `backend/tests/unit/test_connectors.py`.
   The generator does deliberately date 15% of in vivo evidence after the IND date
   so that such a test would have something to catch — but **no test currently
   catches it.** This is the most consequential gap on the list, because the
   `AsOfContext` invariant is what every reported performance number rests on.
4. **`cohort_entry.split` / `split_fold` are never written.** Documented as
   "Assigned by `trs.validation.splitters`"; all 552 rows have `split = NULL`.
   The split is computed in memory during training and discarded, so a historical
   partition cannot be recovered from the database.
5. **`backend/trs/schemas/` is an empty directory.** The API's Pydantic models
   live in `trs/api/schemas.py`; the top-level package appears to be a vestige.
6. **15 orphan feature snapshots.** `feature_snapshot` holds 586 rows for 586
   distinct tracks; the cohort has 552 entries, all of which have a snapshot.
   The extra 15 belong to tracks outside the cohort — benign residue from an
   earlier run, but it means a snapshot count is not a cohort count.
7. **Six target rows carry a protein alias where an HGNC symbol is asserted.**
   `Target.symbol` is documented as "canonicalised on HGNC symbol where one
   exists"; the generator sets `target.external_refs = {"vocabulary": "HGNC
   symbol"}` and writes each symbol as a `target_alias` row with
   `alias_type = "symbol"` and `match_method = exact_identifier`.
   Six of the 42 are not HGNC symbols: `MEK1` (HGNC `MAP2K1`), `TROP2`
   (`TACSTD2`), `BCMA` (`TNFRSF17`), `HIF2A` (`EPAS1`), `SHP2` (`PTPN11`) and
   `STING` (`STING1`). They are real, widely used target names, so the
   vocabulary claim in §8.1 holds — but a live Open Targets or ChEMBL lookup
   keyed on `Target.symbol` will miss those six until an alias mapping is added.
   This is exactly the class of defect the alias tables exist to absorb.

---

*Compiled against the live repository, database and network on 2026-09-24. Every
connector status, row count, licence note and fixture label above was read from
the code, queried from `backend/trs.db`, or obtained by running the commands in
§10. Nothing was transcribed from memory or from any prior document.*
