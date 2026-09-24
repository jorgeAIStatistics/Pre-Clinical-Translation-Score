# Architecture

*Preclinical Translational Risk Scorer — how the system is built, and why each choice was made.*

Companion to [DATA_DICTIONARY.md](DATA_DICTIONARY.md), which documents every table, controlled
vocabulary and feature. This document explains the shape; that one documents the contents.

> **Demonstration data.** The database shipped with this repository
> (`backend/trs.db`) was produced by `trs/seed/generator.py`. Gene symbols, indication names and
> pathway names are real public vocabulary; **every program-level fact and every outcome in it is
> simulated.** Numbers in this document that derive from it are labelled as such. The platform has
> no regulatory standing and no clinical validation.

---

## Contents

- [1. What the system does, and what it is built around](#1-what-the-system-does-and-what-it-is-built-around)
- [2. Layered view](#2-layered-view)
- [3. The five data states](#3-the-five-data-states)
- [4. Persistence: one model, two engines](#4-persistence-one-model-two-engines)
- [5. Temporal integrity: the chokepoint](#5-temporal-integrity-the-chokepoint)
- [6. Ingestion: the connector adapter pattern](#6-ingestion-the-connector-adapter-pattern)
- [7. Entity resolution and why aliases are a first-class table](#7-entity-resolution-and-why-aliases-are-a-first-class-table)
- [8. Feature, cohort and modelling layers](#8-feature-cohort-and-modelling-layers)
- [9. The API layer](#9-the-api-layer)
- [10. Frontend architecture](#10-frontend-architecture)
- [11. Configuration and process topology](#11-configuration-and-process-topology)
- [12. Scaling: what breaks first and what would change](#12-scaling-what-breaks-first-and-what-would-change)
- [13. Decision record](#13-decision-record)
- [14. Known weaknesses](#14-known-weaknesses)

---

## 1. What the system does, and what it is built around

The platform estimates and explains the risk that a preclinical therapeutic program fails to
translate into human clinical efficacy. The primary endpoint is cause-specific:

> Probability that a program entering human development fails to demonstrate sufficient efficacy to
> progress successfully beyond Phase II, **conditional on not being withdrawn first for safety,
> pharmacokinetic, manufacturing or commercial reasons.**

That conditional is not a caveat bolted on at the end. It is the reason the schema separates
`outcome_label` from `discontinuation_reason`, the reason `safety_finding` is a different table from
`evidence_item`, and the reason the cohort builder reports an attrition waterfall rather than a final
N. A system that collapsed every non-success into "failure" would predict corporate behaviour while
claiming to predict biology.

Three properties drove nearly every structural decision:

1. **Temporal integrity.** A feature vector for a decision at time `T` must be a function only of
   facts knowable at `T`. If that fails, every validation number downstream is optimistic by an
   unknown amount. This is enforced at one chokepoint ([§5](#5-temporal-integrity-the-chokepoint)).
2. **Reconstructability.** A score shown to a committee in March must be fully reconstructable in
   November — including after the model has been retrained. Every link in the chain is persisted
   ([§3](#3-the-five-data-states)).
3. **Explanation from the model, not about it.** Every driver shown to a user is a term in the
   deployed model's own linear predictor, and the terms sum to the number on screen. This is a
   constraint on the model family, not a presentation choice ([§8](#8-feature-cohort-and-modelling-layers)).

The backend is 72 Python modules under `backend/trs/`, defining 40 tables, 38 controlled
vocabularies and 63 declared features. The frontend is a Next.js 16 / React 19 application of about
4,100 lines of TypeScript across five routes.

---

## 2. Layered view

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL SOURCES                                                                     │
│  PubMed E-utilities · OpenAlex · Crossref · Open Targets GraphQL · ChEMBL · CTgov v2  │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │  HTTPS, throttled, bounded retries
┌───────────────────────────────▼───────────────────────────────────────────────────────┐
│  CONNECTOR ADAPTERS                           trs/connectors/                         │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │ Connector (ABC)   base_url() · health() -> ConnectorHealth · normalise()      │     │
│  │ request()  →  ResponseCache (content-addressed, on disk, TTL 168 h)           │     │
│  │           →  throttle (min_request_interval) → retry(3, exp backoff)          │     │
│  │           →  stale-cache fallback on failure, flagged from_cache              │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
│   pubmed   openalex   crossref   opentargets   chembl   ctgov      + ConnectorRun log │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │  fetch() — the ONLY method permitted to touch the network
┌───────────────────────────────▼───────────────────────────────────────────────────────┐
│  ① RAW — the landing zone                     table: raw_record  (append-only)        │
│     verbatim bytes · SHA-256 content hash · endpoint · params · http_status           │
│     never rewritten, never deleted → any value is re-derivable, any bug diagnosable   │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │  normalise() — PURE, no I/O, fixture-testable
┌───────────────────────────────▼───────────────────────────────────────────────────────┐
│  ② NORMALISED — NormalisedRecord(kind, external_id, available_at, payload, provenance)│
│     available_at is MANDATORY: a record that cannot be placed in time is inadmissible │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │  entity resolution
┌───────────────────────────────▼───────────────────────────────────────────────────────┐
│  ③ RESOLVED — canonical entities + alias graphs + decision audit                      │
│     target · compound · sponsor · indication · program · biomarker · pathway          │
│     *_alias (match_method, match_confidence, available_at)                            │
│     entity_resolution_decision  ← one row per link, accepted or rejected              │
│     ──► source · trial · evidence_item · target_assessment · pkpd_bridge ·            │
│         program_biomarker · safety_finding · evidence_field_provenance                │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │
                   ┌────────────▼────────────┐
                   │  AsOfContext(session,T) │ ◄── THE temporal chokepoint
                   │  every accessor applies │     trs/features/temporal.py
                   │  available_at <= T and  │     builder takes a context, NOT a session
                   │  re-verifies the rows   │
                   └────────────┬────────────┘
┌───────────────────────────────▼───────────────────────────────────────────────────────┐
│  ④ DERIVED — feature_snapshot  (program_indication_id, as_of_date, schema_version)    │
│     build_features()  → values · missingness · feature_evidence_map · completeness    │
│     effects.standardise_effect() · historical.shrunk_rate() · missingness grading     │
│     cohort_version + cohort_entry ← outcomes.py endpoint definitions, exclusions kept │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │  FeatureTransformer: fit on TRAIN fold only
┌───────────────────────────────▼───────────────────────────────────────────────────────┐
│  ⑤ MODEL-READY → TRAINED                                                              │
│     split (temporal × connected component) → tune (expanding window, inside train)    │
│     → fit → calibrate (cross-validated out-of-fold) → validate → select → artifact    │
│     l2_logistic (default) · hierarchical_logistic · gbdt   ← all three always run     │
│     model_artifact: coefficients, calibrator, transformer, metrics, selection record  │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────────────────────┐
│  SERVING                                       trs/scoring/ · trs/api/                │
│  ScoringService.score_track()                                                         │
│     → rebuild features through AsOfContext (never read a snapshot at another cutoff)  │
│     → predict_with_uncertainty()  parameter | imputation | calibration                │
│     → explain()   asserts Σ contributions == logit(p)                                 │
│     → grade_evidence_confidence()  · propagate is_demonstration                       │
│  FastAPI  /api/v1  ·  5 routers, 29 endpoints  ·  JWT | API key | dev fallback        │
│  score_record + audit_event written on every scoring call                             │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │  HTTP, server-side fetch
┌───────────────────────────────▼───────────────────────────────────────────────────────┐
│  FRONTEND — Next.js 16 App Router, React 19 server components                         │
│  app/ portfolio · programs/[id] · evidence · methods · provenance                     │
│  components/charts — hand-built SVG (d3-scale / d3-shape for arithmetic only)         │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The five data states

Each arrow above is a state transition with a persisted artefact on both sides. That is what makes
the chain in `trs/pipeline/artifacts.py` walkable in reverse:

```
ScoreRecord.model_version → ModelArtifact (coefficients + transformer + calibrator)
ScoreRecord.feature_snapshot_id → FeatureSnapshot.features (the exact vector)
                                → FeatureSnapshot.feature_evidence_map
                                → EvidenceItem ids
                                → Source → RawRecord.content_hash → the verbatim payload
```

The costs are honest and worth naming. Persisting raw payloads means storage grows with ingestion
volume and never shrinks. Persisting feature snapshots means a schema change requires a rebuild pass
rather than a recompute-on-read. Both were accepted because the alternative — recomputing features on
demand — silently changes historical scores whenever evidence is updated, and a decision-support tool
whose past outputs mutate is not auditable.

---

## 4. Persistence: one model, two engines

**The decision.** PostgreSQL 15+ is the production target. SQLite is the zero-dependency local
default, so `git clone && pip install && python -m trs.cli pipeline` produces a working system with
no external service. **One set of SQLAlchemy models and one Alembic history serve both.**

The alternative — Postgres-only, with `docker compose` as a prerequisite — was rejected because the
first thing a sceptical reviewer does is try to run the pipeline, and a reviewer who cannot run it
evaluates the README instead of the system. The alternative of maintaining two schemas was rejected
because divergence is inevitable and the divergence always shows up in production.

Making one schema work on both engines requires four specific devices, all in
[`trs/db/base.py`](backend/trs/db/base.py).

### 4.1 ULID string primary keys instead of native `UUID`

```python
def new_id(prefix: str = "") -> str:
    ts = int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ts << 80) | rand
    ...  # Crockford base-32, 26 characters
```

PostgreSQL has a native `UUID` type; SQLite does not, and SQLAlchemy's portable `Uuid` type stores a
different physical representation on each. Rather than depend on that, primary keys are plain
`VARCHAR(32)` holding a 26-character ULID: 48 bits of millisecond timestamp followed by 80 bits of
randomness.

The second reason is the better one. ULIDs sort lexicographically by creation time. `ORDER BY id` is
therefore a valid proxy for creation order without an extra index, and B-tree inserts stay
append-only rather than scattering across the index the way random UUIDv4 keys do.

**Cost:** 26 bytes of text per key instead of 16 bytes of binary, on every row and every foreign
key. At the scale this system targets — hundreds of thousands of evidence rows, not billions — that
is not the constraint. At a much larger scale it would be, and the migration path is a Postgres-only
`UUID` column with a backfill.

### 4.2 `JSONVariant`

```python
class JSONVariant(TypeDecorator):
    impl = JSON
    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.JSONB())
        return dialect.type_descriptor(JSON())
```

`JSONB` on PostgreSQL, portable TEXT-backed `JSON` on SQLite. Structured payloads — validation
reports, coefficient sets, missingness maps, feature-to-evidence maps — are written once per column
declaration and get the right physical type on each engine.

**Cost:** SQLite cannot index inside the JSON, so any query that filters on a JSON field is a scan
there and an indexable `JSONB` GIN query on Postgres. Nothing on the hot path does this today; the
columns that *are* filtered on are ordinary typed columns.

### 4.3 `UTCDateTime`, and why naive datetimes are rejected outright

```python
def process_bind_param(self, value, dialect):
    if value.tzinfo is None:
        raise ValueError(
            "Naive datetime written to a UTCDateTime column. All timestamps "
            "must be timezone-aware; use trs.db.base.utcnow()."
        )
    return value.astimezone(dt.UTC)
```

This is the most important of the four, and the rejection is deliberate rather than defensive.

SQLite stores datetimes as naive strings. Without this decorator, a value written as aware UTC comes
back naive, and a later comparison against an aware value raises `TypeError` — or, worse, a
comparison against another naive value silently succeeds against a local-time reading. Both failure
modes land in the same place: the `available_at <= as_of` comparison in the temporal layer, which is
the single most important correctness property in the system.

So the type decorator refuses the write. It would have been easy to coerce a naive value to UTC
instead; that would mean a caller who passed a naive local timestamp would get a silently shifted
value and no error. A hard failure at bind time makes the bug appear at the write, not in a
validation metric three layers away. The complementary half, `process_result_value`, re-attaches UTC
to anything SQLite hands back naive.

`utcnow()` is the single clock the ORM layer uses. The models never rely on database-side
`CURRENT_TIMESTAMP`, which would have different semantics and a different timezone posture on each
engine.

### 4.4 No engine-specific DDL

No partial indexes, no array columns, no database `ENUM` types. Controlled vocabularies are stored
as `VARCHAR` and validated in Pydantic and application code. Constraint names come from a
deterministic `MetaData` naming convention, so Alembic autogenerate is stable and a migration can
drop a constraint by name on either engine.

**Cost, stated plainly:** the database will accept `"in_vivo"` where `"animal_in_vivo"` was meant.
There is no `CHECK` constraint standing behind the 38 vocabularies. The benefit bought is that
adding a vocabulary term requires no migration on either engine — which matters because the ontology
is expected to grow — and that SQLite, which cannot `ALTER TYPE`, does not need a table rebuild every
time it does. A Postgres-only deployment could add `CHECK` constraints in a follow-on migration
without touching the models.

### 4.5 Migrations

One Alembic revision (`05c2a99460a1`, 1,353 lines) creates the whole schema. Two details make it
portable:

- `env.py` sets `render_as_batch=settings.is_sqlite`, so SQLite gets the copy-and-rename table
  rebuild it needs for any `ALTER` that it does not support natively, and Postgres gets ordinary
  `ALTER` statements.
- The database URL always comes from `trs.config.settings`, never from `alembic.ini`, so migrations
  and the application cannot disagree about which database they are pointing at.

SQLite is additionally brought closer to Postgres semantics at connect time in
`trs/db/session.py`: `PRAGMA foreign_keys=ON` (they are off by default, which would let the
development database accumulate orphans that production would reject), `journal_mode=WAL`,
`synchronous=NORMAL`, `busy_timeout=5000`.

---

## 5. Temporal integrity: the chokepoint

### The property being defended

For a development track whose decision point is date `T`, the feature vector must be a function only
of facts knowable at `T`. A single leak makes every validation metric optimistic by an unquantifiable
amount, and the product is then worthless in a way that is invisible from its outputs.

### Why `AsOfContext` is a class and not a convention

The obvious implementation is a convention: "remember to add `WHERE available_at <= T`". That
convention has to hold at every call site, forever, including in code written by someone who has not
read this document. It fails silently and it fails in the direction that flatters the results, which
is the worst combination available.

`trs/features/temporal.py` instead makes the correct query the *only reachable* one:

```python
@dataclasses.dataclass(frozen=True, slots=True)
class AsOfContext:
    session: Session
    as_of: dt.date
    strict: bool = True
```

- **It owns the session.** `trs/features/builder.py::build_features()` takes an `AsOfContext`, not a
  `Session`. A call site inside the feature layer has no handle with which to write an unguarded
  query. This is the structural half of the guarantee.
- **Every accessor applies the predicate.** `evidence_for_track`, `latest_pkpd_bridge`,
  `latest_target_assessment`, `biomarkers_for_track` and `safety_findings` each filter on
  `available_at <= self.as_of` before returning.
- **It re-verifies what it returns.** In `strict` mode (the default), `_verify()` walks the returned
  rows and raises `TemporalIntegrityError` if any `available_at` exceeds the cutoff. This is
  belt-and-braces: it catches a mis-written filter rather than trusting the SQL to be right.
- **"Latest at or before", never "latest".** `latest_pkpd_bridge` orders by
  `available_at DESC` *within* the filtered set. A program's exposure argument improves over time;
  scoring a 2014 decision with a 2019 bridge is precisely the leak the system exists to prevent, and
  it is the kind that a naive "get the current bridge" helper introduces without anyone noticing.
- **It is frozen and slotted.** `as_of` cannot be mutated after construction, so a context cannot be
  quietly widened mid-computation.

### `available_at` is not `published_on`

The module docstring is explicit, and the distinction is load-bearing: a 2015 paper reporting a 2011
experiment became *knowable* in 2015. A trial registered in 2013 whose "why stopped" field was
populated in 2018 yields two facts with two different availability dates. `available_at` is always
"when a diligent analyst could have found this", never the date of the underlying event — which is
carried separately as `observed_on`. See
[DATA_DICTIONARY.md §2](DATA_DICTIONARY.md#2-available_at-the-temporal-integrity-contract).

### What the chokepoint does not cover

`AsOfContext` has accessors for `evidence_item`, `pkpd_bridge`, `target_assessment`,
`program_biomarker` and `safety_finding`. It has none for the alias tables, `trial` or `source`, and
`available_at` is nullable on all five alias tables. No declared feature reads an alias today, so
nothing is presently wrong — but the structural guarantee stops at those five tables, and anything
added later that reads outside them is back on the convention. That is recorded in
[§14](#14-known-weaknesses) rather than left to be discovered.

---

## 6. Ingestion: the connector adapter pattern

### The contract

Every source implements the same three-method interface in `trs/connectors/base.py`:

```python
class Connector(abc.ABC):
    name: ClassVar[str]              # stable registry key, stored on every RawRecord
    version: ClassVar[str]           # bump when normalise() changes output for unchanged input
    license_note: ClassVar[str]      # shown alongside every ingested record

    def base_url(self) -> str: ...
    def health(self) -> ConnectorHealth: ...
    def normalise(self, payload: RawPayload) -> list[NormalisedRecord]:
        """Pure transformation. Must not perform I/O."""
```

The pipeline shape is identical for all six:
`fetch → RawRecord → normalise → domain objects → entity resolution → evidence`.

### Why adapters rather than direct calls

Every public biomedical source is unreliable in a different way: rate limits, silent schema drift,
IP-based blocking, partial outages of specific endpoints. Isolating each behind one interface means
an outage degrades one connector instead of the platform. It also means a commercial
trial-intelligence feed — different auth, different shape, different licence — drops in as a sibling
class without touching feature engineering.

The separation of `fetch` from `normalise` buys something more concrete: **`normalise` is pure, so
the connector test suite runs against stored fixtures with no network at all.** That is what makes
CI possible for a system whose inputs are six third-party APIs.

### The raw landing zone

`fetch` is the only method permitted to touch the network, and it always writes a `raw_record` — the
verbatim bytes, keyed by SHA-256 of the body — before anything interprets them. `persist_raw()`
deduplicates on that hash, so re-fetching an unchanged payload does not grow the table.

This costs storage and buys three things: every downstream value is re-derivable, every extraction
bug is diagnosable against exactly the payload that caused it, and a pipeline re-run against the
cache produces byte-identical raw payloads — a precondition for reproducing a historical prediction.

### Caching, throttling, retries and degradation

`Connector.request()` composes them in one place:

| Behaviour | Setting | Default |
| --- | --- | --- |
| Content-addressed on-disk cache, keyed on `(method, url, params, body)` | `TRS_CONNECTOR_CACHE_TTL_HOURS` | 168 h |
| Minimum interval between requests | `Connector.min_request_interval`, overridden per connector | base 0.35 s; OpenAlex 0.11 s; Crossref 0.20 s; PubMed 0.34 s without an NCBI key and 0.11 s with one, decided per instance because the published ceiling depends on the key |
| Retries on network error and on 429/500/502/503/504, exponential backoff capped at 8 s | `TRS_CONNECTOR_MAX_RETRIES`, overridable per instance | 3 |
| Request timeout | `TRS_CONNECTOR_TIMEOUT_SECONDS` | 30 s |
| Offline mode | `TRS_ALLOW_LIVE_NETWORK` | `true` |

The retry budget is a constructor argument (`max_retries`) rather than a global, because a health
probe and an ingestion run want different behaviour: retrying a dead endpoint three times with
exponential backoff turns a 6-second timeout into a 20-second wait, and six connectors into a
40-second page load. `/api/v1/provenance/connectors` therefore constructs its probes with
`max_retries=1`, while ingestion uses the configured budget.

Cache entries are stored as a metadata JSON file beside the raw body, sharded by the first two
characters of the key, so a cached payload can be inspected with ordinary tools rather than through
the application.

**Reachability is a first-class state, not an exception.** `health()` returns a `ConnectorHealth`
with one of `ok`, `degraded`, `unreachable`, `not_configured`, `disabled`. When a source cannot be
reached and a cached payload exists, the connector serves it and marks every derived record
`from_cache`. It never fabricates, and — the important half — it never silently returns an empty
result that a caller could mistake for "no evidence found". That distinction is also why failed runs
are written to `connector_run` rather than swallowed.

### Live status, verified from this machine

`python -m trs.cli connectors`, run while writing this document (2026-09-24T12:27Z):

| Connector | Status | Detail |
| --- | --- | --- |
| Open Targets GraphQL | `ok` | apiVersion 26.9.0, dataVersion 26.09 |
| ChEMBL (EMBL-EBI) | `ok` | direct resource fetch and name search both responding |
| PubMed (NCBI E-utilities) | `degraded` | reachable; no `TRS_NCBI_API_KEY`, so throttled to 3 req/s |
| OpenAlex | `degraded` | reachable (327,970,096 works); no `TRS_OPENALEX_MAILTO`, so anonymous pool |
| Crossref | `degraded` | reachable (186,912,531 works); no `TRS_CROSSREF_MAILTO`, so anonymous pool |
| ClinicalTrials.gov v2 | `unreachable` | **HTTP 403 on every `/api/v2` path including `/version`.** The block is at the network edge, not in the API. |

`degraded` here means "working but without the politeness credentials that guarantee availability" —
the health checks are specific about which credential is missing rather than reporting a generic
warning. ClinicalTrials.gov is genuinely unreachable from this egress, so **trial ingestion is
unavailable**; `normalise()` for that connector remains covered by offline fixtures.

The fixtures record which is which, per file. `backend/tests/fixtures/connectors/*.json` each carry a
`_fixture_note`: real recordings say `RECORDED LIVE FROM <url> ON <date>`, and the three `ctgov_*`
files say `HAND-CONSTRUCTED FIXTURE — shape follows the documented ClinicalTrials.gov API v2 schema;
field VALUES are illustrative and are NOT real trial data.` A fixture that cannot be recorded is
labelled as constructed rather than presented as a recording.

---

## 7. Entity resolution and why aliases are a first-class table

### The problem

A therapeutic program is referred to by a discovery code, then a development code, then an INN, then
a brand name — and it changes sponsor once or twice along the way. Publications cite the discovery
code; the trial registry cites the development code; the approval cites the INN.

Treating those as four programs does three separate kinds of damage:

1. It inflates the cohort with duplicates.
2. It puts the *same biology* in both the training and the test fold, which makes every held-out
   metric optimistic.
3. It destroys the historical-comparator feature, because the prior programs on a target are exactly
   the ones most likely to have been renamed.

Damage (2) is the one that matters most: it is invisible in the outputs and it flatters the results.

### The design

Every nameable entity — `target`, `compound`, `sponsor`, `indication`, `program` — has an explicit
alias table sharing `AliasMixin`:

| Column | Purpose |
| --- | --- |
| `alias` | The surface form as encountered |
| `alias_normalised` | Case/punctuation-folded; **indexed**, and the blocking key for candidate generation |
| `alias_type` | e.g. synonym, development code, INN |
| `match_method` | `EntityMatchMethod` — from `exact_identifier` down to `fuzzy_string` and `unresolved` |
| `match_confidence` | Numeric, so a threshold is explicit rather than implied by the method |
| `source_id` | Where the alias was seen |
| `available_at` | When the alias became publicly knowable |
| `is_primary` | Which surface form to display |

An alias table rather than a synonym column, because the *method and confidence of the link* are
themselves data a reviewer needs. "These two records were merged" is not a useful statement;
"these two were merged on an exact ChEMBL identifier" and "these two were merged on a fuzzy string
match scoring 0.87" are different claims with different failure modes, and a system that stores only
the merged result cannot distinguish them afterwards.

**Nothing is merged silently.** `entity_resolution_decision` takes one row per link — accepted or
rejected — with the method, the score, the decider and the supporting evidence. A spurious merge can
therefore be traced and reversed.

### How resolution connects to validation

Grouped cross-validation keys on the canonical entity id, and
`trs/validation/splitters.py::ConnectedComponentGrouper` takes the transitive closure: two cohort
entries join if they share *any* key in the active scope, or are linked through a chain of such
shared keys. Assigning whole components to folds is what guarantees no unit straddles the boundary.

The scope choice is itself a documented decision. `GroupScope.IDENTITY` uses `(program, compound)` —
the same asset under a different name, which is unambiguous leakage — and is used for the headline
temporal split. `GroupScope.STRICT` adds `target` and is used only for a secondary sensitivity
analysis. Mechanism class is *never* a grouping key, because the transitive closure over it swallows
the cohort and because it would remove from training exactly the class-level knowledge a real analyst
possesses at the decision point.

On the demonstration cohort this yields 466 connected components from 571 rows under identity scope,
largest component 3, 374 singletons. Under the strict (target-inclusive) scope the same 571 rows
collapse to 42 components with a largest of 23 — which is why the strict scope cannot be combined
with a temporal split and is reported as a cross-sectional sensitivity analysis instead.

**The gap.** `trs/db/models/core.py` names a `trs.resolution` module as the writer of these tables.
That module does not exist — `backend/trs/resolution/` is an empty directory. Today the alias tables
and the decision log are populated only by the seed generator. The schema is ready for a resolution
pipeline; the pipeline is not written.

---

## 8. Feature, cohort and modelling layers

This section is a map, not a specification; the substance is in
[DATA_DICTIONARY.md §5–§7](DATA_DICTIONARY.md#5-endpoint-definitions) and in the module docstrings.

| Stage | Module | The decision that shapes it |
| --- | --- | --- |
| Endpoint definition | `trs/cohort/outcomes.py` | Six endpoints, each fully declarative. Exclusions are counted, not discarded. Competing risks are removed, not relabelled. |
| Cohort build | `trs/cohort/build.py` | Produces a frozen `cohort_version` plus an attrition waterfall. Changing a rule creates a new version rather than mutating the old one. |
| Feature schema | `trs/features/spec.py`, `registry.py` | 63 declared features; anything undeclared cannot enter a matrix. Ordinal maps state the scale they encode. `expected_direction` is recorded but **never enforced** — it only flags sign-discordant coefficients for review. |
| Feature build | `trs/features/builder.py` | Reads only through `AsOfContext`. Emits values, per-feature missingness reasons, and a feature→evidence map together. |
| Effect harmonisation | `trs/features/effects.py` | Two documented monotone families (ratio-of-remaining, log-ratio). A comparability device, explicitly not a meta-analysis. |
| Historical rates | `trs/features/historical.py` | Isolated because it is the most dangerous feature family. Excludes future outcomes and the focal program's own fate unconditionally; beta-binomial shrinkage with prior strength 10. |
| Encoding | `trs/features/transform.py` | Imputation statistics, one-hot level sets and standardisation constants are all fitted on the training fold only and serialised into the artifact. |
| Splitting | `trs/validation/splitters.py` | Temporal × connected component. No random split is reported anywhere. |
| Model families | `trs/modeling/` | `l2_logistic` (Laplace posterior, hand-written IRLS), `hierarchical_logistic` (partial pooling), `gbdt` (grouped bootstrap ensemble). All three are fitted every run so "a linear model suffices here" is tested rather than assumed. |
| Calibration | `trs/pipeline/calibrate.py` | Cross-validated out-of-fold on the training partition, not a dedicated slice — a temporal calibration slice produced a held-out slope of 0.53 during development. |
| Selection | `trs/pipeline/select.py` | Log loss, not AUROC. Calibration-slope gate [0.80, 1.25]. Paired component bootstrap must exclude zero. Surcharges for losing exact attribution or a real posterior. |
| Uncertainty | `trs/modeling/uncertainty.py` | Three sources decomposed by ablation on the log-odds scale: parameter, imputation, calibration. |
| Explanation | `trs/scoring/explanation.py` | `explain()` asserts that the contributions sum to `logit(p)` rather than assuming it. |

### The deployed model, on demonstration data

Read from `backend/artifacts/ptrs-pef-001.json`. **All of these describe a simulation.**

| | |
| --- | --- |
| Dataset | 600 programs → 785 development tracks → 5,444 evidence items |
| Cohort | 785 candidates → 571 included (335 positive, 58.7%), 214 excluded with recorded reasons |
| Split | Temporal at 2017-12-23; train 422, test 149, **12 rows dropped**. A component whose rows straddle the boundary is assigned wholly to the *later* partition and its earlier rows are discarded — the conservative direction, since it removes rows from training rather than letting future-resolved siblings inform an earlier prediction |
| Selected family | `l2_logistic`; penalty 128.0 chosen from a grid spanning 0.25–2048 |
| Rejected | `gbdt` (slope 1.452) and `hierarchical_logistic` (slope 0.776) — both disqualified on the calibration gate, neither on discrimination |
| Held-out | AUROC 0.630 [0.537, 0.720] · Brier 0.238 · log loss 0.671 · calibration slope 0.733 · intercept −0.229 · ECE 0.098 |
| Skill vs base rate | scaled Brier **+0.035 [−0.065, +0.112]** — interval includes zero |
| Verdict emitted by the platform | `not_distinguishable_from_base_rate` |
| Oracle ceiling (simulation ground truth) | AUROC 0.737 · log loss 0.592 (base-rate log loss 0.678) |
| Target-grouped sensitivity | AUROC 0.601 vs headline 0.630 |
| Interval honesty check | analytic Laplace widths / bootstrap-refit widths: median 1.88 (p10 1.55, p90 2.44) → `analytic_intervals_too_wide` |

Two of these deserve emphasis because they are the kind of number a product usually hides. The
platform's own usefulness verdict is that the model is **not distinguishable from quoting the cohort
base rate**, and it says so in the API response and on the Methods page. And the interval-agreement
check reports that the published analytic intervals are roughly 1.9× wider than a bootstrap refit
would give — erring toward over-caution, but erring, and reported rather than tuned away.

The architectural point is that both numbers exist because the pipeline computes them unprompted.
An oracle benchmark, a skill-versus-base-rate gate and an interval-agreement check are cheap to leave
out and expensive to add later, after the headline number has been quoted.

---

## 9. The API layer

FastAPI, versioned at `/api/v1`, five routers, 29 endpoints plus `/api/v1/health` and `/`. OpenAPI is
generated from the Pydantic models in `trs/api/schemas.py`, so the published contract cannot drift
from the code that serves it.

| Router | Prefix | Responsibility |
| --- | --- | --- |
| `programs` | `/api/v1/programs` | Track listing and detail, evidence, evidence matrix, PK/PD, target assessment, safety |
| `scores` | `/api/v1` | Score (GET and POST), scenario levers, scenario comparison, comparators |
| `model` | `/api/v1/model` | Version(s), validation report, coefficients, feature schema, endpoints, cohorts, schema version |
| `portfolio` | `/api/v1/portfolio` | Portfolio roll-up and projects |
| `provenance` | `/api/v1/provenance` | Source, evidence, score lineage, feature lineage, connector health and catalogue, ingestion runs |

Versioning posture: a breaking payload change ships as `/api/v2` while `/api/v1` keeps working.
Model versions and feature-schema versions are carried *inside* payloads rather than in the URL,
because a retrained model is not a new API.

### 9.1 Authentication posture, and the development affordance

Three credential paths resolve to one `Principal` (`trs/api/deps.py::get_principal`):

1. `X-API-Key` → `resolve_api_key()`. The stored value is an **HMAC keyed on the application
   secret**, not a bare digest, so a leaked key table is not offline-crackable without also stealing
   `TRS_SECRET_KEY`. Lookup is by indexed `prefix`, comparison by `hmac.compare_digest`.
2. `Authorization: Bearer <jwt>` → HS256, TTL from `TRS_ACCESS_TOKEN_TTL_MINUTES` (720 by default),
   carrying `sub`, `org`, `role`, `iat`, `exp`, `jti`.
3. **Neither** → if `TRS_ENV` is `development` or `test`, the request resolves to the first
   organisation with `role=analyst` and `is_development_fallback=True`.

Path 3 is the affordance, and it exists so the UI runs without a login flow during development.
Conveniences of exactly this kind are how systems ship with authentication effectively disabled, so
it is hard-gated at three levels:

- **It cannot start in production.** `assert_production_safety()` runs in the FastAPI `lifespan`
  hook — at application start, not at first request — and raises `RuntimeError` if
  `settings.is_production` and any of: the fallback is reachable, `TRS_SECRET_KEY` is still
  `"dev-only-insecure-key-change-me"`, or CORS is set to `*`. The process refuses to serve.
- **It cannot be invisible.** The lifespan hook logs a warning naming the active `TRS_ENV`, and
  `/api/v1/health` returns `development_auth_fallback` as a field, so a deployment running with the
  fallback active advertises it.
- **It is flagged on the principal.** `Principal.is_development_fallback` travels with the caller.

The honest caveat: the production guard checks the *environment variable*, so it protects against
the realistic failure (a `.env` copied forward, a container inheriting `TRS_ENV=development`) and not
against someone deliberately setting `TRS_ENV=development` on a production host. That is a
configuration-discipline boundary, and the guard is placed where it can actually catch the mistake
that happens.

Roles are ordered `viewer < analyst < admin < owner` in `ROLE_ORDER`, and `require_role()` compares
by index, so a higher role implies every lower power without enumerating combinations.

### 9.2 Rate limiting and its single-process caveat

`TokenBucketLimiter` is an in-process token bucket keyed by API key id, then user id, then
organisation id, then client address. Defaults: 120 requests/minute, burst 40.

**The caveat is stated in the class docstring, not discovered later:** this is adequate for a
single-process deployment and for making the limit visible in the API contract. Run four uvicorn
workers and the effective limit becomes 4× the configured one, because each process holds its own
`defaultdict` of buckets. There is no shared store. The interface — `check(key) -> (allowed,
retry_after)` — is deliberately the one that would be swapped for a Redis-backed implementation, and
the swap touches one class.

A second consequence: buckets are held in a `defaultdict` that is never evicted, so a long-running
process accumulates one entry per distinct key seen. At this scale that is bounded by the number of
API keys and organisations; behind a public endpoint keyed on client address it would be unbounded.

### 9.3 Audit trail

`record_audit()` appends to `audit_event`, and is wired into both scoring paths in
`trs/api/v1/scores.py`. Three properties:

- **It never raises into the request path.** The whole write is wrapped; on failure it rolls back and
  returns. Auditing that can break a request gets removed the first time it does.
- **Rows are append-only.** `AuditEvent` carries `IdMixin` but deliberately not `TimestampMixin`:
  there is no `updated_at` because there is no update.
- **The actor columns are not foreign keys.** Intentional — the trail must survive deletion of the
  user or key it refers to. The cost is that an orphaned row is indistinguishable from a bad write.

The rationale, from the model docstring: *scoring is an auditable act.* A portfolio committee that
acted on a number must be able to learn, later, exactly which model version and which evidence
snapshot produced it. The audit row is the index into the reconstruction chain in
[§3](#3-the-five-data-states).

### 9.4 Error handling

A global exception handler logs the stack trace and returns `{"detail": "Internal server error.",
"request_id": ...}`. A stack trace in a response body is an information-disclosure bug, so internals
are logged and never returned. Every response carries `X-Request-ID` and `X-Response-Time-Ms`,
correlating a user-visible failure with a server log line and with the `request_id` on the audit row.

---

## 10. Frontend architecture

Next.js 16 App Router, React 19, TypeScript, Tailwind v4. Five routes: portfolio (`/`),
`programs/[id]`, `evidence`, `methods`, `provenance`.

### 10.1 Server components fetch; the client bundle stays thin

All data fetching happens in React Server Components through `lib/api.ts`. Only five modules are
`"use client"`, and each is client-side for a specific reason: `error.tsx` (error boundaries must
be), `PortfolioTable` and `EvidenceTable` (sorting and filtering), `ScenarioPanel` (interactive
lever selection), and `charts/index.tsx` (event handlers on marks).

The intent, stated in the `lib/api.ts` docstring, is that **the browser never talks to the API
directly and no credentials reach the client bundle.** That property holds for every read path.

**It does not currently hold for one write path, and that should be said plainly.**
`components/program/ScenarioPanel.tsx` issues its scenario `POST` with a browser `fetch` straight to
`NEXT_PUBLIC_API_BASE_URL`. In development that works because CORS allows `http://localhost:3000` and
the unauthenticated fallback resolves the principal. In production it would arrive without a
credential and be rejected, because there is none in the client bundle to send — which is, at least,
the safe failure direction. Closing it means either a Next route handler proxying the call
server-side, or a server action. It is a gap, not a design.

### 10.2 Caching is per-call and deliberate

`request()` defaults to `revalidate: 0` and takes `revalidate: false` (`cache: "no-store"`) for
anything that must be live. Scores and portfolio roll-ups are computed live and are never cached.

The interesting case is the **contract-describing** endpoints — the lever catalogue, the feature
schema, the endpoint definitions, the connector catalogue. These look like static reference data and
are explicitly *not* cached, with the reason recorded in a comment in `lib/api.ts`:

> During development this collapsed the adverse-scenario group into the improvement group for an
> hour: the cached payload predated the `direction` field, every lever read as `undefined`, and the
> filter put all fourteen into one bucket with no error anywhere.

Caching a contract survives a backend deploy, after which the interface renders against a contract
the server no longer serves — silently, because every field the UI reads is still present, just
stale. These payloads are a few kilobytes and the API is a local analytical service, so caching buys
nothing worth that failure mode.

### 10.3 Hand-built SVG charts, and why not a chart library

`components/charts/index.tsx` is 405 lines and exports six charts: `DriverWaterfall`,
`CalibrationPlot`, `DecisionCurve`, `ScenarioComparisonChart`, `UncertaintyComposition`,
`DomainContributions`. The only dependencies are `d3-scale`, `d3-shape` and `d3-array` — the
*arithmetic*. Every mark is written out.

The argument, from the module docstring: scientific charts need exact control over axes, reference
lines, interval rendering and annotation, and a general-purpose library spends most of its API
surface on things this product never needs while making the three things it does need awkward.

Three conventions follow from that and are enforced across every chart:

- **Reference lines are drawn first and dashed** — the null, perfect calibration, the cohort average
  — so a reader sees what the data is being compared *against* before seeing the data.
- **Uncertainty is drawn wherever it exists.** A point estimate without its interval is not plotted.
  This is a rule the chart code can enforce and a chart library cannot.
- **Axes are labelled with units.** Nothing is left to a caption.

`DriverWaterfall` is the clearest case for hand-building. Because the contributions sum exactly to
the log-odds difference from the cohort average, **the chart is the arithmetic**, not an illustration
of it. Rendering that through a bar-chart abstraction that normalises or reorders values would break
the identity the explanation layer asserts.

The cost is real: roughly 400 lines to maintain, no free tooltips, no free animation, and any new
chart type is new code. For six charts that are load-bearing scientific claims, that is the right
side of the trade. For twenty dashboard tiles it would not be.

### 10.4 Demonstration banner

`app/layout.tsx` calls `api.health()` on every render and shows a `DemoBanner` when
`is_demonstration_deployment` is true. The flag originates on `model_artifact.trained_on_synthetic`,
which originates on `is_synthetic` on the training rows. The chain is mechanical, so the banner
cannot be out of sync with the data.

---

## 11. Configuration and process topology

Twelve-factor: all configuration is environment-driven with the `TRS_` prefix, via
`pydantic-settings`. `.env.example` at the repository root is the template; nothing secret is
committed. The full variable list is in that file.

Processes in the full stack:

```
uvicorn trs.api.main:app --port 8000          # stateless API (see the rate-limit caveat, §9.2)
next dev/start --port 3000                    # SSR frontend
python -m trs.cli pipeline                    # offline: init-db → seed → build-cohort → train
python -m trs.cli {connectors,oracle,models}  # offline diagnostics
alembic upgrade head                          # schema migration
```

The API process holds no state except the rate-limiter buckets and the SQLAlchemy connection pool.
Training is entirely offline and writes a `model_artifact` row plus a JSON artifact under
`TRS_ARTIFACT_DIR`; the API reads the active artifact and never trains.

---

## 12. Scaling: what breaks first and what would change

In the order the limits would actually be hit.

| # | Limit | Symptom | Change required |
| --- | --- | --- | --- |
| 1 | **Rate limiter is in-process** | Effective limit multiplies by the worker count; a restart resets every bucket | Swap `TokenBucketLimiter` for a Redis-backed implementation behind the same `check(key) -> (allowed, retry_after)` interface. One class. |
| 2 | **SQLite write concurrency** | `database is locked` under concurrent ingestion and serving, despite WAL and a 5 s busy timeout | Move to PostgreSQL. Already supported: change `TRS_DATABASE_URL`, run `alembic upgrade head`. No model changes. |
| 3 | **Connector cache is a local directory** | Multiple API or worker hosts cannot share warm payloads; each re-fetches | `ResponseCache` gains an object-store backend (its interface is `key`/`get`/`put`). |
| 4 | **Ingestion is synchronous and single-process** | A full sweep across six sources is wall-clock bound by the slowest, and a crash loses the run | A task queue (Celery/RQ/arq) with `connector_run` as the existing job ledger — the status and counter columns are already there. |
| 5 | **Scoring rebuilds features per request** | Latency grows with evidence volume per track | Serve from `feature_snapshot` when `(track, as_of, schema_version)` matches and the evidence set is unchanged. The uniqueness constraint already makes the cache key well-defined. |
| 6 | **`raw_record.content_bytes` in-row** | The table dominates database size | Move payloads to object storage, keep the hash and a pointer. `content_hash` is already the identity. |
| 7 | **Training is single-machine, in-memory** | Bounded by RAM at roughly 10⁵–10⁶ rows × ~100 design columns | Out-of-core or distributed fitting. Not close — the demonstration cohort is 571 rows, and the binding constraint on this problem is *sample size*, not compute. |
| 8 | **No read replicas / no per-tenant sharding** | Analytical queries compete with serving | Standard Postgres read replicas; `organization_id` is already on every private table, so tenant sharding has a key if it is ever needed. |

Two things deliberately do **not** appear on that list. The feature schema is versioned
(`fs-1.0.0`) and embedded in every snapshot and artifact, so a schema change is a migration rather
than a breaking incident. And the API is versioned at `/api/v1` with model and schema versions inside
payloads, so retraining does not break clients.

---

## 13. Decision record

Each row: the decision, the alternative that was rejected, why, and what it costs.

| # | Decision | Alternative rejected | Why | Cost |
| --- | --- | --- | --- | --- |
| **ADR-001** | PostgreSQL as production target, SQLite as zero-dependency local default, **one** model set and **one** migration history | Postgres-only with a required `docker compose`; or two divergent schemas | A reviewer who cannot run the pipeline in one command evaluates the README instead of the system. Two schemas diverge, and the divergence surfaces in production. | Every schema decision must be portable; no engine-specific DDL is available (see ADR-004) |
| **ADR-002** | ULID `VARCHAR(32)` primary keys | Native `UUID` (Postgres) / `BLOB` (SQLite); or integer autoincrement | One physical representation on both engines. Time-sortable, so `ORDER BY id` is creation order and B-tree inserts stay append-only. Integers leak row counts across a tenant boundary and collide on merge. | 26 text bytes per key vs 16 binary; larger indexes |
| **ADR-003** | `JSONVariant` type decorator | Postgres-only `JSONB`; or normalising every structured payload into tables | Validation reports, coefficient sets and missingness maps are read as whole documents and change shape with the schema version. Normalising them would mean a migration per report field. | No in-JSON indexing on SQLite; JSON columns are opaque to constraints on both |
| **ADR-004** | Controlled vocabularies as `VARCHAR`, validated in application code | Database `ENUM` types; or `CHECK` constraints | Adding a vocabulary term needs no migration on either engine, and SQLite cannot `ALTER TYPE` at all. The ontology is expected to grow. | **The database will accept a typo.** The 38 vocabularies have no database-level guard |
| **ADR-005** | `UTCDateTime` **raises** on a naive datetime | Silently coerce to UTC; or store naive and normalise on read | A silent coercion gives a caller a shifted value and no error. The failure lands on `available_at <= as_of`, the one comparison that must not be wrong. A hard failure appears at the write. | Every caller must use `utcnow()`; a naive value from a library is a runtime error |
| **ADR-006** | `AsOfContext` as a class that owns the session | A `WHERE available_at <= T` convention at each call site; or a query-builder helper | A convention fails silently, forever, and in the direction that flatters results. The builder takes a context, not a session, so an unguarded query is unreachable from the feature layer. | Adding an evidence table means adding an accessor; the guarantee stops where the accessors stop |
| **ADR-007** | Immutable `raw_record` landing zone before any interpretation | Parse on ingest and store only normalised output | Every value stays re-derivable and every extraction bug is diagnosable against exactly the bytes that caused it. Re-running against cache reproduces payloads byte-for-byte. | Storage grows with ingestion and never shrinks |
| **ADR-008** | `normalise()` is pure and does no I/O | Combined fetch-and-parse per source | CI for a system whose inputs are six third-party APIs. Fixtures make parsing testable offline. | Two-stage code; an intermediate `RawPayload` type |
| **ADR-009** | Connector adapters behind one ABC | Direct API calls at each use site | Six sources fail in six different ways. One outage degrades one connector. A commercial feed drops in as a sibling class. | Boilerplate per source; the abstraction must fit both REST and GraphQL |
| **ADR-010** | Alias tables with `match_method` + `match_confidence`, plus a decision log | A synonym column; or silent canonical-name merging | Duplicate programs put the same biology in train and test, which is invisible in the outputs. Storing *how* a link was made is what makes a bad merge reversible. | Five extra tables; resolution must write a decision row for every link |
| **ADR-011** | Connected-component grouping for splits; identity scope for the headline | Random k-fold; or grouping on mechanism class | Random splitting on this problem is not merely optimistic, it is meaningless. Mechanism-class grouping collapses the cohort and removes knowledge a real analyst has. | 12 of 571 demonstration rows dropped at the temporal boundary; folds are coarse |
| **ADR-012** | `l2_logistic` as the deployed default | `gbdt` on discrimination | The L2 penalty *is* a Gaussian prior, so the fit is a MAP estimate with a defensible Laplace posterior, and the log-odds is an exact additive sum that the explanation layer can assert. A tree ensemble has neither. Both challengers were run and both failed the calibration gate anyway (slopes 1.452 and 0.776). | No interactions unless declared; likely a small discrimination cost on a larger cohort |
| **ADR-013** | Log loss as the selection criterion, with a calibration-slope gate | AUROC | The product's output is a probability a committee reads as a number. AUROC is invariant to monotone distortion of probabilities — precisely the failure this product must avoid. | A model that ranks better can lose |
| **ADR-014** | Cross-validated out-of-fold calibration on the training partition | A dedicated three-way train/calibrate/test split | On a few hundred programs, a temporal calibration slice is a narrow window whose idiosyncrasies the calibrator learns. In development that produced a held-out slope of 0.53. | Out-of-fold predictions are marginally less sharp than the final model, so the calibrator over-corrects slightly. Documented in the artifact rather than corrected |
| **ADR-015** | Development auth fallback, hard-gated by `assert_production_safety()` at app start | No fallback (login flow required in dev); or a fallback with no guard | Without it, the first-run experience requires a login flow before anything can be seen. Without the guard, it is how auth ends up disabled in production. | Checks `TRS_ENV`, so it stops the realistic mistake, not a deliberate one |
| **ADR-016** | In-process token-bucket rate limiter | Redis from day one; or no limiting | Zero external dependencies for the local default, and the limit is visible in the API contract. | **Multiplies by worker count; resets on restart.** Documented here rather than found in production |
| **ADR-017** | Server components fetch; no credentials in the client bundle | Client-side fetching with a token in the browser | A token in a client bundle is a token that leaks. Server-side fetching also keeps the API private to the server network. | Interactivity needs an explicit boundary — and one write path currently crosses it (see [§10.1](#101-server-components-fetch-the-client-bundle-stays-thin)) |
| **ADR-018** | Hand-built SVG charts; d3 for scales only | Recharts / Victory / Plotly | Uncertainty must be drawn wherever it exists and reference lines must be drawn first; the driver waterfall *is* the model's arithmetic and must not be normalised or reordered by an abstraction. | ~400 lines to maintain; no free tooltips or animation |
| **ADR-019** | Contract-describing endpoints are explicitly not cached | ISR / long `revalidate` on "static" reference data | A cached contract survives a backend deploy and then renders silently wrong — observed during development with the scenario levers. | A few kilobytes re-fetched per render |
| **ADR-020** | Persist `feature_snapshot` rather than recompute on read | Recompute features on demand | Recomputation silently changes historical scores when evidence is updated. A decision-support tool whose past outputs mutate is not auditable. | Storage; a schema change requires a rebuild pass |
| **ADR-021** | Oracle benchmark, base-rate skill gate and interval-agreement check computed every run | Report AUROC and calibration and stop | "AUROC 0.63 on demonstration data" is close to meaningless alone — a reader cannot tell a pipeline recovering most of an attainable signal from one failing on an easy problem. These checks are cheap now and impossible to add credibly after a number has been quoted. | Training is slower; the platform publishes an unflattering verdict about itself |

---

## 14. Known weaknesses

Stated here so they are not found later.

1. **The temporal guarantee stops at five tables.** `AsOfContext` has accessors for `evidence_item`,
   `pkpd_bridge`, `target_assessment`, `program_biomarker` and `safety_finding`. Anything reading
   `trial`, `source` or the alias tables must apply its own filter, and `available_at` is nullable on
   all five alias tables and on `program_target`. Nothing currently reads them in the feature path.

2. **There is no entity-resolution pipeline.** `trs.resolution`, named in the model docstrings, is an
   empty directory. Alias rows and `entity_resolution_decision` rows are written only by the seed
   generator. `trs/schemas/` is likewise empty — the Pydantic models live in `trs/api/schemas.py`.

3. **The rate limiter does not survive horizontal scaling**, and its bucket dictionary is never
   evicted. See [§9.2](#92-rate-limiting-and-its-single-process-caveat).

4. **One frontend write path bypasses the server-component boundary.** The scenario `POST` goes from
   the browser to the API directly. See [§10.1](#101-server-components-fetch-the-client-bundle-stays-thin).

5. **No database-level vocabulary enforcement.** A deliberate trade (ADR-004), but it means all 38
   vocabularies are guarded only by application code.

6. **Trial ingestion is unavailable from this network.** ClinicalTrials.gov v2 returns HTTP 403 on
   every path from this egress, so the `ctgov` fixtures are hand-constructed and labelled as such.
   Until egress changes or a proxy is configured, no registry-derived evidence can be ingested live.

7. **The deployed model is not distinguishable from the cohort base rate on demonstration data**,
   and the demonstration data is synthetic. The architecture is the deliverable here; the model's
   performance on real data is unknown and unclaimed.

---

*Cross-references: [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for every table, vocabulary and feature.*
