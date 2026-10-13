"""API integration tests: the HTTP contract, end to end, offline.

These exercise the real FastAPI application against a real (in-memory) database
seeded with the demonstration generator and a real, if small, trained model
artifact. Nothing is mocked below the HTTP boundary, because the failures worth
catching here — a filter that silently ignores its argument, an as-of cutoff
that leaks future evidence into a score, an explanation that does not add up to
the number shown beside it — all live in the wiring between layers rather than
inside any one of them.

Why this module builds its own database instead of using ``seeded_session``
--------------------------------------------------------------------------
``seeded_session`` is function-scoped, and the scoring endpoints need an active
model artifact. Training once per test would dominate the runtime of the suite,
so the whole environment (schema, seed, cohort, model, client) is built once at
session scope. The helpers used are the same ones ``conftest`` uses.

Why the seed is larger than the 60 programs ``seeded_session`` uses
------------------------------------------------------------------
Training is not a formality here. Two gates must be cleared with real data.

``trs.pipeline.select`` rejects a candidate whose calibration slope is outside
the sanity band; at 60 programs the recalibrator separates and the slope
explodes.

``trs.validation.protocol.plan_training`` then requires a development partition
of at least 120 rows, an evaluation partition of at least 60, and at least
three usable forward-chaining folds — *after* purging rows whose outcome had
not resolved by the fitting origin. Label maturity is the expensive constraint:
it removes roughly half the otherwise-eligible development rows, because the
median outcome lag exceeds five years.

``N_SEED_PROGRAMS`` is the smallest round size observed to satisfy
``temporal_outer_maturity`` — the same protocol the production demonstration
run accepts. That matters more than the fixture's speed: a smaller seed would
only satisfy ``retrospective``, and the scoring tests would then be exercising
a protocol path production does not use. The thresholds in
``trs.validation.protocol`` are deliberately not lowered for tests; a test that
cannot meet them is told to supply more data.

Offline
-------
No test here touches the network. Connectors are never invoked; the model is
trained from seeded rows; artifacts are written to a temporary directory.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

#: Programs to seed. See the module docstring for why this is not 60 — and why
#: it is not 150, which was enough for the predecessor's split but leaves only
#: 39 label-mature development rows under the current protocol. 480 satisfies
#: the protocol's row counts and then fails the selection gate with a
#: calibration slope of −0.147: 121 development rows against ~110 encoded
#: columns is a degenerate fit, and the gate is right to refuse it. 700 yields
#: 187 development and 140 evaluation rows and clears both gates in ~4 s.
N_SEED_PROGRAMS = 700

#: Protocol names ``trs.validation.protocol`` can accept. A number whose
#: provenance names none of these cannot be interpreted.
DECLARED_PROTOCOLS = frozenset({
    "strict_temporal", "temporal_outer_maturity", "retrospective",
})

#: The four verdicts ``trs.pipeline.train._usefulness_verdict`` can return.
#: A fifth value appearing here means an unreviewed claim about the model's
#: skill is reaching the Methods page.
USEFULNESS_VERDICTS = frozenset({
    "beats_base_rate",
    "worse_than_base_rate",
    "not_distinguishable_from_base_rate",
    "indeterminate",
})

#: Response keys whose value must be a probability. Signed quantities are
#: deliberately absent: ``probability_delta``, ``delta_probability``,
#: ``contribution`` and the logits may all legitimately be negative.
PROBABILITY_KEYS = frozenset({
    "probability",
    "baseline_probability",
    "interval_low",
    "interval_high",
    "interval_level",
    "interval_width",
    "coverage",
    "overall",
    "prevalence",
    "completeness_score",
    "variance_share",
})


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ApiEnv:
    """The wired-up application under test, plus what was put in the database."""

    client: TestClient
    session: Session
    tracks: list[dict[str, Any]] = field(default_factory=list)
    model_version: str | None = None
    model_error: str | None = None


@pytest.fixture(scope="session")
def api_env(tmp_path_factory) -> ApiEnv:
    """The application, bound to a seeded in-memory database with a live model.

    The engine uses ``StaticPool`` because ``TestClient`` runs the ASGI app on a
    worker thread: with the default SQLite pooling that thread would open its
    own connection and find an empty database.

    Two process-global objects are redirected for the duration of the session
    and restored afterwards:

    * ``trs.db.session``'s engine and session factory, because the health
      endpoint opens its own ``session_scope()`` rather than taking the request
      session, and would otherwise report on the developer's real database;
    * ``trs.api.security.limiter``, because the production token bucket has a
      burst of 40 and this module issues more requests than that. Rate limiting
      is a real behaviour and is not exercised here; it is disabled explicitly
      rather than left to make the suite fail on request number 41.
    """
    import trs.api.security as security
    import trs.db.models  # noqa: F401  (registers the mappers)
    import trs.db.session as db_session_module
    from trs.api.deps import get_db
    from trs.api.main import app
    from trs.config import settings
    from trs.db.base import Base
    from trs.pipeline.artifacts import clear_bundle_cache

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    session = factory()

    saved = (
        db_session_module._engine,
        db_session_module._SessionFactory,
        settings.artifact_dir,
        security.limiter,
    )
    db_session_module._engine = engine
    db_session_module._SessionFactory = factory
    settings.artifact_dir = tmp_path_factory.mktemp("artifacts")
    security.limiter = security.TokenBucketLimiter(rate_per_minute=10**6, burst=10**6)
    clear_bundle_cache()

    env = ApiEnv(client=None, session=session)  # type: ignore[arg-type]
    try:
        from trs.seed.generator import DemoDataGenerator

        DemoDataGenerator(session, seed=4242).run(n_programs=N_SEED_PROGRAMS)
        session.commit()

        try:
            from trs.cohort.build import build_cohort
            from trs.pipeline.train import train

            cohort = build_cohort(session).cohort_version
            session.commit()
            bundle = train(
                session,
                cohort_version_id=cohort.id,
                candidate_families=("l2_logistic",),
                n_bootstrap=0,
                run_interval_check=False,
                run_ablation=False,
                register_candidates=False,
                make_active=True,
                verbose=False,
            )
            session.commit()
            env.model_version = bundle.version
        except Exception as exc:  # noqa: BLE001 - reported as a skip reason below
            session.rollback()
            env.model_error = f"{type(exc).__name__}: {exc}"

        def _override_get_db():
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

        app.dependency_overrides[get_db] = _override_get_db
        with TestClient(app) as client:
            env.client = client
            # Page through the whole listing rather than asking for one
            # oversized page. `/api/v1/programs` caps `limit` at 500, so a
            # single request silently truncated the fixture's view of the
            # cohort once the seed grew past that — and every test that
            # compares a filtered listing against `api_env.tracks` inherited
            # the truncation as a false expectation.
            page_size = 200
            collected: list[dict[str, Any]] = []
            seen: set[str] = set()
            while True:
                response = client.get(
                    "/api/v1/programs",
                    params={"limit": page_size, "offset": len(collected)},
                )
                assert response.status_code == 200, response.text
                page = response.json()
                collected.extend(page)
                if len(page) < page_size:
                    break
                assert len(collected) <= 5000, "pagination did not terminate"
            for row in collected:
                assert row["id"] not in seen, (
                    f"track {row['id']} was returned on two pages; the "
                    "listing's ordering is not stable under offset"
                )
                seen.add(row["id"])

            # Cross-check the paginated total against the database, so a
            # listing that drops rows at a page boundary cannot pass as a
            # complete view.
            from trs.db.models.core import ProgramIndication as _PI

            expected_total = session.query(_PI).count()
            assert len(collected) == expected_total, (
                f"paginated listing returned {len(collected)} tracks but the "
                f"database holds {expected_total}"
            )
            env.tracks = collected
            yield env
    finally:
        app.dependency_overrides.clear()
        clear_bundle_cache()
        session.close()
        engine.dispose()
        (
            db_session_module._engine,
            db_session_module._SessionFactory,
            settings.artifact_dir,
            security.limiter,
        ) = saved


@pytest.fixture(scope="session")
def client(api_env: ApiEnv) -> TestClient:
    return api_env.client


@pytest.fixture(scope="session")
def any_track(api_env: ApiEnv) -> dict[str, Any]:
    """A track with an IND date and evidence attached — the normal case."""
    for row in api_env.tracks:
        if row["ind_or_cta_on"] and row["n_evidence_items"] > 0:
            return row
    pytest.fail("The seeded dataset contains no track with both an IND date and evidence.")


@pytest.fixture(scope="session")
def scorable_track(api_env: ApiEnv, any_track: dict[str, Any]) -> dict[str, Any]:
    """The track the scoring tests use; skips the lot if no model was trained."""
    if api_env.model_version is None:
        pytest.skip(
            "No active model artifact could be trained for this module, so the "
            f"scoring endpoints cannot be exercised. Training failed with: "
            f"{api_env.model_error}"
        )
    return any_track


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_probability_fields(node: Any, path: str = "$") -> list[tuple[str, float]]:
    """Every probability-valued leaf in a decoded JSON payload, with its path.

    Recursive because the score payload nests probabilities four levels deep
    (``scenario.current.uncertainty.variance_share.parameter``), and a check
    that only looked at the top level would pass while the interface displayed
    a nonsense number from inside ``explanation`` or ``uncertainty``.
    """
    found: list[tuple[str, float]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            if (
                key in PROBABILITY_KEYS
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                found.append((child, float(value)))
            elif key == "variance_share" and isinstance(value, dict):
                found.extend(
                    (f"{child}.{k}", float(v))
                    for k, v in value.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                )
            found.extend(collect_probability_fields(value, child))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(collect_probability_fields(item, f"{path}[{index}]"))
    return found


def assert_probabilities_bounded(payload: Any, *, minimum_checked: int) -> None:
    """Assert every probability-valued field is a finite number in [0, 1].

    ``minimum_checked`` guards the guard: if a rename made the walk find
    nothing, this helper would otherwise pass silently for ever.
    """
    fields = collect_probability_fields(payload)
    assert len(fields) >= minimum_checked, (
        f"Only {len(fields)} probability-valued fields were found, expected at "
        f"least {minimum_checked}. Either the payload shrank or the walk is "
        f"looking for the wrong keys. Found: {[p for p, _ in fields]}"
    )
    for path, value in fields:
        assert math.isfinite(value), f"{path} = {value} is not a finite number"
        assert 0.0 <= value <= 1.0, f"{path} = {value} is outside [0, 1]"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_reports_deployment_posture(client: TestClient, api_env: ApiEnv) -> None:
    """If this breaks, a deployment cannot tell you what it is.

    ``is_demonstration_deployment`` and ``development_auth_fallback`` are the
    two facts that distinguish a demonstration instance with authentication
    effectively disabled from a real one. Both must be reported, and reported
    truthfully: a demo instance that answers ``false`` here is indistinguishable
    from production to anyone reading its output.
    """
    from trs import __version__
    from trs.features.spec import FEATURE_SCHEMA_VERSION

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "status", "version", "environment", "database", "active_model",
        "feature_schema_version", "is_demonstration_deployment",
        "development_auth_fallback",
    }
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["version"] == __version__
    assert body["environment"] == "test"
    assert body["feature_schema_version"] == FEATURE_SCHEMA_VERSION

    # TRS_ENV=test keeps the unauthenticated fallback reachable; saying so is
    # the entire point of the flag.
    assert body["development_auth_fallback"] is True

    if api_env.model_version is None:
        assert body["active_model"] is None
        assert body["is_demonstration_deployment"] is False
    else:
        assert body["active_model"] == api_env.model_version
        # The active model was trained on generator output, so the deployment
        # must declare itself a demonstration.
        assert body["is_demonstration_deployment"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Program listing and filters
# ─────────────────────────────────────────────────────────────────────────────

def test_programs_list_returns_tracks(client: TestClient, api_env: ApiEnv) -> None:
    """If this breaks, the portfolio screen is empty or shows malformed rows.

    The list is the entry point to everything else: an id that does not round
    trip to a detail page, or a missing indication, breaks every downstream
    view.
    """
    rows = api_env.tracks
    assert isinstance(rows, list)
    assert len(rows) >= N_SEED_PROGRAMS, (
        f"{N_SEED_PROGRAMS} programs were seeded but only {len(rows)} tracks "
        "came back; tracks are program × indication, so there should be at "
        "least one per program."
    )
    assert len({r["id"] for r in rows}) == len(rows), "duplicate track ids in the listing"

    for row in rows[:20]:
        assert row["indication"]["name"]
        assert row["modality"]
        assert row["outcome_label"]
        assert row["n_evidence_items"] >= 0
        # Everything in this dataset is generated; a row that claims otherwise
        # would be presented to users as real evidence.
        assert row["is_synthetic"] is True


def test_programs_pagination(client: TestClient, api_env: ApiEnv) -> None:
    """If this breaks, the portfolio table either truncates or repeats rows.

    ``limit``/``offset`` are the only way the UI walks a portfolio larger than
    one page; an ignored ``offset`` shows page one for ever.
    """
    first = client.get("/api/v1/programs", params={"limit": 5}).json()
    assert len(first) == 5

    second = client.get("/api/v1/programs", params={"limit": 5, "offset": 5}).json()
    assert len(second) == 5
    assert {r["id"] for r in first}.isdisjoint({r["id"] for r in second})
    assert [r["id"] for r in first + second] == [r["id"] for r in api_env.tracks[:10]]

    over_limit = client.get("/api/v1/programs", params={"limit": 501})
    assert over_limit.status_code == 422, "the documented limit ceiling is not enforced"


def test_programs_filter_by_target(client: TestClient, api_env: ApiEnv) -> None:
    """If this breaks, a target filter shows programs against other targets.

    This is a soundness *and* completeness check: the filtered set must be
    exactly the rows the unfiltered listing attributes to that symbol. A join
    that drops rows is as wrong as one that admits foreign ones.
    """
    symbol = next(r["target"]["symbol"] for r in api_env.tracks if r["target"])
    expected = {r["id"] for r in api_env.tracks if r["target"] and r["target"]["symbol"] == symbol}

    rows = client.get("/api/v1/programs", params={"target": symbol, "limit": 500}).json()
    assert {r["id"] for r in rows} == expected
    assert expected, "no track carries a target symbol, so the filter is untested"
    assert len(expected) < len(api_env.tracks), "the filter narrowed nothing"
    assert all(r["target"]["symbol"] == symbol for r in rows)

    # Symbol matching is case-insensitive on the API surface.
    lowered = client.get(
        "/api/v1/programs", params={"target": symbol.lower(), "limit": 500}
    ).json()
    assert {r["id"] for r in lowered} == expected

    assert client.get("/api/v1/programs", params={"target": "NOT_A_GENE"}).json() == []


def test_programs_filter_by_modality(client: TestClient, api_env: ApiEnv) -> None:
    """If this breaks, a modality filter returns the wrong therapeutic class.

    Modality drives which comparators and which historical base rates a user
    reads the score against; a filter that silently ignores it puts a cell
    therapy in an oligonucleotide list.
    """
    modality = api_env.tracks[0]["modality"]
    expected = {r["id"] for r in api_env.tracks if r["modality"] == modality}

    rows = client.get("/api/v1/programs", params={"modality": modality, "limit": 500}).json()
    assert {r["id"] for r in rows} == expected
    assert 0 < len(expected) < len(api_env.tracks)
    assert all(r["modality"] == modality for r in rows)

    assert client.get("/api/v1/programs", params={"modality": "not_a_modality"}).json() == []


def test_programs_free_text_filter(client: TestClient, api_env: ApiEnv) -> None:
    """If this breaks, search either returns everything or nothing.

    A search box that silently ignores its query is worse than no search box:
    the user believes they are looking at a filtered list.
    """
    name = api_env.tracks[0]["program_name"]
    expected = {r["id"] for r in api_env.tracks if name.lower() in r["program_name"].lower()}

    rows = client.get("/api/v1/programs", params={"q": name, "limit": 500}).json()
    assert {r["id"] for r in rows} == expected
    assert 0 < len(rows) < len(api_env.tracks)

    # Matching is case-insensitive on both sides of the comparison.
    lowered = client.get("/api/v1/programs", params={"q": name.lower(), "limit": 500}).json()
    assert {r["id"] for r in lowered} == expected

    assert client.get("/api/v1/programs", params={"q": "zzz-no-such-program"}).json() == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. Program detail
# ─────────────────────────────────────────────────────────────────────────────

def test_program_detail_round_trips(client: TestClient, any_track: dict[str, Any]) -> None:
    """If this breaks, clicking a portfolio row opens the wrong program.

    The detail payload must extend the summary rather than restate it
    differently: the UI renders both from the same fields.
    """
    response = client.get(f"/api/v1/programs/{any_track['id']}")
    assert response.status_code == 200
    detail = response.json()

    assert detail["id"] == any_track["id"]
    assert detail["program_name"] == any_track["program_name"]
    assert detail["indication"]["name"] == any_track["indication"]["name"]
    assert detail["n_evidence_items"] == any_track["n_evidence_items"]

    # Sibling tracks: the same asset in other indications. Exactly one of them
    # is this one, and every one belongs to the same program.
    assert detail["all_tracks"], "detail payload lists no tracks at all"
    assert sum(1 for t in detail["all_tracks"] if t["is_current"]) == 1
    assert any(t["track_id"] == any_track["id"] for t in detail["all_tracks"])


def test_program_detail_unknown_id_is_404(client: TestClient) -> None:
    """If this breaks, a stale link renders an error page or, worse, a 200.

    A bogus identifier must be a clean 404, not a 500 from an unguarded
    ``None`` and not an empty success.
    """
    response = client.get("/api/v1/programs/01NOT-A-REAL-TRACK-ID")
    assert response.status_code == 404
    assert response.json()["detail"] == "Program not found."


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evidence and the as-of cutoff
# ─────────────────────────────────────────────────────────────────────────────

def test_evidence_respects_the_as_of_cutoff(
    client: TestClient, any_track: dict[str, Any]
) -> None:
    """If this breaks, every score the platform reports is invalid.

    The evidence list is what a user checks the score against. If it shows an
    item published after the decision date, either the score used it — and the
    model's validation statistics are contaminated by hindsight — or the
    interface is lying about which evidence the score rests on. Both are fatal
    to the product's central claim.
    """
    track_id = any_track["id"]
    cutoff = dt.date.fromisoformat(any_track["ind_or_cta_on"])

    default = client.get(f"/api/v1/programs/{track_id}/evidence").json()
    assert default, "the chosen track reports evidence items but returns none"
    for item in default:
        assert dt.date.fromisoformat(item["available_at"]) <= cutoff, (
            f"evidence {item['id']} became available {item['available_at']}, "
            f"after the {cutoff.isoformat()} cutoff, yet is shown as evidence "
            "available at the decision point"
        )

    # An explicit earlier cutoff can only shrink the set, and never admits an
    # item dated after it.
    earlier = cutoff - dt.timedelta(days=365 * 3)
    narrowed = client.get(
        f"/api/v1/programs/{track_id}/evidence", params={"as_of": earlier.isoformat()}
    ).json()
    assert {i["id"] for i in narrowed} <= {i["id"] for i in default}
    assert all(dt.date.fromisoformat(i["available_at"]) <= earlier for i in narrowed)


def test_include_post_cutoff_only_ever_adds(client: TestClient, api_env: ApiEnv) -> None:
    """If this breaks, the context view hides evidence or silently scores it.

    ``include_post_cutoff`` exists so a reader can see what was learned after
    the decision point without that evidence entering the score. It must be a
    strict superset of the scoring view — an item that appears only in the
    default view would mean the two paths disagree about what exists.
    """
    grew_for_at_least_one = False
    checked = 0

    for row in api_env.tracks[:20]:
        if not row["ind_or_cta_on"]:
            continue
        checked += 1
        track_id = row["id"]
        cutoff = dt.date.fromisoformat(row["ind_or_cta_on"])
        default = client.get(f"/api/v1/programs/{track_id}/evidence").json()
        everything = client.get(
            f"/api/v1/programs/{track_id}/evidence", params={"include_post_cutoff": True}
        ).json()

        assert {i["id"] for i in default} <= {i["id"] for i in everything}
        assert len(everything) >= len(default)
        if len(everything) > len(default):
            grew_for_at_least_one = True
            later = [
                i for i in everything
                if dt.date.fromisoformat(i["available_at"]) > cutoff
            ]
            assert later, (
                "the post-cutoff view returned extra items, none of which "
                "postdate the cutoff — the two queries disagree for some other "
                "reason"
            )

    assert checked >= 5, "too few tracks with an IND date to test the cutoff"
    assert grew_for_at_least_one, (
        "no track gained an item when post-cutoff evidence was requested, so "
        "this test proved nothing; the seeded data no longer contains evidence "
        "dated after any IND date"
    )


def test_evidence_matrix_is_internally_consistent(
    client: TestClient, any_track: dict[str, Any]
) -> None:
    """If this breaks, the evidence-gap view misreports what is missing.

    The matrix exists to show negative space — an empty human-derived row is
    the finding. Coverage outside [0, 1], or a cell pointing at an evidence id
    that is not in the track's evidence at that cutoff, means the gaps shown are
    not the real gaps.
    """
    track_id = any_track["id"]
    matrix = client.get(f"/api/v1/programs/{track_id}/evidence-matrix").json()

    assert 0.0 <= matrix["coverage"] <= 1.0
    total_cells = len(matrix["rows"]) * len(matrix["columns"])
    assert total_cells > 0
    occupied = {(c["evidence_class"], c["system"]) for c in matrix["cells"]}
    assert matrix["empty_cells"] == total_cells - len(occupied)
    assert matrix["coverage"] == pytest.approx(len(occupied) / total_cells, abs=5e-5)

    available = client.get(f"/api/v1/programs/{track_id}/evidence").json()
    available_ids = {i["id"] for i in available}
    assert matrix["n_evidence"] == len(available_ids)

    cell_ids: list[str] = []
    for cell in matrix["cells"]:
        assert cell["n_total"] == len(cell["evidence_ids"])
        assert cell["n_supporting"] + cell["n_refuting"] <= cell["n_total"]
        cell_ids.extend(cell["evidence_ids"])

    assert len(cell_ids) == len(set(cell_ids)), "an evidence item appears in two cells"
    assert set(cell_ids) <= available_ids, (
        "the matrix references evidence ids that are not in the track's "
        "evidence at this cutoff"
    )
    assert len(cell_ids) == matrix["n_evidence"], "the matrix drops evidence items"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Scoring
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def score(client: TestClient, scorable_track: dict[str, Any]) -> dict[str, Any]:
    """A score obtained in an explicitly selected demonstration context.

    The fixture model is trained on synthetic data, so `demo_context` is
    required. ``test_score_is_refused_without_a_demonstration_context`` covers
    the other branch — without this flag the API returns a refusal, and that is
    the behaviour the eligibility policy exists to produce.
    """
    response = client.post(
        f"/api/v1/programs/{scorable_track['id']}/score",
        json={"demo_context": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["probability"] is not None
    assert body["eligibility"]["permitted"] is True
    return body


def test_score_returns_an_ordered_interval(
    score: dict[str, Any], api_env: ApiEnv, scorable_track: dict[str, Any]
) -> None:
    """If this breaks, the product ships a probability nobody can act on.

    The central promise is that no estimate is returned bare. An interval whose
    bounds are crossed, or that excludes its own point estimate, is worse than
    no interval: it looks authoritative and is arithmetically impossible.
    """
    assert score["track_id"] == scorable_track["id"]
    assert score["model_version"] == api_env.model_version
    assert score["model_family"] == "l2_logistic"
    assert score["endpoint"] == "ph2_efficacy_failure"
    assert score["estimand"], "the estimand must travel with the estimate"

    assert 0.0 <= score["interval_low"] <= score["probability"] <= score["interval_high"] <= 1.0
    assert score["interval_level"] == pytest.approx(0.95)
    assert score["uncertainty"]["probability"] == pytest.approx(score["probability"])
    assert score["uncertainty"]["interval_low"] == pytest.approx(score["interval_low"])
    assert score["uncertainty"]["interval_high"] == pytest.approx(score["interval_high"])
    assert score["uncertainty"]["interval_width"] == pytest.approx(
        score["interval_high"] - score["interval_low"]
    )

    # The interval's width is decomposed, and the shares are a decomposition:
    # they account for all of it and name the largest contributor.
    shares = score["uncertainty"]["variance_share"]
    assert shares, "the interval is reported without saying where its width comes from"
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)
    assert score["uncertainty"]["dominant_source"] in shares
    assert shares[score["uncertainty"]["dominant_source"]] == pytest.approx(max(shares.values()))


def test_score_is_as_of_the_decision_date(
    score: dict[str, Any], scorable_track: dict[str, Any], client: TestClient
) -> None:
    """If this breaks, the score answers a different question than it claims.

    The default cutoff is the IND/CTA date — the decision the product exists to
    support. Falling back to today's date would quietly turn a prospective
    estimate into a retrospective one computed with hindsight.
    """
    assert score["as_of_date"] == scorable_track["ind_or_cta_on"]

    at_cutoff = client.get(f"/api/v1/programs/{scorable_track['id']}/evidence").json()
    assert score["n_evidence_items"] == len(at_cutoff), (
        "the score counts a different number of evidence items than the "
        "evidence endpoint shows for the same cutoff"
    )


def test_score_explanation_reconstructs_the_logit(score: dict[str, Any]) -> None:
    """If this breaks, the displayed drivers do not explain the displayed number.

    The product's explanation claim is exactness: contributions are terms in the
    model's own linear predictor, not a post-hoc approximation. If the domain
    roll-up plus the baseline plus the residual does not equal the total logit,
    the driver list is decoration and the interface is misattributing risk.
    """
    explanation = score["explanation"]
    assert explanation is not None, score.get("explanation_unavailable_reason")
    assert explanation["method"].startswith("exact_additive_logodds")
    assert explanation["reconstruction_error"] == pytest.approx(0.0, abs=1e-8)

    # by_domain covers every driver; increasing_risk/decreasing_risk are the
    # top-k slices shown in the UI and need not be exhaustive.
    domain_total = sum(d["contribution"] for d in explanation["by_domain"])
    reconstructed = (
        explanation["baseline_logit"]
        + domain_total
        + explanation["other_factors_contribution"]
    )
    assert reconstructed == pytest.approx(explanation["total_logit"], abs=1e-8)
    assert sigmoid(explanation["total_logit"]) == pytest.approx(
        explanation["probability"], abs=1e-9
    )
    assert sigmoid(explanation["baseline_logit"]) == pytest.approx(
        explanation["baseline_probability"], abs=1e-9
    )

    assert all(d["contribution"] > 0 for d in explanation["increasing_risk"])
    assert all(d["contribution"] < 0 for d in explanation["decreasing_risk"])
    for bucket in ("increasing_risk", "decreasing_risk"):
        magnitudes = [abs(d["contribution"]) for d in score["explanation"][bucket]]
        assert magnitudes == sorted(magnitudes, reverse=True), (
            f"{bucket} is not ordered by magnitude, so the UI's 'top drivers' "
            "are not the top drivers"
        )
    for driver in explanation["increasing_risk"] + explanation["decreasing_risk"]:
        assert driver["direction"] == (
            "increases_risk" if driver["contribution"] > 0 else "decreases_risk"
        )
        assert driver["domain"] and driver["label"]
        if driver["is_missing"]:
            assert driver["value"] is None


def test_score_declares_its_demonstration_status(score: dict[str, Any]) -> None:
    """If this breaks, simulated output can be mistaken for a real estimate.

    Every number in this deployment derives from generated data and a model
    trained on it. The response must say so, and the evidence-confidence grade
    must be pinned to its lowest level — an interface showing 'moderate
    confidence' beside a synthetic estimate is the single most dangerous state
    this product can reach.
    """
    assert score["is_demonstration"] is True
    assert score["demonstration_note"]
    assert "DEMONSTRATION" in score["demonstration_note"]
    assert score["evidence_confidence"] == "very_low"
    assert score["confidence_reasons"], "a confidence grade with no stated reasons"
    assert any("synthetic" in reason.lower() for reason in score["confidence_reasons"])

    # The model's own verdict on whether it beats the base rate travels with
    # every estimate, not only with the Methods page.
    assert score["model_usefulness"].get("verdict") in USEFULNESS_VERDICTS


def test_score_completeness_is_a_real_assessment(score: dict[str, Any]) -> None:
    """If this breaks, the evidence-gap readout beside the score is fabricated.

    Completeness is what stops a confident-looking number resting on three
    features. The counts must agree with the declared feature schema.
    """
    from trs.features.registry import FEATURES

    completeness = score["completeness"]
    assert 0.0 <= completeness["overall"] <= 1.0
    assert completeness["n_total"] == len(FEATURES) == 63
    assert 0 <= completeness["n_present"] <= completeness["n_total"]
    assert all(0.0 <= v <= 1.0 for v in completeness["by_domain"].values())


def test_score_unknown_track_is_rejected(client: TestClient, scorable_track) -> None:
    """If this breaks, a bad id yields a 500 and a stack trace in the logs.

    A scoring request for something that does not exist is a client error with
    a readable message, not an unhandled exception.

    ``scorable_track`` is unused except to skip this test when no model exists:
    without an artifact the endpoint fails before it ever looks at the id.
    """
    response = client.post("/api/v1/programs/01NOT-A-REAL-TRACK-ID/score", json={})
    assert response.status_code == 400
    assert "01NOT-A-REAL-TRACK-ID" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 7 & 8. Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_returns_both_sides_of_the_comparison(
    client: TestClient, scorable_track: dict[str, Any]
) -> None:
    """If this breaks, the scenario view shows a number with nothing to compare.

    A scenario is only interpretable as a difference: current state,
    hypothetical state, and exactly what was changed between them. Without the
    change list a user cannot tell what they were shown the effect of.
    """
    response = client.post(
        f"/api/v1/programs/{scorable_track['id']}/scenarios",
        json={"lever_keys": ["independent_replication"], "demo_context": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    current, scenario = body["current"], body["scenario"]
    assert current["track_id"] == scenario["track_id"] == scorable_track["id"]
    assert current["as_of_date"] == scenario["as_of_date"]
    assert current["scenario"] is None
    assert scenario["scenario"]["lever_keys"] == ["independent_replication"]

    assert body["changes"], (
        "applying a lever changed nothing at all, so the comparison is empty"
    )
    changed = {c["feature"] for c in body["changes"]}
    assert changed <= set(scenario["scenario"]["overrides"])
    for change in body["changes"]:
        assert change["from_value"] != change["to_value"]
        assert change["label"] and change["domain_label"]

    assert body["delta_probability"] == pytest.approx(
        scenario["probability"] - current["probability"], abs=1e-12
    )
    assert body["delta_interval_width"] == pytest.approx(
        (scenario["interval_high"] - scenario["interval_low"])
        - (current["interval_high"] - current["interval_low"]),
        abs=1e-12,
    )
    # The scenario must state what it is not. This wording is the guard against
    # reading a sensitivity analysis as a forecast.
    assert "not predict" in body["disclaimer"] or "does not predict" in body["disclaimer"]


def test_scenario_levers_cannot_make_a_program_look_worse(
    client: TestClient, scorable_track: dict[str, Any]
) -> None:
    """If this breaks, commissioning good work appears to raise a program's risk.

    An improvement lever is clamped so it can only move a feature in the
    risk-reducing direction. Without the clamp, applying 'add an independent
    replication' to an already well-replicated program lowers its replication
    score — and the scenario reads as 'doing this study would make it worse',
    which is nonsense a team could act on.
    """
    response = client.post(
        f"/api/v1/programs/{scorable_track['id']}/scenarios",
        json={
            "lever_keys": ["independent_replication", "multi_lab_replication"],
            "demo_context": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    for change in body["changes"]:
        if change["feature"] == "single_lab_dependence":
            # Lower is better for this one.
            assert change["to_value"] <= (change["from_value"] or 1.0)
        if change["feature"] in {"replication_scope_ord", "n_independent_labs"}:
            assert change["from_value"] is None or change["to_value"] >= change["from_value"]


def test_scenario_rejects_overrides_it_must_not_allow(
    client: TestClient, scorable_track: dict[str, Any]
) -> None:
    """If this breaks, users can fabricate a low-risk program.

    Three separate gates, each of which has to hold: unknown features, features
    a team cannot change by doing work (historical class performance, disease
    context), and values outside a feature's declared range. Any of them
    leaking turns scenario analysis into a way to produce whatever number the
    user wants.
    """
    url = f"/api/v1/programs/{scorable_track['id']}/scenarios"

    unknown_feature = client.post(url, json={"overrides": {"not_a_feature": 1.0}})
    assert unknown_feature.status_code == 422
    assert "not_a_feature" in unknown_feature.json()["detail"]

    # Declared in the registry, but deliberately not scenario-adjustable.
    not_adjustable = client.post(url, json={"overrides": {"animal_median_effect_z": 3.0}})
    assert not_adjustable.status_code == 422
    assert "not adjustable" in not_adjustable.json()["detail"]

    # animal_positive_fraction is adjustable but declared on [0, 1].
    out_of_range = client.post(url, json={"overrides": {"animal_positive_fraction": 1.5}})
    assert out_of_range.status_code == 422
    assert "maximum" in out_of_range.json()["detail"]

    unknown_lever = client.post(url, json={"lever_keys": ["no_such_lever"]})
    assert unknown_lever.status_code == 422
    assert "no_such_lever" in unknown_lever.json()["detail"]

    nothing_asked = client.post(url, json={})
    assert nothing_asked.status_code == 422


def test_lever_catalogue_states_each_lever_direction(client: TestClient) -> None:
    """If this breaks, the UI cannot tell good news from bad news.

    ``direction`` is what distinguishes 'a replication succeeded' from 'a
    contradictory study appeared'. Without it every lever renders as an
    improvement, and the adverse scenarios — the ones that stop this being a
    marketing tool — are presented as upside.
    """
    body = client.get("/api/v1/scenarios/levers").json()
    levers = body["levers"]
    assert levers
    assert body["note"], "the catalogue is served without its interpretation caveat"

    keys = [lever["key"] for lever in levers]
    assert len(keys) == len(set(keys))

    for lever in levers:
        assert lever["direction"] in {"improvement", "adverse", "absolute"}
        assert lever["label"] and lever["description"] and lever["action"]
        assert lever["domain"] and lever["domain_label"]
        assert lever["overrides"], f"lever {lever['key']} changes nothing"
        assert {a["name"] for a in lever["affected_features"]} <= set(lever["overrides"])
        for implied in lever["implies"]:
            assert implied in keys, f"lever {lever['key']} implies a lever that does not exist"

    # A catalogue with no adverse lever would only ever show improvement.
    assert any(lever["direction"] == "adverse" for lever in levers)


# ─────────────────────────────────────────────────────────────────────────────
# 9 & 10. Model registry, validation report, feature schema
# ─────────────────────────────────────────────────────────────────────────────

def test_validation_report_publishes_the_unflattering_numbers(
    client: TestClient, api_env: ApiEnv
) -> None:
    """If this breaks, the Methods page stops telling users not to trust the model.

    The usefulness verdict is the threshold question — does this beat quoting
    the base rate? — and the limitations are how a reader learns the data is
    synthetic. A report that serves a verdict it cannot support, or drops the
    limitations, converts a documented weak model into an undocumented one.
    """
    if api_env.model_version is None:
        pytest.skip(f"no model artifact: {api_env.model_error}")

    response = client.get("/api/v1/model/validation")
    assert response.status_code == 200, response.text
    report = response.json()

    assert report["model_version"] == api_env.model_version
    assert report["cohort"]["n_entries"] > 0

    # The split description must name a declared protocol, not merely describe
    # a date boundary. A reader who cannot tell which protocol produced a
    # number cannot tell whether it may be read as deployment performance.
    design = report["evaluation_design"]
    assert any(
        proto in design["split"] for proto in DECLARED_PROTOCOLS
    ), f"split description names no declared protocol: {design['split']!r}"
    sizes = design["sizes"]
    assert sizes["development"] > 0
    assert sizes["evaluation"] > 0
    # Selection must have happened on fewer rows than the whole cohort and must
    # not have been given the evaluation partition.
    assert 0 < sizes["n_selection_rows"] <= sizes["development"]

    usefulness = report["usefulness"]
    assert usefulness["verdict"] in USEFULNESS_VERDICTS
    assert usefulness["headline"]
    scaled_brier = usefulness["scaled_brier"]
    if scaled_brier.get("ci_low") is None or scaled_brier.get("ci_high") is None:
        # No interval was estimated, so no claim about beating the base rate is
        # supportable. Anything other than 'indeterminate' here is an
        # unsupported claim — and until this was fixed, training with the
        # bootstrap switched off raised TypeError while formatting the headline.
        assert usefulness["verdict"] == "indeterminate"
    assert "base rate" in usefulness["headline"]

    primary = report["primary_metrics"]
    assert primary["n"] == report["evaluation_design"]["sizes"]["evaluation"]
    assert 0.0 <= primary["discrimination"]["auroc"]["value"] <= 1.0
    assert 0.0 <= primary["accuracy"]["brier"]["value"] <= 1.0
    assert 0.0 <= primary["prevalence"] <= 1.0

    limitations = report["limitations"]
    assert limitations, "a validation report with no stated limitations"
    assert any("synthetic" in item.lower() for item in limitations), (
        "the report does not state that the model was trained on synthetic data"
    )
    assert any("component" in item.lower() for item in limitations)

    assert report["coefficient_sign_audit"], "the sign audit is not published"


def test_feature_schema_is_the_declared_63(client: TestClient) -> None:
    """If this breaks, the Methods page and the model disagree about the inputs.

    The feature schema is the published contract between evidence and model.
    Serving a different set than the registry declares means the documented
    derivation of a driver no longer describes the number shown.
    """
    from trs.features.registry import FEATURES
    from trs.features.spec import DOMAIN_LABELS

    response = client.get("/api/v1/model/features")
    assert response.status_code == 200
    specs = response.json()

    assert len(specs) == 63 == len(FEATURES)
    assert [s["name"] for s in specs] == [f.name for f in FEATURES]
    assert len({s["name"] for s in specs}) == 63

    labels = {str(k): v for k, v in DOMAIN_LABELS.items()}
    for spec in specs:
        assert spec["label"] and spec["description"] and spec["derivation"]
        assert spec["domain"] in labels
        assert spec["domain_label"] == labels[spec["domain"]]
        assert spec["expected_direction"] in {"increases_risk", "decreases_risk", "uncertain"}
        assert isinstance(spec["scenario_adjustable"], bool)
        assert spec["completeness_weight"] >= 0
        if spec["valid_min"] is not None and spec["valid_max"] is not None:
            assert spec["valid_min"] <= spec["valid_max"]

    assert sum(1 for s in specs if s["scenario_adjustable"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. Probability bounds, everywhere
# ─────────────────────────────────────────────────────────────────────────────

def test_every_probability_in_a_score_response_is_bounded(score: dict[str, Any]) -> None:
    """If this breaks, the interface can display a 130% chance of failure.

    Probabilities reach the UI from four subsystems — the calibrator, the
    uncertainty decomposition, the explanation and the completeness report —
    and a bound violation in any of them is a visible, credibility-destroying
    defect. The walk is recursive because most of these live several levels
    deep in the payload.
    """
    assert_probabilities_bounded(score, minimum_checked=12)


def test_every_probability_in_a_scenario_response_is_bounded(
    client: TestClient, scorable_track: dict[str, Any]
) -> None:
    """If this breaks, an overridden feature can push a probability out of range.

    A scenario feeds hand-supplied feature values through the same model. It is
    the path most likely to produce an extreme design row, and it nests two
    complete score payloads, so the bound check has to cover both.
    """
    response = client.post(
        f"/api/v1/programs/{scorable_track['id']}/scenarios",
        json={
            "lever_keys": ["exposure_gap_identified", "contradictory_study_appears"],
            "demo_context": True,
        },
    )
    assert response.status_code == 200, response.text
    assert_probabilities_bounded(response.json(), minimum_checked=24)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Eligibility, refusal and the prediction contract
# ─────────────────────────────────────────────────────────────────────────────

def test_score_is_refused_without_a_demonstration_context(
    client: TestClient, api_env: ApiEnv, scorable_track: dict[str, Any]
) -> None:
    """If this breaks, a simulation's number appears without being asked for.

    The fixture model is trained on synthetic data. Asking for a score without
    declaring a demonstration context must return a refusal — ``probability``
    absent, a machine-readable ``refusal_code``, and a reason a person can
    read. Not a number with a banner next to it: a banner is something a client
    can fail to render, and a null is not.
    """
    if api_env.model_version is None:
        pytest.skip(f"no model artifact: {api_env.model_error}")

    response = client.post(
        f"/api/v1/programs/{scorable_track['id']}/score", json={}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["probability"] is None
    assert body["interval_low"] is None and body["interval_high"] is None
    assert body["uncertainty"] is None
    assert body["prediction_contract"] is None
    assert body["explanation"] is None

    elig = body["eligibility"]
    assert elig["permitted"] is False
    assert elig["refusal_code"] == "demo_context_not_selected"
    assert elig["model_status"] == "demo_only"
    assert elig["reason"].strip()


def test_refusal_still_returns_evidence_review_material(
    client: TestClient, api_env: ApiEnv, scorable_track: dict[str, Any]
) -> None:
    """If this breaks, declining to score becomes declining to be useful.

    The whole argument for a default-deny score policy is that everything
    *else* the platform knows stays available: completeness, the evidence
    count, the snapshot id for provenance, the endpoint's estimand. A refusal
    that blanks the page is a worse product and an easier one to work around.
    """
    if api_env.model_version is None:
        pytest.skip(f"no model artifact: {api_env.model_error}")

    body = client.post(
        f"/api/v1/programs/{scorable_track['id']}/score", json={}
    ).json()

    assert body["completeness"]["overall"] is not None
    assert 0.0 <= body["completeness"]["overall"] <= 1.0
    assert body["n_evidence_items"] >= 0
    assert body["feature_snapshot_id"]
    assert body["estimand"].strip()
    assert body["evidence_confidence"]
    assert body["model_usefulness"], "the model's own verdict was dropped"
    assert body["explanation_unavailable_reason"]


def test_scenario_is_refused_when_no_score_may_be_produced(
    client: TestClient, api_env: ApiEnv, scorable_track: dict[str, Any]
) -> None:
    """If this breaks, a scenario reports 'no change' where the truth is 'no estimate'.

    Returning a comparison whose two sides are both null, inside a shape that
    promises ``delta_probability``, invites a client to render zero.
    """
    if api_env.model_version is None:
        pytest.skip(f"no model artifact: {api_env.model_error}")

    response = client.post(
        f"/api/v1/programs/{scorable_track['id']}/scenarios",
        json={"lever_keys": ["independent_replication"]},
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "scenario_unavailable"
    assert detail["refusal_code"] == "demo_context_not_selected"
    assert detail["reason"].strip()


def test_served_probability_is_the_functional_the_metrics_describe(
    score: dict[str, Any], client: TestClient
) -> None:
    """If this breaks, the displayed number is not the one that was evaluated.

    The predecessor served ``np.mean(sigmoid(draws))`` while every metric in
    the artifact — and the driver attribution — described the calibrated
    plug-in value. The two differ systematically by Jensen's inequality, most
    for the programmes with the least evidence, which are exactly the ones
    whose number is read most carefully.
    """
    contract = score["prediction_contract"]
    assert contract["functional"] == "calibrated_plug_in"
    assert contract["evaluated_functional"] == contract["functional"], (
        "the served functional and the evaluated functional have diverged"
    )
    assert contract["probability"] == pytest.approx(score["probability"])

    # The predictive mean must be present under its own name and must not be
    # the served value's twin by accident.
    assert contract["predictive_mean_probability"] is not None
    assert score["uncertainty"]["predictive_mean_probability"] == pytest.approx(
        contract["predictive_mean_probability"]
    )
    assert contract["functional_gap"] == pytest.approx(
        contract["predictive_mean_probability"] - contract["probability"]
    )

    # And the artifact must agree about which functional it reports on.
    validation = client.get("/api/v1/model/validation").json()
    assert validation["prediction_contract"]["evaluated_functional"] == (
        contract["functional"]
    )

    assert "NOT a prediction interval" in contract["interval_estimand"]


def test_explanation_is_checked_against_the_served_number(
    score: dict[str, Any]
) -> None:
    """If this breaks, drivers are shown for a number nobody can see.

    The internal reconstruction check compares a partition of a sum against
    that sum — both built from the same terms — so it cannot detect that the
    decomposition targets the wrong quantity. Only the external check can, and
    a ``None`` there means it did not run rather than that it passed.
    """
    explanation = score["explanation"]
    if explanation is None:
        pytest.skip(f"no attribution: {score['explanation_unavailable_reason']}")

    assert explanation["external_check_ran"] is True, (
        "the service did not hand the explainer the number it serves, so the "
        "only check capable of catching a wrong-quantity explanation was "
        "skipped"
    )
    assert explanation["served_reconstruction_error"] == pytest.approx(0.0, abs=1e-9)
    assert explanation["explained_quantity"] == "calibrated_plug_in"
    assert explanation["explained_quantity_description"].strip()


def test_portfolio_pagination_and_summary_agree_about_scope(
    client: TestClient, api_env: ApiEnv
) -> None:
    """If this breaks, a page-level mean is read as a portfolio-level one.

    The response field was previously named `total` and returned `len(rows)` —
    the page size — so a caller asking for 60 rows was told the portfolio held
    60 programmes when it held 765, with no indication the other 705 existed.
    The page then described its 60-row mean as covering "every program in the
    portfolio".

    Three properties are asserted together, because each is useless alone: the
    portfolio-level count is independent of pagination, the page-level count is
    not, and the summary block says which of the two it describes.
    """
    first = client.get(
        "/api/v1/portfolio", params={"limit": 5, "offset": 0, "score": False}
    )
    assert first.status_code == 200, first.text
    a = first.json()

    assert a["n_returned"] == len(a["rows"]) == 5
    assert a["total_matching"] > a["n_returned"], (
        "this fixture does not have enough programmes to test pagination"
    )
    assert a["page_offset"] == 0
    assert a["page_limit"] == 5
    assert a["has_more"] is True

    summary = a["page_summary"]
    assert summary["scope"] == "this_page_only"
    assert summary["n_rows_on_page"] == a["n_returned"]
    assert str(a["total_matching"]) in summary["note"], (
        "the summary does not disclose the portfolio-level denominator it is "
        "*not* describing"
    )

    # The portfolio-level count must not move with the page.
    second = client.get(
        "/api/v1/portfolio", params={"limit": 5, "offset": 5, "score": False}
    ).json()
    assert second["total_matching"] == a["total_matching"]
    assert second["page_offset"] == 5
    assert {r["track_id"] for r in second["rows"]}.isdisjoint(
        {r["track_id"] for r in a["rows"]}
    ), "consecutive pages overlap, so the ordering is not stable under offset"

    # Walking off the end yields no rows and no further pages, and still
    # reports the same denominator.
    tail = client.get(
        "/api/v1/portfolio",
        params={"limit": 5, "offset": a["total_matching"], "score": False},
    ).json()
    assert tail["rows"] == []
    assert tail["has_more"] is False
    assert tail["total_matching"] == a["total_matching"]


def test_portfolio_rows_report_refusal_rather_than_a_silent_null(
    client: TestClient, api_env: ApiEnv
) -> None:
    """If this breaks, "not scored" and "refused" are indistinguishable.

    A null probability has two quite different causes: scoring was not
    requested, or it was requested and declined. A client that cannot tell them
    apart will render an empty cell for both.
    """
    if api_env.model_version is None:
        pytest.skip(f"no model artifact: {api_env.model_error}")

    unscored = client.get(
        "/api/v1/portfolio", params={"limit": 3, "score": False}
    ).json()
    for row in unscored["rows"]:
        assert row["probability"] is None
        assert row["score_permitted"] is None, (
            "a row that was never scored claims a permission decision"
        )

    refused = client.get(
        "/api/v1/portfolio",
        params={"limit": 3, "score": True, "demo_context": False},
    ).json()
    assert refused["page_summary"]["n_refused_on_page"] == len(refused["rows"])
    for row in refused["rows"]:
        assert row["probability"] is None
        assert row["score_permitted"] is False
        assert row["refusal_code"] == "demo_context_not_selected"

    permitted = client.get(
        "/api/v1/portfolio",
        params={"limit": 3, "score": True, "demo_context": True},
    ).json()
    assert permitted["page_summary"]["n_refused_on_page"] == 0
    assert permitted["page_summary"]["n_scored_on_page"] == len(permitted["rows"])
    for row in permitted["rows"]:
        assert row["score_permitted"] is True
        assert row["probability"] is not None
        assert 0.0 <= row["probability"] <= 1.0
