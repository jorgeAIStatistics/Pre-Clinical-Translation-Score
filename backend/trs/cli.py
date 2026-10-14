"""Command-line interface.

Everything the platform does can be driven from here without a running web
server: seeding, cohort construction, feature materialisation, training,
scoring and connector health. This is deliberate — the modelling layer must be
usable from a notebook or a batch job by a statistician who has no interest in
the web application.

    python -m trs.cli --help
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import Any

from trs.config import settings


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_init_db(args: argparse.Namespace) -> int:
    """Create all tables directly (development shortcut for `alembic upgrade head`)."""
    import trs.db.models  # noqa: F401
    from trs.db.base import Base
    from trs.db.session import get_engine

    Base.metadata.create_all(get_engine())
    print(f"Schema created on {settings.database_url}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    from trs.db.session import session_scope
    from trs.seed import seed_database
    from trs.seed.diagnostics import save_ground_truth
    from trs.seed.generator import verify_base_rate

    with session_scope() as session:
        result = seed_database(session, n_programs=args.programs, seed=args.seed)
        print("DEMO / SYNTHETIC DATA — NOT FOR SCIENTIFIC INTERPRETATION")
        _print(result.summary())
        path = save_ground_truth(result.ground_truth)
        print(f"Simulation ground truth (diagnostic only) written to {path}")
        check = verify_base_rate(session)
        _print({"base_rate_check": check})
        if not check["within_tolerance"]:
            print(
                "WARNING: realised failure rate is far from the simulation's "
                "target; the generator's intercept may need recalibration.",
                file=sys.stderr,
            )
    return 0


def cmd_build_cohort(args: argparse.Namespace) -> int:
    from trs.cohort.build import build_cohort
    from trs.cohort.outcomes import ENDPOINTS
    from trs.db.session import session_scope

    endpoint = ENDPOINTS[args.endpoint]
    with session_scope() as session:
        result = build_cohort(
            session,
            name=args.name,
            version=args.version,
            endpoint=endpoint,
            as_of_rule=args.as_of_rule,
            data_cutoff=dt.date.fromisoformat(args.data_cutoff) if args.data_cutoff else None,
        )
        cv = result.cohort_version
        print(f"Cohort {cv.name}/{cv.version}  id={cv.id}")
        _print({
            "endpoint": endpoint.name,
            "estimand": endpoint.estimand,
            **result.summary,
            "n_positive": cv.n_positive,
            "n_negative": cv.n_negative,
            "positive_rate": (cv.n_positive / cv.n_entries) if cv.n_entries else None,
        })
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from trs.cohort.outcomes import ENDPOINTS
    from trs.db.session import session_scope
    from trs.pipeline.train import train

    with session_scope() as session:
        bundle = train(
            session,
            cohort_version_id=args.cohort,
            endpoint=ENDPOINTS[args.endpoint],
            make_active=not args.no_activate,
            n_bootstrap=args.bootstrap,
            run_interval_check=not args.skip_interval_check,
            run_ablation=not args.skip_ablation,
            verbose=True,
        )
        m = bundle.metrics["primary_metrics"]
        print("\n" + "=" * 72)
        print(f"MODEL {bundle.version}  family={bundle.family}")
        print("=" * 72)
        _print({
            "auroc": m["discrimination"]["auroc"],
            "auprc": m["discrimination"]["auprc"],
            "brier": m["accuracy"]["brier"],
            "scaled_brier": m["accuracy"]["scaled_brier"],
            "log_loss": m["accuracy"]["log_loss"],
            "calibration_slope": m["calibration"]["calibration_slope"],
            "calibration_intercept": m["calibration"]["calibration_intercept"],
            "ece": m["calibration"]["ece"],
            "n_test": m["n"],
        })
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from trs.db.session import session_scope
    from trs.scoring.service import ScoringService

    with session_scope() as session:
        service = ScoringService(session)
        result = service.score_track(
            args.track,
            as_of=dt.date.fromisoformat(args.as_of) if args.as_of else None,
            persist=not args.no_persist,
        )
        _print(result.to_dict())
    return 0


def cmd_connectors(args: argparse.Namespace) -> int:
    from trs.connectors.base import available_connectors, connector_catalogue, get_connector

    if args.list:
        _print(connector_catalogue())
        return 0

    names = [args.name] if args.name else available_connectors()
    report = []
    for name in names:
        try:
            with get_connector(name) as c:
                report.append(c.health().to_dict())
        except Exception as exc:  # noqa: BLE001
            report.append({"connector": name, "status": "error", "detail": str(exc)})
    _print(report)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Seed → cohort → train, in one command. The demo bootstrap."""
    rc = cmd_init_db(args)
    if rc:
        return rc
    print("\n── Seeding demonstration data ──")
    rc = cmd_seed(args)
    if rc:
        return rc
    print("\n── Building cohort ──")
    rc = cmd_build_cohort(args)
    if rc:
        return rc
    print("\n── Training ──")
    return cmd_train(args)


def cmd_facts(args: argparse.Namespace) -> int:
    """Print every checkable number, so documentation can be verified against it."""
    from trs.db.session import session_scope
    from trs.reporting.facts import fact_sheet, fact_sheet_json, render_fact_sheet

    with session_scope() as session:
        if args.json:
            print(fact_sheet_json(session))
        else:
            print(render_fact_sheet(fact_sheet(session)))
    return 0


def cmd_oracle(args: argparse.Namespace) -> int:
    from trs.db.session import session_scope
    from trs.seed.diagnostics import oracle_benchmark

    with session_scope() as session:
        _print(oracle_benchmark(session, cohort_version_id=args.cohort))
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    """Measure how often the published interval contains the true probability.

    Only meaningful on the synthetic demonstration cohort, where the generator
    recorded the probability it actually used. The result establishes that the
    implementation propagates the uncertainty it claims to under a known
    data-generating process; it says nothing about real programmes.
    """
    import datetime as dt

    from sqlalchemy import select

    from trs.db.models.analytics import CohortEntry, CohortVersion
    from trs.db.session import session_scope
    from trs.scoring.service import ScoringService
    from trs.seed.diagnostics import load_ground_truth
    from trs.validation.coverage import interval_coverage

    truth = load_ground_truth()
    if not truth:
        _print({
            "available": False,
            "reason": (
                "No simulation ground-truth file. Coverage against a true "
                "probability can only be measured on synthetic data; re-run "
                "seeding, or accept that this check is unavailable on a real "
                "corpus."
            ),
        })
        return 0

    with session_scope() as session:
        cohort = (
            session.get(CohortVersion, args.cohort) if args.cohort
            else session.scalars(
                select(CohortVersion).order_by(CohortVersion.built_at.desc()).limit(1)
            ).first()
        )
        if cohort is None:
            _print({"available": False, "reason": "No cohort version."})
            return 1

        service = ScoringService(session)
        artifact = service.active_artifact(args.endpoint)
        manifest = (artifact.metrics or {}).get("protocol", {}).get("split_manifest", {})
        evaluation_ids = set(manifest.get("evaluation_row_ids") or [])

        entries = list(session.scalars(
            select(CohortEntry).where(CohortEntry.cohort_version_id == cohort.id)
        ))
        if evaluation_ids:
            entries = [e for e in entries if e.id in evaluation_ids]
            scope = "evaluation partition"
        else:
            scope = "whole cohort (the artifact records no evaluation row ids)"

        predicted, low, high, true_p = [], [], [], []
        skipped = 0
        for entry in entries:
            tp = truth.get(entry.program_indication_id)
            if tp is None:
                skipped += 1
                continue
            result = service.score_track(
                entry.program_indication_id, as_of=entry.as_of_date,
                endpoint=args.endpoint, persist=False, compare_families=False,
                demo_context=True,
            )
            if result.probability is None:
                skipped += 1
                continue
            predicted.append(result.probability)
            low.append(result.interval_low)
            high.append(result.interval_high)
            true_p.append(tp)

        if not predicted:
            _print({
                "available": False,
                "reason": f"No scorable rows ({skipped} skipped).",
            })
            return 1

        payload = interval_coverage(predicted, low, high, true_p).to_dict()
        payload["scope_of_rows"] = scope
        payload["model_version"] = artifact.version
        payload["n_skipped"] = skipped
        payload["generated_at"] = dt.datetime.now(dt.UTC).isoformat()
        _print(payload)
    return 0


def cmd_model_info(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from trs.db.models.analytics import ModelArtifact
    from trs.db.session import session_scope

    with session_scope() as session:
        rows = session.scalars(
            select(ModelArtifact).order_by(ModelArtifact.trained_at.desc())
        ).all()
        _print([
            {
                "version": r.version, "family": r.family, "endpoint": r.endpoint,
                "active": r.is_active, "trained_at": r.trained_at,
                "n_train": r.n_train, "n_test": r.n_test,
                "synthetic": r.trained_on_synthetic,
                "auroc": (r.metrics or {}).get("primary_metrics", {})
                          .get("discrimination", {}).get("auroc", {}).get("value"),
            }
            for r in rows
        ])
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trs",
        description="Preclinical Translational Risk Scorer — command line interface.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init-db", help="Create all tables (dev shortcut).")
    sp.set_defaults(func=cmd_init_db)

    sp = sub.add_parser("seed", help="Generate the synthetic demonstration dataset.")
    sp.add_argument("--programs", type=int, default=600)
    sp.add_argument("--seed", type=int, default=settings.seed_random_state)
    sp.set_defaults(func=cmd_seed)

    sp = sub.add_parser("build-cohort", help="Construct a labelled cohort version.")
    sp.add_argument("--name", default="oncology_ph2_efficacy")
    sp.add_argument("--version", default="v1")
    sp.add_argument("--endpoint", default="ph2_efficacy_failure")
    sp.add_argument("--as-of-rule", default="ind_or_cta_on")
    sp.add_argument("--data-cutoff", default=None)
    sp.set_defaults(func=cmd_build_cohort)

    sp = sub.add_parser("train", help="Train, validate and register a model.")
    sp.add_argument("--cohort", default=None)
    sp.add_argument("--endpoint", default="ph2_efficacy_failure")
    sp.add_argument("--bootstrap", type=int, default=600)
    sp.add_argument("--no-activate", action="store_true")
    sp.add_argument("--skip-interval-check", action="store_true")
    sp.add_argument("--skip-ablation", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("score", help="Score one development track.")
    sp.add_argument("track")
    sp.add_argument("--as-of", default=None)
    sp.add_argument("--no-persist", action="store_true")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("connectors", help="Connector catalogue and health.")
    sp.add_argument("--name", default=None)
    sp.add_argument("--list", action="store_true")
    sp.set_defaults(func=cmd_connectors)

    sp = sub.add_parser("facts", help="Print every checkable number for documentation review.")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_facts)

    sp = sub.add_parser("oracle", help="Maximum attainable performance on the demo cohort.")
    sp.add_argument("--cohort", default=None)
    sp.set_defaults(func=cmd_oracle)

    sp = sub.add_parser(
        "coverage",
        help="Does the published interval contain the simulation's true probability?",
    )
    sp.add_argument("--cohort", default=None)
    sp.add_argument("--endpoint", default="ph2_efficacy_failure")
    sp.set_defaults(func=cmd_coverage)

    sp = sub.add_parser("models", help="List registered model artifacts.")
    sp.set_defaults(func=cmd_model_info)

    sp = sub.add_parser("pipeline", help="init-db → seed → build-cohort → train.")
    sp.add_argument("--programs", type=int, default=600)
    sp.add_argument("--seed", type=int, default=settings.seed_random_state)
    sp.add_argument("--name", default="oncology_ph2_efficacy")
    sp.add_argument("--version", default="v1")
    sp.add_argument("--endpoint", default="ph2_efficacy_failure")
    sp.add_argument("--as-of-rule", default="ind_or_cta_on")
    sp.add_argument("--data-cutoff", default=None)
    sp.add_argument("--cohort", default=None)
    sp.add_argument("--bootstrap", type=int, default=600)
    sp.add_argument("--no-activate", action="store_true")
    sp.add_argument("--skip-interval-check", action="store_true")
    sp.add_argument("--skip-ablation", action="store_true")
    sp.set_defaults(func=cmd_pipeline)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
