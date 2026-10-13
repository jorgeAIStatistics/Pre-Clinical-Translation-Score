"""Model artifacts: serialise, hash, register, reload.

An artifact is the complete, self-contained thing needed to reproduce a score:
the fitted encoder (including its imputation statistics and one-hot level set),
the fitted model, the fitted calibrator, and the provenance of all three. A
score cannot be reproduced from coefficients alone — the encoder is half the
model — so they travel together and are hashed together.

Reproducing a historical prediction
-----------------------------------
Given a ``ScoreRecord``, the chain is: model version → artifact (this module)
→ feature snapshot id → the exact feature vector → the contributing evidence
ids → the raw connector payloads. Every link is persisted, so a number shown to
a committee in March can be fully reconstructed in November, including when the
model has since been retrained.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.config import settings
from trs.db.models.analytics import ModelArtifact
from trs.features.spec import FEATURE_SCHEMA_VERSION
from trs.features.transform import FeatureTransformer, TransformerState
from trs.modeling import MODEL_FAMILIES
from trs.modeling.base import RiskModel
from trs.modeling.calibration import Calibrator

#: Artifact payload format. Bumped to "2" when the split manifest, model
#: status and prediction contract became first-class fields. Format "1" is
#: still readable: :meth:`ModelBundle.from_dict` fills the new fields with
#: denying defaults and records that it did, so an old artifact is labelled
#: rather than rejected — and cannot be served, because `unknown` status is not
#: in SERVING_PERMITTED.
ARTIFACT_FORMAT_VERSION = "2"
READABLE_ARTIFACT_FORMATS: frozenset[str] = frozenset({"1", "2"})


@dataclasses.dataclass(slots=True)
class ModelBundle:
    """Everything required to score a program, in one serialisable object."""

    version: str
    endpoint: str
    family: str
    transformer: FeatureTransformer
    model: RiskModel
    calibrator: Calibrator
    feature_schema_version: str
    cohort_version_id: str | None
    dataset_hash: str
    trained_at: dt.datetime
    metrics: dict[str, Any]
    selection_record: dict[str, Any]
    calibration_record: dict[str, Any]
    limitations: list[str]
    trained_on_synthetic: bool
    library_versions: dict[str, str]
    git_commit: str | None
    n_train: int
    n_calibration: int
    n_test: int
    #: Grouping key used by the hierarchical family, when applicable.
    group_key_name: str | None = None

    # ---- format 2 additions ------------------------------------------------
    # All default so a format-1 artifact can be read back and labelled rather
    # than rejected. The defaults are deliberately the *denying* values: an old
    # artifact is an unknown-status artifact on a legacy protocol, which the
    # eligibility policy refuses rather than waves through.

    #: Named split protocol from :mod:`trs.validation.protocol`.
    protocol: str = "legacy_exploratory"
    #: Full manifest dict, including its hash. Empty for legacy artifacts.
    split_manifest: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Status from :mod:`trs.pipeline.status`. ``unknown`` denies serving.
    status: str = "unknown"
    status_record: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Which functional this artifact's metrics describe, and which the server
    #: must therefore return. See :mod:`trs.scoring.contract`.
    prediction_functional: str = "unknown"
    prediction_contract_version: str | None = None

    @property
    def split_manifest_hash(self) -> str | None:
        return self.split_manifest.get("manifest_hash")

    @property
    def permits_deployment_claim(self) -> bool:
        return bool(self.split_manifest.get("permits_deployment_claim", False))

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "version": self.version,
            "endpoint": self.endpoint,
            "family": self.family,
            "feature_schema_version": self.feature_schema_version,
            "cohort_version_id": self.cohort_version_id,
            "dataset_hash": self.dataset_hash,
            "trained_at": self.trained_at.isoformat(),
            "transformer": self.transformer.state.to_dict(),
            "model": self.model.to_state(),
            "calibrator": self.calibrator.to_dict(),
            "metrics": self.metrics,
            "selection_record": self.selection_record,
            "calibration_record": self.calibration_record,
            "limitations": self.limitations,
            "trained_on_synthetic": self.trained_on_synthetic,
            "library_versions": self.library_versions,
            "git_commit": self.git_commit,
            "n_train": self.n_train,
            "n_calibration": self.n_calibration,
            "n_test": self.n_test,
            "group_key_name": self.group_key_name,
            "protocol": self.protocol,
            "split_manifest": self.split_manifest,
            "status": self.status,
            "status_record": self.status_record,
            "prediction_functional": self.prediction_functional,
            "prediction_contract_version": self.prediction_contract_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelBundle:
        fmt = str(payload.get("artifact_format_version"))
        if fmt not in READABLE_ARTIFACT_FORMATS:
            raise ValueError(
                f"Artifact format {payload.get('artifact_format_version')!r} is not "
                f"readable by this build (readable: "
                f"{sorted(READABLE_ARTIFACT_FORMATS)})."
            )
        legacy = fmt == "1"
        if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Artifact was trained under feature schema "
                f"{payload['feature_schema_version']!r} but this build declares "
                f"{FEATURE_SCHEMA_VERSION!r}. Scoring across a schema change "
                "would silently misalign columns; retrain instead."
            )
        family = payload["family"]
        model_cls = MODEL_FAMILIES[family]
        return cls(
            version=payload["version"],
            endpoint=payload["endpoint"],
            family=family,
            transformer=FeatureTransformer(TransformerState.from_dict(payload["transformer"])),
            model=model_cls.from_state(payload["model"]),
            calibrator=Calibrator.from_dict(payload["calibrator"]),
            feature_schema_version=payload["feature_schema_version"],
            cohort_version_id=payload.get("cohort_version_id"),
            dataset_hash=payload["dataset_hash"],
            trained_at=dt.datetime.fromisoformat(payload["trained_at"]),
            metrics=payload.get("metrics", {}),
            selection_record=payload.get("selection_record", {}),
            calibration_record=payload.get("calibration_record", {}),
            limitations=payload.get("limitations", []),
            trained_on_synthetic=payload.get("trained_on_synthetic", False),
            library_versions=payload.get("library_versions", {}),
            git_commit=payload.get("git_commit"),
            n_train=payload.get("n_train", 0),
            n_calibration=payload.get("n_calibration", 0),
            n_test=payload.get("n_test", 0),
            group_key_name=payload.get("group_key_name"),
            protocol=payload.get(
                "protocol", "legacy_exploratory" if legacy else "unknown"
            ),
            split_manifest=payload.get("split_manifest", {}),
            status=payload.get("status", "unknown"),
            status_record=payload.get("status_record", {}) or (
                {
                    "status": "unknown",
                    "may_serve": False,
                    "rationale": (
                        "Artifact written in payload format 1, before the split "
                        "manifest, the model-status policy and the prediction "
                        "contract existed. Its reported metrics were produced by "
                        "a procedure that selected the model family on the same "
                        "partition it then reported, so they are exploratory. "
                        "Retrain to obtain a servable artifact."
                    ),
                }
                if legacy else {}
            ),
            prediction_functional=payload.get(
                "prediction_functional",
                "legacy_ambiguous" if legacy else "unknown",
            ),
            prediction_contract_version=payload.get("prediction_contract_version"),
        )

    # ---- disk --------------------------------------------------------------

    def path(self, root: Path | None = None) -> Path:
        root = Path(root or settings.artifact_dir)
        return root / f"{self.version}.json"

    def save(self, root: Path | None = None) -> tuple[Path, str]:
        path = self.path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        path.write_text(blob)
        return path, hashlib.sha256(blob.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ─────────────────────────────────────────────────────────────────────────────
# Environment capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_library_versions() -> dict[str, str]:
    """Record the numerical stack. A model refit under a different BLAS or a
    different scikit-learn is not bit-identical, and the artifact should say
    which one produced it."""
    import platform

    versions = {"python": platform.python_version()}
    for module in ("numpy", "scipy", "pandas", "sklearn", "statsmodels"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            versions[module] = "unavailable"
    return versions


def capture_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def register_artifact(
    session: Session,
    bundle: ModelBundle,
    *,
    make_active: bool = False,
    display_name: str | None = None,
    description: str | None = None,
) -> ModelArtifact:
    """Persist the artifact to disk and record it in the model registry."""
    path, sha = bundle.save()

    coefficients = None
    if hasattr(bundle.model, "coefficient_table"):
        coefficients = {
            "columns": bundle.model.coefficient_table(),
            "intercept": float(getattr(bundle.model, "intercept_", 0.0)),
            "note": (
                "Coefficients are on the standardised design-matrix scale: each "
                "is the change in log-odds per one training standard deviation "
                "of that column, holding the others fixed."
            ),
        }

    row = ModelArtifact(
        version=bundle.version,
        family=bundle.family,
        endpoint=bundle.endpoint,
        display_name=display_name or f"{bundle.family} · {bundle.version}",
        description=description,
        feature_schema_version=bundle.feature_schema_version,
        cohort_version_id=bundle.cohort_version_id,
        dataset_hash=bundle.dataset_hash,
        hyperparameters=bundle.model.hyperparameters,
        coefficients=coefficients,
        calibration=bundle.calibration_record,
        metrics=bundle.metrics,
        subgroup_metrics=bundle.metrics.get("subgroup_performance"),
        selection_record=bundle.selection_record,
        trained_at=bundle.trained_at,
        n_train=bundle.n_train,
        n_calibration=bundle.n_calibration,
        n_test=bundle.n_test,
        artifact_path=str(path.relative_to(settings.artifact_dir))
        if path.is_relative_to(settings.artifact_dir) else str(path),
        artifact_sha256=sha,
        is_active=False,
        trained_on_synthetic=bundle.trained_on_synthetic,
        limitations={"items": bundle.limitations},
        git_commit=bundle.git_commit,
        library_versions=bundle.library_versions,
    )
    session.add(row)
    session.flush()

    if make_active:
        activate_artifact(session, bundle.version)
    return row


def activate_artifact(session: Session, version: str) -> ModelArtifact:
    """Make one artifact the serving default, deactivating any other.

    Exactly one artifact per endpoint is active at a time; scores record the
    version they used, so flipping the pointer never rewrites history.
    """
    target = session.scalars(
        select(ModelArtifact).where(ModelArtifact.version == version)
    ).first()
    if target is None:
        raise KeyError(f"No model artifact with version {version!r}")
    for row in session.scalars(
        select(ModelArtifact).where(
            ModelArtifact.endpoint == target.endpoint, ModelArtifact.is_active.is_(True)
        )
    ):
        row.is_active = False
    target.is_active = True
    session.flush()
    return target


def get_active_artifact(session: Session, endpoint: str) -> ModelArtifact | None:
    return session.scalars(
        select(ModelArtifact).where(
            ModelArtifact.endpoint == endpoint, ModelArtifact.is_active.is_(True)
        )
    ).first()


_BUNDLE_CACHE: dict[str, ModelBundle] = {}


def load_bundle(session: Session, version: str, *, use_cache: bool = True) -> ModelBundle:
    """Load an artifact by version, verifying its on-disk integrity."""
    if use_cache and version in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[version]

    row = session.scalars(
        select(ModelArtifact).where(ModelArtifact.version == version)
    ).first()
    if row is None or not row.artifact_path:
        raise KeyError(f"No artifact on disk for model version {version!r}")

    path = Path(row.artifact_path)
    if not path.is_absolute():
        path = Path(settings.artifact_dir) / path

    blob = path.read_text()
    digest = hashlib.sha256(blob.encode()).hexdigest()
    if row.artifact_sha256 and digest != row.artifact_sha256:
        raise RuntimeError(
            f"Artifact {version!r} on disk does not match its registered hash. "
            "The file has been modified since training; refusing to score with it."
        )

    bundle = ModelBundle.from_dict(json.loads(blob))
    if use_cache:
        _BUNDLE_CACHE[version] = bundle
    return bundle


def clear_bundle_cache() -> None:
    _BUNDLE_CACHE.clear()


def next_version(session: Session, endpoint: str, prefix: str = "ptrs") -> str:
    """Monotonic ``prefix-endpointinitials-NNN`` version string."""
    existing = session.scalars(
        select(ModelArtifact.version).where(ModelArtifact.endpoint == endpoint)
    ).all()
    n = len(existing) + 1
    tag = "".join(part[0] for part in endpoint.split("_"))[:4]
    return f"{prefix}-{tag}-{n:03d}"
