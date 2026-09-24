"""Entity resolution: linking an incoming name to a canonical entity.

Why this is a separate, auditable layer
----------------------------------------
A therapeutic program is referred to by a discovery code, then a development
code, then an INN, then a brand name, and it changes sponsor once or twice on
the way. Publications cite the discovery code; the trial registry cites the
development code. Treating those as four programs would inflate the cohort,
break historical comparators, and — worst — let the *same biology* appear in
both the training and test folds under different names, which silently inflates
every reported metric.

So resolution is not a helper function. It is a decision with consequences, and
every decision is written to ``entity_resolution_decision`` with the method and
the score that produced it. Nothing is ever merged silently: a spurious merge
can be found, explained and reversed.

The precision/recall asymmetry that shapes every threshold here
----------------------------------------------------------------
A **false merge** puts two different assets into one program. That program then
carries a mixture of two evidence bases, and its grouping key wrongly binds two
unrelated rows together in cross-validation. A **missed merge** leaves one asset
as two programs — the cohort is slightly larger, comparators are slightly
weaker, and grouping is *more* conservative than it needed to be.

The first error corrupts results. The second only wastes them. Every threshold
below is therefore set to prefer a missed merge, and anything in the ambiguous
band is returned for human review rather than decided.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Sequence
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.db.base import utcnow
from trs.db.models.core import (
    Compound,
    CompoundAlias,
    EntityResolutionDecision,
    Indication,
    IndicationAlias,
    Program,
    ProgramAlias,
    Sponsor,
    SponsorAlias,
    Target,
    TargetAlias,
)
from trs.ontology.enums import EntityMatchMethod
from trs.resolution.normalise import (
    jaccard,
    normalise_compound,
    normalise_general,
    normalise_sponsor,
    normalise_target,
    normalised_edit_similarity,
    token_set,
)

RESOLVER_VERSION = "res-1.0.0"

#: At or above this score a fuzzy match is accepted automatically.
AUTO_ACCEPT_SCORE = 0.92

#: Between this and AUTO_ACCEPT_SCORE a candidate is returned as `needs_review`
#: — plausible, but not strong enough to merge two assets without a human.
REVIEW_SCORE = 0.80

EntityType = Literal["program", "compound", "target", "sponsor", "indication"]

#: Per-entity normaliser. Compound and target keys drop separators entirely;
#: sponsor keys drop legal forms; everything else uses the generic rule.
_NORMALISERS = {
    "program": normalise_compound,
    "compound": normalise_compound,
    "target": normalise_target,
    "sponsor": normalise_sponsor,
    "indication": normalise_general,
}

_ALIAS_MODELS = {
    "program": (Program, ProgramAlias, "program_id"),
    "compound": (Compound, CompoundAlias, "compound_id"),
    "target": (Target, TargetAlias, "target_id"),
    "sponsor": (Sponsor, SponsorAlias, "sponsor_id"),
    "indication": (Indication, IndicationAlias, "indication_id"),
}

#: Identifier fields that constitute proof of identity when they match. A shared
#: InChIKey or ChEMBL id is not evidence of similarity — it is the same molecule.
_STRONG_IDENTIFIERS: dict[str, tuple[str, ...]] = {
    "compound": ("inchikey", "chembl_id", "unii", "pubchem_cid", "inn"),
    "target": ("hgnc_id", "ensembl_gene_id", "uniprot_id", "chembl_target_id"),
    "indication": ("ontology_id",),
    "program": ("code",),
    "sponsor": (),
}


@dataclasses.dataclass(slots=True)
class Resolution:
    """The outcome of one resolution attempt, decided or not."""

    entity_type: str
    query: str
    normalised: str
    entity_id: str | None
    method: str
    score: float
    status: Literal["resolved", "needs_review", "unresolved"]
    #: Runner-up candidates, for the review queue.
    alternatives: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    rationale: str = ""

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class EntityResolver:
    """Resolves names to canonical entities, recording every decision.

    A single instance caches its alias index for the life of the object, so a
    bulk ingestion run does one query per entity type rather than one per name.
    Call :meth:`invalidate` after writing new entities.
    """

    def __init__(self, session: Session, *, record_decisions: bool = True) -> None:
        self.session = session
        self.record_decisions = record_decisions
        self._alias_index: dict[str, dict[str, str]] = {}
        self._name_index: dict[str, list[tuple[str, str]]] = {}

    # ---- index ------------------------------------------------------------

    def invalidate(self) -> None:
        """Drop cached indexes. Call after inserting entities or aliases."""
        self._alias_index.clear()
        self._name_index.clear()

    def _aliases(self, entity_type: str) -> dict[str, str]:
        """``{normalised_alias: entity_id}`` for one entity type."""
        if entity_type in self._alias_index:
            return self._alias_index[entity_type]
        _, alias_model, fk = _ALIAS_MODELS[entity_type]
        index: dict[str, str] = {}
        for norm, eid in self.session.execute(
            select(alias_model.alias_normalised, getattr(alias_model, fk))
        ).all():
            # First writer wins. A later duplicate alias pointing elsewhere is a
            # genuine conflict and must not silently reassign an existing link.
            index.setdefault(norm, eid)
        self._alias_index[entity_type] = index
        return index

    def _names(self, entity_type: str) -> list[tuple[str, str]]:
        """``[(normalised_name, entity_id)]`` for fuzzy candidate generation."""
        if entity_type in self._name_index:
            return self._name_index[entity_type]
        model, _, _ = _ALIAS_MODELS[entity_type]
        field = {
            "program": Program.name, "compound": Compound.preferred_name,
            "target": Target.symbol, "sponsor": Sponsor.name,
            "indication": Indication.name,
        }[entity_type]
        norm = _NORMALISERS[entity_type]
        rows = [
            (norm(name), eid)
            for name, eid in self.session.execute(select(field, model.id)).all()
            if name
        ]
        self._name_index[entity_type] = rows
        return rows

    # ---- resolution --------------------------------------------------------

    def resolve(
        self,
        entity_type: EntityType,
        name: str | None,
        *,
        identifiers: dict[str, str] | None = None,
        allow_fuzzy: bool = True,
    ) -> Resolution:
        """Resolve one name, in descending order of evidential strength.

        1. **Strong identifier.** A matching InChIKey, HGNC id or ontology id is
           proof of identity, not a similarity signal.
        2. **Registered alias.** A previously recorded designation for a known
           entity — including a merge a human already approved.
        3. **Normalised name.** An exact match after normalisation.
        4. **Fuzzy.** Blocked on shared tokens, scored, and thresholded. Only
           reached when the caller allows it.
        """
        norm = _NORMALISERS[entity_type]
        key = norm(name)
        if not key and not identifiers:
            return self._record(Resolution(
                entity_type=entity_type, query=name or "", normalised="",
                entity_id=None, method=EntityMatchMethod.UNRESOLVED, score=0.0,
                status="unresolved", rationale="Empty name and no identifiers supplied.",
            ))

        if identifiers:
            hit = self._by_identifier(entity_type, identifiers)
            if hit is not None:
                entity_id, field, value = hit
                return self._record(Resolution(
                    entity_type=entity_type, query=name or "", normalised=key,
                    entity_id=entity_id,
                    method=EntityMatchMethod.EXACT_IDENTIFIER, score=1.0,
                    status="resolved",
                    rationale=f"Matched on {field}={value}, which is an identity claim.",
                ))

        alias_hit = self._aliases(entity_type).get(key)
        if alias_hit:
            return self._record(Resolution(
                entity_type=entity_type, query=name or "", normalised=key,
                entity_id=alias_hit,
                method=EntityMatchMethod.REGISTERED_ALIAS, score=1.0,
                status="resolved",
                rationale="Matched a registered alias for this entity.",
            ))

        for candidate_key, entity_id in self._names(entity_type):
            if candidate_key == key:
                return self._record(Resolution(
                    entity_type=entity_type, query=name or "", normalised=key,
                    entity_id=entity_id,
                    method=EntityMatchMethod.NORMALISED_NAME, score=1.0,
                    status="resolved",
                    rationale="Exact match after normalisation.",
                ))

        if not allow_fuzzy:
            return self._record(Resolution(
                entity_type=entity_type, query=name or "", normalised=key,
                entity_id=None, method=EntityMatchMethod.UNRESOLVED, score=0.0,
                status="unresolved", rationale="No exact match; fuzzy matching disabled.",
            ))

        return self._record(self._fuzzy(entity_type, name or "", key))

    # ---- strategies --------------------------------------------------------

    def _by_identifier(
        self, entity_type: str, identifiers: dict[str, str]
    ) -> tuple[str, str, str] | None:
        model, _, _ = _ALIAS_MODELS[entity_type]
        for field in _STRONG_IDENTIFIERS.get(entity_type, ()):
            value = identifiers.get(field)
            if not value or not hasattr(model, field):
                continue
            row = self.session.scalars(
                select(model).where(getattr(model, field) == value).limit(1)
            ).first()
            if row is not None:
                return row.id, field, value
        return None

    def _fuzzy(self, entity_type: str, name: str, key: str) -> Resolution:
        """Blocked fuzzy match, scored and thresholded conservatively.

        Blocking on a shared token keeps this linear in practice: a name with no
        token in common with any candidate is not scored at all. The score
        combines token overlap with a bounded edit similarity, so neither a
        reordered multi-word name nor a single transposed character is missed,
        while unrelated names score near zero on both.
        """
        query_tokens = token_set(name)
        scored: list[tuple[float, str, str]] = []

        for candidate_key, entity_id in self._names(entity_type):
            if not candidate_key:
                continue
            candidate_tokens = token_set(candidate_key)
            overlap = jaccard(query_tokens, candidate_tokens)
            # Block: require either a shared token or a close length match.
            if overlap == 0.0 and abs(len(candidate_key) - len(key)) > 3:
                continue
            edit = normalised_edit_similarity(key, candidate_key)
            if edit == 0.0 and overlap == 0.0:
                continue
            score = max(edit, 0.5 * edit + 0.5 * overlap)
            scored.append((score, entity_id, candidate_key))

        if not scored:
            return Resolution(
                entity_type=entity_type, query=name, normalised=key,
                entity_id=None, method=EntityMatchMethod.UNRESOLVED, score=0.0,
                status="unresolved", rationale="No candidate shared a token or a length band.",
            )

        scored.sort(reverse=True)
        best_score, best_id, best_key = scored[0]
        alternatives = [
            {"entity_id": eid, "key": ckey, "score": round(sc, 4)}
            for sc, eid, ckey in scored[1:4]
        ]

        # An ambiguous best — one barely ahead of the runner-up — is not a
        # decision. Merging on a 0.01 margin is how two different assets become
        # one program.
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= AUTO_ACCEPT_SCORE and (best_score - runner_up) >= 0.05:
            return Resolution(
                entity_type=entity_type, query=name, normalised=key,
                entity_id=best_id, method=EntityMatchMethod.FUZZY_STRING,
                score=best_score, status="resolved", alternatives=alternatives,
                rationale=(
                    f"Fuzzy match to {best_key!r} at {best_score:.3f}, clear of the "
                    f"runner-up at {runner_up:.3f}."
                ),
            )
        if best_score >= REVIEW_SCORE:
            return Resolution(
                entity_type=entity_type, query=name, normalised=key,
                entity_id=None, method=EntityMatchMethod.FUZZY_STRING,
                score=best_score, status="needs_review", alternatives=alternatives,
                rationale=(
                    f"Best candidate {best_key!r} scored {best_score:.3f}"
                    + (
                        f", too close to the runner-up at {runner_up:.3f}"
                        if best_score >= AUTO_ACCEPT_SCORE else ""
                    )
                    + ". Held for human review rather than merged."
                ),
            )
        return Resolution(
            entity_type=entity_type, query=name, normalised=key,
            entity_id=None, method=EntityMatchMethod.UNRESOLVED,
            score=best_score, status="unresolved", alternatives=alternatives,
            rationale=f"Best candidate scored {best_score:.3f}, below the review threshold.",
        )

    # ---- audit -------------------------------------------------------------

    def _record(self, resolution: Resolution) -> Resolution:
        if not self.record_decisions:
            return resolution
        self.session.add(EntityResolutionDecision(
            entity_type=resolution.entity_type,
            canonical_id=resolution.entity_id or "",
            candidate_key=resolution.normalised or resolution.query,
            method=resolution.method,
            score=resolution.score,
            accepted=resolution.resolved,
            decided_at=utcnow(),
            decided_by=f"auto:{RESOLVER_VERSION}",
            evidence={
                "query": resolution.query,
                "status": resolution.status,
                "rationale": resolution.rationale,
                "alternatives": resolution.alternatives,
            },
        ))
        return resolution

    # ---- alias registration ------------------------------------------------

    def register_alias(
        self,
        entity_type: EntityType,
        entity_id: str,
        alias: str,
        *,
        alias_type: str = "synonym",
        method: str = EntityMatchMethod.HUMAN_ASSERTED,
        confidence: float = 1.0,
        available_at: dt.date | None = None,
        source_id: str | None = None,
    ) -> bool:
        """Record a new designation for a known entity.

        Returns False without writing if the alias already points somewhere —
        reassigning an existing alias would silently move evidence between
        entities, which is the one thing this layer exists to prevent. The
        conflict is recorded as a rejected decision so it surfaces for review.
        """
        norm = _NORMALISERS[entity_type](alias)
        if not norm:
            return False

        existing = self._aliases(entity_type).get(norm)
        if existing is not None:
            if existing != entity_id and self.record_decisions:
                self.session.add(EntityResolutionDecision(
                    entity_type=entity_type, canonical_id=entity_id,
                    candidate_key=norm, method=method, score=confidence,
                    accepted=False, decided_at=utcnow(),
                    decided_by=f"auto:{RESOLVER_VERSION}",
                    evidence={
                        "conflict": (
                            f"Alias {alias!r} already resolves to {existing}; refusing "
                            f"to reassign it to {entity_id}."
                        ),
                    },
                ))
            return False

        _, alias_model, fk = _ALIAS_MODELS[entity_type]
        self.session.add(alias_model(**{
            fk: entity_id, "alias": alias, "alias_normalised": norm,
            "alias_type": alias_type, "match_method": method,
            "match_confidence": confidence, "available_at": available_at,
            "source_id": source_id, "is_primary": False,
        }))
        self._aliases(entity_type)[norm] = entity_id
        return True

    # ---- bulk --------------------------------------------------------------

    def resolve_many(
        self, entity_type: EntityType, names: Sequence[str]
    ) -> list[Resolution]:
        return [self.resolve(entity_type, n) for n in names]

    def review_queue(self, limit: int = 100) -> list[dict[str, Any]]:
        """Decisions held for a human: ambiguous or rejected links."""
        rows = self.session.scalars(
            select(EntityResolutionDecision)
            .where(EntityResolutionDecision.accepted.is_(False))
            .order_by(EntityResolutionDecision.decided_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": r.id, "entity_type": r.entity_type,
                "candidate_key": r.candidate_key, "method": r.method,
                "score": r.score, "decided_at": r.decided_at.isoformat(),
                "evidence": r.evidence,
            }
            for r in rows
        ]
