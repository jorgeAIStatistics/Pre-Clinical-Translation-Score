"""Entity resolution.

The property under test throughout is the precision/recall asymmetry stated in
:mod:`trs.resolution.resolver`: a false merge corrupts results (two assets
become one program, and that program's evidence and grouping key are both
wrong), whereas a missed merge only wastes them. Every threshold is set to
prefer the second, and these tests fail if that preference ever inverts.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select

from trs.db.models.core import (
    Compound,
    CompoundAlias,
    EntityResolutionDecision,
    Program,
    ProgramAlias,
    Sponsor,
    Target,
    TargetAlias,
)
from trs.ontology.enums import EntityMatchMethod
from trs.resolution import EntityResolver
from trs.resolution.normalise import (
    jaccard,
    normalise_compound,
    normalise_general,
    normalise_sponsor,
    normalise_target,
    normalised_edit_similarity,
    token_set,
)

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "variants",
    [
        ("ABT-199", "ABT199", "abt 199", "  ABT-199  "),
        ("GDC-0199", "gdc0199", "GDC 0199"),
    ],
)
def test_compound_separator_variants_share_a_key(variants: tuple[str, ...]) -> None:
    """If this fails, one molecule enters the cohort as several programs.

    Development codes are written with and without separators across sources.
    Treating them as different assets inflates the cohort and lets the same
    biology land on both sides of a train/test split.
    """
    keys = {normalise_compound(v) for v in variants}
    assert len(keys) == 1, f"{variants} produced {keys}"


def test_compound_normalisation_does_not_merge_different_numbers() -> None:
    """If this fails, two different molecules become one program.

    This is the error the whole layer is tuned to avoid: dropping digits or
    over-collapsing would make ABT-199 and ABT-1990 the same asset.
    """
    assert normalise_compound("ABT-199") != normalise_compound("ABT-1990")
    assert normalise_compound("BMS-986165") != normalise_compound("BMS-986156")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("EGFR", "egfr"),
        ("human EGFR", "egfr"),
        ("hsaKRAS", "kras"),
        ("EGFR protein", "egfr"),
        ("TNF-α", "tnfalpha"),
        ("TNF-alpha", "tnfalpha"),
    ],
)
def test_target_symbols_fold_multichar_species_prefixes_and_greek(
    raw: str, expected: str
) -> None:
    """If this fails, the same gene is tracked as several targets.

    Sources write the same symbol with a species qualifier, a trailing
    "protein"/"receptor" or a Greek glyph. Each variant would otherwise get its
    own target row and its own target-validation evidence.
    """
    assert normalise_target(raw) == expected


def test_single_letter_species_prefixes_are_deliberately_not_stripped() -> None:
    """If this fails, real gene symbols are destroyed by prefix stripping.

    This documents an accepted missed merge. Stripping a leading h/m/r turned
    MET into "et", RET into "et" — colliding two distinct targets — and MYC
    into "yc". Those are false merges, which corrupt every feature derived from
    the target. A missed merge on "hEGFR" only costs a human one alias
    registration, so the conservative rule wins.
    """
    assert normalise_target("hEGFR") == "hegfr"
    assert normalise_target("hEGFR") != normalise_target("EGFR")


def test_target_normalisation_keeps_genuinely_different_symbols_apart() -> None:
    """If this fails, unrelated targets merge and their evidence pools together."""
    assert normalise_target("CDK4") != normalise_target("CDK6")
    assert normalise_target("MEK1") != normalise_target("MEK2")
    # Short symbols must survive intact — these are the ones prefix stripping broke.
    assert normalise_target("MET") == "met"
    assert normalise_target("RET") == "ret"
    assert normalise_target("MYC") == "myc"
    assert normalise_target("MTOR") == normalise_target("mTOR") == "mtor"
    assert normalise_target("MET") != normalise_target("RET")


@pytest.mark.parametrize(
    "variants",
    [
        ("Kestrel Ridge Therapeutics", "Kestrel Ridge Therapeutics, Inc.",
         "Kestrel Ridge Therapeutics Ltd", "KESTREL RIDGE THERAPEUTICS LLC"),
        ("Northreach Pharmaceuticals", "Northreach Pharmaceuticals Co., Ltd.",
         "Northreach Pharma GmbH"),
    ],
)
def test_sponsor_legal_forms_are_stripped(variants: tuple[str, ...]) -> None:
    """If this fails, one developer appears as several sponsors.

    A legal-form suffix records a jurisdiction, not an identity. Sponsor is a
    grouping key in the validation report, so splitting one company into four
    understates how concentrated the evidence base is.
    """
    keys = {normalise_sponsor(v) for v in variants}
    assert len(keys) == 1, f"{variants} produced {keys}"


def test_sponsor_normalisation_keeps_different_companies_apart() -> None:
    """If this fails, unrelated sponsors merge."""
    assert normalise_sponsor("Kestrel Ridge Therapeutics") != normalise_sponsor(
        "Kestrel Valley Therapeutics"
    )


def test_normalisers_are_total_on_empty_input() -> None:
    """If this fails, an unnamed record raises instead of simply not resolving."""
    for fn in (normalise_general, normalise_compound, normalise_target, normalise_sponsor):
        assert fn(None) == ""
        assert fn("") == ""
        assert fn("   ") == ""


def test_edit_similarity_is_bounded_and_symmetric() -> None:
    """If this fails, fuzzy scores leave [0, 1] and the thresholds stop meaning anything."""
    for a, b in [("abt199", "abt1990"), ("kras", "nras"), ("", "x"), ("same", "same")]:
        s1, s2 = normalised_edit_similarity(a, b), normalised_edit_similarity(b, a)
        assert 0.0 <= s1 <= 1.0
        assert s1 == pytest.approx(s2)
    assert normalised_edit_similarity("same", "same") == 1.0


def test_jaccard_handles_empty_sets() -> None:
    """If this fails, blocking raises on a name that normalises to nothing."""
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert jaccard(token_set("alpha beta"), token_set("beta alpha")) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Resolution strategies, in order of evidential strength
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def resolver_fixture(db_session):
    """A minimal universe: two distinct programs and their compounds."""
    target = Target(symbol="KRAS", name="KRAS proto-oncogene", target_class="gtpase")
    db_session.add(target)
    db_session.flush()
    db_session.add(TargetAlias(
        target_id=target.id, alias="KRAS", alias_normalised=normalise_target("KRAS"),
        alias_type="symbol", match_method=EntityMatchMethod.EXACT_IDENTIFIER,
        match_confidence=1.0, is_primary=True,
    ))

    compound = Compound(
        preferred_name="ABT-199", modality="small_molecule",
        inchikey="OGZPYBSJPMXTAG-UHFFFAOYSA-N", chembl_id="CHEMBL3137309",
    )
    db_session.add(compound)
    db_session.flush()
    db_session.add(CompoundAlias(
        compound_id=compound.id, alias="ABT-199",
        alias_normalised=normalise_compound("ABT-199"), alias_type="development_code",
        match_method=EntityMatchMethod.EXACT_IDENTIFIER, is_primary=True,
    ))

    programs = []
    for name in ("DEMO-KRAS-001", "DEMO-KRAS-002"):
        p = Program(name=name, code=name, modality="small_molecule", primary_target_id=target.id)
        db_session.add(p)
        db_session.flush()
        db_session.add(ProgramAlias(
            program_id=p.id, alias=name, alias_normalised=normalise_compound(name),
            alias_type="development_code",
            match_method=EntityMatchMethod.EXACT_IDENTIFIER, is_primary=True,
        ))
        programs.append(p)

    sponsor = Sponsor(
        name="Kestrel Ridge Therapeutics",
        name_normalised=normalise_sponsor("Kestrel Ridge Therapeutics"),
        sponsor_type="biotech_small",
    )
    db_session.add(sponsor)
    db_session.flush()
    return db_session, EntityResolver(db_session), target, compound, programs, sponsor


def test_strong_identifier_beats_everything(resolver_fixture) -> None:
    """If this fails, a matching InChIKey is treated as a similarity hint.

    A shared InChIKey is not evidence that two records are alike; it is the
    same molecule. It must resolve even when the supplied name is unrelated.
    """
    _, resolver, _, compound, _, _ = resolver_fixture
    res = resolver.resolve(
        "compound", "a completely unrelated name",
        identifiers={"inchikey": compound.inchikey},
    )
    assert res.resolved
    assert res.entity_id == compound.id
    assert res.method == EntityMatchMethod.EXACT_IDENTIFIER
    assert res.score == 1.0


def test_registered_alias_resolves_without_fuzzy_matching(resolver_fixture) -> None:
    """If this fails, previously approved links stop being honoured.

    A recorded alias is a decision someone already made. Re-deriving it by
    string similarity each time would make resolution non-deterministic as the
    entity set grows.
    """
    _, resolver, _, _, programs, _ = resolver_fixture
    res = resolver.resolve("program", "DEMO-KRAS-001")
    assert res.resolved
    assert res.entity_id == programs[0].id
    assert res.method == EntityMatchMethod.REGISTERED_ALIAS


def test_separator_variant_of_a_registered_alias_still_resolves(resolver_fixture) -> None:
    """If this fails, 'DEMO KRAS 001' becomes a second program."""
    _, resolver, _, _, programs, _ = resolver_fixture
    for variant in ("demo kras 001", "DEMOKRAS001", "Demo-Kras-001"):
        res = resolver.resolve("program", variant)
        assert res.resolved, variant
        assert res.entity_id == programs[0].id, variant


def test_near_miss_is_held_for_review_not_merged(resolver_fixture) -> None:
    """If this fails, two different assets are merged on a string coincidence.

    'DEMO-KRAS-003' is one character from two existing programs. Merging it
    into either would be a false merge; the resolver must decline and say so.
    """
    _, resolver, _, _, _, _ = resolver_fixture
    res = resolver.resolve("program", "DEMO-KRAS-003")
    assert not res.resolved, (
        f"resolved to {res.entity_id} at {res.score:.3f} — a one-character "
        "difference must never auto-merge"
    )
    assert res.status in {"needs_review", "unresolved"}


def test_ambiguous_best_candidate_is_refused_even_when_it_scores_high(
    resolver_fixture,
) -> None:
    """If this fails, a tie between two candidates is broken arbitrarily.

    When the best and second-best candidates are within a hair of each other,
    picking the best is a coin flip with a corrupted cohort on one face.
    """
    _, resolver, _, _, _, _ = resolver_fixture
    # Equidistant from DEMO-KRAS-001 and DEMO-KRAS-002.
    res = resolver.resolve("program", "DEMO-KRAS-00")
    if res.resolved:
        pytest.fail(
            f"auto-merged an ambiguous name into {res.entity_id} "
            f"(score {res.score:.3f}, alternatives {res.alternatives})"
        )


def test_unrelated_name_does_not_resolve(resolver_fixture) -> None:
    """If this fails, every unknown name is absorbed into some existing entity."""
    _, resolver, _, _, _, _ = resolver_fixture
    res = resolver.resolve("program", "ZZZ-UNRELATED-9999")
    assert not res.resolved
    assert res.entity_id is None


def test_fuzzy_can_be_disabled_for_strict_callers(resolver_fixture) -> None:
    """If this fails, an ingestion path that wants exact-only gets fuzzy matches."""
    _, resolver, _, _, _, _ = resolver_fixture
    res = resolver.resolve("program", "DEMO-KRAS-00X", allow_fuzzy=False)
    assert not res.resolved
    assert res.method == EntityMatchMethod.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# Audit trail and alias registration
# ─────────────────────────────────────────────────────────────────────────────

def test_every_resolution_attempt_is_recorded(resolver_fixture) -> None:
    """If this fails, a bad merge cannot be traced or reversed.

    The audit row is the only thing that makes an automated merge reviewable
    after the fact.
    """
    session, resolver, _, _, _, _ = resolver_fixture
    before = session.scalar(select(func.count()).select_from(EntityResolutionDecision))
    resolver.resolve("program", "DEMO-KRAS-001")
    resolver.resolve("program", "ZZZ-UNRELATED-9999")
    session.flush()
    after = session.scalar(select(func.count()).select_from(EntityResolutionDecision))
    assert after == before + 2, "both the hit and the miss must be recorded"


def test_recorded_decision_carries_method_score_and_rationale(resolver_fixture) -> None:
    """If this fails, the audit row exists but explains nothing."""
    session, resolver, _, _, programs, _ = resolver_fixture
    resolver.resolve("program", "DEMO-KRAS-001")
    session.flush()
    row = session.scalars(
        select(EntityResolutionDecision).order_by(EntityResolutionDecision.decided_at.desc())
    ).first()
    assert row.canonical_id == programs[0].id
    assert row.accepted is True
    assert row.method == EntityMatchMethod.REGISTERED_ALIAS
    assert row.decided_by.startswith("auto:")
    assert row.evidence and row.evidence.get("rationale")


def test_alias_conflict_is_refused_and_recorded(resolver_fixture) -> None:
    """If this fails, an alias silently moves evidence between two programs.

    Re-pointing an existing alias is the single most destructive operation in
    this layer: it reassigns every record that resolves through that name.
    """
    session, resolver, _, _, programs, _ = resolver_fixture
    accepted = resolver.register_alias("program", programs[1].id, "DEMO-KRAS-001")
    assert accepted is False, "an alias already bound to another program was reassigned"

    session.flush()
    conflicts = session.scalars(
        select(EntityResolutionDecision).where(
            EntityResolutionDecision.accepted.is_(False),
            EntityResolutionDecision.candidate_key == normalise_compound("DEMO-KRAS-001"),
        )
    ).all()
    assert conflicts, "the refused reassignment must be recorded for review"
    assert "already resolves to" in conflicts[0].evidence["conflict"]

    # And the original binding is untouched.
    assert resolver.resolve("program", "DEMO-KRAS-001").entity_id == programs[0].id


def test_new_alias_is_registered_and_immediately_resolvable(resolver_fixture) -> None:
    """If this fails, a newly linked designation is not usable until a restart."""
    session, resolver, _, _, programs, _ = resolver_fixture
    assert resolver.register_alias(
        "program", programs[0].id, "sotorasib-demo",
        alias_type="inn_style", available_at=dt.date(2019, 1, 1),
    )
    session.flush()
    res = resolver.resolve("program", "Sotorasib Demo")
    assert res.resolved
    assert res.entity_id == programs[0].id


def test_review_queue_surfaces_refused_decisions(resolver_fixture) -> None:
    """If this fails, conflicts are recorded but nobody ever sees them."""
    session, resolver, _, _, programs, _ = resolver_fixture
    resolver.register_alias("program", programs[1].id, "DEMO-KRAS-001")
    session.flush()
    queue = resolver.review_queue()
    assert queue, "a refused link must appear in the review queue"
    assert all(item["score"] is not None for item in queue)


def test_recording_can_be_disabled_for_dry_runs(db_session) -> None:
    """If this fails, a read-only resolution pass writes audit rows."""
    resolver = EntityResolver(db_session, record_decisions=False)
    resolver.resolve("program", "anything at all")
    db_session.flush()
    assert db_session.scalar(select(func.count()).select_from(EntityResolutionDecision)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Consistency with the rest of the platform
# ─────────────────────────────────────────────────────────────────────────────

def test_seeded_alias_keys_match_the_resolvers_normalisation(seeded_session) -> None:
    """If this fails, the generator and the resolver disagree on identity.

    They did once: the generator wrote space-separated alias keys while the
    resolver produced separator-free ones, so every alias lookup missed and
    conflicting aliases were never detected. Any writer of alias rows must use
    trs.resolution.normalise.
    """
    session, _ = seeded_session
    resolver = EntityResolver(session, record_decisions=False)

    mismatches = []
    for alias in session.scalars(select(ProgramAlias).limit(60)):
        if normalise_compound(alias.alias) != alias.alias_normalised:
            mismatches.append((alias.alias, alias.alias_normalised))
    assert not mismatches, f"stored keys disagree with the resolver: {mismatches[:5]}"

    for alias in session.scalars(select(ProgramAlias).limit(20)):
        res = resolver.resolve("program", alias.alias)
        assert res.method == EntityMatchMethod.REGISTERED_ALIAS, (
            f"{alias.alias!r} did not hit the alias index; it resolved via {res.method}"
        )
        assert res.entity_id == alias.program_id


def test_target_alias_keys_are_consistent_too(seeded_session) -> None:
    """If this fails, target lookups fall through to fuzzy matching."""
    session, _ = seeded_session
    aliases = list(session.scalars(select(TargetAlias).limit(40)))
    assert aliases, "no target aliases were seeded; the check below would be vacuous"
    for alias in aliases:
        assert normalise_target(alias.alias) == alias.alias_normalised
