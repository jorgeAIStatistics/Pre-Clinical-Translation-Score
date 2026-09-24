"""Name normalisation for entity resolution.

Normalisation is where most resolution errors are made or avoided. The rules
here are deliberately conservative: they fold away variation that is certainly
meaningless (case, punctuation, corporate suffixes, unicode dashes) and leave
everything else alone. An over-eager normaliser that strips digits or collapses
whitespace inside identifiers would merge ``ABT-199`` with ``ABT-1990``, and a
false merge is far more damaging than a missed one — it puts two different
assets into one program and then lets that program straddle a train/test split.

Every function is pure and deterministic. The same input always normalises to
the same key, which is what makes a stored alias lookup meaningful across runs.
"""

from __future__ import annotations

import re
import unicodedata

#: Legal-form suffixes stripped from sponsor names. Order matters: longer forms
#: are removed first so "Co., Ltd." does not leave a stray "Co".
_SPONSOR_SUFFIXES = (
    "co ltd", "co limited", "pty ltd", "pte ltd", "s a s", "s p a", "n v",
    "a s", "a b", "k k", "gmbh co kg",
    "incorporated", "corporation", "limited", "holdings", "holding",
    "pharmaceuticals", "pharmaceutical", "biosciences", "bioscience",
    "therapeutics", "biopharma", "biotech", "pharma", "sciences", "labs",
    "laboratories", "laboratory", "company", "group", "gmbh", "ltda", "sarl",
    "bvba", "oyj", "asa", "plc", "llc", "inc", "corp", "ltd", "spa", "sas",
    "srl", "nv", "bv", "ab", "as", "ag", "sa", "kk", "co",
)

#: Greek letters spelled out, because "TNF-alpha" and "TNF-α" are the same target.
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
}

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _fold(text: str) -> str:
    """Lowercase, decompose accents, normalise dashes and Greek letters."""
    text = text.translate(_DASHES)
    for glyph, name in _GREEK.items():
        text = text.replace(glyph, name).replace(glyph.upper(), name)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


def normalise_general(text: str | None) -> str:
    """Generic key: folded, punctuation collapsed to single spaces.

    Used for indications and anything without a domain-specific rule.
    """
    if not text:
        return ""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", _fold(text))).strip()


def normalise_compound(text: str | None) -> str:
    """Compound / program designation key.

    Development codes are alphanumeric with separators that vary freely across
    sources: ``ABT-199``, ``ABT199`` and ``ABT 199`` are one molecule. Separators
    are therefore removed entirely rather than collapsed to spaces — but only
    for this key, because doing the same to a sponsor name would merge
    "Bio Gen" with "Biogen" on no evidence.
    """
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", _fold(text))


#: Species qualifiers safe to strip from a gene symbol. Multi-character only —
#: see `normalise_target` for why single letters are not on this list.
_SPECIES_PREFIXES = ("human", "mouse", "rat", "hsa", "mmu", "rno", "hs", "mm", "rn")

#: Trailing qualifiers sources append inconsistently to the same symbol.
_TARGET_SUFFIXES = ("protein", "receptor", "gene")

#: Below this length a symbol is not eligible for any stripping at all.
_MIN_SYMBOL_LENGTH = 2


def normalise_target(text: str | None) -> str:
    """Gene / protein symbol key.

    Strips multi-character species qualifiers ("human EGFR" → "egfr") and the
    "protein"/"receptor"/"gene" suffixes that sources append inconsistently.

    **Single-letter species prefixes are deliberately not stripped**, even
    though "hEGFR" is common usage. An earlier version of this function removed
    a leading ``h``/``m``/``r`` before any symbol of two or more letters, and
    that silently destroyed real gene symbols: ``MET`` became ``et``, ``RET``
    became ``et`` — colliding two distinct targets — and ``MYC`` became ``yc``.
    Merging MET with RET would have pooled two unrelated evidence bases into
    one target and corrupted every feature derived from it.

    The rule this module is built on says a missed merge is recoverable and a
    false merge is not, so the single-letter case is left alone. ``hEGFR`` will
    simply not fold to ``EGFR`` here; it is handled by registering it as an
    alias, which is a decision a human makes once and the resolver then honours
    forever.
    """
    if not text:
        return ""
    key = re.sub(r"[^a-z0-9]", "", _fold(text))
    if len(key) <= _MIN_SYMBOL_LENGTH:
        return key

    for prefix in _SPECIES_PREFIXES:  # ordered longest-first
        if key.startswith(prefix) and len(key) - len(prefix) >= 3:
            key = key[len(prefix):]
            break

    for suffix in _TARGET_SUFFIXES:
        if key.endswith(suffix) and len(key) - len(suffix) >= _MIN_SYMBOL_LENGTH:
            key = key[: -len(suffix)]
            break
    return key


def normalise_sponsor(text: str | None) -> str:
    """Organisation-name key, with legal forms removed.

    "Kestrel Ridge Therapeutics, Inc." and "Kestrel Ridge Therapeutics Ltd"
    are the same developer; the suffix records a jurisdiction, not an identity.
    Stripping continues until no suffix matches, because names routinely carry
    two ("Pharma Co., Ltd.").
    """
    if not text:
        return ""
    key = normalise_general(text)
    changed = True
    while changed and key:
        changed = False
        for suffix in _SPONSOR_SUFFIXES:
            if key.endswith(f" {suffix}"):
                key = key[: -(len(suffix) + 1)].strip()
                changed = True
                break
    return key


def token_set(text: str | None) -> frozenset[str]:
    """Token bag used for blocking. Empty for empty input."""
    key = normalise_general(text)
    return frozenset(key.split()) if key else frozenset()


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalised_edit_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] from a bounded Levenshtein distance.

    Implemented directly rather than pulled from a dependency: the inputs are
    short identifiers, the algorithm is twenty lines, and the resolver's
    behaviour should not change because a transitive dependency did.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.5:
        return 0.0  # length gap too large to be a plausible variant

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))
