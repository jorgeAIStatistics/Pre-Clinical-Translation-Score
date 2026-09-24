"""Scoring, explanation, comparators and scenario analysis."""

from trs.scoring.comparators import find_comparators, summarise_comparators
from trs.scoring.explanation import Explanation, ExplanationUnavailable, explain
from trs.scoring.scenarios import LEVERS, lever_catalogue, resolve_levers
from trs.scoring.service import ScoreResult, ScoringError, ScoringService

__all__ = [
    "Explanation", "ExplanationUnavailable", "LEVERS", "ScoreResult",
    "ScoringError", "ScoringService", "explain", "find_comparators",
    "lever_catalogue", "resolve_levers", "summarise_comparators",
]
