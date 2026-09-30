"""Phase 3: Evidence grading for discovered rules.

Assigns A/B/C/D grades based on sensitivity, specificity, sample size,
and violation rates.
"""

from __future__ import annotations

from typing import List

from discovery.types import CandidateRule, EquivalenceCandidate, MutexCandidate


def grade_rules(candidates: List[CandidateRule]) -> None:
    """Grade threshold/multi-feature rules in-place.

    Grade thresholds:
        A: sensitivity >= 0.80, specificity >= 0.80, n_positive >= 50
        B: sensitivity >= 0.60, specificity >= 0.60, n_positive >= 20
        C: sensitivity >= 0.40, specificity >= 0.40, n_positive >= 10
        D: below C
    """
    for r in candidates:
        if r.sensitivity >= 0.80 and r.specificity >= 0.80 and r.n_positive >= 50:
            r.evidence_grade = "A"
        elif r.sensitivity >= 0.60 and r.specificity >= 0.60 and r.n_positive >= 20:
            r.evidence_grade = "B"
        elif r.sensitivity >= 0.40 and r.specificity >= 0.40 and r.n_positive >= 10:
            r.evidence_grade = "C"
        else:
            r.evidence_grade = "D"


def grade_mutex(candidates: List[MutexCandidate]) -> None:
    """Grade mutex groups in-place by violation rate.

    A: violation_rate <= 0.01 (< 1% co-occurrence)
    B: violation_rate <= 0.05
    C: violation_rate <= 0.10
    D: above 10%
    """
    for m in candidates:
        if m.violation_rate <= 0.01:
            m.evidence_grade = "A"
        elif m.violation_rate <= 0.05:
            m.evidence_grade = "B"
        elif m.violation_rate <= 0.10:
            m.evidence_grade = "C"
        else:
            m.evidence_grade = "D"


def grade_equivalence(candidates: List[EquivalenceCandidate]) -> None:
    """Grade equivalence groups in-place by substitution + similarity.

    A: substitution_rate >= 0.80, feature_similarity >= 0.95
    B: substitution_rate >= 0.60, feature_similarity >= 0.90
    C: substitution_rate >= 0.40, feature_similarity >= 0.85
    D: below C
    """
    for e in candidates:
        if e.substitution_rate >= 0.80 and e.feature_similarity >= 0.95:
            e.evidence_grade = "A"
        elif e.substitution_rate >= 0.60 and e.feature_similarity >= 0.90:
            e.evidence_grade = "B"
        elif e.substitution_rate >= 0.40 and e.feature_similarity >= 0.85:
            e.evidence_grade = "C"
        else:
            e.evidence_grade = "D"
