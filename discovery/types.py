"""Data transfer objects for the rule discovery pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class FeatureDiagnosisCorrelation:
    """Correlation between a feature and a diagnosis."""

    feature: str
    diagnosis: str
    correlation_r: float
    p_value: float
    cohens_d: float
    direction: str  # "positive" | "negative"
    n_positive: int
    n_negative: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "diagnosis": self.diagnosis,
            "correlation_r": round(self.correlation_r, 4),
            "p_value": self.p_value,
            "cohens_d": round(self.cohens_d, 4),
            "direction": self.direction,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


@dataclass
class CandidateRule:
    """A candidate rule discovered from data."""

    rule_id: str
    rule_type: str  # "THRESHOLD" | "MULTI_FEATURE"
    text: str
    diagnosis: str
    feature: str
    threshold: float = 0.0
    operator: str = ">"  # ">" | "<" | ">=" | "<="
    sensitivity: float = 0.0
    specificity: float = 0.0
    f1: float = 0.0
    n_positive: int = 0
    n_total: int = 0
    evidence_grade: str = "D"
    secondary_features: List[str] = field(default_factory=list)
    secondary_thresholds: List[float] = field(default_factory=list)
    secondary_operators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "text": self.text,
            "diagnosis": self.diagnosis,
            "feature": self.feature,
            "threshold": round(self.threshold, 4),
            "operator": self.operator,
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "f1": round(self.f1, 4),
            "n_positive": self.n_positive,
            "n_total": self.n_total,
            "evidence_grade": self.evidence_grade,
        }
        if self.secondary_features:
            d["secondary_features"] = self.secondary_features
            d["secondary_thresholds"] = [round(t, 4) for t in self.secondary_thresholds]
            d["secondary_operators"] = self.secondary_operators
        return d


@dataclass
class MutexCandidate:
    """A mutually exclusive group discovered from co-occurrence analysis."""

    group_name: str
    members: Set[str]
    cooccurrence_ratio: float  # Fraction of patients with >1 member
    violations: int  # Count of patients with >1 member
    violation_rate: float
    evidence_grade: str = "D"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "members": sorted(self.members),
            "cooccurrence_ratio": round(self.cooccurrence_ratio, 4),
            "violations": self.violations,
            "violation_rate": round(self.violation_rate, 6),
            "evidence_grade": self.evidence_grade,
        }


@dataclass
class EquivalenceCandidate:
    """An equivalence group discovered from substitution patterns."""

    group_name: str
    members: Set[str]
    scope: str  # "scoring" | "clustering" | "scoring_and_clustering"
    substitution_rate: float
    feature_similarity: float
    evidence_grade: str = "D"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "members": sorted(self.members),
            "scope": self.scope,
            "substitution_rate": round(self.substitution_rate, 4),
            "feature_similarity": round(self.feature_similarity, 4),
            "evidence_grade": self.evidence_grade,
        }


@dataclass
class DiscoveryReport:
    """Aggregate output from a rule discovery run."""

    domain: str
    n_patients: int
    n_features: int
    n_diagnoses: int
    correlations: List[FeatureDiagnosisCorrelation] = field(default_factory=list)
    threshold_rules: List[CandidateRule] = field(default_factory=list)
    mutex_groups: List[MutexCandidate] = field(default_factory=list)
    equivalence_groups: List[EquivalenceCandidate] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "domain": self.domain,
                "n_patients": self.n_patients,
                "n_features": self.n_features,
                "n_diagnoses": self.n_diagnoses,
            },
            "correlations": [c.to_dict() for c in self.correlations],
            "threshold_rules": [r.to_dict() for r in self.threshold_rules],
            "mutex_groups": [m.to_dict() for m in self.mutex_groups],
            "equivalence_groups": [e.to_dict() for e in self.equivalence_groups],
            "validation_results": self.validation_results,
        }

    def summary(self) -> str:
        """Return a concise text summary."""
        lines = [
            f"Discovery Report: {self.domain}",
            f"  Patients: {self.n_patients}, Features: {self.n_features}, Diagnoses: {self.n_diagnoses}",
            f"  Correlations: {len(self.correlations)} (top by |r|)",
            f"  Threshold rules: {len(self.threshold_rules)}",
            f"  Mutex groups: {len(self.mutex_groups)}",
            f"  Equivalence groups: {len(self.equivalence_groups)}",
        ]
        # Grade breakdown
        for label, rules in [
            ("Threshold", self.threshold_rules),
            ("Mutex", self.mutex_groups),
            ("Equivalence", self.equivalence_groups),
        ]:
            grades = {}
            for r in rules:
                g = r.evidence_grade
                grades[g] = grades.get(g, 0) + 1
            if grades:
                parts = [f"{g}:{n}" for g, n in sorted(grades.items())]
                lines.append(f"    {label} grades: {', '.join(parts)}")
        return "\n".join(lines)
