"""Phase 5: Assemble discovered rules into standard rules.json format.

Converts discovery output into the Layer 2 artifact format and
checks for internal consistency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from discovery.types import (
    CandidateRule,
    DiscoveryReport,
    EquivalenceCandidate,
    MutexCandidate,
)


def assemble_rules_json(
    domain: str,
    threshold_rules: List[CandidateRule],
    mutex_groups: List[MutexCandidate],
    equivalence_groups: List[EquivalenceCandidate],
    grade_cutoff: str = "C",
) -> Dict[str, Any]:
    """Assemble discovered rules into standard rules.json format.

    Args:
        domain: Domain name.
        threshold_rules: Discovered threshold rules.
        mutex_groups: Discovered mutex groups.
        equivalence_groups: Discovered equivalence groups.
        grade_cutoff: Minimum evidence grade to include (A, B, C, or D).

    Returns:
        Dict matching the Layer 2 rules.json schema.
    """
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    cutoff_rank = grade_order.get(grade_cutoff, 2)

    def passes(grade: str) -> bool:
        return grade_order.get(grade, 3) <= cutoff_rank

    # Filter by grade
    filtered_thresholds = [r for r in threshold_rules if passes(r.evidence_grade)]
    filtered_mutex = [m for m in mutex_groups if passes(m.evidence_grade)]
    filtered_equiv = [e for e in equivalence_groups if passes(e.evidence_grade)]

    # Convert to rules.json format
    thr_rules = []
    for r in filtered_thresholds:
        safe_name = r.diagnosis.lower().replace(" ", "_").replace("/", "_")
        thr_rules.append({
            "id": r.rule_id,
            "type": "THRESHOLD",
            "diagnosis": r.diagnosis,
            "threshold": round(r.threshold, 4),
            "feature": r.feature,
            "operator": r.operator,
            "description": r.text,
            "sensitivity": round(r.sensitivity, 4),
            "specificity": round(r.specificity, 4),
            "f1": round(r.f1, 4),
            "evidence_grade": r.evidence_grade,
            "source": "discovery",
        })

    mx_rules = []
    for m in filtered_mutex:
        mx_rules.append({
            "id": f"rule:mutex_{m.group_name}",
            "type": "MUTEX",
            "group": m.group_name,
            "members": sorted(m.members),
            "violation_rate": round(m.violation_rate, 6),
            "description": f"Discovered mutex group: {', '.join(sorted(m.members))}",
            "evidence_grade": m.evidence_grade,
            "source": "discovery",
        })

    eq_rules = []
    for e in filtered_equiv:
        eq_rules.append({
            "id": f"rule:equiv_{e.group_name}",
            "type": "EQUIVALENCE",
            "members": sorted(e.members),
            "scope": e.scope,
            "substitution_rate": round(e.substitution_rate, 4),
            "feature_similarity": round(e.feature_similarity, 4),
            "description": f"Discovered equivalence: {', '.join(sorted(e.members))}",
            "evidence_grade": e.evidence_grade,
            "source": "discovery",
        })

    return {
        "meta": {
            "domain": domain,
            "layer": 2,
            "created_by": "factory/discovery",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "grade_cutoff": grade_cutoff,
            "counts": {
                "threshold_rules": len(thr_rules),
                "mutex_rules": len(mx_rules),
                "equivalence_rules": len(eq_rules),
            },
        },
        "threshold_rules": thr_rules,
        "mutex_rules": mx_rules,
        "equivalence_rules": eq_rules,
    }


def check_consistency(rules_json: Dict[str, Any]) -> List[str]:
    """Check for internal consistency issues.

    Returns:
        List of warning strings (empty = consistent).
    """
    warnings: List[str] = []

    # Check for duplicate rule IDs
    seen_ids: set = set()
    for section in ["threshold_rules", "mutex_rules", "equivalence_rules"]:
        for rule in rules_json.get(section, []):
            rid = rule.get("id", "")
            if rid in seen_ids:
                warnings.append(f"Duplicate rule ID: {rid}")
            seen_ids.add(rid)

    # Check mutex groups don't overlap
    mutex_rules = rules_json.get("mutex_rules", [])
    for i, m1 in enumerate(mutex_rules):
        for j, m2 in enumerate(mutex_rules):
            if j <= i:
                continue
            overlap = set(m1["members"]) & set(m2["members"])
            if overlap:
                warnings.append(
                    f"Mutex groups {m1['id']} and {m2['id']} overlap: "
                    f"{sorted(overlap)}"
                )

    # Check equivalence groups don't contain mutex-excluded pairs
    equiv_members: set = set()
    for eq in rules_json.get("equivalence_rules", []):
        for m in eq["members"]:
            equiv_members.add(m)

    for mx in mutex_rules:
        for m in mx["members"]:
            if m in equiv_members:
                warnings.append(
                    f"Diagnosis '{m}' appears in both mutex and equivalence groups"
                )

    return warnings


def classify_handoff(report: DiscoveryReport) -> Dict[str, List[str]]:
    """Classify rules by handoff destination.

    Returns:
        Dict with keys "production" (B+), "rlvr_candidate" (C),
        "layer4_gap" (D).
    """
    result: Dict[str, List[str]] = {
        "production": [],
        "rlvr_candidate": [],
        "layer4_gap": [],
    }

    for rule in report.threshold_rules:
        if rule.evidence_grade in ("A", "B"):
            result["production"].append(rule.rule_id)
        elif rule.evidence_grade == "C":
            result["rlvr_candidate"].append(rule.rule_id)
        else:
            result["layer4_gap"].append(rule.rule_id)

    return result
