"""Phase 4: Validation of discovered rules on held-out data.

Applies each candidate rule to a validation set and computes
helped/hurt metrics. Also performs regression testing of rule sets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from discovery.types import CandidateRule


def validate_rules(
    candidates: List[CandidateRule],
    X_val: np.ndarray,
    Y_val: np.ndarray,
    feature_names: List[str],
    diagnosis_names: List[str],
) -> Dict[str, Any]:
    """Validate each candidate rule on a validation set.

    Args:
        candidates: Rules to validate.
        X_val: Validation feature matrix.
        Y_val: Validation label matrix.
        feature_names: Feature names.
        diagnosis_names: Diagnosis names.

    Returns:
        Dict with per-rule validation metrics.
    """
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    dx_idx = {name: i for i, name in enumerate(diagnosis_names)}
    results: Dict[str, Any] = {"rules": [], "summary": {}}

    for rule in candidates:
        fi = feat_idx.get(rule.feature)
        di = dx_idx.get(rule.diagnosis)
        if fi is None or di is None:
            continue

        x_col = X_val[:, fi]
        y_col = Y_val[:, di]

        # Apply rule
        if rule.operator == ">":
            pred = x_col > rule.threshold
        elif rule.operator == "<":
            pred = x_col < rule.threshold
        elif rule.operator == ">=":
            pred = x_col >= rule.threshold
        else:
            pred = x_col <= rule.threshold

        tp = int((pred & (y_col > 0.5)).sum())
        fp = int((pred & (y_col < 0.5)).sum())
        fn = int((~pred & (y_col > 0.5)).sum())
        tn = int((~pred & (y_col < 0.5)).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

        # Overfitting check: drop > 0.05 from discovery F1
        overfit = rule.f1 - f1 > 0.05

        results["rules"].append({
            "rule_id": rule.rule_id,
            "diagnosis": rule.diagnosis,
            "discovery_f1": round(rule.f1, 4),
            "validation_f1": round(f1, 4),
            "val_sensitivity": round(sens, 4),
            "val_specificity": round(spec, 4),
            "val_tp": tp,
            "val_fp": fp,
            "val_fn": fn,
            "val_tn": tn,
            "overfitting_flag": overfit,
            "f1_delta": round(f1 - rule.f1, 4),
        })

    # Summary
    if results["rules"]:
        val_f1s = [r["validation_f1"] for r in results["rules"]]
        overfit_count = sum(1 for r in results["rules"] if r["overfitting_flag"])
        results["summary"] = {
            "n_rules_validated": len(results["rules"]),
            "mean_val_f1": round(np.mean(val_f1s), 4),
            "median_val_f1": round(np.median(val_f1s), 4),
            "overfitting_count": overfit_count,
        }

    return results


def regression_test_rule_set(
    candidates: List[CandidateRule],
    X_val: np.ndarray,
    Y_val: np.ndarray,
    feature_names: List[str],
    diagnosis_names: List[str],
    baseline_f1: Optional[Dict[str, float]] = None,
    max_degradation: float = 0.05,
) -> Dict[str, Any]:
    """Test a set of rules together for regression.

    Applies all rules and checks that no diagnosis F1 degrades
    beyond max_degradation.

    Args:
        candidates: All rules to apply together.
        X_val: Validation features.
        Y_val: Validation labels.
        feature_names: Feature names.
        diagnosis_names: Diagnosis names.
        baseline_f1: Optional per-diagnosis baseline F1 for comparison.
        max_degradation: Maximum acceptable F1 drop.

    Returns:
        Dict with regression test results.
    """
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    dx_idx = {name: i for i, name in enumerate(diagnosis_names)}
    n = X_val.shape[0]
    k = len(diagnosis_names)

    # Build prediction matrix from rules
    pred_matrix = np.zeros((n, k), dtype=np.float32)

    for rule in candidates:
        fi = feat_idx.get(rule.feature)
        di = dx_idx.get(rule.diagnosis)
        if fi is None or di is None:
            continue

        x_col = X_val[:, fi]
        if rule.operator == ">":
            fires = x_col > rule.threshold
        elif rule.operator == "<":
            fires = x_col < rule.threshold
        elif rule.operator == ">=":
            fires = x_col >= rule.threshold
        else:
            fires = x_col <= rule.threshold

        pred_matrix[fires, di] = 1.0

    # Compute per-diagnosis F1
    per_dx: Dict[str, Dict[str, float]] = {}
    regressions: List[str] = []

    for j, dx in enumerate(diagnosis_names):
        y = Y_val[:, j]
        p = pred_matrix[:, j]
        tp = ((p > 0.5) & (y > 0.5)).sum()
        fp = ((p > 0.5) & (y < 0.5)).sum()
        fn = ((p < 0.5) & (y > 0.5)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        entry = {"f1": round(float(f1), 4), "precision": round(float(prec), 4), "recall": round(float(rec), 4)}

        if baseline_f1 and dx in baseline_f1:
            delta = f1 - baseline_f1[dx]
            entry["delta"] = round(float(delta), 4)
            if delta < -max_degradation:
                regressions.append(dx)

        per_dx[dx] = entry

    return {
        "per_diagnosis": per_dx,
        "regressions": regressions,
        "pass": len(regressions) == 0,
    }
