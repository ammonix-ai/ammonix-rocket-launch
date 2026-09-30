"""Phase 2: Candidate rule formulation from correlation data.

Generates threshold rules, discovers mutex groups, and identifies
equivalence groups from feature and label matrices.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from discovery.types import (
    CandidateRule,
    EquivalenceCandidate,
    FeatureDiagnosisCorrelation,
    MutexCandidate,
)


def generate_threshold_rules(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: List[str],
    diagnosis_names: List[str],
    correlations: List[FeatureDiagnosisCorrelation],
    top_k: int = 5,
    n_thresholds: int = 50,
    min_positive: int = 10,
) -> List[CandidateRule]:
    """Generate threshold rules by grid-searching optimal cutoffs.

    For each diagnosis, takes the top-k correlated features and sweeps
    thresholds to maximize F1.

    Args:
        X: Feature matrix (n, m).
        Y: Label matrix (n, k).
        feature_names: Feature names.
        diagnosis_names: Diagnosis names.
        correlations: Pre-computed correlations (used to pick top features).
        top_k: Features per diagnosis to try.
        n_thresholds: Grid resolution.
        min_positive: Minimum positive examples.

    Returns:
        List of CandidateRule.
    """
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    dx_idx = {name: i for i, name in enumerate(diagnosis_names)}

    # Group correlations by diagnosis
    dx_corrs: Dict[str, List[FeatureDiagnosisCorrelation]] = {}
    for c in correlations:
        dx_corrs.setdefault(c.diagnosis, []).append(c)

    rules: List[CandidateRule] = []
    rule_counter = 0

    for dx_name in diagnosis_names:
        j = dx_idx[dx_name]
        y = Y[:, j]
        n_pos = int(y.sum())

        if n_pos < min_positive:
            continue

        # Get top correlated features for this diagnosis
        top_feats = dx_corrs.get(dx_name, [])[:top_k]

        for corr in top_feats:
            fi = feat_idx.get(corr.feature)
            if fi is None:
                continue

            x_col = X[:, fi]
            best = _sweep_threshold(x_col, y, n_thresholds)
            if best is None:
                continue

            threshold, operator, sens, spec, f1 = best
            rule_counter += 1
            safe_dx = dx_name.replace(" ", "_")
            safe_feat = corr.feature.replace(" ", "_")[:30]

            rules.append(CandidateRule(
                rule_id=f"disc:thr_{safe_dx}_{rule_counter}",
                rule_type="THRESHOLD",
                text=f"IF {corr.feature} {operator} {threshold:.4f} THEN {dx_name}",
                diagnosis=dx_name,
                feature=corr.feature,
                threshold=threshold,
                operator=operator,
                sensitivity=sens,
                specificity=spec,
                f1=f1,
                n_positive=n_pos,
                n_total=len(y),
            ))

    rules.sort(key=lambda r: r.f1, reverse=True)
    return rules


def discover_mutex_groups(
    Y: np.ndarray,
    diagnosis_names: List[str],
    threshold: float = 0.02,
    min_group_size: int = 2,
) -> List[MutexCandidate]:
    """Discover mutually exclusive diagnosis groups from co-occurrence.

    Diagnoses that rarely co-occur in the same patient are candidates
    for mutex groups.

    Args:
        Y: Label matrix (n, k).
        diagnosis_names: Diagnosis names.
        threshold: Maximum co-occurrence rate to consider mutually exclusive.
        min_group_size: Minimum group size.

    Returns:
        List of MutexCandidate.
    """
    n, k = Y.shape
    if n == 0 or k < 2:
        return []

    # Compute pairwise co-occurrence rates
    # cooccur[i,j] = fraction of patients with both diagnosis i and j
    cooccur = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            both = ((Y[:, i] > 0.5) & (Y[:, j] > 0.5)).sum()
            either = ((Y[:, i] > 0.5) | (Y[:, j] > 0.5)).sum()
            if either > 0:
                cooccur[i, j] = both / either
                cooccur[j, i] = cooccur[i, j]

    # Find anti-correlated pairs (low co-occurrence)
    anti_pairs: List[Tuple[int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            # Both must have positive examples
            if Y[:, i].sum() < 5 or Y[:, j].sum() < 5:
                continue
            if cooccur[i, j] <= threshold:
                anti_pairs.append((i, j))

    # Cluster into groups via greedy union
    groups: List[Set[int]] = []
    for i, j in anti_pairs:
        # Check if either belongs to an existing group
        merged = False
        for group in groups:
            if i in group or j in group:
                # Only add if ALL pairs in group are also anti-correlated
                can_add_i = all(cooccur[i, m] <= threshold for m in group if m != i)
                can_add_j = all(cooccur[j, m] <= threshold for m in group if m != j)
                if can_add_i and can_add_j:
                    group.add(i)
                    group.add(j)
                    merged = True
                    break
        if not merged:
            groups.append({i, j})

    # Convert to candidates
    candidates: List[MutexCandidate] = []
    for idx, group in enumerate(groups):
        if len(group) < min_group_size:
            continue

        members = {diagnosis_names[i] for i in group}
        # Count violations (patients with >1 member)
        member_cols = Y[:, list(group)]
        multi = (member_cols.sum(axis=1) > 1).sum()
        total_any = (member_cols.sum(axis=1) > 0).sum()
        rate = multi / total_any if total_any > 0 else 0.0

        candidates.append(MutexCandidate(
            group_name=f"mutex_discovered_{idx}",
            members=members,
            cooccurrence_ratio=rate,
            violations=int(multi),
            violation_rate=rate,
        ))

    return candidates


def discover_equivalence_groups(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: List[str],
    diagnosis_names: List[str],
    substitution_threshold: float = 0.15,
    similarity_threshold: float = 0.85,
    min_examples: int = 20,
) -> List[EquivalenceCandidate]:
    """Discover equivalence groups from substitution patterns.

    Diagnoses that often substitute for each other (similar patients get
    one or the other) with similar feature profiles are candidates for
    equivalence groups.

    Args:
        X: Feature matrix.
        Y: Label matrix.
        feature_names: Feature names.
        diagnosis_names: Diagnosis names.
        substitution_threshold: Min substitution rate.
        similarity_threshold: Min feature cosine similarity.
        min_examples: Min patients per diagnosis.

    Returns:
        List of EquivalenceCandidate.
    """
    _, k = Y.shape
    candidates: List[EquivalenceCandidate] = []

    for i in range(k):
        for j in range(i + 1, k):
            n_i = int(Y[:, i].sum())
            n_j = int(Y[:, j].sum())
            if n_i < min_examples or n_j < min_examples:
                continue

            # Substitution rate: patients with one but not both
            has_i = Y[:, i] > 0.5
            has_j = Y[:, j] > 0.5
            has_both = (has_i & has_j).sum()
            has_either = (has_i | has_j).sum()

            if has_either == 0:
                continue

            # Substitution = (either - both) / either
            sub_rate = (has_either - has_both) / has_either
            if sub_rate < substitution_threshold:
                continue

            # Feature profile similarity (cosine of mean feature vectors)
            mean_i = X[has_i].mean(axis=0)
            mean_j = X[has_j].mean(axis=0)
            cos_sim = _cosine_similarity(mean_i, mean_j)

            if cos_sim < similarity_threshold:
                continue

            candidates.append(EquivalenceCandidate(
                group_name=f"equiv_{diagnosis_names[i]}__{diagnosis_names[j]}",
                members={diagnosis_names[i], diagnosis_names[j]},
                scope="scoring",
                substitution_rate=float(sub_rate),
                feature_similarity=float(cos_sim),
            ))

    return candidates


def _sweep_threshold(
    x: np.ndarray,
    y: np.ndarray,
    n_steps: int = 50,
) -> Optional[Tuple[float, str, float, float, float]]:
    """Find optimal threshold for a single feature → diagnosis rule.

    Returns:
        (threshold, operator, sensitivity, specificity, f1) or None.
    """
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < 1e-10:
        return None

    n_pos = int(y.sum())
    if n_pos < 2:
        return None

    best_f1 = 0.0
    best: Optional[Tuple[float, str, float, float, float]] = None

    thresholds = np.linspace(x_min, x_max, n_steps + 2)[1:-1]

    for thr in thresholds:
        for op in [">", "<"]:
            pred = (x > thr) if op == ">" else (x < thr)
            tp = (pred & (y > 0.5)).sum()
            fp = (pred & (y < 0.5)).sum()
            fn = (~pred & (y > 0.5)).sum()
            tn = (~pred & (y < 0.5)).sum()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best = (float(thr), op, float(sens), float(spec), float(f1))

    return best


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))
