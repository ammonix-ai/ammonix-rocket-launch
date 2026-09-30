"""Phase 1: Correlation analysis between features and diagnoses.

Computes point-biserial correlation, Cohen's d effect size, and
Mann-Whitney U statistics for all (feature, diagnosis) pairs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from discovery.types import FeatureDiagnosisCorrelation


def compute_all_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: List[str],
    diagnosis_names: List[str],
    min_positive: int = 10,
    top_k: int = 20,
) -> List[FeatureDiagnosisCorrelation]:
    """Compute correlations for all (feature, diagnosis) pairs.

    Args:
        X: (n_patients, n_features) feature matrix.
        Y: (n_patients, n_diagnoses) binary label matrix.
        feature_names: Feature names matching X columns.
        diagnosis_names: Diagnosis names matching Y columns.
        min_positive: Skip diagnoses with fewer positive examples.
        top_k: Return top-k correlations per diagnosis (by |r|).

    Returns:
        List of FeatureDiagnosisCorrelation, sorted by |correlation_r| descending.
    """
    n, m = X.shape
    _, k = Y.shape
    results: List[FeatureDiagnosisCorrelation] = []

    for j in range(k):
        y = Y[:, j]
        n_pos = int(y.sum())
        n_neg = n - n_pos

        if n_pos < min_positive or n_neg < min_positive:
            continue

        dx_name = diagnosis_names[j]
        pos_mask = y > 0.5
        neg_mask = ~pos_mask

        # Compute correlations for all features at once
        dx_corrs: List[Tuple[float, int]] = []  # (abs_r, feat_idx)
        for i in range(m):
            feat = X[:, i]

            # Point-biserial correlation
            r, p = _point_biserial(feat, y, n_pos, n_neg)

            # Cohen's d
            pos_vals = feat[pos_mask]
            neg_vals = feat[neg_mask]
            d = _cohens_d(pos_vals, neg_vals)

            dx_corrs.append((abs(r), i, r, p, d))

        # Keep top_k by |r|
        dx_corrs.sort(key=lambda t: t[0], reverse=True)
        for _, feat_i, r, p, d in dx_corrs[:top_k]:
            results.append(FeatureDiagnosisCorrelation(
                feature=feature_names[feat_i],
                diagnosis=dx_name,
                correlation_r=r,
                p_value=p,
                cohens_d=d,
                direction="positive" if r > 0 else "negative",
                n_positive=n_pos,
                n_negative=n_neg,
            ))

    # Sort overall by |r|
    results.sort(key=lambda c: abs(c.correlation_r), reverse=True)
    return results


def cluster_feature_families(
    correlations: List[FeatureDiagnosisCorrelation],
    prefix_separator: str = "_",
    depth: int = 3,
) -> Dict[str, List[FeatureDiagnosisCorrelation]]:
    """Group correlations by feature name prefix.

    Args:
        correlations: List of correlations.
        prefix_separator: Character to split feature names on.
        depth: How many prefix parts to use for grouping.

    Returns:
        Dict mapping prefix → list of correlations.
    """
    families: Dict[str, List[FeatureDiagnosisCorrelation]] = {}
    for corr in correlations:
        parts = corr.feature.split(prefix_separator)
        prefix = prefix_separator.join(parts[:depth])
        families.setdefault(prefix, []).append(corr)
    return families


def _point_biserial(
    x: np.ndarray, y: np.ndarray, n1: int, n0: int
) -> Tuple[float, float]:
    """Compute point-biserial correlation and p-value.

    Equivalent to Pearson correlation between continuous x and binary y.
    """
    n = len(x)
    if n < 3 or n1 < 1 or n0 < 1:
        return 0.0, 1.0

    mask = y > 0.5
    mean1 = x[mask].mean()
    mean0 = x[~mask].mean()
    std_x = x.std(ddof=1)

    if std_x < 1e-12:
        return 0.0, 1.0

    r = (mean1 - mean0) / std_x * np.sqrt(n1 * n0 / (n * n))

    # t-test approximation for p-value
    if abs(r) > 0.9999:
        return float(np.clip(r, -1, 1)), 0.0
    t_stat = r * np.sqrt((n - 2) / (1 - r * r))
    # Two-sided p-value approximation (normal for large n)
    p = 2.0 * _norm_sf(abs(t_stat))
    return float(r), float(p)


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.var(ddof=1), group2.var(ddof=1)

    pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd < 1e-12:
        return 0.0
    return float((m1 - m2) / pooled_sd)


def _norm_sf(x: float) -> float:
    """Survival function for standard normal (1 - CDF), approximation."""
    # Abramowitz & Stegun approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    result = d * np.exp(-0.5 * x * x) * poly
    return float(result)
