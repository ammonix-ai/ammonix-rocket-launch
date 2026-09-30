"""Shared per-class metric battery."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# sklearn provides the threshold-free metrics + curves. The count-based metrics
# (confusion, sens/spec/PPV/NPV/F1/balanced-acc/MCC, Brier, Wilson CI) are pure
# numpy and work without it; only ``threshold_free``, ``curve_data`` and the
# bootstrap CI require sklearn and raise a clear ImportError if it is absent.
try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only in a sklearn-less env
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════
# Documented defaults (NO magic numbers — every knob is a named constant)
# ═══════════════════════════════════════════════════════════════════════

#: Two-sided confidence level for all interval estimates (0.95 = 95% CI).
DEFAULT_CI_LEVEL: float = 0.95

#: Resamples for the AUROC/AUPRC bootstrap CI. 1000 is the conventional
#: lower bound for a stable 95% percentile interval; raise for tighter tails.
DEFAULT_N_BOOTSTRAP: int = 1000

#: Seed for the bootstrap RNG. Fixed so every call is reproducible — the module
#: must be deterministic for the factory-primitive lift and for golden tests.
DEFAULT_BOOTSTRAP_SEED: int = 0

#: Partial-AUROC is restricted to the high-specificity band ``FPR <= this``
#: (i.e. ``specificity >= 1 - this``). 0.1 ⇒ specificity ≥ 0.90, the clinically
#: relevant low-false-alarm operating region where a screening model must
#: live. Reported as sklearn's McClish-standardised pAUC (rescaled to [0.5, 1]).
DEFAULT_PARTIAL_AUROC_MAX_FPR: float = 0.1

#: Number of evenly spaced thresholds in the F1/sens/spec sweep curve
#: (101 ⇒ 0.00, 0.01, …, 1.00 inclusive).
DEFAULT_THRESHOLD_SWEEP_POINTS: int = 101

#: The three aggregation modes ``aggregate`` understands.
AGGREGATION_MODES = ("macro", "micro", "weighted")

#: Proportion metrics that take a Wilson interval (each is a ratio of two cells
#: of the 2x2). Mapped to (numerator-cell, denominator-cells) in ``_PROPORTION_COUNTS``.
PROPORTION_METRICS = ("sensitivity", "specificity", "ppv", "npv")

#: Threshold-free metrics that take a bootstrap interval.
RANK_METRICS = ("auroc", "auprc")


# ═══════════════════════════════════════════════════════════════════════
# 1. The 2x2 confusion cell — the thing nothing forms inline
# ═══════════════════════════════════════════════════════════════════════

def _as_binary_true(y_true: Sequence[Any]) -> np.ndarray:
    """Coerce a label vector to a boolean positive-mask (``== 1``).

    Matches the original inline convention ``(y_true == 1)`` — labels are 0/1
    (or 0.0/1.0) so this is exact, but it is robust to float labels too.
    """
    return np.asarray(y_true) == 1


def per_class_confusion(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    threshold: float,
) -> Dict[str, int]:
    """Form the full per-class 2x2 confusion at ``threshold``.

    Returns ``{tp, fp, fn, tn}`` as plain ints. **TN is the cell no original
    inline site formed** — materialising it here is what unlocks specificity,
    NPV, balanced accuracy and MCC downstream.

    Positive prediction iff ``y_prob >= threshold`` (inclusive ``>=``, the
    repository-wide convention — see module docstring).
    """
    yt = _as_binary_true(y_true)
    yp = np.asarray(y_prob) >= threshold
    tp = int(np.count_nonzero(yp & yt))
    fp = int(np.count_nonzero(yp & ~yt))
    fn = int(np.count_nonzero(~yp & yt))
    tn = int(np.count_nonzero(~yp & ~yt))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _safe_ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` with the codebase's ``denom == 0 -> 0.0``
    convention (matches the original ``_compute_f1`` / confusion-analysis sites)."""
    return numerator / denominator if denominator > 0 else 0.0


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """The count-derived metric battery from a single 2x2.

    Split out from :func:`per_class_metrics` so that micro-aggregation (which
    pools the 2x2 across classes) shares exactly one definition of every metric
    — no second inline copy can drift from this one.

    Returns sensitivity (recall), specificity, ppv (precision), npv, f1,
    balanced_acc, mcc, support (= n positives = tp + fn). Edge cases follow the
    existing convention: a zero denominator yields 0.0; MCC yields 0.0 when any
    factor of its denominator is 0 (matching ``sklearn.matthews_corrcoef``).
    """
    sensitivity = _safe_ratio(tp, tp + fn)          # recall / TPR
    specificity = _safe_ratio(tn, tn + fp)          # TNR
    ppv = _safe_ratio(tp, tp + fp)                  # precision
    npv = _safe_ratio(tn, tn + fn)
    f1 = _safe_ratio(2 * ppv * sensitivity, ppv + sensitivity)
    balanced_acc = (sensitivity + specificity) / 2.0
    mcc_denom = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "support": int(tp + fn),
    }


def per_class_metrics(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    """Operating-point metric battery for one class at ``threshold``.

    Returns ``{sensitivity, specificity, ppv, npv, f1, balanced_acc, mcc,
    support}``. ``support`` is the positive count (n_positives), the denominator
    that the C5 confidence interval is conditioned on.

    Clinical reading: sensitivity/PPV answer "when it fires, is it right and does
    it catch the disease"; specificity/NPV answer "when it stays silent, is the
    patient really clear" — the over-calling failure mode of a multi-label model
    lives in the specificity/NPV column, which is exactly what forming TN makes
    visible.
    """
    c = per_class_confusion(y_true, y_prob, threshold)
    return metrics_from_counts(c["tp"], c["fp"], c["fn"], c["tn"])


# ═══════════════════════════════════════════════════════════════════════
# 2. Threshold-free metrics (rank quality, independent of the operating point)
# ═══════════════════════════════════════════════════════════════════════

def _require_sklearn(what: str) -> None:
    if not HAS_SKLEARN:
        raise ImportError(
            f"scikit-learn is required for {what}. Install scikit-learn."
        )


def threshold_free(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    *,
    partial_auroc_max_fpr: float = DEFAULT_PARTIAL_AUROC_MAX_FPR,
) -> Dict[str, float]:
    """Threshold-independent rank metrics: AUROC, AUPRC, partial-AUROC.

    - **AUROC** — rank-invariant under any monotone calibration (Platt is
      monotone), so it is *unaffected* by the tune-on-OOF/report-on-holdout
      threshold bug; it measures pure ranking quality.
    - **AUPRC** (average precision) — the rare-class truth-teller; unlike AUROC
      it is not flattered by a large true-negative pool, so for low-prevalence
      classes it is the honest headline.
    - **partial-AUROC** — AUROC restricted to the high-specificity band
      (``FPR <= partial_auroc_max_fpr``), McClish-standardised to [0.5, 1]. This
      is the only region a screening model may operate in, so a class can have a
      strong full AUROC yet a weak partial one — that gap matters clinically.

    Undefined when only one class is present (no positive or no negative): all
    three return ``nan`` so the caller can flag rather than fabricate a 0.5.
    """
    _require_sklearn("threshold-free metrics (AUROC/AUPRC/partial-AUROC)")
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    n_pos = int(np.count_nonzero(yt == 1))
    n_neg = int(np.count_nonzero(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return {"auroc": math.nan, "auprc": math.nan, "partial_auroc": math.nan}
    return {
        "auroc": float(roc_auc_score(yt, yp)),
        "auprc": float(average_precision_score(yt, yp)),
        "partial_auroc": float(roc_auc_score(yt, yp, max_fpr=partial_auroc_max_fpr)),
    }


def brier(y_true: Sequence[Any], y_prob: Sequence[float]) -> float:
    """Brier score — mean squared error of the probabilities.

    ``mean((p - y)^2)``. The single proper scoring rule that rewards both
    calibration *and* sharpness at once; lower is better, 0 is perfect. Equal to
    ``sklearn.metrics.brier_score_loss`` for 0/1 labels with ``pos_label=1``.
    Returns ``nan`` for an empty input.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if yt.size == 0:
        return math.nan
    return float(np.mean((yp - yt) ** 2))


# ═══════════════════════════════════════════════════════════════════════
# 3. Per-class record + macro / micro / weighted aggregation
# ═══════════════════════════════════════════════════════════════════════

def per_class_record(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    threshold: float,
    *,
    partial_auroc_max_fpr: float = DEFAULT_PARTIAL_AUROC_MAX_FPR,
) -> Dict[str, Any]:
    """One class's full record: 2x2 + operating-point battery + rank metrics.

    The canonical per-class unit a producer builds per class per regime, and the
    element type :func:`aggregate` consumes. Bundles the confusion counts (so
    micro-aggregation can pool them), the operating-point metrics, and the
    threshold-free metrics into one flat dict.
    """
    conf = per_class_confusion(y_true, y_prob, threshold)
    rec: Dict[str, Any] = dict(conf)
    rec.update(metrics_from_counts(conf["tp"], conf["fp"], conf["fn"], conf["tn"]))
    if HAS_SKLEARN:
        rec.update(threshold_free(
            y_true, y_prob, partial_auroc_max_fpr=partial_auroc_max_fpr,
        ))
    else:  # pragma: no cover - sklearn-less fallback keeps the record shape
        rec.update({"auroc": math.nan, "auprc": math.nan, "partial_auroc": math.nan})
    return rec


def _mean_skip_nan(values: List[float], weights: Optional[List[float]] = None) -> float:
    """(Optionally weighted) mean over the finite entries; nan if none survive.

    Macro/weighted AUROC must skip classes whose AUROC is undefined (single-class
    columns ⇒ nan) rather than poison the average — sklearn would raise; we
    report the mean over the classes that *can* be scored and surface the count.
    """
    pairs = [
        (v, (1.0 if weights is None else weights[i]))
        for i, v in enumerate(values)
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    pairs = [(v, w) for (v, w) in pairs if w > 0]
    if not pairs:
        return math.nan
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return math.nan
    return sum(v * w for v, w in pairs) / total_w


def aggregate(
    per_class_list: Sequence[Dict[str, Any]],
    how: str,
    *,
    matrices: Optional[Any] = None,
) -> Dict[str, Any]:
    """Aggregate a list of per-class records into one row, ``how`` ways.

    ``how`` ∈ ``{"macro", "micro", "weighted"}`` — applied to **both** the
    count-based metrics (sens/spec/PPV/NPV/F1/balanced-acc/MCC) **and** the rank
    metrics (AUROC/AUPRC). Micro is net-new for both.

    - **macro** — unweighted mean across classes. Every class counts the same,
      so rare classes pull the headline as hard as common ones (the strict bar).
    - **weighted** — support-weighted mean (weight = positive count). The headline
      the prevalence-mix actually produces.
    - **micro** — pool the 2x2 counts across all classes, then compute the
      count-metrics once from the pooled cells (every decision weighted equally).
      Micro AUROC/AUPRC are genuinely *raveled* quantities, not a function of
      per-class summaries, so they require the full matrices: pass
      ``matrices=(Y_true, Y_prob)`` (each ``(n_samples, n_classes)``). Without it,
      micro AUROC/AUPRC come back ``nan`` (the count-metrics are still exact).

    The macro↔micro gap is itself the diagnostic — a high micro-F1 with a low
    macro-F1 means the common classes are carrying a model that is weak on the
    rare ones.

    Returns a flat dict of the aggregated metrics plus ``how``, ``n_classes``,
    ``n_classes_scored_auroc`` (how many had a defined AUROC), and ``support_total``.
    """
    if how not in AGGREGATION_MODES:
        raise ValueError(f"how must be one of {AGGREGATION_MODES}, got {how!r}")

    records = list(per_class_list)
    n_classes = len(records)
    supports = [float(r.get("support", 0)) for r in records]
    support_total = int(sum(supports))
    aurocs = [r.get("auroc", math.nan) for r in records]
    auprcs = [r.get("auprc", math.nan) for r in records]
    n_scored_auroc = sum(
        1 for v in aurocs if v is not None and not (isinstance(v, float) and math.isnan(v))
    )

    out: Dict[str, Any] = {
        "how": how,
        "n_classes": n_classes,
        "n_classes_scored_auroc": n_scored_auroc,
        "support_total": support_total,
    }

    count_keys = ("sensitivity", "specificity", "ppv", "npv", "f1", "balanced_acc", "mcc")

    if how == "micro":
        tp = int(sum(r.get("tp", 0) for r in records))
        fp = int(sum(r.get("fp", 0) for r in records))
        fn = int(sum(r.get("fn", 0) for r in records))
        tn = int(sum(r.get("tn", 0) for r in records))
        pooled = metrics_from_counts(tp, fp, fn, tn)
        for k in count_keys:
            out[k] = pooled[k]
        out["auroc"], out["auprc"] = _micro_rank_metrics(matrices)
    elif how == "macro":
        for k in count_keys:
            out[k] = _mean_skip_nan([float(r.get(k, 0.0)) for r in records])
        out["auroc"] = _mean_skip_nan(aurocs)
        out["auprc"] = _mean_skip_nan(auprcs)
    else:  # weighted
        for k in count_keys:
            out[k] = _mean_skip_nan(
                [float(r.get(k, 0.0)) for r in records], weights=supports,
            )
        out["auroc"] = _mean_skip_nan(aurocs, weights=supports)
        out["auprc"] = _mean_skip_nan(auprcs, weights=supports)

    return out


def _micro_rank_metrics(matrices: Optional[Any]) -> tuple:
    """Raveled micro AUROC/AUPRC from the full (Y_true, Y_prob) matrices.

    Micro-averaging for a rank metric flattens the entire (n_samples x n_classes)
    decision grid into one binary problem and scores it once — it cannot be
    recovered from per-class scalar summaries, which is why it needs the matrices.
    Returns ``(nan, nan)`` when the matrices are absent or degenerate.
    """
    if matrices is None:
        return math.nan, math.nan
    _require_sklearn("micro-averaged AUROC/AUPRC")
    Y_true, Y_prob = matrices
    yt = np.asarray(Y_true).ravel()
    yp = np.asarray(Y_prob).ravel()
    if int(np.count_nonzero(yt == 1)) == 0 or int(np.count_nonzero(yt == 0)) == 0:
        return math.nan, math.nan
    return float(roc_auc_score(yt, yp)), float(average_precision_score(yt, yp))


# ═══════════════════════════════════════════════════════════════════════
# 4. Confidence intervals (C5 — rare-class statistical power, quantified)
# ═══════════════════════════════════════════════════════════════════════

#: For each proportion metric, the (numerator cell, denominator cells) of the 2x2.
_PROPORTION_COUNTS = {
    "sensitivity": ("tp", ("tp", "fn")),   # tp / (tp + fn)
    "specificity": ("tn", ("tn", "fp")),   # tn / (tn + fp)
    "ppv": ("tp", ("tp", "fp")),           # tp / (tp + fp)
    "npv": ("tn", ("tn", "fn")),           # tn / (tn + fn)
}


def _z_for_level(level: float) -> float:
    """Two-sided normal quantile for a confidence ``level`` (no scipy needed)."""
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def wilson_interval(k: int, n: int, *, level: float = DEFAULT_CI_LEVEL) -> Dict[str, float]:
    """Wilson score interval for a binomial proportion ``k`` of ``n``.

    Preferred over the normal (Wald) interval because it stays inside [0, 1] and
    behaves sanely for tiny ``n`` and proportions near 0 or 1 — exactly the
    rare-class regime where the honest split produces small holdout counts.
    ``n == 0`` ⇒ ``{nan, nan}`` (the proportion is undefined).
    """
    if n <= 0:
        return {"lo": math.nan, "hi": math.nan}
    z = _z_for_level(level)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + (z * z) / (4 * n * n))) / denom
    return {"lo": max(0.0, center - half), "hi": min(1.0, center + half)}


def confidence_interval(
    metric_name: str,
    counts: Dict[str, Any],
    *,
    level: float = DEFAULT_CI_LEVEL,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Confidence interval for one metric, **with its positive count**.

    This is the C5 mechanism: a thin rare class is reported as *a count plus an
    interval*, never as a binary "underpowered" flag — the caller decides what
    width is too wide for its purpose.

    Dispatch by ``metric_name``:

    - **proportion metrics** (sensitivity/specificity/ppv/npv) → **Wilson**
      interval. ``counts`` carries the 2x2: ``{tp, fp, fn, tn}``. The relevant
      numerator/denominator cells are selected per metric.
    - **rank metrics** (auroc/auprc) → **bootstrap** percentile interval.
      ``counts`` carries the raw vectors: ``{"y_true": ..., "y_prob": ...}``.
      Resampling is seeded (``seed``) so the interval is reproducible; resamples
      with a single class present are skipped (the metric is undefined there).

    Returns ``{lo, hi, level, n_pos}``. ``n_pos`` is the positive count the
    estimate rests on (``tp + fn`` for proportions, ``sum(y_true == 1)`` for
    rank metrics) — the headline number for judging statistical power.
    """
    name = metric_name.lower()
    if name in _PROPORTION_COUNTS:
        num_cell, den_cells = _PROPORTION_COUNTS[name]
        k = int(counts.get(num_cell, 0))
        n = int(sum(counts.get(c, 0) for c in den_cells))
        ci = wilson_interval(k, n, level=level)
        n_pos = int(counts.get("tp", 0) + counts.get("fn", 0))
        return {"lo": ci["lo"], "hi": ci["hi"], "level": level, "n_pos": n_pos}

    if name in RANK_METRICS:
        return _bootstrap_rank_ci(
            name, counts, level=level, n_bootstrap=n_bootstrap, seed=seed,
        )

    raise ValueError(
        f"confidence_interval: unsupported metric {metric_name!r}; "
        f"supported = {PROPORTION_METRICS + RANK_METRICS}"
    )


def _bootstrap_rank_ci(
    metric_name: str,
    counts: Dict[str, Any],
    *,
    level: float,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    """Seeded percentile bootstrap CI for AUROC / AUPRC."""
    _require_sklearn(f"bootstrap CI for {metric_name}")
    yt = np.asarray(counts["y_true"])
    yp = np.asarray(counts["y_prob"])
    n_pos = int(np.count_nonzero(yt == 1))
    metric_fn = roc_auc_score if metric_name == "auroc" else average_precision_score

    n = yt.shape[0]
    if n == 0 or n_pos == 0 or n_pos == n:
        return {"lo": math.nan, "hi": math.nan, "level": level, "n_pos": n_pos}

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    stats: List[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(idx, size=n, replace=True)
        yt_s = yt[sample]
        # AUROC needs both classes; AUPRC needs at least one positive.
        if metric_name == "auroc" and len(np.unique(yt_s)) < 2:
            continue
        if metric_name == "auprc" and int(np.count_nonzero(yt_s == 1)) == 0:
            continue
        stats.append(float(metric_fn(yt_s, yp[sample])))

    if len(stats) < 2:
        return {"lo": math.nan, "hi": math.nan, "level": level, "n_pos": n_pos}
    lo_q = (1.0 - level) / 2.0 * 100.0
    hi_q = (1.0 + level) / 2.0 * 100.0
    lo, hi = np.percentile(stats, [lo_q, hi_q])
    return {"lo": float(lo), "hi": float(hi), "level": level, "n_pos": n_pos}


# ═══════════════════════════════════════════════════════════════════════
# 5. Renderable curve data (curves are data, not pictures)
# ═══════════════════════════════════════════════════════════════════════

def _finite(value: float) -> float:
    """JSON-safe float: sklearn's ROC thresholds lead with ``+inf`` (predict
    nothing); clip the non-finite endpoints so the arrays serialise."""
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    return float(value)


def curve_data(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    *,
    threshold_sweep_points: int = DEFAULT_THRESHOLD_SWEEP_POINTS,
) -> Dict[str, Any]:
    """Renderable ROC / PR / threshold-sweep arrays for one class.

    Returns::

        {
          "roc": {"fpr": [...], "tpr": [...], "thr": [...]},
          "pr":  {"precision": [...], "recall": [...], "thr": [...]},
          "threshold_sweep": [{"thr": t, "f1": ..., "sens": ..., "spec": ...}, ...],
        }

    ROC/PR come from sklearn's curve builders (the same points the scalar AUROC/
    AUPRC integrate, so a panel showing a curve and its area is self-consistent).
    The threshold-sweep is the operating-point view: F1, sensitivity and
    specificity at each candidate threshold — it makes the threshold *choice*
    legible, which is the whole point of tune-on-OOF/report-on-holdout. ROC/PR
    come back empty when only one class is present (undefined). ``thr`` is a
    keyword knob so resolution is never a magic literal.
    """
    _require_sklearn("curve data (ROC/PR)")
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    has_both = int(np.count_nonzero(yt == 1)) > 0 and int(np.count_nonzero(yt == 0)) > 0

    if has_both:
        fpr, tpr, roc_thr = roc_curve(yt, yp)
        precision, recall, pr_thr = precision_recall_curve(yt, yp)
        roc = {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thr": [_finite(v) for v in roc_thr],
        }
        pr = {
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
            "thr": [float(v) for v in pr_thr],
        }
    else:
        roc = {"fpr": [], "tpr": [], "thr": []}
        pr = {"precision": [], "recall": [], "thr": []}

    sweep: List[Dict[str, float]] = []
    for thr in np.linspace(0.0, 1.0, threshold_sweep_points):
        m = per_class_metrics(yt, yp, float(thr))
        sweep.append({
            "thr": float(thr),
            "f1": m["f1"],
            "sens": m["sensitivity"],
            "spec": m["specificity"],
        })

    return {"roc": roc, "pr": pr, "threshold_sweep": sweep}


__all__ = [
    # confusion + operating-point battery
    "per_class_confusion",
    "metrics_from_counts",
    "per_class_metrics",
    "per_class_record",
    # threshold-free
    "threshold_free",
    "brier",
    # aggregation
    "aggregate",
    # confidence intervals (C5)
    "wilson_interval",
    "confidence_interval",
    # curves
    "curve_data",
    # documented defaults
    "DEFAULT_CI_LEVEL",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_PARTIAL_AUROC_MAX_FPR",
    "DEFAULT_THRESHOLD_SWEEP_POINTS",
    "AGGREGATION_MODES",
    "PROPORTION_METRICS",
    "RANK_METRICS",
]
