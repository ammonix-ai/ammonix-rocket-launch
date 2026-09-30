"""Parametric stratified holdout split + K-fold CV."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("factory.cross_validation.splitting")

# Multi-label stratification (optional but strongly recommended). Mirrors the
# detection so behaviour is identical.
try:
    from iterstrat.ml_stratifiers import (
        MultilabelStratifiedKFold,
        MultilabelStratifiedShuffleSplit,
    )
    HAS_ITERSTRAT = True
except ImportError:  # pragma: no cover - exercised only in an iterstrat-less env
    HAS_ITERSTRAT = False
    log.warning(
        "iterstrat not installed — falling back to single-label stratification. "
        "Install: pip install iterative-stratification"
    )


def resolve_holdout_size(
    n_samples: int,
    n_folds: int,
    holdout_frac: Optional[float] = None,
    holdout_size: Optional[int] = None,
) -> int:
    """Resolve the absolute holdout count from a fraction (preferred) or override.

    WHY a *fraction*, never a fixed count: the cohort
    is not fixed, so the holdout must scale with it. The split runs identically
    at any cohort size because the holdout is a stratified *fraction* — never a
    magic absolute count. A small cohort yields a small holdout; that is honest
    (surfaced downstream by the per-class positive count + confidence interval),
    NOT a defect — so there is deliberately no minimum-holdout floor beyond
    ``>= 1``.

    WHY the default is ``1/(n_folds + 1)`` (parametric "sixths"): the cohort is
    partitioned into ``n_folds + 1`` equal stratified portions — ``n_folds`` work
    folds for the CV (→ OOF) plus ONE pristine holdout (the final-test "6th" for
    the default 5 folds). Tying the holdout to the fold count keeps every portion
    the same size and makes the split a single parametric machine, with no second
    hardcoded knob to drift from ``n_folds``.

    Precedence: an explicit ``holdout_size`` (int) wins — for a caller or test
    that wants an exact count; otherwise the count is derived from
    ``holdout_frac`` when given, else the ``1/(n_folds + 1)`` default. The derived
    count is floored at 1 so even a tiny cohort still yields a holdout.
    """
    if holdout_size is not None:
        return int(holdout_size)
    frac = holdout_frac if holdout_frac is not None else 1.0 / (n_folds + 1)
    return max(1, int(n_samples * frac))


def _primary_label_keys(
    Y: np.ndarray,
    class_names: Optional[Sequence[str]],
) -> List[Any]:
    """Single-label stratification key per row, for the iterstrat-less fallback.

    Reproduces the pre-lift convention exactly: the key is the class at
    ``argmax`` of the row (``class_names[argmax]`` when names are supplied, else
    the integer column index), and a label-less row gets the ``"_none_"``
    sentinel (or ``-1`` when names are absent). Used only when ``iterstrat`` is
    unavailable; with it installed the partition is a pure function of ``Y`` and
    these keys are never formed.
    """
    keys: List[Any] = []
    for i in range(Y.shape[0]):
        row = Y[i]
        if row.sum() > 0:
            j = int(np.argmax(row))
            keys.append(class_names[j] if class_names is not None else j)
        else:
            keys.append("_none_" if class_names is not None else -1)
    return keys


# ─────────────────────────── group-atomic helpers (WS-1) ───────────────────────────
#
# TODO(WS-6): a QIS-on-predicted-set split can reuse this exact collapse→split→
# expand machinery — pass the predicted-positive subset's subject ids as
# ``groups`` and the same atomicity guarantee holds for that arm.

def _collapse_to_groups(
    Y: np.ndarray,
    groups: Sequence[Any],
) -> Tuple[List[Any], np.ndarray, Dict[Any, List[int]]]:
    """Collapse rows into unique subjects for group-atomic splitting (WS-1).

    Returns ``(group_keys, Y_g, group_to_rows)``:

    - ``group_keys`` — the unique group identifiers in FIRST-APPEARANCE order.
      First-appearance (not sorted) is deliberate: when every row is its own
      group (``groups=[0..n-1]``) the group order equals the row order, so the
      splitter output is byte-identical to the ungrouped path.
    - ``Y_g`` — the ``(n_groups, n_classes)`` SUBJECT SIGNATURE: a subject is
      positive for class ``c`` iff ANY of its rows is (logical OR). This is what
      multilabel stratification balances, so subject-level prevalence is kept
      even across work/holdout and folds.
    - ``group_to_rows`` — ``{group_key: [row indices, ascending]}``.
    """
    if len(groups) != Y.shape[0]:
        raise ValueError(
            f"groups length ({len(groups)}) != n_samples ({Y.shape[0]})"
        )
    group_keys: List[Any] = []
    group_to_rows: Dict[Any, List[int]] = {}
    for row_idx, g in enumerate(groups):
        rows = group_to_rows.get(g)
        if rows is None:
            group_to_rows[g] = [row_idx]
            group_keys.append(g)
        else:
            rows.append(row_idx)
    Y_g = np.zeros((len(group_keys), Y.shape[1]), dtype=Y.dtype)
    for gi, g in enumerate(group_keys):
        Y_g[gi] = (Y[group_to_rows[g]].sum(axis=0) > 0)
    return group_keys, Y_g, group_to_rows


def _expand_groups_to_rows(
    group_indices: Sequence[int],
    group_keys: Sequence[Any],
    group_to_rows: Dict[Any, List[int]],
) -> List[int]:
    """Expand group-level indices back to row indices.

    The expansion PRESERVES the order the splitter returned groups in (it does
    NOT sort) so the all-singleton case reproduces the ungrouped path's exact
    index order — the byte-identity guarantee for ``groups=[0..n-1]``.
    """
    rows: List[int] = []
    for gi in group_indices:
        rows.extend(group_to_rows[group_keys[gi]])
    return rows


def _shuffle_split_indices(
    M: np.ndarray,
    test_count: int,
    seed: int,
    class_names: Optional[Sequence[str]],
) -> Tuple[List[int], List[int]]:
    """One stratified shuffle split of matrix ``M`` into ``(train, test)`` rows.

    The shared holdout core: ``MultilabelStratifiedShuffleSplit`` when
    ``iterstrat`` is available (the production path) else a single-label
    ``train_test_split`` with rare-singleton merging. Calling it with ``M=Y``
    reproduces the pre-WS-1 ``stratified_holdout_split`` BYTE-FOR-BYTE; the group
    path calls it with the collapsed subject matrix ``M=Y_g``.
    """
    n = M.shape[0]
    if HAS_ITERSTRAT:
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_count, random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(np.zeros(n), M))
        return train_idx.tolist(), test_idx.tolist()
    # pragma: no cover - iterstrat-less fallback
    from sklearn.model_selection import train_test_split

    primary_labels = _primary_label_keys(M, class_names)
    # Merge rare classes (<2 members) into the most common key to avoid
    # sklearn's "too few members" error in stratified splitting.
    label_counts = Counter(primary_labels)
    rare_classes = {lbl for lbl, cnt in label_counts.items() if cnt < 2}
    if rare_classes:
        most_common = label_counts.most_common(1)[0][0]
        safe_labels = [
            most_common if lbl in rare_classes else lbl for lbl in primary_labels
        ]
        log.warning(
            "  Merged %d rare singleton classes into %r for stratification: %s",
            len(rare_classes), most_common, sorted(map(str, rare_classes)),
        )
    else:
        safe_labels = primary_labels
    train_idx, test_idx = train_test_split(
        list(range(n)), test_size=test_count, stratify=safe_labels, random_state=seed,
    )
    return train_idx, test_idx


def _kfold_indices(
    M: np.ndarray,
    n_folds: int,
    seed: int,
    class_names: Optional[Sequence[str]],
) -> List[Tuple[List[int], List[int]]]:
    """K-fold split of matrix ``M`` into ``(train, val)`` row tuples.

    The shared CV core: ``MultilabelStratifiedKFold`` when ``iterstrat`` is
    available else a single-label ``StratifiedKFold`` fallback. ``M=Y``
    reproduces the pre-WS-1 ``stratified_cv_folds`` byte-for-byte; the group path
    passes the collapsed subject matrix ``M=Y_g``.
    """
    n = M.shape[0]
    if HAS_ITERSTRAT:
        kfold = MultilabelStratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed,
        )
        return [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in kfold.split(np.zeros(n), M)
        ]
    # pragma: no cover - iterstrat-less fallback
    from sklearn.model_selection import StratifiedKFold

    primary_labels = _primary_label_keys(M, class_names)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in skf.split(np.zeros(n), primary_labels)
    ]


def stratified_holdout_split(
    Y: np.ndarray,
    holdout_size: int,
    *,
    seed: int = 42,
    class_names: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[Any]] = None,
) -> Tuple[List[int], List[int]]:
    """Stratified multi-label split of ``Y`` into a working set and a holdout.

    Sample-level: every row of ``Y`` lands in exactly one split. Uses
    ``MultilabelStratifiedShuffleSplit`` when ``iterstrat`` is available (the
    production path), otherwise a single-label ``train_test_split`` fallback with
    rare-singleton merging.

    ``holdout_size`` is an explicit absolute count and is REQUIRED — there is no
    magic default (the retired ``2800``). Callers fraction-ize it via
    :func:`resolve_holdout_size` and pass the resolved int, keeping this splitter
    size-agnostic. ``class_names`` (optional) only affects the fallback grouping
    key + log text; the iterstrat partition is a pure function of ``Y`` + ``seed``.

    ``groups`` (optional, WS-1) — one subject id per row. ``None`` (default) is
    BYTE-IDENTICAL to the pre-WS-1 behaviour. When supplied, whole subjects are
    assigned atomically (no patient straddles work/holdout): rows collapse to
    subjects, the subject signatures are split to a target subject-count whose
    rows ≈ ``holdout_size``, then expand back to rows. Because subjects are
    atomic the realized holdout row-count is APPROXIMATE (surfaced in the log, not
    asserted exact).

    Returns ``(work_indices, holdout_indices)`` — row indices into ``Y``.
    """
    Y = np.asarray(Y)
    n = Y.shape[0]
    if holdout_size >= n:
        raise ValueError(f"holdout_size ({holdout_size}) >= total samples ({n})")

    if groups is None:
        log.info(
            "Using %s (holdout=%d/%d)",
            "MultilabelStratifiedShuffleSplit" if HAS_ITERSTRAT
            else "single-label train_test_split",
            holdout_size, n,
        )
        work_indices, holdout_indices = _shuffle_split_indices(
            Y, holdout_size, seed, class_names,
        )
    else:
        group_keys, Y_g, group_to_rows = _collapse_to_groups(Y, groups)
        n_groups = len(group_keys)
        if n_groups < 2:
            raise ValueError(
                f"group-aware holdout needs >= 2 distinct groups, got {n_groups}"
            )
        # Target subject count whose rows ≈ holdout_size (atomic → approximate).
        target_groups = int(round(holdout_size * n_groups / n))
        target_groups = max(1, min(target_groups, n_groups - 1))
        work_g, hold_g = _shuffle_split_indices(
            Y_g, target_groups, seed, class_names,
        )
        work_indices = _expand_groups_to_rows(work_g, group_keys, group_to_rows)
        holdout_indices = _expand_groups_to_rows(hold_g, group_keys, group_to_rows)
        log.info(
            "Group-aware holdout: %d/%d subjects → %d/%d rows "
            "(requested %d holdout rows; atomic-subject realized count)",
            len(hold_g), n_groups, len(holdout_indices), n, holdout_size,
        )

    # Audit: warn when any class is absent from a split (too rare to stratify).
    Y_work = Y[work_indices]
    Y_hold = Y[holdout_indices]
    for ci in range(Y.shape[1]):
        n_work = int(Y_work[:, ci].sum())
        n_hold = int(Y_hold[:, ci].sum())
        if n_work == 0 or n_hold == 0:
            label = class_names[ci] if class_names is not None else f"class[{ci}]"
            log.warning("  %s: work=%d, holdout=%d — may be too rare!", label, n_work, n_hold)

    log.info("Split: %d working + %d holdout", len(work_indices), len(holdout_indices))
    return work_indices, holdout_indices


def stratified_cv_folds(
    Y: np.ndarray,
    n_folds: int = 5,
    *,
    seed: int = 42,
    class_names: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[Any]] = None,
) -> List[Tuple[List[int], List[int]]]:
    """Stratified multi-label K-fold split of ``Y`` (the working set).

    Uses ``MultilabelStratifiedKFold`` when ``iterstrat`` is available, otherwise
    a single-label ``StratifiedKFold`` fallback. ``class_names`` (optional) only
    affects the fallback grouping key; the iterstrat folds are a pure function of
    ``Y`` + ``seed``.

    ``groups`` (optional, WS-1) — one subject id per row of ``Y`` (here the rows
    of the WORKING set). ``None`` (default) is BYTE-IDENTICAL to the pre-WS-1
    behaviour. When supplied, whole subjects are assigned to folds atomically so
    no patient's recordings straddle two folds; the subject signatures are
    K-fold-split then expanded back to row indices.

    Returns a list of ``(train_idx, val_idx)`` tuples whose indices are LOCAL to
    the rows of ``Y`` (``0 .. n_samples-1``).
    """
    Y = np.asarray(Y)
    n = Y.shape[0]

    if groups is None:
        log.info(
            "Using %s (k=%d)",
            "MultilabelStratifiedKFold" if HAS_ITERSTRAT else "single-label StratifiedKFold",
            n_folds,
        )
        folds = _kfold_indices(Y, n_folds, seed, class_names)
    else:
        group_keys, Y_g, group_to_rows = _collapse_to_groups(Y, groups)
        n_groups = len(group_keys)
        if n_groups < n_folds:
            raise ValueError(
                f"group-aware CV needs >= n_folds distinct groups, got "
                f"{n_groups} groups for {n_folds} folds"
            )
        log.info("Group-aware MultilabelStratifiedKFold (k=%d, %d subjects → row folds)",
                 n_folds, n_groups)
        group_folds = _kfold_indices(Y_g, n_folds, seed, class_names)
        folds = [
            (
                _expand_groups_to_rows(train_g, group_keys, group_to_rows),
                _expand_groups_to_rows(val_g, group_keys, group_to_rows),
            )
            for train_g, val_g in group_folds
        ]

    for fi, (train_idx, val_idx) in enumerate(folds):
        log.info("  Fold %d: train=%d, val=%d", fi, len(train_idx), len(val_idx))

    return folds


__all__ = [
    "resolve_holdout_size",
    "stratified_holdout_split",
    "stratified_cv_folds",
    "HAS_ITERSTRAT",
]
