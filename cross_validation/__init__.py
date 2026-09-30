"""Cross-validation and evaluation primitives of the Ammonix platform.

Domain-agnostic: they speak only label matrices, probability vectors and counts, so any
domain inherits the same stratified-split and leak-aware cross-validation discipline and the
same per-class metric suite.

- :mod:`cross_validation.splitting`: the parametric stratified holdout split + K-fold CV.
  The holdout is a stratified *fraction* of the cohort (default ``1/(n_folds+1)``), never a
  fixed absolute count. See :func:`~cross_validation.splitting.resolve_holdout_size`.
- :mod:`cross_validation.metrics`: the per-class 2x2 and operating-point battery
  (sens/spec/PPV/NPV/F1/balanced-acc/MCC), threshold-free AUROC/AUPRC/partial-AUROC, Brier,
  macro/micro/weighted aggregation, confidence intervals (Wilson + bootstrap) and curve data.
"""

from __future__ import annotations

from cross_validation.splitting import (
    HAS_ITERSTRAT,
    resolve_holdout_size,
    stratified_cv_folds,
    stratified_holdout_split,
)

__all__ = [
    "resolve_holdout_size",
    "stratified_holdout_split",
    "stratified_cv_folds",
    "HAS_ITERSTRAT",
]
