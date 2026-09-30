#!/usr/bin/env python
"""Run the Ammonix platform code on the launch-vehicle domain: Layers 1-4.

Simulates a campaign of launches, then runs the platform's own code on it:

  Layer 1  layer-1/taxonomy.json         failure-mode catalog with counts (>= 50 per class)
  Layer 2  layer-2/rules.json            discovery/ unchanged: correlation -> threshold
                                         rules -> evidence grades -> holdout validation
  Layer 3  layer-3/knowledge-graph.json  subsystems, channels, failure modes, rules, actions
  Layer 4  layer-4/                      ammonix-style models: one gradient-boosted classifier
                                         per failure mode, trained on time-truncated rows and
                                         split by cross_validation/ (launch-level
                                         holdout + 5 folds). Alert thresholds are calibrated on
                                         out-of-fold *streaming* replays to a false-alarm budget.

The trained models go to layer-4/model.pkl, and every launch's score at every streaming tick
to layer-4/streaming_scores.npz for the knowledge universe (ControlRoom/build_universe.py).
Neither is committed: regenerate them with this script.

Usage:
    python run_factory.py [--n-launches 3000] [--seed 2026] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

DOMAIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOMAIN_DIR
sys.path[:0] = [str(DOMAIN_DIR), str(REPO_ROOT)]

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.stats import poisson  # noqa: E402

from discovery.correlation import compute_all_correlations  # noqa: E402
from discovery.formulation import (  # noqa: E402
    discover_equivalence_groups, discover_mutex_groups, generate_threshold_rules)
from discovery.grading import grade_equivalence, grade_mutex, grade_rules  # noqa: E402
from discovery.integration import (  # noqa: E402
    assemble_rules_json, check_consistency, classify_handoff)
from discovery.types import DiscoveryReport  # noqa: E402
from discovery.validation import validate_rules  # noqa: E402
from cross_validation.metrics import aggregate, per_class_record  # noqa: E402
from cross_validation.splitting import (  # noqa: E402
    HAS_ITERSTRAT, resolve_holdout_size, stratified_cv_folds, stratified_holdout_split)
from launch_sim import vehicle as V  # noqa: E402
from launch_sim.anomalies import BY_KEY, FAILURE_KEYS, FAILURE_MODES, NOMINAL_KEY  # noqa: E402
from launch_sim.features import FEATURE_INDEX, FEATURE_NAMES, FEATURE_SPECS, feature_matrix  # noqa: E402
from launch_sim.streaming import DEBOUNCE_TICKS, TICKS, TICK_S, debounced  # noqa: E402
from launch_sim.simulate import launch_seed, simulate_launch  # noqa: E402

DOMAIN = "launch-vehicle"
VERSION = "0.1.0"
TRAIN_FIXED_TIMES = (-1.0, 6.5, 150.0)   # decision gates and end of window, always trained on
N_RANDOM_TIMES = 18                       # plus random evaluation times per launch (t >= -50 s)
RANDOM_FROM = -50.0
FA_BUDGET = 0.0015                        # per class: share of launches without that failure
                                          # on which a (debounced) alert may fire
REF_TIMES = tuple(sorted(set(np.arange(-57.5, -27.4, 2.5).tolist())
                         | set(np.arange(-25.0, 150.1, 5.0).tolist())
                         | {-1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5}))
N_FOLDS = 5
MIN_POSITIVE = 50               # Layer 1 -> 2 acceptance
MUTEX_MAX_COOCCURRENCE = 0.005  # stricter than the platform default: independent failures
                                # co-occur ~3 %, so 2 % would admit sampling noise as "mutex"
F1_TARGET = 0.80                # spec.yaml acceptance: holdout macro F1 at H0 + 150 s
HGB_PARAMS = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                  l2_regularization=1.0, random_state=0)
CATEGORY_LABELS = {
    "pressurization": "Propellant pressurization", "electrical": "Electrical",
    "core_engine": "Core engine", "instrumentation": "Instrumentation",
    "boosters": "Solid boosters", "flight_control": "Flight control",
    "structural_dynamics": "Structural dynamics", "guidance": "Guidance & navigation",
    "nominal": "Nominal",
}
CALL_ACTIONS = {
    "HOLD": "Stop the countdown clock (hold) and recycle",
    "ABORT": "Shut the core engine down before booster ignition",
    "CRITICAL": "Alert mission director and flight safety: critical in-flight anomaly",
    "WARNING": "Alert mission director: degraded in-flight performance",
    "ADVISORY": "Flag the sensor as failed; do not act on its redline without corroboration",
}


# ---------------------------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------------------------

def _simulate_rows(args: Tuple[int, int]) -> Tuple[int, np.ndarray, Dict[str, Dict[str, float]]]:
    """One launch: its feature vector at every streaming tick, and its labels."""
    i, base = args
    seed = launch_seed(i, base)
    launch = simulate_launch(seed)
    x = feature_matrix(launch.tel, TICKS).astype(np.float32)            # (T, F)
    labels = {k: {"severity": v["severity"], "onset": v["onset"], "manifest": v["manifest"]}
              for k, v in launch.labels.items()}
    return seed, x, labels


def simulate_campaign(n: int, base: int, workers: int):
    with Pool(workers) as pool:
        results = pool.map(_simulate_rows, [(i, base) for i in range(n)], chunksize=8)
    seeds = np.array([r[0] for r in results], dtype=np.int64)
    x_all = np.stack([r[1] for r in results])                           # (N, T, F)
    labels = [r[2] for r in results]
    y_launch = np.array([[k in lab for k in FAILURE_KEYS] for lab in labels], dtype=int)
    manifest = np.full((n, len(FAILURE_KEYS)), np.inf)
    for i, lab in enumerate(labels):
        for j, k in enumerate(FAILURE_KEYS):
            if k in lab:
                manifest[i, j] = lab[k]["manifest"]
    return seeds, x_all, y_launch, manifest, labels


def training_ticks(seed: int) -> np.ndarray:
    """Tick indices a launch contributes as training rows: gates + random times."""
    fixed = [int(np.argmin(np.abs(TICKS - t))) for t in TRAIN_FIXED_TIMES]
    pool = np.nonzero(TICKS >= RANDOM_FROM)[0]
    rng = np.random.default_rng(seed)
    extra = rng.choice(pool, size=N_RANDOM_TIMES, replace=False)
    return np.unique(np.concatenate([fixed, extra]))


def row_labels(manifest_row: np.ndarray, tick_idx: np.ndarray) -> np.ndarray:
    """A failure is a positive row once it is visible (manifest <= t)."""
    return (manifest_row[None, :] <= TICKS[tick_idx][:, None]).astype(int)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def _clean(obj: Any) -> Any:
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python, 4-6 significant digits."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if not math.isfinite(f) else float(f"{f:.6g}")
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(data), indent=2) + "\n", encoding="utf-8")


def _fmt(x: float) -> str:
    return f"{x:.3g}" if abs(x) < 1000 else f"{x:.0f}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------------------------

def build_taxonomy(y_launch: np.ndarray, labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(labels)
    diagnoses = []
    for j, fm in enumerate(FAILURE_MODES):
        count = int(y_launch[:, j].sum())
        sev = [lab[fm.key]["severity"] for lab in labels if fm.key in lab]
        diagnoses.append({
            "canonical": fm.key, "display": fm.display, "category": fm.category,
            "phase": fm.phase, "call": fm.call, "severity": fm.severity,
            "subsystem": fm.subsystem, "channels": list(fm.channels),
            "description": fm.description, "positive_count": count,
            "prevalence": count / n, "median_severity": float(np.median(sev)) if sev else None,
            "is_default": False, "meets_min_positive": count >= MIN_POSITIVE,
        })
    n_nominal = int((y_launch.sum(axis=1) == 0).sum())
    diagnoses.append({
        "canonical": NOMINAL_KEY, "display": "Nominal", "category": "nominal", "phase": "all",
        "call": "GO", "severity": "none", "subsystem": None, "channels": [],
        "description": "No failure injected: the launch runs within normal variation.",
        "positive_count": n_nominal, "prevalence": n_nominal / n, "median_severity": None,
        "is_default": True, "meets_min_positive": n_nominal >= MIN_POSITIVE,
    })
    categories: Dict[str, Dict[str, Any]] = {}
    for d in diagnoses:
        cat = categories.setdefault(d["category"], {"label": CATEGORY_LABELS[d["category"]],
                                                    "diagnoses": []})
        cat["diagnoses"].append(d["canonical"])
    multi = int((y_launch.sum(axis=1) >= 2).sum())
    return {
        "meta": {
            "domain": DOMAIN, "version": VERSION, "generated_by": "run_factory.py",
            "created_at": _now(), "canonical_count": len(diagnoses), "launch_count": n,
            "multi_failure_launches": multi, "min_positive": MIN_POSITIVE,
            "acceptance_pass": all(d["meets_min_positive"] for d in diagnoses),
            "note": "Synthetic campaign from a notional heavy-launcher model. "
                    "Failures are oversampled (6 % base rate each).",
        },
        "diagnoses": diagnoses,
        "categories": categories,
    }


# ---------------------------------------------------------------------------------------------
# Layer 2 (platform discovery/ code, unchanged)
# ---------------------------------------------------------------------------------------------

def run_discovery(x_work: np.ndarray, y_work: np.ndarray, x_hold: np.ndarray,
                  y_hold: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cols = [i for i, name in enumerate(FEATURE_NAMES) if not name.startswith("CTX__")]
    names = [FEATURE_NAMES[i] for i in cols]
    xw = x_work[:, cols].astype(np.float64)
    xh = x_hold[:, cols].astype(np.float64)
    med = np.nanmedian(xw, axis=0)
    xw = np.where(np.isnan(xw), med, xw)
    xh = np.where(np.isnan(xh), med, xh)
    keys = list(FAILURE_KEYS)

    correlations = compute_all_correlations(xw, y_work, names, keys, min_positive=10, top_k=20)
    rules = generate_threshold_rules(xw, y_work, names, keys, correlations, top_k=5,
                                     n_thresholds=50)
    grade_rules(rules)
    mutex = discover_mutex_groups(y_work, keys, threshold=MUTEX_MAX_COOCCURRENCE)
    grade_mutex(mutex)
    equiv = discover_equivalence_groups(StandardScaler().fit_transform(xw), y_work, names, keys)
    grade_equivalence(equiv)
    validation = validate_rules(rules, xh, y_hold, names, keys)
    rules_json = assemble_rules_json(DOMAIN, rules, mutex, equiv, grade_cutoff="B")
    warnings = check_consistency(rules_json)
    report = DiscoveryReport(domain=DOMAIN, n_patients=len(xw), n_features=len(names),
                             n_diagnoses=len(keys), correlations=correlations,
                             threshold_rules=rules, mutex_groups=mutex,
                             equivalence_groups=equiv, validation_results=validation)
    handoff = classify_handoff(report)

    by_id = {v["rule_id"]: v for v in validation.get("rules", [])}
    rules_json["threshold_rules"].sort(key=lambda r: (keys.index(r["diagnosis"]), -r["f1"]))
    for n, r in enumerate(rules_json["threshold_rules"], start=1):
        spec = FEATURE_SPECS[FEATURE_INDEX[r["feature"]]]
        fm = BY_KEY[r["diagnosis"]]
        r["short_id"] = f"R-{n:02d}"
        r["sentence"] = (f"IF {spec.description} {r['operator']} {_fmt(r['threshold'])} "
                         f"{spec.unit} THEN {fm.display}")
        r["feature_unit"] = spec.unit
        r["feature_valid_from"] = spec.valid_from
        r["channels"] = list(spec.channels)
        v = by_id.get(r["id"], {})
        r["holdout"] = {k: v.get(k) for k in ("validation_f1", "val_sensitivity",
                                               "val_specificity", "overfitting_flag")}
    for m in rules_json["mutex_rules"]:
        m["sentence"] = " and ".join(BY_KEY[k].display for k in m["members"]) + \
            " were never seen together"
    covered = {r["diagnosis"] for r in rules_json["threshold_rules"]}
    rules_json["meta"].update({
        "source_launches": len(xw), "feature_count": len(names),
        "mutex_max_cooccurrence": MUTEX_MAX_COOCCURRENCE,
        "classes_with_b_plus_rule": sorted(covered),
        "layer4_gap": sorted(set(keys) - covered),
        "acceptance_pass": covered == set(keys),
        "consistency_warnings": warnings,
        "handoff_counts": {k: len(v) for k, v in handoff.items()},
    })
    discovery_report = report.to_dict()
    discovery_report["correlations"] = discovery_report["correlations"][:200]
    discovery_report["handoff"] = handoff
    return rules_json, discovery_report


# ---------------------------------------------------------------------------------------------
# Layer 4 (ammonix-style one-vs-rest gradient boosting on time-truncated rows)
# ---------------------------------------------------------------------------------------------

def fit_models(x: np.ndarray, y: np.ndarray) -> List[HistGradientBoostingClassifier]:
    models = []
    for j in range(y.shape[1]):
        clf = HistGradientBoostingClassifier(**HGB_PARAMS)
        clf.fit(x, y[:, j])
        models.append(clf)
    return models


def predict(models: Sequence[HistGradientBoostingClassifier], x: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict_proba(x)[:, 1] for m in models])


def calibrate_thresholds(p_stream: np.ndarray, y_launch: np.ndarray
                         ) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Per class, the lowest threshold whose debounced alert fires on at most FA_BUDGET of the
    launches without that failure (out-of-fold streaming replays)."""
    peak = debounced(p_stream, axis=1).max(axis=1)                      # (n, K)
    thresholds, detail = {}, {}
    for j, k in enumerate(FAILURE_KEYS):
        neg, pos = peak[y_launch[:, j] == 0, j], peak[y_launch[:, j] == 1, j]
        thr = float(np.clip(np.quantile(neg, 1.0 - FA_BUDGET, method="higher") + 1e-4, 0.05, 0.995))
        thresholds[k] = thr
        detail[k] = {"oof_false_alarm_rate": float((neg >= thr).mean()),
                     "oof_detection_rate": float((pos >= thr).mean()) if pos.size else None,
                     "n_negative": int(neg.size), "n_positive": int(pos.size)}
    return thresholds, detail


def streaming_metrics(p_stream: np.ndarray, y_launch: np.ndarray, manifest: np.ndarray,
                      thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Debounced first-alert times on replayed launches: detection, latency, false alarms."""
    held = debounced(p_stream, axis=1)
    out: Dict[str, Any] = {}
    gates = {"countdown": -1.0, "engine_start": 6.5}
    for j, fm in enumerate(FAILURE_MODES):
        above = held[:, :, j] >= thresholds[fm.key]
        any_alert = above.any(axis=1)
        first_t = np.where(any_alert, TICKS[np.argmax(above, axis=1)], np.nan)
        pos = y_launch[:, j] == 1
        lat = first_t[pos] - manifest[pos, j]
        gate = gates.get(fm.phase)
        if fm.key == "COMBUSTION_INSTABILITY":
            in_time = np.where(manifest[pos, j] <= 6.5, first_t[pos] <= 6.5, np.isfinite(first_t[pos]))
        else:
            in_time = first_t[pos] <= gate if gate is not None else np.isfinite(first_t[pos])
        out[fm.key] = {
            "n_launches": int(pos.sum()),
            "detection_rate": float(np.isfinite(first_t[pos]).mean()) if pos.any() else None,
            "in_time_rate": float(np.mean(in_time)) if pos.any() else None,
            "gate": gate,
            "median_latency_s": float(np.nanmedian(lat)) if np.isfinite(lat).any() else None,
            "false_alerts_per_1000": 1000.0 * float(any_alert[~pos].mean()),
        }
    return out


def _records(y: np.ndarray, p: np.ndarray, thresholds: Dict[str, float],
             keys: Sequence[str]) -> Dict[str, Any]:
    cols = [FAILURE_KEYS.index(k) for k in keys]
    per = {k: per_class_record(y[:, j], p[:, j], thresholds[k]) for k, j in zip(keys, cols)}
    recs = list(per.values())
    return {"per_class": per, "macro": aggregate(recs, "macro"),
            "micro": aggregate(recs, "micro", matrices=(y[:, cols], p[:, cols]))}


# ---------------------------------------------------------------------------------------------
# Layer 3
# ---------------------------------------------------------------------------------------------

def build_knowledge_graph(rules_json: Dict[str, Any], correlations: List[Dict[str, Any]],
                          y_work: np.ndarray, thresholds: Dict[str, float],
                          n_launches: int) -> Dict[str, Any]:
    keys = list(FAILURE_KEYS)
    edges: List[Dict[str, Any]] = []
    top: Dict[str, List[Dict[str, Any]]] = {k: [] for k in keys}
    for c in correlations:
        if len(top[c["diagnosis"]]) < 5:
            top[c["diagnosis"]].append(c)

    feature_ids = {r["feature"] for r in rules_json["threshold_rules"]}
    feature_ids |= {c["feature"] for cs in top.values() for c in cs}
    features = []
    for name in sorted(feature_ids):
        spec = FEATURE_SPECS[FEATURE_INDEX[name]]
        features.append({"id": f"f:{name}", "label": spec.description, "unit": spec.unit,
                         "phase": spec.phase, "channels": list(spec.channels)})
        for ch in spec.channels:
            edges.append({"type": "MEASURED_ON", "from": f"f:{name}", "to": f"ch:{ch}"})

    subsystems = [{"id": f"sub:{k}", "label": v} for k, v in V.SUBSYSTEMS.items()]
    channels = [{"id": f"ch:{c.key}", "label": c.label, "unit": c.unit,
                 "subsystem": c.subsystem} for c in V.CHANNELS]
    for c in V.CHANNELS:
        edges.append({"type": "PART_OF", "from": f"ch:{c.key}", "to": f"sub:{c.subsystem}"})

    base = y_work.mean(axis=0)
    modes = []
    for j, fm in enumerate(FAILURE_MODES):
        modes.append({
            "id": f"fm:{fm.key}", "label": fm.display, "canonical": fm.key,
            "category": fm.category, "phase": fm.phase, "call": fm.call,
            "severity": fm.severity, "subsystem": fm.subsystem, "base_rate": float(base[j]),
            "threshold": thresholds[fm.key],
            "top_features": [{"feature_id": f"f:{c['feature']}", "r": c["correlation_r"],
                              "cohens_d": c["cohens_d"]} for c in top[fm.key]],
        })
        edges.append({"type": "LOCATED_IN", "from": f"fm:{fm.key}", "to": f"sub:{fm.subsystem}"})
        for ch in fm.channels:
            edges.append({"type": "SEEN_ON", "from": f"fm:{fm.key}", "to": f"ch:{ch}"})
        for c in top[fm.key]:
            edges.append({"type": "DISCRIMINATED_BY", "from": f"fm:{fm.key}",
                          "to": f"f:{c['feature']}", "r": c["correlation_r"],
                          "cohens_d": c["cohens_d"]})
        calls = ["ABORT", "CRITICAL"] if fm.phase == "engine_start_or_flight" else [fm.call]
        for call in calls:
            edges.append({"type": "TRIGGERS", "from": f"fm:{fm.key}", "to": f"act:{call}"})
        phases = {"engine_start_or_flight": ["engine_start", "flight"]}.get(fm.phase, [fm.phase])
        for ph in phases:
            edges.append({"type": "OCCURS_IN", "from": f"fm:{fm.key}", "to": f"phase:{ph}"})

    rules = []
    for r in rules_json["threshold_rules"]:
        rules.append({"id": r["id"], "short_id": r["short_id"], "type": "THRESHOLD",
                      "text": r["sentence"], "evidence_grade": r["evidence_grade"],
                      "sensitivity": r["sensitivity"], "specificity": r["specificity"]})
        edges.append({"type": "SUPPORTS", "from": r["id"], "to": f"fm:{r['diagnosis']}",
                      "sensitivity": r["sensitivity"], "specificity": r["specificity"],
                      "evidence_grade": r["evidence_grade"]})
        edges.append({"type": "USES", "from": r["id"], "to": f"f:{r['feature']}"})
    for m in rules_json["mutex_rules"]:
        rules.append({"id": m["id"], "type": "MUTEX", "text": m["sentence"],
                      "evidence_grade": m["evidence_grade"]})
        for a in m["members"]:
            for b in m["members"]:
                if a < b:
                    edges.append({"type": "EXCLUDES", "from": f"fm:{a}", "to": f"fm:{b}",
                                  "rule_id": m["id"]})
    for e in rules_json["equivalence_rules"]:
        a, b = e["members"][:2]
        edges.append({"type": "EQUIVALENT", "from": f"fm:{a}", "to": f"fm:{b}", "rule_id": e["id"]})

    n = len(y_work)
    n_pairs = len(keys) * (len(keys) - 1) // 2
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            both = int((y_work[:, i] & y_work[:, j]).sum())
            expected = base[i] * base[j] * n
            if both < 10 or expected <= 0 or both / expected < 2.0:
                continue
            p_value = float(poisson.sf(both - 1, expected))   # P(X >= both) if independent
            if p_value * n_pairs >= 0.05:                      # Bonferroni over all pairs
                continue
            edges.append({"type": "CO_OCCURS", "from": f"fm:{keys[i]}", "to": f"fm:{keys[j]}",
                          "count": both, "expected": expected, "lift": both / expected,
                          "p_value": p_value,
                          "p_given_from": both / max(int(y_work[:, i].sum()), 1),
                          "p_given_to": both / max(int(y_work[:, j].sum()), 1)})

    actions = [{"id": f"act:{k}", "label": k, "description": v} for k, v in CALL_ACTIONS.items()]
    phases = [{"id": "phase:countdown", "label": "Final countdown", "window": [-60, 0]},
              {"id": "phase:engine_start", "label": "Engine start", "window": [0, 7]},
              {"id": "phase:flight", "label": "Boosted ascent", "window": [7, 150]}]
    nodes = {"subsystems": subsystems, "channels": channels, "failure_modes": modes,
             "features": features, "rules": rules, "actions": actions, "phases": phases}
    edge_types: Dict[str, int] = {}
    for e in edges:
        edge_types[e["type"]] = edge_types.get(e["type"], 0) + 1
    return {
        "meta": {
            "domain": DOMAIN, "version": VERSION, "created_by": "run_factory.py",
            "created_at": _now(),
            "source_artifacts": {"taxonomy": "layer-1/taxonomy.json",
                                 "rules": "layer-2/rules.json", "model": "layer-4/model.pkl",
                                 "thresholds": "layer-4/thresholds.json"},
            "launch_count": n_launches, "node_counts": {k: len(v) for k, v in nodes.items()},
            "edge_count": len(edges), "edge_types": edge_types,
        },
        "nodes": nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-launches", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=DOMAIN_DIR,
                    help="where layer-1..layer-4 are written (default: this domain)")
    args = ap.parse_args()
    out = args.out_dir
    keys = list(FAILURE_KEYS)
    clock = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - clock:6.1f}s] {msg}", flush=True)

    log(f"simulating {args.n_launches} launches (seed base {args.seed}), features every "
        f"{TICK_S} s")
    seeds, x_all, y_launch, manifest, labels = simulate_campaign(args.n_launches, args.seed,
                                                                 args.workers)
    n, n_t, n_f = x_all.shape
    i_end = int(np.argmin(np.abs(TICKS - 150.0)))

    # Layer 1
    taxonomy = build_taxonomy(y_launch, labels)
    _write(out / "layer-1" / "taxonomy.json", taxonomy)
    log(f"Layer 1: {len(taxonomy['diagnoses'])} classes, acceptance "
        f"{'PASS' if taxonomy['meta']['acceptance_pass'] else 'FAIL'}")

    # Split: launch-level stratified holdout + K folds (platform primitives)
    n_hold = resolve_holdout_size(n, N_FOLDS)
    work, hold = stratified_holdout_split(y_launch, n_hold, seed=args.seed, class_names=keys)
    work, hold = np.array(work), np.array(hold)
    folds = stratified_cv_folds(y_launch[work], N_FOLDS, seed=args.seed, class_names=keys)
    log(f"split: {len(work)} work / {len(hold)} holdout launches, {N_FOLDS} folds "
        f"(iterstrat={HAS_ITERSTRAT})")

    # Layer 2: discovery on the complete window (H0 + 150 s)
    rules_json, discovery_report = run_discovery(x_all[work, i_end], y_launch[work],
                                                 x_all[hold, i_end], y_launch[hold])
    _write(out / "layer-2" / "rules.json", rules_json)
    _write(out / "layer-2" / "discovery_report.json", discovery_report)
    log(f"Layer 2: {len(rules_json['threshold_rules'])} B+ threshold rules, "
        f"{len(rules_json['mutex_rules'])} mutex, gaps {rules_json['meta']['layer4_gap']}")

    # Layer 4: train on time-truncated rows; out-of-fold streaming replays -> thresholds
    ticks_of = {int(i): training_ticks(int(seeds[i])) for i in range(n)}

    def rows(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs = [x_all[i, ticks_of[int(i)]] for i in idx]
        ys = [row_labels(manifest[i], ticks_of[int(i)]) for i in idx]
        return np.vstack(xs), np.vstack(ys)

    def stream(models, idx: np.ndarray) -> np.ndarray:
        return predict(models, x_all[idx].reshape(-1, n_f)).reshape(len(idx), n_t, len(keys))

    p_oof = np.zeros((len(work), n_t, len(keys)), dtype=np.float32)
    for fi, (tr, va) in enumerate(folds):
        tr, va = np.array(tr), np.array(va)
        models = fit_models(*rows(work[tr]))
        p_oof[va] = stream(models, work[va])
        log(f"  fold {fi + 1}/{N_FOLDS} done")
    thresholds, calibration = calibrate_thresholds(p_oof, y_launch[work])
    xw, yw = rows(work)
    models = fit_models(xw, yw)
    log(f"Layer 4: final models trained on {len(xw)} rows; thresholds calibrated to a "
        f"{FA_BUDGET:.2%} false-alarm budget per class")

    # Holdout: end-of-window and gate snapshots, over time, and streaming replays
    p_hold = stream(models, hold)
    y_rows_hold = (manifest[hold][:, None, :] <= TICKS[None, :, None]).astype(int)

    def snap(t: float, subset: Sequence[str]) -> Dict[str, Any]:
        i = int(np.argmin(np.abs(TICKS - t)))
        return _records(y_rows_hold[:, i], p_hold[:, i], thresholds, subset)

    countdown = [fm.key for fm in FAILURE_MODES if fm.phase == "countdown"]
    start = ["SLOW_ENGINE_START", "LOX_PUMP_CAVITATION", "COMBUSTION_INSTABILITY"]
    end = snap(150.0, keys)
    gates = {"ignition_gate_t-1": snap(-1.0, countdown),
             "booster_commit_gate_t6.5": snap(6.5, start)}
    over_time = []
    for t in (-45.0, -30.0, -15.0, -1.0, 2.0, 4.0, 6.5, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0):
        i = int(np.argmin(np.abs(TICKS - t)))
        active = [k for j, k in enumerate(keys) if y_rows_hold[:, i, j].sum() >= 5]
        if active:
            r = _records(y_rows_hold[:, i], p_hold[:, i], thresholds, active)
            over_time.append({"t": t, "classes_scored": len(active),
                              "macro_f1": r["macro"]["f1"], "micro_f1": r["micro"]["f1"]})
    streaming = streaming_metrics(p_hold, y_launch[hold], manifest[hold], thresholds)
    macro_f1 = end["macro"]["f1"]
    evaluation = {
        "meta": {"domain": DOMAIN, "version": VERSION, "created_by": "run_factory.py",
                 "created_at": _now(), "sklearn": sklearn.__version__},
        "model": {"type": "ammonix-style one-vs-rest HistGradientBoosting",
                  "params": HGB_PARAMS, "n_features": n_f, "training_rows": int(len(xw)),
                  "training_times": {"fixed": list(TRAIN_FIXED_TIMES),
                                     "random_per_launch": N_RANDOM_TIMES,
                                     "random_from": RANDOM_FROM},
                  "stream_tick_s": TICK_S, "debounce_ticks": DEBOUNCE_TICKS},
        "split": {"n_launches": n, "n_work": len(work), "n_holdout": len(hold),
                  "n_folds": N_FOLDS, "iterstrat": HAS_ITERSTRAT, "seed": args.seed},
        "thresholds": thresholds,
        "threshold_calibration": {"false_alarm_budget": FA_BUDGET, "per_class": calibration},
        "holdout_end_of_window": end,
        "holdout_gates": gates,
        "holdout_over_time": over_time,
        "holdout_streaming": streaming,
        "acceptance": {"overall_f1_target": F1_TARGET, "holdout_macro_f1": macro_f1,
                       "pass": bool(macro_f1 >= F1_TARGET)},
    }
    _write(out / "layer-4" / "evaluation.json", evaluation)
    _write(out / "layer-4" / "thresholds.json",
           {"thresholds": thresholds, "false_alarm_budget": FA_BUDGET,
            "debounce_ticks": DEBOUNCE_TICKS, "tick_s": TICK_S, "created_at": _now()})
    _write(out / "layer-4" / "split.json",
           {"seed_base": args.seed, "work_seeds": seeds[work].tolist(),
            "holdout_seeds": seeds[hold].tolist()})
    ref_idx = [int(np.argmin(np.abs(TICKS - t))) for t in REF_TIMES]
    nominal_rows = x_all[work][y_launch[work].sum(axis=1) == 0][:, ref_idx]   # (n_nom, R, F)
    with warnings.catch_warnings():            # features whose window has not started are NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        nom_mean, nom_std = np.nanmean(nominal_rows, axis=0), np.nanstd(nominal_rows, axis=0)
    _write(out / "layer-4" / "nominal_reference.json",
           {"times": list(REF_TIMES), "features": list(FEATURE_NAMES), "mean": nom_mean,
            "std": nom_std, "n_launches": int(len(nominal_rows))})
    with open(out / "layer-4" / "model.pkl", "wb") as fh:
        pickle.dump({"models": models, "keys": keys, "features": list(FEATURE_NAMES),
                     "thresholds": thresholds, "sklearn": sklearn.__version__}, fh)
    # Every launch's scores at every tick, as the agent would have seen them: out-of-fold for
    # the work launches, the final models for the holdout. Input of ControlRoom/build_universe.py.
    p_all = np.zeros((n, n_t, len(keys)), dtype=np.float16)
    p_all[work], p_all[hold] = p_oof, p_hold
    np.savez_compressed(out / "layer-4" / "streaming_scores.npz", p=p_all, ticks=TICKS,
                        keys=np.array(keys), thresholds=np.array([thresholds[k] for k in keys]),
                        seeds=seeds, holdout=np.isin(np.arange(n), hold), y=y_launch,
                        manifest=manifest)
    log(f"Layer 4: holdout macro F1 at H0+150 s = {macro_f1:.3f} "
        f"(target {F1_TARGET}: {'PASS' if macro_f1 >= F1_TARGET else 'FAIL'})")

    # Layer 3
    kg = build_knowledge_graph(rules_json, discovery_report["correlations"], y_launch[work],
                               thresholds, n)
    _write(out / "layer-3" / "knowledge-graph.json", kg)
    log(f"Layer 3: {kg['meta']['node_counts']} nodes, {kg['meta']['edge_count']} edges, "
        f"CO_OCCURS {[(e['from'], e['to'], round(float(e['lift']), 1)) for e in kg['edges'] if e['type'] == 'CO_OCCURS']}")

    for k in keys:
        r, st = end["per_class"][k], streaming[k]
        print(f"  {k:26s} thr={thresholds[k]:.3f} end: sens={r['sensitivity']:.2f} "
              f"spec={r['specificity']:.4f} auroc={r['auroc']:.3f} | stream: "
              f"det={st['detection_rate']:.2f} in_time={st['in_time_rate']:.2f} "
              f"lat={st['median_latency_s']} FA/1000={st['false_alerts_per_1000']:.1f}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
