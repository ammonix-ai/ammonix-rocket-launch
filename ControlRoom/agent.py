"""Control-room agent: replays a launch tick by tick and makes the call for the phase.

Everything the agent knows comes from the factory's artifacts:
  layer-4/model.pkl, thresholds    ammonix-style classifiers, one per failure mode
  layer-2/rules.json               clear-text graded rules, shown as evidence
  layer-3/knowledge-graph.json     discovered couplings (CO_OCCURS) and exclusions
  layer-4/nominal_reference.json   nominal feature values for "measured vs nominal" text

No LLM makes or phrases a decision. Explanations are templates filled with the
fired rules and measured values, so every sentence traces to a rule and a number.

Calls by phase:
  countdown (t < 0)        GO / HOLD
  engine start (0-7 s)     ENGINE_START -> COMMIT at 6.5 s, or ABORT (engine shutdown)
  flight (t >= 7 s)        NOMINAL / ADVISORY / WARNING / CRITICAL, for the mission director
                           and flight safety; an uncrewed launcher cannot abort in flight
"""

from __future__ import annotations

import json
import operator
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DOMAIN_DIR = Path(__file__).resolve().parents[1]
if str(DOMAIN_DIR) not in sys.path:
    sys.path.insert(0, str(DOMAIN_DIR))

from launch_sim import vehicle as V  # noqa: E402
from launch_sim.anomalies import BY_KEY  # noqa: E402
from launch_sim.features import FEATURE_INDEX, FEATURE_SPECS, LaunchFeatures  # noqa: E402
from launch_sim.streaming import debounced, stream_ticks  # noqa: E402

COMMIT_T = V.EVENTS["BOOSTER_COMMIT_GATE"]
FLIGHT_T = V.EVENTS["BOOSTER_IGNITION"]
CRITICAL_IN_FLIGHT = {"COMBUSTION_INSTABILITY", "TURBINE_OVERTEMP", "LOX_PUMP_CAVITATION",
                      "SLOW_ENGINE_START", "TRAJECTORY_DEVIATION"}
WARNING_IN_FLIGHT = {"SRB_THRUST_ASYMMETRY", "TVC_ACTUATOR_DEGRADATION", "POGO_ONSET"}
CALLS = ("GO", "HOLD", "ENGINE_START", "COMMIT", "ABORT", "NOMINAL", "ADVISORY", "WARNING",
         "CRITICAL")
OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}


def phase_of(t: float) -> str:
    return "countdown" if t < 0.0 else ("engine_start" if t < FLIGHT_T else "flight")


def alert_level(key: str, t: float) -> str:
    """The call an alert on ``key`` raises when first confirmed at time ``t``."""
    ph = phase_of(t)
    if ph == "countdown":
        return "HOLD"
    if ph == "engine_start":
        return "ABORT"
    if key in CRITICAL_IN_FLIGHT:
        return "CRITICAL"
    if key in WARNING_IN_FLIGHT:
        return "WARNING"
    return "ADVISORY"


def _num(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


@dataclass
class Replay:
    ticks: np.ndarray
    probs: np.ndarray                       # (T, K) classifier probabilities
    active: np.ndarray                      # (T, K) debounced, mutex-corrected alerts
    calls: List[str]
    first_alert: Dict[str, float]
    events: List[Dict[str, Any]] = field(default_factory=list)


class ControlRoomAgent:
    def __init__(self, domain_dir: Path = DOMAIN_DIR):
        with open(domain_dir / "layer-4" / "model.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        self.models = bundle["models"]
        self.keys: List[str] = list(bundle["keys"])
        self.thresholds = np.array([bundle["thresholds"][k] for k in self.keys])
        rules = json.loads((domain_dir / "layer-2" / "rules.json").read_text())
        self.rules_by_class: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.keys}
        for r in rules["threshold_rules"]:
            self.rules_by_class[r["diagnosis"]].append(r)
        self.mutex = [[self.keys.index(m) for m in r["members"]] for r in rules["mutex_rules"]]
        kg = json.loads((domain_dir / "layer-3" / "knowledge-graph.json").read_text())
        self.cooccurs = [e for e in kg["edges"] if e["type"] == "CO_OCCURS"]
        self.linked: Dict[str, List[str]] = {k: [] for k in self.keys}
        for e in kg["edges"]:
            if e["type"] == "DISCRIMINATED_BY":
                self.linked[e["from"][3:]].append(e["to"][2:])
        for k, rs in self.rules_by_class.items():
            self.linked[k] = list(dict.fromkeys([r["feature"] for r in rs] + self.linked[k]))
        ref = json.loads((domain_dir / "layer-4" / "nominal_reference.json").read_text())
        self.ref_times = np.array(ref["times"])
        self.ref_mean = np.array(ref["mean"], dtype=float)
        self.ref_std = np.array(ref["std"], dtype=float)

    # ---- inference ------------------------------------------------------------------------
    def probabilities(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict_proba(x)[:, 1] for m in self.models])

    def replay(self, tel: np.ndarray, t_end: float = V.T_END, explain: bool = True) -> Replay:
        ticks = stream_ticks(t_end)
        lf = LaunchFeatures(tel)
        x = np.vstack([lf.vector(t) for t in ticks])
        return self.decide(ticks, x, self.probabilities(x), explain=explain)

    def decide(self, ticks: np.ndarray, x: np.ndarray, p: np.ndarray,
               explain: bool = True) -> Replay:
        held = debounced(p, axis=0)                  # probability held for the debounce window
        active = held >= self.thresholds
        rows = np.arange(len(ticks))
        for group in self.mutex:                     # keep only the most probable member
            sub = active[:, group]
            best = np.argmax(np.where(sub, held[:, group], -1.0), axis=1)
            keep = np.zeros_like(sub)
            keep[rows, best] = sub[rows, best]
            active[:, group] = keep
        latched = np.logical_or.accumulate(active, axis=0)
        first_alert = {k: float(ticks[np.argmax(active[:, j])])
                       for j, k in enumerate(self.keys) if active[:, j].any()}
        calls = [self._call(float(t), latched[i]) for i, t in enumerate(ticks)]
        replay = Replay(ticks, p, active, calls, first_alert)
        if explain:
            replay.events = self._events(replay, x)
        return replay

    def _call(self, t: float, latched_row: np.ndarray) -> str:
        on = {k for j, k in enumerate(self.keys) if latched_row[j]}
        if t < 0.0:
            return "HOLD" if on else "GO"
        if t < FLIGHT_T:
            if on:
                return "ABORT"
            return "COMMIT" if t >= COMMIT_T else "ENGINE_START"
        if on & CRITICAL_IN_FLIGHT:
            return "CRITICAL"
        if on & WARNING_IN_FLIGHT:
            return "WARNING"
        return "ADVISORY" if on else "NOMINAL"

    # ---- explanations -----------------------------------------------------------------------
    def _reference(self, t: float) -> Optional[int]:
        i = int(np.searchsorted(self.ref_times, t, side="right")) - 1
        return i if i >= 0 else None

    def evidence(self, key: str, t: float, xrow: np.ndarray, n: int = 3) -> List[Dict[str, Any]]:
        """Class-linked features furthest from nominal at time ``t``."""
        ri = self._reference(t)
        out = []
        for name in self.linked[key]:
            fi = FEATURE_INDEX[name]
            spec = FEATURE_SPECS[fi]
            val = xrow[fi]
            if t < spec.valid_from or not np.isfinite(val) or ri is None:
                continue
            mu, sd = self.ref_mean[ri, fi], self.ref_std[ri, fi]
            if not (np.isfinite(mu) and np.isfinite(sd)) or sd < 1e-6:
                continue                # no spread in nominal launches: not a comparison
            z = (val - mu) / max(sd, 1e-9)
            out.append({"feature": name, "label": spec.description, "unit": spec.unit,
                        "value": float(val), "nominal_mean": float(mu), "nominal_std": float(sd),
                        "z": float(z)})
        out.sort(key=lambda e: -abs(e["z"]))
        return out[:n]

    def fired_rules(self, key: str, t: float, xrow: np.ndarray) -> List[Dict[str, Any]]:
        fired = []
        for r in self.rules_by_class[key]:
            val = xrow[FEATURE_INDEX[r["feature"]]]
            if t >= r["feature_valid_from"] and np.isfinite(val) and \
                    OPS[r["operator"]](val, r["threshold"]):
                fired.append({"short_id": r["short_id"], "sentence": r["sentence"],
                              "grade": r["evidence_grade"], "sensitivity": r["sensitivity"],
                              "specificity": r["specificity"], "value": float(val),
                              "unit": r["feature_unit"]})
        fired.sort(key=lambda r: (r["grade"], -r["sensitivity"] - r["specificity"]))
        return fired

    def graph_notes(self, key: str) -> List[str]:
        notes = []
        for e in self.cooccurs:
            a, b = e["from"][3:], e["to"][3:]
            if key in (a, b):
                other = b if key == a else a
                notes.append(f"Knowledge graph: {BY_KEY[key].display.lower()} and "
                             f"{BY_KEY[other].display.lower()} occur together {e['lift']:.1f}x "
                             f"more often than chance ({e['count']} launches)")
        return notes

    def explain(self, key: str, t: float, prob: float, xrow: np.ndarray) -> Dict[str, Any]:
        level = alert_level(key, t)
        fm = BY_KEY[key]
        ev = self.evidence(key, t, xrow)
        rules = self.fired_rules(key, t, xrow)
        notes = self.graph_notes(key)
        parts = [f"{level}: {fm.display} (p = {prob:.2f})."]
        for e in ev[:2]:
            parts.append(f"{e['label']}: {_num(e['value'])} {e['unit']} "
                         f"(nominal {_num(e['nominal_mean'])} ± {_num(e['nominal_std'])}).")
        if rules:
            r = rules[0]
            parts.append(f"Rule {r['short_id']} (grade {r['grade']}; sensitivity "
                         f"{r['sensitivity']:.2f}, specificity {r['specificity']:.2f}): "
                         f"{r['sentence']}.")
        elif self.rules_by_class[key]:
            parts.append("No graded rule for this mode can be checked yet at this point in the "
                         "window; the early call rests on the model.")
        else:
            parts.append("No single-feature rule reaches grade B for this mode (a Layer 4 gap); "
                         "the call rests on the model.")
        parts.extend(n + "." for n in notes[:1])
        return {"key": key, "display": fm.display, "level": level, "p": prob,
                "evidence": ev, "rules": rules[:3], "graph": notes,
                "has_rules": bool(self.rules_by_class[key]), "text": " ".join(parts)}

    def confirmations(self, key: str, ticks: np.ndarray, x: np.ndarray, i0: int,
                      max_n: int = 2) -> List[Dict[str, Any]]:
        """Graded rules that start to fire after the call: the evidence catching up."""
        already = {r["short_id"] for r in self.fired_rules(key, float(ticks[i0]), x[i0])}
        out: List[Dict[str, Any]] = []
        for i in range(i0 + 1, len(ticks)):
            for r in self.fired_rules(key, float(ticks[i]), x[i]):
                if r["short_id"] not in already:
                    already.add(r["short_id"])
                    out.append({"t": float(ticks[i]), **r})
            if len(out) >= max_n:
                break
        return out[:max_n]

    def _events(self, rp: Replay, x: np.ndarray) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        idx = {float(t): i for i, t in enumerate(rp.ticks)}
        for key, t in sorted(rp.first_alert.items(), key=lambda kv: kv[1]):
            i = idx[t]
            j = self.keys.index(key)
            e = self.explain(key, t, float(rp.probs[i, j]), x[i])
            events.append({"t": t, "kind": "alert", **e})
            for c in self.confirmations(key, rp.ticks, x, i):
                events.append({"t": c["t"], "kind": "rule", "key": key, "display": e["display"],
                               "rule": {k: v for k, v in c.items() if k != "t"}})
        prev = None
        for i, (t, call) in enumerate(zip(rp.ticks, rp.calls)):
            if call != prev:
                events.append({"t": float(t), "kind": "call", "call": call})
                prev = call
        events.sort(key=lambda e: (e["t"], e["kind"] != "call"))
        return events


def summarize(rp: Replay, labels: Dict[str, Dict[str, Any]], redlines: Sequence[dict]
              ) -> Dict[str, Any]:
    """Per-launch outcome used by the streaming evaluation."""
    def first(call_names, t0, t1):
        for t, c in zip(rp.ticks, rp.calls):
            if t0 <= t < t1 and c in call_names:
                return float(t)
        return None

    # A hold stops the countdown and an abort stops the launch: later calls on the
    # (counterfactual) telemetry that keeps running in the simulation do not count.
    hold_at = first({"HOLD"}, V.T_START, 0.0)
    abort_at = first({"ABORT"}, 0.0, FLIGHT_T) if hold_at is None else None
    flying = hold_at is None and abort_at is None
    return {
        "labels": {k: {"manifest": v["manifest"], "severity": v["severity"]}
                   for k, v in labels.items()},
        "first_alert": rp.first_alert,
        "hold_at": hold_at,
        "abort_at": abort_at,
        "critical_at": first({"CRITICAL"}, FLIGHT_T, V.T_END + 1) if flying else None,
        "warning_at": first({"WARNING", "CRITICAL"}, FLIGHT_T, V.T_END + 1) if flying else None,
        "advisory_at": first({"ADVISORY", "WARNING", "CRITICAL"}, FLIGHT_T, V.T_END + 1)
        if flying else None,
        "redlines": [{"t": r["t"], "call": r["call"], "text": r["text"]} for r in redlines],
    }
