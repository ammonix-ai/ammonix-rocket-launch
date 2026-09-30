#!/usr/bin/env python
"""Streaming evaluation of the control-room agent on the unseen (holdout) launches.

Every holdout launch is replayed at 2 Hz exactly as in the control room: features use
only the telemetry received so far. The fixed-limit redline monitor runs on the same
telemetry for comparison. Writes ControlRoom/agent_evaluation.json.

Usage:
    python ControlRoom/evaluate.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
DOMAIN_DIR = HERE.parent
sys.path[:0] = [str(HERE), str(DOMAIN_DIR)]

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from agent import ControlRoomAgent, summarize  # noqa: E402
from launch_sim.anomalies import BY_KEY, FAILURE_KEYS, FAILURE_MODES  # noqa: E402
from launch_sim.simulate import simulate_launch  # noqa: E402
from redline import redline_events  # noqa: E402

_AGENT: Optional[ControlRoomAgent] = None
_LIMITS = None
COUNTDOWN = [f.key for f in FAILURE_MODES if f.phase == "countdown"]
START = ["SLOW_ENGINE_START", "LOX_PUMP_CAVITATION"]
FLIGHT = [f.key for f in FAILURE_MODES if f.phase == "flight"]
IGNITION_GATE, COMMIT_GATE = -1.0, 6.5


def _init() -> None:
    # One thread per worker. Several processes each spinning a full OpenMP thread pool on
    # the same cores make every model call ~1000x slower. The limit must be set after the
    # agent has loaded scikit-learn, because threadpoolctl only sees loaded runtimes.
    global _AGENT, _LIMITS
    os.environ["OMP_NUM_THREADS"] = "1"
    _AGENT = ControlRoomAgent(DOMAIN_DIR)
    _LIMITS = threadpool_limits(1)


def _one(seed: int) -> Dict[str, Any]:
    launch = simulate_launch(int(seed))
    rp = _AGENT.replay(launch.tel, explain=False)
    out = summarize(rp, launch.labels, redline_events(launch.tel))
    out["seed"] = int(seed)
    return out


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def _redline_first(r: Dict[str, Any], calls, t0: float, t1: float, text: Optional[str] = None):
    ts = [e["t"] for e in r["redlines"] if e["call"] in calls and t0 <= e["t"] < t1
          and (text is None or text in e["text"])]
    return min(ts) if ts else None


def _red_hold(r: Dict[str, Any]) -> bool:
    return _redline_first(r, {"HOLD"}, -1e9, 0.0) is not None


def _red_flies(r: Dict[str, Any]) -> bool:
    return not _red_hold(r) and _redline_first(r, {"ABORT"}, 0.0, 7.0) is None


def evaluate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    nominal = [r for r in results if not r["labels"]]
    per_class: Dict[str, Any] = {}
    for key in FAILURE_KEYS:
        pos = [r for r in results if key in r["labels"]]
        lat, in_time = [], 0
        for r in pos:
            t = r["first_alert"].get(key)
            if t is None:
                continue
            lat.append(t - r["labels"][key]["manifest"])
            gate = IGNITION_GATE if key in COUNTDOWN else COMMIT_GATE if key in START else 1e9
            if key == "COMBUSTION_INSTABILITY" and r["labels"][key]["manifest"] > COMMIT_GATE:
                gate = 1e9
            in_time += t <= gate
        false_alerts = sum(1 for r in results if key not in r["labels"] and key in r["first_alert"])
        per_class[key] = {
            "display": BY_KEY[key].display, "n_launches": len(pos), "detected": len(lat),
            "detection_rate": _rate(len(lat), len(pos)),
            "in_time_rate": _rate(int(in_time), len(pos)),
            "median_latency_s": float(np.median(lat)) if lat else None,
            "false_alerts_per_1000": 1000.0 * false_alerts / max(len(results) - len(pos), 1),
        }

    def caught(keys, gate_t):
        """Launches stopped in time: held in the countdown, or aborted before the gate."""
        pos = [r for r in results if any(
            k in r["labels"] and (k != "COMBUSTION_INSTABILITY" or r["labels"][k]["manifest"] <= COMMIT_GATE)
            for k in keys)]
        def agent_stops(r):
            if r["hold_at"] is not None and r["hold_at"] <= min(gate_t, IGNITION_GATE):
                return True
            return r["abort_at"] is not None and r["abort_at"] <= gate_t
        def red_stops(r):
            if _redline_first(r, {"HOLD"}, -1e9, 0.0) is not None:
                return True
            return gate_t > 0 and _redline_first(r, {"ABORT"}, 0.0, gate_t + 0.5) is not None
        agent = sum(1 for r in pos if agent_stops(r))
        red = sum(1 for r in pos if red_stops(r))
        return {"n_launches": len(pos), "agent": _rate(agent, len(pos)), "redline": _rate(red, len(pos))}

    n_nom = len(nominal)
    false_calls = {
        "n_nominal_launches": n_nom,
        "agent_holds_per_1000": 1000.0 * sum(r["hold_at"] is not None for r in nominal) / max(n_nom, 1),
        "agent_aborts_per_1000": 1000.0 * sum(r["abort_at"] is not None for r in nominal) / max(n_nom, 1),
        "agent_flight_warnings_per_1000": 1000.0 * sum(r["warning_at"] is not None for r in nominal) / max(n_nom, 1),
        "redline_holds_per_1000": 1000.0 * sum(_red_hold(r) for r in nominal) / max(n_nom, 1),
        "redline_aborts_per_1000": 1000.0 * sum(not _red_hold(r) and _redline_first(r, {"ABORT"}, 0.0, 7.0) is not None for r in nominal) / max(n_nom, 1),
        "redline_flight_alarms_per_1000": 1000.0 * sum(_red_flies(r) and _redline_first(r, {"ALARM"}, 7.0, 1e9) is not None for r in nominal) / max(n_nom, 1),
    }

    sensor = [r for r in results if "TEMP_SENSOR_FAULT" in r["labels"]]
    overtemp = [r for r in results if "TURBINE_OVERTEMP" in r["labels"]]
    temp_text = "Turbine inlet temperature above 950 K"
    sensor_story = {
        "n_sensor_fault_launches": len(sensor),
        "agent_called_it_sensor_fault": _rate(sum("TEMP_SENSOR_FAULT" in r["first_alert"] for r in sensor), len(sensor)),
        "agent_called_it_overtemp": _rate(sum("TURBINE_OVERTEMP" in r["first_alert"] for r in sensor), len(sensor)),
        "redline_temperature_alarm": _rate(sum(_redline_first(r, {"ALARM", "ABORT"}, -1e9, 1e9, temp_text) is not None for r in sensor), len(sensor)),
        "n_overtemp_launches": len(overtemp),
        "agent_detected_overtemp": _rate(sum("TURBINE_OVERTEMP" in r["first_alert"] for r in overtemp), len(overtemp)),
        "redline_detected_overtemp": _rate(sum(_redline_first(r, {"ALARM", "ABORT"}, -1e9, 1e9, temp_text) is not None for r in overtemp), len(overtemp)),
    }
    lat_agent, lat_red = [], []
    for r in overtemp:
        m = r["labels"]["TURBINE_OVERTEMP"]["manifest"]
        if "TURBINE_OVERTEMP" in r["first_alert"]:
            lat_agent.append(r["first_alert"]["TURBINE_OVERTEMP"] - m)
        t = _redline_first(r, {"ALARM", "ABORT"}, -1e9, 1e9, temp_text)
        if t is not None:
            lat_red.append(t - m)
    sensor_story["agent_overtemp_median_latency_s"] = float(np.median(lat_agent)) if lat_agent else None
    sensor_story["redline_overtemp_median_latency_s"] = float(np.median(lat_red)) if lat_red else None

    flight_lat = [per_class[k]["median_latency_s"] for k in FLIGHT
                  if per_class[k]["median_latency_s"] is not None]
    return {
        "n_launches": len(results),
        "per_class": per_class,
        "countdown_failures_caught_before_ignition": caught(COUNTDOWN, IGNITION_GATE),
        "start_failures_caught_before_booster_ignition": caught(
            START + ["COMBUSTION_INSTABILITY"], COMMIT_GATE),
        "false_calls_on_nominal_launches": false_calls,
        "sensor_fault_vs_overtemp": sensor_story,
        "flight_median_latency_s": float(np.median(flight_lat)) if flight_lat else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    split = json.loads((DOMAIN_DIR / "layer-4" / "split.json").read_text())
    seeds = split["holdout_seeds"]
    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(_one, seeds, chunksize=4)
    report = evaluate(results)
    report["launches"] = results
    (HERE / "agent_evaluation.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    summary = {k: v for k, v in report.items() if k not in ("launches", "per_class")}
    print(json.dumps(summary, indent=1, default=float))
    for k, v in report["per_class"].items():
        print(f"  {k:26s} n={v['n_launches']:3d} det={v['detection_rate'] or 0:.2f} "
              f"in_time={v['in_time_rate'] or 0:.2f} lat={v['median_latency_s']} "
              f"FA/1000={v['false_alerts_per_1000']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
