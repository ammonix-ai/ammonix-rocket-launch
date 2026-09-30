#!/usr/bin/env python
"""Build the offline dashboard: ControlRoom/dashboard.html.

Replays the showcase stories and one unseen (held-out) launch per failure type through the
control-room agent and the fixed-limit redline monitor, then writes one self-contained
HTML file: inline CSS, JS and data, no network needed. The page also lists all 500
held-out launches with their truth and the agent's result; ControlRoom/serve.py replays
any of them on demand with the functions here.

A HOLD stops the countdown, so the replay ends there. An ABORT shuts the engine down,
so the launch is re-simulated with that abort and the replay shows the shutdown.

Each launch also carries feature snapshots (the features furthest from the nominal
launches) for the analyst chat, which ControlRoom/serve.py adds when it serves the page.

Usage:
    python ControlRoom/build_dashboard.py [--artifact PATH]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
DOMAIN_DIR = HERE.parent
REPO_ROOT = DOMAIN_DIR
sys.path[:0] = [str(HERE), str(DOMAIN_DIR)]

import numpy as np  # noqa: E402

from agent import CALLS, ControlRoomAgent, Replay, alert_level  # noqa: E402
from launch_sim import vehicle as V  # noqa: E402
from launch_sim.anomalies import BY_KEY, FAILURE_MODES  # noqa: E402
from launch_sim.features import FEATURE_SPECS, LaunchFeatures  # noqa: E402
from launch_sim.simulate import simulate_launch  # noqa: E402
from redline import REDLINES, redline_events  # noqa: E402
from showcase import MODE_LAY, SHOWCASE, TEST_LAY  # noqa: E402

BIN = 10                                     # samples per display bin (0.2 s at 50 Hz)
N_BINS = (V.N_SAMPLES - 1) // BIN            # 1050 bins, H0 - 60 s to H0 + 150 s
N_BAND_LAUNCHES = 120
HOLD_TAIL_S, ABORT_TAIL_S = 12.0, 8.0
SNAP_FEATURES = 8                            # most unusual features kept per snapshot
REDLINE_CODES = {"HOLD": 1, "ABORT": 2, "ALARM": 3}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _channels_b64(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> str:
    """Every channel's display bins as ONE base64 string, channel after channel. A list of
    strings would put two base64 strings side by side ('...","...'); gitleaks' generic-api-key
    rule then reads a random 'kEy' or 'API' near the end of one as a key name and the next
    string as its secret, and the push-run secret scan fails."""
    return _b64(np.stack([_quantize(x[i], lo[i], hi[i]) for i in range(len(lo))]))


def trajectory_corridor() -> Dict[str, Any]:
    """The illustrative flight-safety corridor on the display bins, for the trajectory strip:
    the same function the fixed limit in redline.py trips on. None before liftoff."""
    t = V.T_START + (np.arange(N_BINS) + 0.5) * BIN / V.FS
    km = V.trajectory_corridor(t)
    return {"channel": "TRAJ_DEV",
            "km": [round(float(v), 3) if ti >= V.EVENTS["BOOSTER_IGNITION"] else None
                   for ti, v in zip(t, km)]}


def alerts_shown_until(stop_type: Optional[str], stop_t: Optional[float]) -> float:
    """The last moment whose alerts a replay shows: to the end of a hold's tail (the clock
    stays stopped, never past ignition), to the abort itself (after it the engine is shut
    down, see pack_launch), else the whole window."""
    if stop_type == "hold":
        return min(stop_t + HOLD_TAIL_S, -0.5)
    if stop_type == "abort":
        return stop_t
    return V.T_END


def case_meta(seed: int) -> Dict[str, Any]:
    """How the page names a held-out launch; serve.py uses it for on-demand replays too."""
    return {"id": f"case-{seed}", "kind": "test", "title": f"Unseen launch #{seed % 100_000:04d}",
            "summary": "One of the 500 held-out launches the agent never saw.", "lay": TEST_LAY}


def holdout_index(evaluation: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Every held-out launch for the page's selector (from ControlRoom/evaluate.py): truth
    [[mode, severity, visible from]], the agent's first alert per mode [[mode, t, call]] in
    time order, and its stop (hold or abort). The evaluation keeps scoring a stopped launch as if
    it flew on; the index keeps only the alerts its replay shows.
    Groups: exactly one failure (its type), none, or two or more. One example per group is
    built into the page, so it can play without the server: the median-severity case of a
    type, the first normal launch without an alert, the first launch with several failures
    that the agent caught all of."""
    seeds, truth, alerts, stops = [], [], [], []
    for r in evaluation["launches"]:
        labs = sorted(r["labels"].items(), key=lambda kv: keys.index(kv[0]))
        seeds.append(int(r["seed"]))
        truth.append([[keys.index(k), round(float(v["severity"]), 2), _round(v["manifest"], 1)]
                      for k, v in labs])
        stop = (["hold", r["hold_at"]] if r["hold_at"] is not None else
                ["abort", r["abort_at"]] if r["abort_at"] is not None else None)
        until = alerts_shown_until(*(stop or (None, None)))
        alerts.append(sorted(([keys.index(k), round(float(t), 1), CALLS.index(alert_level(k, t))]
                              for k, t in r["first_alert"].items() if t <= until),
                             key=lambda a: (a[1], a[0])))
        stops.append(stop)
    examples: Dict[str, int] = {}
    for j, key in enumerate(keys):
        single = sorted((tr[0][1], s) for s, tr in zip(seeds, truth) if len(tr) == 1 and tr[0][0] == j)
        if single:
            examples[key] = single[len(single) // 2][1]
    quiet = [s for s, tr, a in zip(seeds, truth, alerts) if not tr and not a]
    several = [s for s, tr, a in zip(seeds, truth, alerts)
               if len(tr) >= 2 and {m for m, *_ in tr} <= {m for m, *_ in a}] or \
        [s for s, tr in zip(seeds, truth) if len(tr) >= 2]
    if quiet:
        examples["NOMINAL"] = quiet[0]
    if several:
        examples["MULTI"] = several[0]
    return {"seeds": seeds, "truth": truth, "alerts": alerts, "stops": stops, "examples": examples}


def universe_payload() -> Optional[Dict[str, Any]]:
    """The knowledge universe from build_universe.py, embedded (0.3 MB) so its tab also works
    when the page is opened as a file. None when it has not been built."""
    meta, blob = HERE / "universe.json", HERE / "universe.bin.gz"
    if not (meta.exists() and blob.exists()):
        return None
    return {"meta": json.loads(meta.read_text(encoding="utf-8")),
            "gz": base64.b64encode(blob.read_bytes()).decode("ascii")}


def _quantize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """uint16 with 0 reserved for 'no telemetry'."""
    q = 1 + np.round((np.clip(x, lo, hi) - lo) / (hi - lo) * 65534.0)
    return np.where(np.isfinite(x), q, 0).astype(np.uint16)


def _bins(tel: np.ndarray) -> np.ndarray:
    return tel[:, : N_BINS * BIN].reshape(tel.shape[0], N_BINS, BIN)


def _envelope(tel: np.ndarray):
    b = _bins(tel)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmin(b, axis=2), np.nanmax(b, axis=2), np.nanmean(b, axis=2)


def _round(x: Optional[float], nd: int = 3) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _sig(x: float, digits: int = 4) -> float:
    return float(f"{float(x):.{digits}g}")


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if not np.isfinite(f) else round(f, 4)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


# ---------------------------------------------------------------------------------------------
# Launch replays
# ---------------------------------------------------------------------------------------------

def replay_launch(agent: ControlRoomAgent, seed: int, forced: Optional[dict],
                  random_anomalies: bool) -> Dict[str, Any]:
    """Run the agent; apply the consequence of its call (hold stops, abort shuts down)."""
    launch = simulate_launch(seed, forced=forced, random_anomalies=random_anomalies)
    rp = agent.replay(launch.tel)
    stop: Dict[str, Any] = {"type": None, "t": None}
    t_end = V.T_END
    hold = next((float(t) for t, c in zip(rp.ticks, rp.calls) if c == "HOLD"), None)
    abort = next((float(t) for t, c in zip(rp.ticks, rp.calls) if c == "ABORT"), None)
    if hold is not None:
        stop = {"type": "hold", "t": hold}
        t_end = alerts_shown_until("hold", hold)    # the simulation does not stop the clock
    elif abort is not None:
        stop = {"type": "abort", "t": abort}
        launch = simulate_launch(seed, forced=forced, random_anomalies=random_anomalies,
                                 abort_at=abort)
        t_end = abort + ABORT_TAIL_S
        rp = agent.replay(launch.tel, t_end=t_end)
    else:
        rp = agent.replay(launch.tel, t_end=t_end) if t_end < V.T_END else rp
    k_end = V.k_of(t_end)
    tel = launch.tel.copy()
    tel[:, k_end + 1:] = np.nan
    # After an abort the fixed limits are silent too: what trips then is the shutdown itself.
    until = abort if stop["type"] == "abort" else t_end
    return {"launch": launch, "replay": rp, "tel": tel, "stop": stop, "t_end": t_end,
            "redlines": [e for e in redline_events(tel) if e["t"] <= until]}


def vehicle_events(launch, stop: Dict[str, Any], t_end: float) -> List[Dict[str, Any]]:
    ev = []

    def add(t: float, text: str) -> None:
        if t <= t_end:
            ev.append({"t": t, "kind": "vehicle", "text": text})

    add(V.EVENTS["LOX_PRESS_START"], "LOX tank pressurization starts")
    add(V.EVENTS["LH2_PRESS_START"], "LH2 tank pressurization starts")
    add(V.EVENTS["POWER_TRANSFER"], "Switch to onboard power")
    if stop["type"] == "hold":
        add(stop["t"], "Countdown held")
        return ev
    add(0.0, "H0: core engine ignition")
    if stop["type"] == "abort":
        add(stop["t"] + 0.3, "Core engine shutdown; boosters not ignited")
        return ev
    add(V.EVENTS["BOOSTER_IGNITION"], "Booster ignition")
    if launch.meta.get("liftoff") is not None:
        add(float(launch.meta["liftoff"]), "Liftoff")
    add(V.EVENTS["MAX_Q"], "Max-Q")
    add(V.EVENTS["BOOSTER_SEP"], "Booster separation")
    return ev


def snapshot_times(events: List[Dict[str, Any]], t_end: float) -> List[float]:
    """Every 5 s, every 1 s through the engine start, and at every alert and rule event."""
    grid = set(np.arange(-55.0, V.T_END + 0.1, 5.0).tolist()) | set(np.arange(0.0, 7.1, 1.0).tolist())
    grid |= {float(e["t"]) for e in events if e["kind"] in ("alert", "rule")}
    return sorted(t for t in grid if t <= t_end + 1e-9)


def feature_snapshots(agent: ControlRoomAgent, tel: np.ndarray, times: List[float]
                      ) -> Dict[str, Any]:
    """The features furthest from the nominal launches at each time, for the analyst.

    Each entry is [feature index, value, nominal mean, nominal sd] with the same nominal
    reference the agent uses for its evidence text.
    """
    lf = LaunchFeatures(tel)
    rows = []
    for t in times:
        x, ri = lf.vector(t), agent._reference(t)
        picked = []
        if ri is not None:
            for fi, spec in enumerate(FEATURE_SPECS):
                mu, sd = agent.ref_mean[ri, fi], agent.ref_std[ri, fi]
                if t < spec.valid_from or not np.isfinite(x[fi]) or not np.isfinite(mu) \
                        or not np.isfinite(sd) or sd < 1e-6:
                    continue
                picked.append((abs(x[fi] - mu) / sd, fi, x[fi], mu, sd))
        picked.sort(reverse=True)
        rows.append([[fi, _sig(v), _sig(mu), _sig(sd)] for _, fi, v, mu, sd in picked[:SNAP_FEATURES]])
    return {"t": times, "f": rows}


def pack_launch(res: Dict[str, Any], meta: Dict[str, Any], lo: np.ndarray, hi: np.ndarray,
                agent: ControlRoomAgent) -> Dict[str, Any]:
    launch, rp = res["launch"], res["replay"]
    mn, mx, _ = _envelope(res["tel"])
    probs = np.clip(np.round(rp.probs * 250.0), 0, 250).astype(np.uint8)
    calls = np.array([CALLS.index(c) for c in rp.calls], dtype=np.uint8)
    # After an ABORT the engine is shut down and the vehicle stays on the pad: the call stays
    # ABORT, and later alerts are dropped. The models never saw a shutdown (training launches
    # run on), so what they say about one means nothing; the evaluation stops there too.
    safed = res["stop"]["t"] if res["stop"]["type"] == "abort" else None
    if safed is not None:
        calls[rp.ticks >= safed] = CALLS.index("ABORT")
    red = np.zeros(len(rp.ticks), dtype=np.uint8)
    for e in res["redlines"]:
        code = REDLINE_CODES[e["call"]]
        red[rp.ticks >= e["t"]] = np.maximum(red[rp.ticks >= e["t"]], code)
    events: List[Dict[str, Any]] = vehicle_events(launch, res["stop"], res["t_end"])
    for e in rp.events:
        if e["t"] > res["t_end"] or safed is not None and e["t"] > safed:
            continue        # a hold's replay ends early; after an abort, see above
        if e["kind"] == "call":
            events.append({"t": e["t"], "kind": "call", "call": e["call"]})
        elif e["kind"] == "rule":
            r = e["rule"]
            events.append({"t": e["t"], "kind": "rule", "mode": e["key"],
                           "rule": {"id": r["short_id"], "grade": r["grade"],
                                    "sentence": r["sentence"]}})
        else:
            events.append({"t": e["t"], "kind": "alert", "mode": e["key"], "level": e["level"],
                           "p": _round(e["p"]), "text": e["text"],
                           "rules": [{"id": r["short_id"], "grade": r["grade"],
                                      "sentence": r["sentence"]} for r in e["rules"]],
                           "evidence": [{"label": x["label"], "unit": x["unit"],
                                         "value": _round(x["value"], 4),
                                         "mean": _round(x["nominal_mean"], 4),
                                         "std": _round(x["nominal_std"], 4)}
                                        for x in e["evidence"]],
                           "graph": e["graph"], "has_rules": e["has_rules"]})
    for e in res["redlines"]:
        events.append({"t": e["t"], "kind": "redline", "call": e["call"], "text": e["text"]})
    events.sort(key=lambda e: (e["t"], {"vehicle": 0, "redline": 1, "alert": 2, "rule": 3, "call": 4}[e["kind"]]))
    truth = [{"mode": k, "display": BY_KEY[k].display, "severity": _round(v["severity"], 2),
              "onset": _round(v["onset"], 1), "manifest": _round(v["manifest"], 1)}
             for k, v in sorted(launch.labels.items(), key=lambda kv: kv[1]["manifest"])]
    return {
        **meta,
        "seed": int(launch.seed),
        "t_end": res["t_end"],
        "stop": res["stop"],
        "liftoff": launch.meta.get("liftoff") if res["stop"]["type"] is None else None,
        "truth": truth,
        "tel": {"min": _channels_b64(mn, lo, hi), "max": _channels_b64(mx, lo, hi)},
        "agent": {"n": int(len(rp.ticks)), "probs": _b64(probs), "calls": _b64(calls),
                  "redline": _b64(red)},
        "first_alert": {k: t for k, t in rp.first_alert.items()
                        if t <= res["t_end"] and (safed is None or t <= safed)},
        "events": events,
        "snap": feature_snapshots(agent, launch.tel, snapshot_times(events, res["t_end"])),
    }


# ---------------------------------------------------------------------------------------------
# Factory and performance summaries
# ---------------------------------------------------------------------------------------------

def factory_summary() -> Dict[str, Any]:
    tax = json.loads((DOMAIN_DIR / "layer-1" / "taxonomy.json").read_text())
    rules = json.loads((DOMAIN_DIR / "layer-2" / "rules.json").read_text())
    kg = json.loads((DOMAIN_DIR / "layer-3" / "knowledge-graph.json").read_text())
    ev = json.loads((DOMAIN_DIR / "layer-4" / "evaluation.json").read_text())
    edges = [e for e in kg["edges"] if e["type"] in
             ("PART_OF", "SEEN_ON", "TRIGGERS", "CO_OCCURS", "EXCLUDES")]
    return {
        "taxonomy": [{"mode": d["canonical"], "display": d["display"],
                      "category": tax["categories"][d["category"]]["label"], "phase": d["phase"],
                      "call": d["call"], "count": d["positive_count"],
                      "prevalence": d["prevalence"], "severity": d["median_severity"],
                      "description": d["description"]} for d in tax["diagnoses"]],
        "taxonomy_meta": tax["meta"],
        "rules": [{"id": r["short_id"], "mode": r["diagnosis"],
                   "display": BY_KEY[r["diagnosis"]].display, "sentence": r["sentence"],
                   "grade": r["evidence_grade"], "sens": r["sensitivity"],
                   "spec": r["specificity"], "holdout_f1": r["holdout"]["validation_f1"],
                   "channels": r["channels"]} for r in rules["threshold_rules"]],
        "mutex": [{"members": m["members"], "sentence": m["sentence"],
                   "grade": m["evidence_grade"]} for m in rules["mutex_rules"]],
        "rules_meta": rules["meta"],
        "kg": {"nodes": {k: kg["nodes"][k] for k in ("subsystems", "channels", "failure_modes",
                                                        "actions")},
               "edges": edges, "meta": kg["meta"]},
        "model": ev["model"], "split": ev["split"], "acceptance": ev["acceptance"],
        "end": {k: {m: v.get(m) for m in ("sensitivity", "specificity", "f1", "auroc", "auprc",
                                           "support")}
                for k, v in ev["holdout_end_of_window"]["per_class"].items()},
        "end_macro": {m: ev["holdout_end_of_window"]["macro"].get(m)
                      for m in ("sensitivity", "specificity", "f1", "auroc")},
        "streaming": ev["holdout_streaming"],
        "over_time": ev["holdout_over_time"],
        "calibration": ev["threshold_calibration"],
        "thresholds": ev["thresholds"],
    }


def performance_summary() -> Dict[str, Any]:
    ag = json.loads((HERE / "agent_evaluation.json").read_text())
    per_class = {}
    gates = {"countdown": (-1.0, "HOLD"), "engine_start": (6.5, "ABORT")}
    for fm in FAILURE_MODES:
        pos = [r for r in ag["launches"] if fm.key in r["labels"]]
        red = 0
        for r in pos:
            m = r["labels"][fm.key]["manifest"]
            gate, _ = gates.get(fm.phase, (V.T_END, None))
            if fm.key == "COMBUSTION_INSTABILITY" and m <= 6.5:
                gate = 6.5
            red += any(m - 0.5 <= e["t"] <= gate for e in r["redlines"])
        a = ag["per_class"][fm.key]
        per_class[fm.key] = {"display": fm.display, "phase": fm.phase, "n": a["n_launches"],
                             "agent_detection": a["detection_rate"], "agent_in_time": a["in_time_rate"],
                             "agent_latency": a["median_latency_s"],
                             "agent_false_per_1000": a["false_alerts_per_1000"],
                             "redline_detection": red / len(pos) if pos else None}
    return {k: v for k, v in ag.items() if k not in ("launches", "per_class")} | {"per_class": per_class}


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, default=None,
                    help="also write the page body (no <html>/<head>) for publishing")
    args = ap.parse_args()
    agent = ControlRoomAgent(DOMAIN_DIR)
    lo = np.array([c.lo for c in V.CHANNELS])
    hi = np.array([c.hi for c in V.CHANNELS])

    # nominal band: 5th-95th percentile of nominal launches, per display bin
    means = []
    for i in range(N_BAND_LAUNCHES):
        tel = simulate_launch(9_200_000 + i, random_anomalies=False).tel
        means.append(_envelope(tel)[2])
    band_lo = np.nanpercentile(np.stack(means), 5, axis=0)
    band_hi = np.nanpercentile(np.stack(means), 95, axis=0)

    launches = []
    for s in SHOWCASE:
        res = replay_launch(agent, s["seed"], s["forced"], random_anomalies=False)
        launches.append(pack_launch(res, {"id": s["id"], "kind": "showcase", "title": s["title"],
                                          "summary": s["summary"], "lay": s["lay"]},
                                    lo, hi, agent))
        print(f"  showcase {s['id']:20s} stop={res['stop']} alerts={res['replay'].first_alert}")

    # Unseen test launches: an index of all 500 held-out launches for the page's selector, and
    # one replay per group built in; the demo server replays any other one on demand.
    holdout = holdout_index(json.loads((HERE / "agent_evaluation.json").read_text()), agent.keys)
    for group, seed in holdout["examples"].items():
        res = replay_launch(agent, seed, None, random_anomalies=True)
        launches.append(pack_launch(res, case_meta(seed), lo, hi, agent))
        print(f"  example {group:26s} seed={seed} truth={sorted(res['launch'].labels)} "
              f"stop={res['stop']} alerts={res['replay'].first_alert}")

    channels = []
    for c in V.CHANNELS:
        channels.append({"id": c.key, "label": c.label, "unit": c.unit,
                         "subsystem": V.SUBSYSTEMS[c.subsystem], "lo": c.lo, "hi": c.hi})
    data = {
        "meta": {"title": "Launch Control Agent", "t0": V.T_START, "t1": V.T_END,
                 "bin_s": BIN / V.FS, "n_bins": N_BINS, "tick_s": 0.5,
                 "events": V.EVENTS},
        "channels": channels,
        "band": {"lo": _channels_b64(band_lo, lo, hi), "hi": _channels_b64(band_hi, lo, hi)},
        "classes": [{"mode": fm.key, "display": fm.display, "phase": fm.phase, "call": fm.call,
                     "threshold": agent.thresholds[agent.keys.index(fm.key)],
                     "subsystem": V.SUBSYSTEMS[fm.subsystem], "description": fm.description,
                     "lay": MODE_LAY[fm.key]} for fm in FAILURE_MODES],
        "features": [{"label": f.description, "unit": f.unit} for f in FEATURE_SPECS],
        "corridor": trajectory_corridor(),
        "redlines": [{"text": r.text, "call": r.call, "from": r.t_from, "to": r.t_to}
                     for r in REDLINES],
        "class_order": agent.keys,
        "calls": list(CALLS),
        "launches": launches,
        "factory": factory_summary(),
        "performance": performance_summary(),
        "universe": universe_payload(),
        "holdout": holdout,
    }
    if data["universe"] is None:
        print("no knowledge universe: run ControlRoom/build_universe.py first to include it")
    payload = json.dumps(_clean(data), separators=(",", ":"))
    template = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    body = template.replace("/*__DATA__*/null", payload)
    standalone = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                  "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, "
                  "viewport-fit=cover\">\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    out = HERE / "dashboard.html"
    out.write_text(standalone, encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")
    if args.artifact:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(body, encoding="utf-8")
        print(f"wrote {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
