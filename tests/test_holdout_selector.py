"""The unseen-launch selector: the page's index of the 500 held-out launches, its built-in
examples, and replays that stop where the vehicle stops.

The index test uses a made-up evaluation, so it needs no model; the others read the
committed dashboard.html.
"""

from __future__ import annotations

import base64
import json
import re

import numpy as np
import pytest

import build_dashboard as bd
from agent import CALLS
from launch_sim.anomalies import FAILURE_KEYS

KEYS = list(FAILURE_KEYS)
J = KEYS.index


def _alert(key: str, t: float, call: str) -> list:
    return [J(key), t, CALLS.index(call)]


def page_data() -> dict:
    html = (bd.HERE / "dashboard.html").read_text(encoding="utf-8")
    return json.loads(re.search(r"const DATA = (\{.*?\});\s*\nconst SERVER", html, re.S).group(1))


@pytest.fixture(scope="module")
def page() -> dict:
    return page_data()


def _record(seed, labels=None, alerts=None, hold=None, abort=None) -> dict:
    return {"seed": seed, "first_alert": alerts or {}, "hold_at": hold, "abort_at": abort,
            "labels": {k: {"severity": s, "manifest": m} for k, (s, m) in (labels or {}).items()}}


def test_the_index_keeps_only_what_a_replay_shows():
    ev = {"launches": [
        _record(1),                                                    # normal, quiet
        _record(2, alerts={"POGO_ONSET": 90.0}),                       # normal, false alarm
        _record(3, {"SLOW_ENGINE_START": (0.2, 2.0)},
                {"SLOW_ENGINE_START": 3.5, "TEMP_SENSOR_FAULT": 9.0}, abort=3.5),
        _record(4, {"LH2_PRESS_OSCILLATION": (0.4, -30.0), "POGO_ONSET": (0.5, 100.0)},
                {"LH2_PRESS_OSCILLATION": -29.0, "BUS_VOLTAGE_SAG": -20.0, "POGO_ONSET": 104.0},
                hold=-29.0),
        _record(5, {"POGO_ONSET": (0.1, 110.0)}),
        _record(6, {"POGO_ONSET": (0.6, 100.0)}, {"POGO_ONSET": 104.0}),
        _record(7, {"POGO_ONSET": (0.3, 105.0)}, {"POGO_ONSET": 108.0}),
        _record(8, {"POGO_ONSET": (0.3, 105.0), "TVC_ACTUATOR_DEGRADATION": (0.2, 50.0)},
                {"POGO_ONSET": 108.0, "TVC_ACTUATOR_DEGRADATION": 55.0}),
    ]}
    idx = bd.holdout_index(ev, KEYS)
    assert idx["seeds"] == list(range(1, 9))
    assert idx["stops"][:4] == [None, None, ["abort", 3.5], ["hold", -29.0]]
    # After an abort nothing new; a hold shows 12 s more of the countdown, not the flight.
    # Each alert carries the call it raised then: the same failure can be ABORT or CRITICAL.
    assert idx["alerts"][2] == [_alert("SLOW_ENGINE_START", 3.5, "ABORT")]
    assert idx["alerts"][3] == [_alert("LH2_PRESS_OSCILLATION", -29.0, "HOLD"),
                                _alert("BUS_VOLTAGE_SAG", -20.0, "HOLD")]
    assert idx["alerts"][7] == [_alert("TVC_ACTUATOR_DEGRADATION", 55.0, "WARNING"),
                                _alert("POGO_ONSET", 108.0, "WARNING")]
    assert idx["truth"][3] == [[J("LH2_PRESS_OSCILLATION"), 0.4, -30.0], [J("POGO_ONSET"), 0.5, 100.0]]
    # Built-in examples: the median-severity launch with only that failure (POGO: 0.1, 0.3,
    # 0.6), the first quiet normal launch, the first multi-failure launch caught in full.
    assert idx["examples"] == {"SLOW_ENGINE_START": 3, "POGO_ONSET": 7, "NOMINAL": 1, "MULTI": 8}


def test_the_page_lists_every_held_out_launch_and_plays_one_per_group(page):
    ho = page["holdout"]
    split = json.loads((bd.DOMAIN_DIR / "layer-4" / "split.json").read_text())
    assert len(ho["seeds"]) == 500 and set(ho["seeds"]) == set(split["holdout_seeds"])
    assert set(ho["examples"]) == set(KEYS) | {"NOMINAL", "MULTI"}
    built = {L["seed"]: L for L in page["launches"] if L["kind"] == "test"}
    assert set(built) == set(ho["examples"].values())
    for n, seed in enumerate(ho["seeds"]):
        if seed not in built:
            continue
        L = built[seed]
        assert L["id"] == f"case-{seed}" and L["title"] == f"Unseen launch #{seed % 100_000:04d}"
        shown = {}
        for e in L["events"]:
            if e["kind"] == "alert":
                shown.setdefault(e["mode"], [e["t"], e["level"]])
        assert shown == {KEYS[j]: [t, CALLS[c]] for j, t, c in ho["alerts"][n]}
        assert sorted(x["mode"] for x in L["truth"]) == sorted(KEYS[j] for j, *_ in ho["truth"][n])
        assert ([L["stop"]["type"], L["stop"]["t"]] if L["stop"]["type"] else None) == ho["stops"][n]


def test_after_an_abort_the_call_stays_abort_and_nothing_new_alarms(page):
    aborted = [L for L in page["launches"] if L["stop"]["type"] == "abort"]
    assert len(aborted) >= 3
    for L in aborted:
        t_abort = L["stop"]["t"]
        ticks = page["meta"]["t0"] + page["meta"]["tick_s"] * np.arange(L["agent"]["n"])
        calls = np.frombuffer(base64.b64decode(L["agent"]["calls"]), np.uint8)
        assert set(calls[ticks >= t_abort]) == {page["calls"].index("ABORT")}, L["id"]
        late = [e for e in L["events"] if e["t"] > t_abort and e["kind"] != "vehicle"]
        assert not late, (L["id"], late)
        red = np.frombuffer(base64.b64decode(L["agent"]["redline"]), np.uint8)
        assert len(set(red[ticks >= t_abort])) == 1, L["id"]
