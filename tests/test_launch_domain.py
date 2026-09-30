"""Tests for the launch-vehicle domain: simulator, causal features, artifacts, agent.

Run from the repository root:  python -m pytest tests -q
The agent tests need layer-4/model.pkl (regenerate with run_factory.py); they skip
without it. The end-to-end factory run is marked slow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from launch_sim import vehicle as V
from launch_sim.anomalies import FAILURE_KEYS
from launch_sim.features import FEATURE_INDEX, FEATURE_NAMES, LaunchFeatures
from launch_sim.simulate import simulate_launch
from launch_sim.streaming import debounced
from redline import redline_events

DOMAIN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DOMAIN_DIR
HAS_MODEL = (DOMAIN_DIR / "layer-4" / "model.pkl").exists()


def _load(rel: str):
    return json.loads((DOMAIN_DIR / rel).read_text())


# ---- simulator ------------------------------------------------------------------------------

def test_simulator_is_deterministic():
    a, b = simulate_launch(4242), simulate_launch(4242)
    assert a.tel.shape == (len(V.CHANNELS), V.N_SAMPLES)
    assert np.array_equal(a.tel, b.tel, equal_nan=True)
    assert a.labels == b.labels


def test_abort_keeps_the_telemetry_before_the_abort():
    forced = {"SLOW_ENGINE_START": {"severity": 0.5}}
    full = simulate_launch(77, forced=forced, random_anomalies=False)
    aborted = simulate_launch(77, forced=forced, random_anomalies=False, abort_at=3.0)
    k = V.k_of(3.3)
    assert np.array_equal(full.tel[:, :k], aborted.tel[:, :k], equal_nan=True)
    assert aborted.meta["liftoff"] is None
    assert np.nanmax(aborted.tel[V.CH["PC_SRB_L"]]) < 5.0          # boosters never lit


def test_nominal_launch_reaches_liftoff_and_booster_burnout():
    launch = simulate_launch(5, random_anomalies=False, confusers=False)
    assert 7.0 <= launch.meta["liftoff"] <= 7.5
    acc = launch.tel[V.CH["ACC_AX"]]
    assert 3.0 < np.nanmax(acc[V.k_of(110.0):V.k_of(125.0)]) < 4.0


DRIFT = {"TRAJECTORY_DEVIATION": {"severity": 0.5, "onset": 30.0, "mode": "drift"}}


def test_a_trajectory_failure_leaves_every_other_channel_alone():
    # Own random streams: adding the channel and its failure changed nothing else in a launch.
    plain = simulate_launch(313, random_anomalies=False)
    drift = simulate_launch(313, random_anomalies=False, forced=DRIFT)
    k = V.CH["TRAJ_DEV"]
    others = [i for i in range(len(V.CHANNELS)) if i != k]
    assert np.array_equal(plain.tel[others], drift.tel[others], equal_nan=True)
    assert np.nanmax(drift.tel[k]) > 3.0 * np.nanmax(plain.tel[k])


def test_normal_flights_stay_in_the_corridor_and_a_drift_leaves_it():
    t, k = V.TIME, V.CH["TRAJ_DEV"]
    for seed in range(20):
        dev = simulate_launch(1000 + seed, random_anomalies=False).tel[k]
        assert np.nanmax(dev / V.trajectory_corridor(t)) < 0.6        # nowhere near it
    drift = simulate_launch(99, random_anomalies=False, forced=DRIFT)
    over = np.nonzero((t >= 7.0) & (drift.tel[k] > V.trajectory_corridor(t)))[0]
    visible = drift.labels["TRAJECTORY_DEVIATION"]["manifest"]
    assert 30.0 < visible < t[over[0]] < V.T_END          # first visible, later beyond the corridor
    trips = [e for e in redline_events(drift.tel) if e["text"].startswith("Trajectory")]
    assert trips and abs(trips[0]["t"] - t[over[0]]) < 0.5


# ---- features -------------------------------------------------------------------------------

@pytest.mark.parametrize("t_now", [-45.0, -30.0, -1.0, 2.5, 6.5, 15.0, 60.0, 150.0])
def test_features_are_causal(t_now):
    """The vector at t equals the one computed from telemetry cut at t."""
    tel = simulate_launch(31).tel
    full = LaunchFeatures(tel).vector(t_now)
    cut = LaunchFeatures(tel[:, : V.k_of(t_now) + 1]).vector(t_now)
    assert np.array_equal(full, cut, equal_nan=True)


def test_windows_that_have_not_started_are_nan():
    x = LaunchFeatures(simulate_launch(3).tel).vector(-45.0)
    assert np.isnan(x[FEATURE_INDEX["ASC__PC_CORE__rise_max"]])
    assert np.isnan(x[FEATURE_INDEX["IGN__PC_CORE__t90"]])
    assert np.isfinite(x[FEATURE_INDEX["CD__P_LOX_TANK__t_ready"]])


def test_slow_start_is_visible_in_the_rise_time():
    i = FEATURE_INDEX["IGN__PC_CORE__t90"]
    nominal = LaunchFeatures(simulate_launch(8, random_anomalies=False).tel).vector(6.5)[i]
    slow = LaunchFeatures(simulate_launch(8, forced={"SLOW_ENGINE_START": {"severity": 0.6}},
                                          random_anomalies=False).tel).vector(6.5)[i]
    assert slow > nominal + 0.5


def test_debounce_needs_consecutive_ticks():
    p = np.array([[0.0], [0.9], [0.1], [0.9], [0.9]])
    held = debounced(p, axis=0)[:, 0]
    assert held.tolist() == [0.0, 0.0, 0.1, 0.1, 0.9]


# ---- specification and committed artifacts --------------------------------------------------

def test_spec_yaml_is_valid():
    pytest.importorskip("yaml")
    from spec_schema import DomainSpec
    spec = DomainSpec.from_yaml(DOMAIN_DIR / "spec.yaml")
    assert spec.domain == "launch-vehicle"
    assert spec.signal_processing.expected_feature_count == len(FEATURE_NAMES)


def test_layer_artifacts_are_consistent():
    from discovery.integration import check_consistency
    taxonomy = _load("layer-1/taxonomy.json")
    assert taxonomy["meta"]["acceptance_pass"]
    assert {d["canonical"] for d in taxonomy["diagnoses"]} == set(FAILURE_KEYS) | {"NOMINAL"}
    rules = _load("layer-2/rules.json")
    assert check_consistency(rules) == []
    for r in rules["threshold_rules"]:
        assert r["feature"] in FEATURE_INDEX
        assert r["evidence_grade"] in ("A", "B")
    kg = _load("layer-3/knowledge-graph.json")
    ids = {n["id"] for group in kg["nodes"].values() for n in group}
    for e in kg["edges"]:
        assert e["from"] in ids and e["to"] in ids, e
    evaluation = _load("layer-4/evaluation.json")
    assert evaluation["acceptance"]["pass"]


def test_discovered_coupling_is_the_planted_one():
    kg = _load("layer-3/knowledge-graph.json")
    pairs = {frozenset((e["from"], e["to"])) for e in kg["edges"] if e["type"] == "CO_OCCURS"}
    assert frozenset(("fm:LOX_TANK_UNDERPRESSURE", "fm:LOX_PUMP_CAVITATION")) in pairs


# ---- agent ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agent():
    if not HAS_MODEL:
        pytest.skip("layer-4/model.pkl missing: run run_factory.py first")
    from agent import ControlRoomAgent
    return ControlRoomAgent(DOMAIN_DIR)


def _calls(agent, forced, seed):
    launch = simulate_launch(seed, forced=forced, random_anomalies=False)
    rp = agent.replay(launch.tel)
    return rp, dict(zip([float(t) for t in rp.ticks], rp.calls))


def test_agent_stays_quiet_on_a_nominal_launch(agent):
    rp, calls = _calls(agent, {}, 9_100_001)
    assert rp.first_alert == {}
    assert calls[-1.0] == "GO" and calls[6.5] == "COMMIT" and calls[150.0] == "NOMINAL"


def test_agent_aborts_a_slow_start_before_booster_ignition(agent):
    rp, calls = _calls(agent, {"SLOW_ENGINE_START": {"severity": 0.35}}, 9_100_004)
    assert "ABORT" in {calls[t] for t in calls if 0.0 <= t <= 6.5}


def test_agent_separates_sensor_fault_from_overtemperature(agent):
    rp, calls = _calls(agent, {"TEMP_SENSOR_FAULT": {"severity": 0.4, "onset": 62.0,
                                                     "mode": "step"}}, 9_100_005)
    assert "TEMP_SENSOR_FAULT" in rp.first_alert
    assert "TURBINE_OVERTEMP" not in rp.first_alert
    assert calls[150.0] == "ADVISORY"
    rp, calls = _calls(agent, {"TURBINE_OVERTEMP": {"severity": 0.3, "onset": 70.0,
                                                    "ramp": 8.0}}, 9_100_006)
    assert "TURBINE_OVERTEMP" in rp.first_alert
    assert calls[150.0] == "CRITICAL"


def test_every_alert_is_explained(agent):
    rp, _ = _calls(agent, {"LOX_TANK_UNDERPRESSURE": {"severity": 0.3}}, 9_100_002)
    alerts = [e for e in rp.events if e["kind"] == "alert"]
    assert alerts and alerts[0]["level"] == "HOLD"
    assert "HOLD: LOX tank under-pressure" in alerts[0]["text"]
    if not alerts[0]["rules"]:                      # an early call says so ...
        assert "can be checked yet" in alerts[0]["text"]
    confirmed = [e for e in rp.events if e["kind"] == "rule" and e["t"] < -1.0]
    assert alerts[0]["rules"] or confirmed          # ... and a graded rule backs it before ignition


# ---- end to end -----------------------------------------------------------------------------

@pytest.mark.slow
def test_factory_runs_end_to_end(tmp_path):
    cmd = [sys.executable, str(DOMAIN_DIR / "run_factory.py"), "--n-launches", "240",
           "--workers", "2", "--out-dir", str(tmp_path)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, timeout=900)
    for rel in ("layer-1/taxonomy.json", "layer-2/rules.json", "layer-3/knowledge-graph.json",
                "layer-4/evaluation.json", "layer-4/model.pkl"):
        assert (tmp_path / rel).exists(), rel
