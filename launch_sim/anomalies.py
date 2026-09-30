"""Failure-mode catalog (Layer 1 metadata) and the random draws that inject them.

Every signature below is a hypothesis written from public engineering knowledge,
not a measurement. Severity ``s`` in [0, 1] scales a signature from barely
visible to strong. ``manifest`` is the time from which a failure is visible in
the telemetry; streaming labels switch on at that time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FailureMode:
    key: str
    display: str
    category: str
    phase: str            # countdown | engine_start | flight | engine_start_or_flight
    call: str             # control-room call when the agent detects it
    severity: str         # medium | high | critical
    subsystem: str
    channels: Tuple[str, ...]
    description: str


FAILURE_MODES: Tuple[FailureMode, ...] = (
    FailureMode(
        "LOX_TANK_UNDERPRESSURE", "LOX tank under-pressure", "pressurization", "countdown",
        "HOLD", "high", "tanks", ("P_LOX_TANK",),
        "LOX tank pressure rises too slowly and settles below the flight setpoint. Low tank "
        "pressure lowers the pump inlet pressure and raises the cavitation risk at engine start."),
    FailureMode(
        "LH2_PRESS_OSCILLATION", "LH2 pressure regulator oscillation", "pressurization",
        "countdown", "HOLD", "medium", "tanks", ("P_LH2_TANK",),
        "The LH2 tank pressure regulator hunts: a slow 0.2-0.4 Hz pressure oscillation around "
        "the setpoint."),
    FailureMode(
        "BUS_VOLTAGE_SAG", "Bus voltage sag at power transfer", "electrical", "countdown",
        "HOLD", "medium", "avionics_power", ("BUS_28V",),
        "At the switch to onboard batteries the 28 V bus dips deeply, recovers slowly and stays "
        "low: a weak battery or a high-resistance connection."),
    FailureMode(
        "SLOW_ENGINE_START", "Slow core engine start", "core_engine", "engine_start",
        "ABORT", "critical", "core_engine", ("PC_CORE", "N_LOX_TP", "T_TURB_IN"),
        "Chamber pressure and turbopump speed build up too slowly after ignition."),
    FailureMode(
        "LOX_PUMP_CAVITATION", "LOX turbopump cavitation", "core_engine", "engine_start",
        "ABORT", "critical", "core_engine", ("N_LOX_TP", "PC_CORE", "VIB_ENG"),
        "The LOX pump loses head in bursts: pump speed spikes, chamber pressure dips and "
        "vibration rises. Most likely during the engine start, when inlet pressure is lowest."),
    FailureMode(
        "COMBUSTION_INSTABILITY", "Combustion instability", "core_engine",
        "engine_start_or_flight", "ABORT", "critical", "core_engine", ("PC_CORE", "VIB_ENG"),
        "Rough combustion: chamber pressure ripple and a jump in engine-bay vibration."),
    FailureMode(
        "TURBINE_OVERTEMP", "Turbine over-temperature", "core_engine", "flight",
        "CRITICAL", "critical", "core_engine", ("T_TURB_IN", "N_LOX_TP", "PC_CORE"),
        "Gas-generator mixture shift: turbine inlet temperature rises, and with the extra "
        "turbine power the pump speed and chamber pressure rise too."),
    FailureMode(
        "TEMP_SENSOR_FAULT", "Turbine temperature sensor fault", "instrumentation", "flight",
        "ADVISORY", "medium", "core_engine", ("T_TURB_IN",),
        "The turbine temperature sensor fails (step, spikes, or stuck at full scale) while "
        "pump speed and chamber pressure stay nominal. A fixed redline cannot tell this from a "
        "real over-temperature."),
    FailureMode(
        "SRB_THRUST_ASYMMETRY", "Booster thrust imbalance", "boosters", "flight",
        "WARNING", "high", "boosters", ("PC_SRB_L", "PC_SRB_R", "TVC_YAW"),
        "One booster's chamber pressure diverges; the core engine TVC takes a yaw offset to "
        "compensate the imbalance."),
    FailureMode(
        "TVC_ACTUATOR_DEGRADATION", "TVC actuator degradation", "flight_control", "flight",
        "WARNING", "high", "flight_control", ("TVC_PITCH",),
        "The pitch actuator develops a limit cycle: a 1-2.2 Hz oscillation in deflection."),
    FailureMode(
        "POGO_ONSET", "POGO onset", "structural_dynamics", "flight",
        "WARNING", "high", "structure", ("ACC_AX", "PC_CORE"),
        "A growing longitudinal oscillation (12-18 Hz) in acceleration, coupled with chamber "
        "pressure, late in the boosted phase."),
    FailureMode(
        "TRAJECTORY_DEVIATION", "Trajectory deviation", "guidance", "flight",
        "CRITICAL", "critical", "guidance", ("TRAJ_DEV",),
        "The guidance follows a wrong direction reference (a drifting gyro, or a wrong heading "
        "setting) and the vehicle flies it faithfully. It leaves its planned "
        "trajectory while engines and steering look normal; a sudden offset shows a short "
        "steering kick. Beyond the flight-safety corridor it is a destruct decision."),
)
FAILURE_KEYS: Tuple[str, ...] = tuple(f.key for f in FAILURE_MODES)
BY_KEY: Dict[str, FailureMode] = {f.key: f for f in FAILURE_MODES}
NOMINAL_KEY = "NOMINAL"
TRAJECTORY_KEY = "TRAJECTORY_DEVIATION"

BASE_RATE = 0.06
CAVITATION_BASE = 0.035
CAVITATION_COUPLING = 0.55     # extra probability per unit of LOX under-pressure severity
CONFUSER_RATE = 0.12


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def draw_failure(name: str, rng: np.random.Generator, severity: Optional[float] = None,
                 **overrides: Any) -> Dict[str, Any]:
    """Draw the parameters of one failure. ``overrides`` pin onset/mode/etc. for showcases."""
    # Mild cases dominate, as in real operations: Beta(1, 1.8) has mean 0.36.
    s = float(rng.beta(1.0, 1.8)) if severity is None else float(severity)
    p: Dict[str, Any] = {}
    if name == "LOX_TANK_UNDERPRESSURE":
        onset, p = -58.0, {"tau_factor": 1.0 + 2.0 * s, "set_drop": 0.02 + 0.42 * s}
        manifest = -52.0
    elif name == "LH2_PRESS_OSCILLATION":
        onset = _u(rng, -50.0, -15.0)
        p = {"freq": _u(rng, 0.2, 0.4), "amp": 0.004 + 0.12 * s}
        manifest = None
    elif name == "BUS_VOLTAGE_SAG":
        onset = -30.0
        p = {"dip": 0.3 + 3.7 * s, "tau": 1.0 + 8.0 * s, "residual": 0.05 + 0.9 * s}
        manifest = -30.0
    elif name == "SLOW_ENGINE_START":
        onset, p = 0.0, {"mid_delay": 0.08 + 1.5 * s, "width_factor": 1.0 + 0.6 * s}
        manifest = 2.0
    elif name == "LOX_PUMP_CAVITATION":
        onset = _u(rng, 0.8, 4.0)
        p = {"rate": 0.4 + 1.5 * s, "amp_n": 0.6 + 4.4 * s, "amp_p": 0.8 + 6.0 * s,
             "amp_v": 0.1 + 0.8 * s}
        manifest = None      # first burst, set by the simulator
    elif name == "COMBUSTION_INSTABILITY":
        in_start = bool(rng.uniform() < 0.6)
        onset = _u(rng, 1.0, 5.0) if in_start else _u(rng, 10.0, 120.0)
        p = {"sigma": 0.2 + 2.6 * s, "vib_factor": 1.08 + 2.6 * s}
        manifest = None
    elif name == "TURBINE_OVERTEMP":
        onset = _u(rng, 10.0, 110.0)
        p = {"ramp": _u(rng, 3.0, 15.0), "delta_t": 10.0 + 150.0 * s}
        manifest = None
    elif name == "TEMP_SENSOR_FAULT":
        onset = _u(rng, 8.0, 130.0)
        mode = str(rng.choice(["step", "spikes", "rail"], p=[0.5, 0.3, 0.2]))
        p = {"mode": mode, "step": 15.0 + 260.0 * s, "spike_rate": 0.2 + 2.5 * s,
             "spike_amp": 80.0 + 400.0 * s, "rail": 1300.0}
        manifest = onset
    elif name == "SRB_THRUST_ASYMMETRY":
        onset = _u(rng, 15.0, 90.0)
        p = {"ramp": _u(rng, 2.0, 10.0), "side": str(rng.choice(["L", "R"])),
             "sign": -1.0 if rng.uniform() < 0.7 else 1.0, "frac": 0.01 + 0.08 * s}
        manifest = None
    elif name == "TVC_ACTUATOR_DEGRADATION":
        onset = _u(rng, 15.0, 110.0)
        p = {"freq": _u(rng, 1.0, 2.2), "amp": 0.05 + 0.45 * s}
        manifest = onset + 2.0
    elif name == "POGO_ONSET":
        onset = _u(rng, 95.0, 118.0)
        p = {"freq": _u(rng, 12.0, 18.0), "amp": 0.006 + 0.24 * s, "tau": _u(rng, 4.0, 9.0),
             "phase": _u(rng, 0.0, 2 * np.pi)}
        manifest = None
    elif name == "TRAJECTORY_DEVIATION":
        # A drifting attitude reference (deg/s) or a wrong one from the onset (deg), pointing
        # the thrust off the plan in direction phi. Estimates; see vehicle.trajectory_band.
        onset = _u(rng, 12.0, 80.0)
        p = {"mode": str(rng.choice(["drift", "offset"], p=[0.6, 0.4])),
             "rate": 0.1 + 0.5 * s, "offset": 1.5 + 6.0 * s, "phi": _u(rng, 0.0, 2 * np.pi)}
        manifest = None      # when the drift leaves the normal scatter, set by the simulator
    else:
        raise KeyError(name)

    p.update({k: v for k, v in overrides.items() if k not in ("onset", "manifest")})
    onset = float(overrides.get("onset", onset))
    if manifest is None or "onset" in overrides:
        manifest = _manifest(name, onset, p)
    return {"severity": s, "onset": onset, "manifest": float(manifest), "params": p}


def _manifest(name: str, onset: float, p: Dict[str, Any]) -> float:
    if name == "LOX_TANK_UNDERPRESSURE":
        return -52.0
    if name == "LH2_PRESS_OSCILLATION":
        return onset + 1.0 / p["freq"]
    if name in ("BUS_VOLTAGE_SAG", "TEMP_SENSOR_FAULT", "LOX_PUMP_CAVITATION"):
        return onset
    if name == "SLOW_ENGINE_START":
        return 2.0
    if name == "COMBUSTION_INSTABILITY":
        return onset + 0.5
    if name == "TURBINE_OVERTEMP":
        return onset + 0.3 * p["ramp"]
    if name == "SRB_THRUST_ASYMMETRY":
        return onset + 0.5 * p["ramp"]
    if name == "TVC_ACTUATOR_DEGRADATION":
        return onset + 2.0
    if name == "POGO_ONSET":
        return onset + p["tau"]
    if name == "TRAJECTORY_DEVIATION":
        return onset         # a placeholder: simulate_launch replaces it
    raise KeyError(name)


def draw_random_failures(rng: np.random.Generator) -> Dict[str, Dict[str, Any]]:
    """Independent failures at BASE_RATE, the planted LOX -> cavitation coupling, and the
    over-temperature / sensor-fault exclusivity (one temperature story per launch)."""
    labels: Dict[str, Dict[str, Any]] = {}
    for name in FAILURE_KEYS:
        if name in ("LOX_PUMP_CAVITATION", TRAJECTORY_KEY):     # drawn separately
            continue
        if rng.uniform() < BASE_RATE:
            labels[name] = draw_failure(name, rng)
    s_under = labels.get("LOX_TANK_UNDERPRESSURE", {}).get("severity", 0.0)
    has_under = "LOX_TANK_UNDERPRESSURE" in labels
    p_cav = CAVITATION_BASE + (CAVITATION_COUPLING * s_under if has_under else 0.0)
    if rng.uniform() < p_cav:
        labels["LOX_PUMP_CAVITATION"] = draw_failure("LOX_PUMP_CAVITATION", rng)
    if "TURBINE_OVERTEMP" in labels and "TEMP_SENSOR_FAULT" in labels:
        labels.pop("TURBINE_OVERTEMP" if rng.uniform() < 0.5 else "TEMP_SENSOR_FAULT")
    return labels


def draw_trajectory_failure(rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    """The trajectory failure at BASE_RATE, from its own random stream (see simulate.py), so
    adding it left every other failure, channel and lost frame of every launch unchanged."""
    return draw_failure(TRAJECTORY_KEY, rng) if rng.uniform() < BASE_RATE else None


def draw_confusers(rng: np.random.Generator) -> Dict[str, Dict[str, Any]]:
    """Small, unlabeled disturbances inside normal variation: they tempt false alarms."""
    c: Dict[str, Dict[str, Any]] = {}

    def hit() -> bool:
        return bool(rng.uniform() < CONFUSER_RATE)

    if hit():
        c["lh2_mini_osc"] = {"amp": _u(rng, 0.004, 0.015), "freq": _u(rng, 0.2, 0.4),
                             "onset": _u(rng, -50.0, -15.0)}
    if hit():
        c["lox_set_low"] = {"drop": _u(rng, 0.02, 0.06)}
    if hit():
        c["start_delay"] = {"mid_delay": _u(rng, 0.05, 0.25)}
    if hit():
        c["t_drift"] = {"delta_t": _u(rng, 5.0, 25.0), "onset": _u(rng, 10.0, 120.0),
                        "ramp": _u(rng, 3.0, 15.0)}
    if hit():
        c["t_glitch"] = {"time": _u(rng, 8.0, 148.0), "amp": _u(rng, 40.0, 120.0)}
    if hit():
        c["srb_mismatch"] = {"side": str(rng.choice(["L", "R"])), "frac": _u(rng, 0.005, 0.02)}
    if hit():
        c["tvc_mini_lc"] = {"amp": _u(rng, 0.01, 0.05), "freq": _u(rng, 1.0, 2.2),
                            "onset": _u(rng, 15.0, 110.0)}
    if hit():
        c["bus_dip"] = {"dip": _u(rng, 0.2, 0.9)}
    if hit():
        c["vib_elevated"] = {"factor": _u(rng, 1.05, 1.20)}
    if hit():
        c["pogo_mini"] = {"amp": _u(rng, 0.005, 0.02), "freq": _u(rng, 12.0, 18.0),
                          "onset": _u(rng, 95.0, 120.0)}
    return c
