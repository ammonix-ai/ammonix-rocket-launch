"""Notional launch vehicle: time base, events, telemetry channels, reference profiles.

A cryogenic core stage (LOX/LH2 gas-generator engine) with two solid strap-on
boosters: a notional heavy-launcher *architecture* with illustrative, order-of-magnitude
numbers. Nothing here is data from a real vehicle.

Time is expressed relative to H0 (core engine ignition). The simulated window
runs from the end of the synchronized countdown to booster separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

FS = 50.0                       # telemetry rate [Hz]
T_START, T_END = -60.0, 150.0   # window relative to H0 [s]
N_SAMPLES = int(round((T_END - T_START) * FS)) + 1
TIME = T_START + np.arange(N_SAMPLES) / FS
G0 = 9.80665                    # [m/s^2]


def k_of(t: float) -> int:
    """Sample index of time ``t`` (times on the 20 ms grid map exactly)."""
    k = int(round((t - T_START) * FS))
    return min(max(k, 0), N_SAMPLES - 1)


EVENTS: Dict[str, float] = {
    "LOX_PRESS_START": -58.0,     # LOX tank pressurization to flight pressure
    "LH2_PRESS_START": -55.0,     # LH2 tank pressurization
    "POWER_TRANSFER": -30.0,      # switch from ground power to onboard batteries
    "IGNITION_GATE": -1.0,        # last GO/HOLD decision before ignition
    "H0": 0.0,                    # core engine ignition
    "BOOSTER_COMMIT_GATE": 6.5,   # last point an abort is possible
    "BOOSTER_IGNITION": 7.0,      # solid boosters light; liftoff follows
    "MAX_Q": 60.0,                # maximum dynamic pressure
    "BOOSTER_TAILOFF": 124.0,
    "BOOSTER_SEP": 136.0,
}


@dataclass(frozen=True)
class Channel:
    key: str
    label: str
    unit: str
    subsystem: str
    noise: float        # sensor noise sigma (relative for VIB_ENG)
    lo: float           # display / quantization range
    hi: float


CHANNELS: Tuple[Channel, ...] = (
    Channel("PC_CORE", "Core chamber pressure", "bar", "core_engine", 0.25, 0.0, 135.0),
    Channel("N_LOX_TP", "LOX turbopump speed", "%", "core_engine", 0.12, 0.0, 115.0),
    Channel("T_TURB_IN", "Turbine inlet temperature", "K", "core_engine", 1.5, 200.0, 1400.0),
    Channel("P_LOX_TANK", "LOX tank pressure", "bar", "tanks", 0.008, 1.0, 4.0),
    Channel("P_LH2_TANK", "LH2 tank pressure", "bar", "tanks", 0.006, 1.0, 2.8),
    Channel("PC_SRB_L", "Booster L chamber pressure", "bar", "boosters", 0.35, 0.0, 110.0),
    Channel("PC_SRB_R", "Booster R chamber pressure", "bar", "boosters", 0.35, 0.0, 110.0),
    Channel("TVC_PITCH", "TVC pitch deflection", "deg", "flight_control", 0.012, -4.0, 4.0),
    Channel("TVC_YAW", "TVC yaw deflection", "deg", "flight_control", 0.012, -4.0, 4.0),
    Channel("ACC_AX", "Axial acceleration", "g", "structure", 0.008, 0.0, 5.0),
    Channel("VIB_ENG", "Engine-bay vibration", "g rms", "core_engine", 0.06, 0.0, 12.0),
    Channel("BUS_28V", "Avionics bus voltage", "V", "avionics_power", 0.03, 22.0, 30.0),
    # Distance between where tracking puts the vehicle and where the planned trajectory has it
    # at the same moment. Noise is per tracking axis; the simulator applies it (own stream).
    Channel("TRAJ_DEV", "Trajectory deviation", "km", "guidance", 0.015, 0.0, 40.0),
)
CH: Dict[str, int] = {c.key: i for i, c in enumerate(CHANNELS)}

SUBSYSTEMS: Dict[str, str] = {
    "core_engine": "Core engine",
    "tanks": "Propellant tanks & pressurization",
    "boosters": "Solid boosters",
    "flight_control": "Flight control (TVC)",
    "structure": "Structure & dynamics",
    "avionics_power": "Avionics power",
    "guidance": "Guidance & trajectory",
}

# Nominal values (means of the vehicle-to-vehicle distributions in simulate.py).
NOMINAL = {
    "pc": 115.0, "n": 100.0, "t_turb": 880.0, "t_amb": 285.0,
    "p_lox_vent": 1.35, "p_lox_set": 3.50, "tau_lox": 4.0,
    "p_lh2_vent": 1.25, "p_lh2_set": 2.30, "tau_lh2": 5.0,
    "start_mid": 1.60, "start_w": 0.30,
    "bus_ground": 28.0, "bus_batt": 28.6,
    "liftoff": 7.1,
}

# Booster chamber pressure profile [bar]: ignition, thrust bucket at max-Q, tail-off.
BOOSTER_KNOTS = np.array([
    (7.0, 88.0), (20.0, 92.0), (35.0, 90.0), (50.0, 72.0), (70.0, 72.0), (85.0, 84.0),
    (110.0, 82.0), (124.0, 76.0), (128.0, 55.0), (131.0, 22.0), (133.0, 6.0),
    (134.5, 1.5), (136.0, 1.0),
])

# Mass and thrust model (orders of magnitude for a two-booster configuration).
LIFTOFF_MASS_T = 530.0
CORE_MDOT_T_S = 0.32
BOOSTER_PROPELLANT_T = 142.0
BOOSTER_CASING_T = 11.0
CORE_THRUST_SL_KN = 960.0
CORE_THRUST_GAIN_KN = 400.0      # sea level -> near vacuum over the first ~100 s of flight
BOOSTER_KN_PER_BAR = 40.0
DRAG_PEAK_KN = 180.0


# Trajectory deviation: OUR ESTIMATES, not values of any real launcher or range (real flight-safety corridors
# are mission-specific and not public). A normal flight points 0.1-0.3 deg off its plan (wind,
# thrust misalignment); at 15-33 m/s^2 of acceleration that drifts like the double integral of
# the acceleration, to ~2 km by booster separation. The corridor keeps a margin of 1 km plus 2.5 x that normal spread, so
# normal flights never touch it; leaving it is a flight-safety case (in reality the destruct
# decision). Checked on 400 simulated normal flights: their 99.7 % quantile stays inside the band.
TRAJ_BAND_SEP_KM = 2.2           # normal 3-sigma deviation at booster separation
TRAJ_CORRIDOR_MARGIN_KM = 1.0
TRAJ_CORRIDOR_BANDS = 2.5
# Mean axial acceleration of normal flights [g], measured on 50 simulated launches. A constant
# pointing error drifts like its double integral, so the band takes exactly that shape: a
# normal flight then keeps a flat deviation-to-band ratio, and a failure makes it rise.
ACC_KNOTS_G = np.array([
    (7.1, 1.50), (10.0, 1.57), (20.0, 1.72), (30.0, 1.81), (40.0, 1.80), (50.0, 1.67),
    (60.0, 1.77), (70.0, 1.91), (80.0, 2.28), (90.0, 2.60), (100.0, 2.85), (110.0, 3.13),
    (120.0, 3.34), (124.0, 3.42), (127.0, 2.93), (130.0, 1.93), (132.0, 1.19), (134.0, 0.76),
    (136.0, 0.73), (140.0, 0.77), (150.0, 0.79),
])


def _drift_shape() -> np.ndarray:
    """Double integral of the mean acceleration from liftoff, 1.0 at booster separation."""
    acc = np.where(TIME >= NOMINAL["liftoff"], np.interp(TIME, ACC_KNOTS_G[:, 0], ACC_KNOTS_G[:, 1]), 0.0)
    drift = np.cumsum(np.cumsum(acc * G0) / FS) / FS
    return drift / drift[k_of(EVENTS["BOOSTER_SEP"])]


_DRIFT_SHAPE = _drift_shape()


def trajectory_band(t: np.ndarray) -> np.ndarray:
    """Normal 3-sigma trajectory deviation [km] at time ``t``; tracking noise before liftoff."""
    return 0.03 + TRAJ_BAND_SEP_KM * np.interp(np.asarray(t, dtype=float), TIME, _DRIFT_SHAPE)


def trajectory_corridor(t: np.ndarray) -> np.ndarray:
    """Flight-safety corridor half-width [km] at time ``t`` (illustrative)."""
    return TRAJ_CORRIDOR_MARGIN_KM + TRAJ_CORRIDOR_BANDS * trajectory_band(t)


def logistic(t: np.ndarray, mid: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - mid) / width))


def engine_state(t: np.ndarray, mid: float, width: float) -> np.ndarray:
    """Core engine power level: 0 before H0, S-shaped start, ~3 % overshoot, 1.0 steady."""
    e = logistic(t, mid, width) + 0.03 * np.exp(-((t - (mid + 1.2)) / 0.5) ** 2)
    return np.where(t >= 0.0, e, 0.0)


def dynamic_pressure(t: np.ndarray) -> np.ndarray:
    """Normalized dynamic pressure q(t) with its peak at max-Q."""
    return np.exp(-((t - EVENTS["MAX_Q"]) / 22.0) ** 2)


def booster_profile(t: np.ndarray) -> np.ndarray:
    """Nominal booster chamber pressure [bar]; 1.0 (ambient) before ignition."""
    t_ig = EVENTS["BOOSTER_IGNITION"]
    prof = np.interp(t, BOOSTER_KNOTS[:, 0], BOOSTER_KNOTS[:, 1])
    ignition = 1.0 - np.exp(-np.clip(t - t_ig, 0.0, None) / 0.12)
    return np.where(t >= t_ig, 1.0 + (prof - 1.0) * ignition, 1.0)


def reference_lox_ramp(t: np.ndarray) -> np.ndarray:
    """Engineering reference for LOX tank pressurization (nominal setpoint and time constant)."""
    t0 = EVENTS["LOX_PRESS_START"]
    rise = 1.0 - np.exp(-np.clip(t - t0, 0.0, None) / NOMINAL["tau_lox"])
    return NOMINAL["p_lox_vent"] + (NOMINAL["p_lox_set"] - NOMINAL["p_lox_vent"]) * rise


def reference_vibration(t: np.ndarray) -> np.ndarray:
    """Nominal engine-bay vibration profile [g rms] (no noise, nominal timing)."""
    e = engine_state(t, NOMINAL["start_mid"], NOMINAL["start_w"])
    lo = NOMINAL["liftoff"]
    flying = t >= lo
    vib = 0.03 + 0.9 * e
    vib = vib + np.where(t >= EVENTS["BOOSTER_IGNITION"],
                         2.6 * np.exp(-np.clip(t - 7.2, 0.0, None) / 6.0), 0.0)
    vib = vib + np.where(flying, 0.7 * np.exp(-((t - 52.0) / 10.0) ** 2), 0.0)
    srb = booster_profile(t)
    vib = vib + 0.6 * (srb - 1.0) / 90.0
    vib = vib + np.where(t >= EVENTS["BOOSTER_SEP"],
                         2.0 * np.exp(-np.clip(t - EVENTS["BOOSTER_SEP"], 0.0, None) / 0.8), 0.0)
    return vib
