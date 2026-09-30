"""Simulate one launch: 13 telemetry channels, H0 - 60 s to H0 + 150 s, at 50 Hz.

Physics-lite: every channel follows a nominal profile, and failures act through
physically consistent couplings (more turbine power -> faster pump -> higher
chamber pressure; booster imbalance -> TVC yaw correction; thrust / mass ->
acceleration; acceleration x pointing error -> trajectory deviation). It is not
an engine or flight simulation.

Determinism: all random draws happen in a fixed order that does not depend on
``abort_at``, so re-simulating a launch with an abort reproduces the telemetry
up to the abort exactly. The trajectory channel and its failure draw from two
streams of their own, so adding them left every other channel, failure and lost
frame of every launch exactly as it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from launch_sim import vehicle as V
from launch_sim.anomalies import (TRAJECTORY_KEY, draw_confusers, draw_failure,
                                  draw_random_failures, draw_trajectory_failure)

SEED_BASE = 2026
# Trajectory deviation (estimates, see vehicle.trajectory_band): normal pointing errors per axis.
TRAJ_BIAS_DEG = 0.22          # fixed bias, 1 sigma: thrust misalignment, mean wind
TRAJ_GUST_FRAC = 0.05         # share of the gust the steering answers that still bends the path
TRAJ_IMBALANCE_FRAC = 0.03    # share of a booster-imbalance correction the steering leaves over
TRAJ_MAX_ERROR_DEG = 25.0
TRAJ_STREAM, TRAJ_FAIL_STREAM = 1, 2


def launch_seed(i: int, base: int = SEED_BASE) -> int:
    return base * 100_000 + i


@dataclass
class Launch:
    launch_id: str
    seed: int
    tel: np.ndarray                                   # float32 (13, N_SAMPLES); NaN = lost frame
    labels: Dict[str, Dict[str, Any]]                 # failure key -> severity/onset/manifest/params
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_nominal(self) -> bool:
        return not self.labels


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ramp_in(t: np.ndarray, t0: float, duration: float) -> np.ndarray:
    return _smoothstep((t - t0) / duration)


def _lowpass_noise(rng: np.random.Generator, n: int, fc: float) -> np.ndarray:
    """Unit-variance low-pass noise (2nd-order Butterworth), filter transient discarded."""
    sos = signal.butter(2, fc, btype="low", fs=V.FS, output="sos")
    y = signal.sosfilt(sos, rng.standard_normal(n + 2000))[2000:]
    return y / (y.std() + 1e-12)


def _double_integral(a: np.ndarray) -> np.ndarray:
    """Position [m] from acceleration [m/s^2] on the telemetry grid, starting at rest."""
    dt = 1.0 / V.FS
    return np.cumsum(np.cumsum(a) * dt) * dt


def _first_order_lag(x: np.ndarray, tau: float) -> np.ndarray:
    a = (1.0 / V.FS) / (tau + 1.0 / V.FS)
    return signal.lfilter([a], [1.0, -(1.0 - a)], x, zi=[x[0] * (1.0 - a)])[0]


def simulate_launch(
    seed: int,
    forced: Optional[Dict[str, Dict[str, Any]]] = None,
    random_anomalies: bool = True,
    confusers: bool = True,
    abort_at: Optional[float] = None,
    launch_id: Optional[str] = None,
) -> Launch:
    """Simulate one launch.

    ``forced`` pins failures for showcase stories, e.g.
    ``{"TEMP_SENSOR_FAULT": {"severity": 0.7, "onset": 62.0, "mode": "step"}}``.
    ``abort_at`` shuts the core engine down after that time (plus 0.3 s) and, if it
    comes before booster ignition, the boosters never light and the vehicle stays on
    the pad.
    """
    rng = np.random.default_rng(seed)
    t = V.TIME
    n = V.N_SAMPLES
    nom = V.NOMINAL

    # ---- vehicle-to-vehicle variation --------------------------------------------------
    pc_nom = rng.normal(nom["pc"], 0.8)
    n_nom = rng.normal(nom["n"], 0.5)
    t_nom = rng.normal(nom["t_turb"], 6.0)
    plox_set = rng.normal(nom["p_lox_set"], 0.025)
    plh2_set = rng.normal(nom["p_lh2_set"], 0.018)
    mid = rng.normal(nom["start_mid"], 0.05)
    width = rng.normal(nom["start_w"], 0.02)
    srb_shared = rng.normal(1.0, 0.008)
    srb_scale = {"L": srb_shared * rng.normal(1.0, 0.006), "R": srb_shared * rng.normal(1.0, 0.006)}
    batt_v = rng.normal(nom["bus_batt"], 0.12)
    wind = float(rng.lognormal(0.0, 0.35))

    # ---- failures and confusers ---------------------------------------------------------
    labels = draw_random_failures(rng) if random_anomalies else {}
    for key, spec in (forced or {}).items():
        if key == TRAJECTORY_KEY:
            continue
        spec = dict(spec)
        severity = spec.pop("severity", None)
        labels[key] = draw_failure(key, rng, severity=severity, **spec)
    conf = draw_confusers(rng) if confusers else {}
    rng_traj_fail = np.random.default_rng([seed, TRAJ_FAIL_STREAM])
    if TRAJECTORY_KEY in (forced or {}):
        spec = dict(forced[TRAJECTORY_KEY])
        severity = spec.pop("severity", None)
        labels[TRAJECTORY_KEY] = draw_failure(TRAJECTORY_KEY, rng_traj_fail, severity=severity, **spec)
    elif random_anomalies:
        traj_label = draw_trajectory_failure(rng_traj_fail)
        if traj_label is not None:
            labels[TRAJECTORY_KEY] = traj_label

    # ---- core engine ----------------------------------------------------------------------
    if "SLOW_ENGINE_START" in labels:
        p = labels["SLOW_ENGINE_START"]["params"]
        mid += p["mid_delay"]
        width *= p["width_factor"]
    if "start_delay" in conf:
        mid += conf["start_delay"]["mid_delay"]
    e = V.engine_state(t, mid, width)
    e_n = np.where(t >= 0.0, V.logistic(t, mid - 0.15, 0.9 * width), 0.0)
    e_t = np.where(t >= 0.0, V.logistic(t, mid - 0.30, 0.8 * width), 0.0)

    d_temp = np.zeros(n)
    if "TURBINE_OVERTEMP" in labels:
        lab = labels["TURBINE_OVERTEMP"]
        d_temp += lab["params"]["delta_t"] * _ramp_in(t, lab["onset"], lab["params"]["ramp"])
    if "t_drift" in conf:
        c = conf["t_drift"]
        d_temp += c["delta_t"] * _ramp_in(t, c["onset"], c["ramp"])
    power = 1.0 + 0.22 * d_temp / t_nom       # extra turbine power -> faster pump -> higher Pc

    pc_true = 1.0 + (pc_nom - 1.0) * e * power
    n_true = n_nom * e_n * power
    t_true = (nom["t_amb"] + (t_nom - nom["t_amb"]) * e_t
              + 35.0 * np.exp(-((t - (mid + 0.4)) / 0.5) ** 2) * (t >= 0.0) + d_temp)

    vib_extra = np.zeros(n)
    if "LOX_PUMP_CAVITATION" in labels:
        lab = labels["LOX_PUMP_CAVITATION"]
        p = lab["params"]
        t0, t1 = lab["onset"], nom["liftoff"] + 3.0   # acceleration head ends it after liftoff
        n_bursts = max(1, int(rng.poisson(p["rate"] * max(t1 - t0, 0.5))))
        times = np.sort(np.concatenate([[t0], rng.uniform(t0, t1, n_bursts - 1)]))
        durations = rng.uniform(0.1, 0.3, len(times))
        shape = np.zeros(n)
        for tb, d in zip(times, durations):
            seg = slice(V.k_of(tb - 2 * d), V.k_of(tb + 2 * d) + 1)
            shape[seg] = np.maximum(shape[seg], np.exp(-((t[seg] - tb) / (d / 2)) ** 2))
        n_true = n_true + p["amp_n"] * shape * n_nom / 100.0
        pc_true = pc_true - p["amp_p"] * shape
        vib_extra += p["amp_v"] * shape
        lab["manifest"] = float(times[0])
        p["n_bursts"] = int(len(times))

    vib_factor = np.ones(n)
    if "COMBUSTION_INSTABILITY" in labels:
        lab = labels["COMBUSTION_INSTABILITY"]
        p = lab["params"]
        instab = _ramp_in(t, lab["onset"], 0.5)
        pc_true = pc_true + p["sigma"] * rng.standard_normal(n) * instab * (e > 0.5)
        vib_factor = 1.0 + (p["vib_factor"] - 1.0) * instab

    # ---- abort: engine shutdown (deterministic, after all random draws above) -----------
    t_off = None if abort_at is None else abort_at + 0.3
    if t_off is not None:
        decay = np.where(t > t_off, np.exp(-np.clip(t - t_off, 0.0, None) / 0.4), 1.0)
        cool = np.where(t > t_off, np.exp(-np.clip(t - t_off, 0.0, None) / 3.0), 1.0)
        e = e * decay
        pc_true = 1.0 + (pc_true - 1.0) * decay
        n_true = n_true * decay
        t_true = nom["t_amb"] + (t_true - nom["t_amb"]) * cool
        vib_factor = vib_factor * decay
        vib_extra = vib_extra * decay
    boosters_lit = t_off is None or t_off > V.EVENTS["BOOSTER_IGNITION"]

    # ---- tanks ------------------------------------------------------------------------------
    tau_lox, set_lox = nom["tau_lox"], plox_set
    if "LOX_TANK_UNDERPRESSURE" in labels:
        p = labels["LOX_TANK_UNDERPRESSURE"]["params"]
        tau_lox *= p["tau_factor"]
        set_lox -= p["set_drop"]
    if "lox_set_low" in conf:
        set_lox -= conf["lox_set_low"]["drop"]
    t0 = V.EVENTS["LOX_PRESS_START"]
    p_lox = nom["p_lox_vent"] + (set_lox - nom["p_lox_vent"]) * (
        1.0 - np.exp(-np.clip(t - t0, 0.0, None) / tau_lox))
    saw = 0.015 * (1.0 - 2.0 * (((t - t0 + rng.uniform(0.0, 8.0)) / 8.0) % 1.0))
    p_lox += saw * _ramp_in(t, t0 + 12.0, 6.0)
    p_lox -= 0.05 * np.exp(-((t - 2.0) / 1.0) ** 2) * (t >= 0.0)

    t0 = V.EVENTS["LH2_PRESS_START"]
    p_lh2 = nom["p_lh2_vent"] + (plh2_set - nom["p_lh2_vent"]) * (
        1.0 - np.exp(-np.clip(t - t0, 0.0, None) / nom["tau_lh2"]))
    saw = 0.010 * (1.0 - 2.0 * (((t - t0 + rng.uniform(0.0, 11.0)) / 11.0) % 1.0))
    p_lh2 += saw * _ramp_in(t, t0 + 15.0, 6.0)
    for osc in (labels.get("LH2_PRESS_OSCILLATION"), conf.get("lh2_mini_osc")):
        if osc is None:
            continue
        q = osc.get("params", osc)
        onset = osc["onset"]
        p_lh2 += q["amp"] * np.sin(2 * np.pi * q["freq"] * (t - onset)) * _ramp_in(t, onset, 5.0)

    # ---- boosters ---------------------------------------------------------------------------
    base = V.booster_profile(t)
    srb: Dict[str, np.ndarray] = {}
    for side in ("L", "R"):
        factor = np.full(n, srb_scale[side])
        if "srb_mismatch" in conf and conf["srb_mismatch"]["side"] == side:
            factor *= 1.0 + conf["srb_mismatch"]["frac"]
        if "SRB_THRUST_ASYMMETRY" in labels:
            lab = labels["SRB_THRUST_ASYMMETRY"]
            p = lab["params"]
            if p["side"] == side:
                factor = factor * (1.0 + p["sign"] * p["frac"] * _ramp_in(t, lab["onset"], p["ramp"]))
        srb[side] = 1.0 + (base - 1.0) * factor if boosters_lit else np.ones(n)

    # ---- thrust, mass, liftoff, acceleration -------------------------------------------------
    dt = 1.0 / V.FS
    vac = np.clip((t - V.EVENTS["BOOSTER_IGNITION"]) / 100.0, 0.0, 1.0)
    f_core = (pc_true - 1.0) / (pc_nom - 1.0) * (V.CORE_THRUST_SL_KN + V.CORE_THRUST_GAIN_KN * vac)
    f_srb = V.BOOSTER_KN_PER_BAR * ((srb["L"] - 1.0) + (srb["R"] - 1.0))
    mass = V.LIFTOFF_MASS_T - V.CORE_MDOT_T_S * np.cumsum(e) * dt
    for side in ("L", "R"):
        w = np.clip(srb[side] - 1.0, 0.0, None)
        if w.sum() > 0:
            mass = mass - V.BOOSTER_PROPELLANT_T * np.cumsum(w) / w.sum()
    sep = t >= V.EVENTS["BOOSTER_SEP"]
    if boosters_lit:
        mass = mass - sep * 2 * V.BOOSTER_CASING_T
    weight = mass * V.G0
    thrust = f_core + f_srb
    candidates = np.nonzero((t >= V.EVENTS["BOOSTER_IGNITION"]) & (thrust > weight))[0]
    liftoff = float(t[candidates[0]]) if boosters_lit and len(candidates) else None
    flying = t >= liftoff if liftoff is not None else np.zeros(n, dtype=bool)
    q_dyn = V.dynamic_pressure(t) * flying
    acc = np.where(flying, (thrust - V.DRAG_PEAK_KN * q_dyn) / weight, 1.0)

    pc_meas = pc_true.copy()
    for pogo, tau in ((labels.get("POGO_ONSET"), None), (conf.get("pogo_mini"), 5.0)):
        if pogo is None or liftoff is None:
            continue
        q = pogo.get("params", pogo)
        onset = pogo["onset"]
        grow = (1.0 - np.exp(-np.clip(t - onset, 0.0, None) / (tau or q["tau"]))) * (t >= onset)
        fade = 1.0 - _ramp_in(t, 132.0, 3.0)
        amp = q["amp"] * grow * fade
        phase = q.get("phase", 0.0)
        acc = acc + amp * np.sin(2 * np.pi * q["freq"] * t + phase)
        pc_meas = pc_meas + 12.0 * amp * np.sin(2 * np.pi * q["freq"] * t + phase - 0.8)

    # ---- TVC ---------------------------------------------------------------------------------
    gust_p = _lowpass_noise(rng, n, 0.5)
    gust_y = _lowpass_noise(rng, n, 0.5)
    pad_noise = rng.standard_normal((2, n))
    env = 0.6 * wind * (0.15 + 0.85 * V.dynamic_pressure(t)) * flying
    program = 0.35 * np.sin(np.pi * np.clip((t - 10.0) / 20.0, 0.0, 1.0)) ** 2
    mean_srb = np.maximum((srb["L"] + srb["R"]) / 2.0, 20.0)
    burning = (t >= V.EVENTS["BOOSTER_IGNITION"] + 0.2) & (t < V.EVENTS["BOOSTER_SEP"])
    yaw_bias = 12.0 * (srb["L"] - srb["R"]) / mean_srb * burning
    on_pad = (~flying) * e
    pitch = _first_order_lag(program * flying + env * gust_p + 0.02 * on_pad * pad_noise[0], 0.05)
    yaw = _first_order_lag(yaw_bias + env * gust_y + 0.02 * on_pad * pad_noise[1], 0.05)
    for lc, is_failure in ((labels.get("TVC_ACTUATOR_DEGRADATION"), True), (conf.get("tvc_mini_lc"), False)):
        if lc is None:
            continue
        q = lc.get("params", lc)
        onset = lc["onset"]
        on = _ramp_in(t, onset, 3.0) * flying
        pitch = pitch + q["amp"] * np.sin(2 * np.pi * q["freq"] * (t - onset)) * on
        if is_failure:
            pitch = pitch + 0.03 * rng.standard_normal(n) * (t >= onset) * flying
    traj = labels.get(TRAJECTORY_KEY)
    if traj is not None and traj["params"]["mode"] == "offset":   # the turn onto a wrong heading
        q = traj["params"]
        kick = (min(0.4 * q["offset"], 3.0) * np.exp(-np.clip(t - traj["onset"], 0.0, None) / 1.5)
                * (t >= traj["onset"]) * flying)
        pitch = pitch + kick * np.cos(q["phi"])
        yaw = yaw + kick * np.sin(q["phi"])

    # ---- trajectory deviation ------------------------------------------------------------------
    # Lateral acceleration = acceleration x pointing error, integrated twice, per tracking axis.
    # Normal pointing errors: a fixed bias, the part of the gusts the steering answers, and what
    # a booster-imbalance correction leaves over. A wrong direction reference adds its own error,
    # which the guidance does not correct: it believes it is on its path. Own random stream.
    rng_traj = np.random.default_rng([seed, TRAJ_STREAM])
    deg = np.pi / 180.0
    a_ms2 = acc * V.G0 * flying
    bias = rng_traj.normal(0.0, TRAJ_BIAS_DEG * deg, 2) * (0.75 + 0.25 * wind)
    theta = (bias[0] + TRAJ_GUST_FRAC * deg * env * gust_p,
             bias[1] + TRAJ_GUST_FRAC * deg * env * gust_y + TRAJ_IMBALANCE_FRAC * deg * yaw_bias)
    x_axes = [_double_integral(a_ms2 * th) for th in theta]
    if traj is not None:
        q = traj["params"]
        since = np.clip(t - traj["onset"], 0.0, None)
        angle = q["rate"] * since if q["mode"] == "drift" else q["offset"] * _ramp_in(t, traj["onset"], 1.5)
        lateral = a_ms2 * np.sin(np.minimum(angle, TRAJ_MAX_ERROR_DEG) * deg * (t >= traj["onset"]))
        x_fail = (_double_integral(lateral * np.cos(q["phi"])), _double_integral(lateral * np.sin(q["phi"])))
        x_axes = [x_axes[0] + x_fail[0], x_axes[1] + x_fail[1]]
        d_fail = np.hypot(*x_fail) / 1000.0
        seen = np.nonzero((t >= traj["onset"]) & (d_fail >= np.maximum(0.15, V.trajectory_band(t) / 3.0)))[0]
        traj["manifest"] = float(t[seen[0]]) if seen.size else float("inf")   # no flight: never seen
    sigma = V.CHANNELS[V.CH["TRAJ_DEV"]].noise
    track = rng_traj.standard_normal((2, n)) * sigma
    traj_dev = np.hypot(x_axes[0] / 1000.0 + track[0], x_axes[1] / 1000.0 + track[1])

    # ---- vibration ---------------------------------------------------------------------------
    vib = 0.03 + 0.9 * e * vib_factor + vib_extra
    if boosters_lit:
        ig = t >= V.EVENTS["BOOSTER_IGNITION"]
        vib = vib + np.where(ig, 2.6 * np.exp(-np.clip(t - 7.2, 0.0, None) / 6.0), 0.0)
        vib = vib + 0.7 * np.exp(-((t - 52.0) / 10.0) ** 2) * flying
        vib = vib + 0.6 * ((srb["L"] + srb["R"]) / 2.0 - 1.0) / 90.0
        vib = vib + np.where(sep, 2.0 * np.exp(-np.clip(t - V.EVENTS["BOOSTER_SEP"], 0.0, None) / 0.8), 0.0)
    if "vib_elevated" in conf:
        vib = vib * conf["vib_elevated"]["factor"]

    # ---- avionics bus -----------------------------------------------------------------------
    tp = V.EVENTS["POWER_TRANSFER"]
    after = t >= tp
    since = np.clip(t - tp, 0.0, None)
    bus = np.where(after, batt_v - 0.05 * since / 60.0 - 0.35 * np.exp(-since / 0.3), nom["bus_ground"])
    bus -= 0.15 * e
    if "BUS_VOLTAGE_SAG" in labels:
        p = labels["BUS_VOLTAGE_SAG"]["params"]
        bus -= after * (p["dip"] * np.exp(-since / p["tau"]) + p["residual"] * (1.0 - np.exp(-since / 1.0)))
    if "bus_dip" in conf:
        bus -= after * conf["bus_dip"]["dip"] * np.exp(-since / 0.8)

    # ---- sensors: noise, sensor faults, lost frames -----------------------------------------
    clean = {
        "PC_CORE": pc_meas, "N_LOX_TP": n_true, "T_TURB_IN": t_true, "P_LOX_TANK": p_lox,
        "P_LH2_TANK": p_lh2, "PC_SRB_L": srb["L"], "PC_SRB_R": srb["R"], "TVC_PITCH": pitch,
        "TVC_YAW": yaw, "ACC_AX": acc, "VIB_ENG": vib, "BUS_28V": bus, "TRAJ_DEV": traj_dev,
    }
    tel = np.empty((len(V.CHANNELS), n))
    for i, ch in enumerate(V.CHANNELS):
        if ch.key == "TRAJ_DEV":          # tracking noise already added, from its own stream
            tel[i] = clean[ch.key]
            continue
        z = rng.standard_normal(n)
        x = clean[ch.key]
        tel[i] = np.clip(x * (1.0 + ch.noise * z), 0.0, None) if ch.key == "VIB_ENG" else x + ch.noise * z

    k_t = V.CH["T_TURB_IN"]
    if "TEMP_SENSOR_FAULT" in labels:
        lab = labels["TEMP_SENSOR_FAULT"]
        p = lab["params"]
        k0 = V.k_of(lab["onset"])
        if p["mode"] == "step":
            tel[k_t, k0:] += p["step"] + 3.0 * rng.standard_normal(n - k0)
        elif p["mode"] == "spikes":
            count = 1 + int(rng.poisson(p["spike_rate"] * (V.T_END - lab["onset"])))
            starts = np.concatenate([[k0], rng.integers(k0, n, count - 1)])
            for ks in starts:
                w = int(rng.integers(1, 4))
                tel[k_t, ks:ks + w] += p["spike_amp"] * rng.uniform(0.5, 1.0)
        else:  # rail: open circuit, stuck at full scale
            tel[k_t, k0:] = p["rail"]
    if "t_glitch" in conf:
        k = V.k_of(conf["t_glitch"]["time"])
        tel[k_t, k] += conf["t_glitch"]["amp"]

    gaps: List[Tuple[float, float]] = []
    if rng.uniform() < 0.35:
        for _ in range(int(rng.integers(1, 4))):
            gaps.append((float(rng.uniform(V.T_START, V.T_END)), float(rng.uniform(0.1, 0.8))))
    if rng.uniform() < 0.5:
        gaps.append((float(rng.uniform(7.0, 12.0)), float(rng.uniform(0.1, 0.6))))
    for g0, d in gaps:
        tel[:, V.k_of(g0):V.k_of(g0 + d) + 1] = np.nan

    meta = {
        "pc_nom": pc_nom, "n_nom": n_nom, "t_nom": t_nom, "plox_set": plox_set,
        "plh2_set": plh2_set, "start_mid": mid, "start_width": width, "wind": wind,
        "srb_scale": srb_scale, "batt_v": batt_v, "liftoff": liftoff, "gaps": gaps,
        "traj_bias_deg": [float(b / deg) for b in bias],
        "confusers": sorted(conf), "abort_at": abort_at,
    }
    return Launch(launch_id or f"L{seed}", seed, tel.astype(np.float32), labels, meta)
