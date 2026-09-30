"""Causal telemetry features for the launch domain.

Every derived series is computed causally (hold-last-value over lost frames,
trailing windows, forward IIR filters). The feature vector at ``t_now`` therefore
equals the one computed from telemetry truncated at ``t_now``, which is what lets
the same features drive both the batch factory run and the streaming agent.

Feature names follow ``PHASE__CHANNEL__stat``:
  CD  countdown       [-60, -1)      IGN engine start [0, 6.5)
  ASC boosted ascent  [7, 136)       LATE late ascent [100, 136)
  FLT the whole flight [7, 150]       (trajectory: it keeps drifting after separation)
  XC  cross-channel consistency      CTX context
A window that has not started, or holds less than one second of data, yields NaN.

``valid_from`` says when a partial-window value is representative. Running
maxima, counts and censored crossing times are valid almost immediately; means
and standard deviations only once their window has closed. The agent shows a
fired rule only when its feature is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from launch_sim import vehicle as V

MIN_WINDOW_S = 1.0
TEMP_PER_PUMP_PCT = 40.0   # K of turbine-temperature rise explained by 1 % of pump speed
LOX_READY_BAR = 3.30
PC_T50_BAR = 57.5          # 50 % of nominal chamber pressure
PC_T90_BAR = 103.5         # 90 % of nominal chamber pressure
TRAJ_RATE_S = 5.0          # trajectory deviation growth is measured over this many seconds
TRAJ_TREND_S = 20.0        # ... and its ratio to the normal spread over this many
TRAJ_NOISE_KM = 0.019      # mean of the tracking-noise magnitude (15 m per axis, Rayleigh)
TRAJ_IMPLIED_FROM = 30.0   # the implied pointing error is trusted from this time on
FLT = (7.0, V.T_END)       # the flight, for the trajectory: it keeps drifting after separation

CD, IGN, ASC, LATE = (-60.0, -1.0), (0.0, 6.5), (7.0, 136.0), (100.0, 136.0)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    unit: str
    channels: Tuple[str, ...]
    phase: str
    valid_from: float


def _ffill(x: np.ndarray) -> np.ndarray:
    """Hold the last valid value over lost frames (leading gaps take the first valid value)."""
    out = x.astype(np.float64).copy()
    for row in out:
        bad = np.isnan(row)
        if not bad.any():
            continue
        idx = np.where(~bad, np.arange(row.size), 0)
        np.maximum.accumulate(idx, out=idx)
        first = int(np.argmax(~bad)) if (~bad).any() else 0
        filled = row[idx]
        filled[:first] = row[first]
        row[:] = filled
    return out


def _delayed(x: np.ndarray, lag: int, fill: float) -> np.ndarray:
    """``x`` delayed by ``lag`` samples, padded with ``fill``; any length (causal)."""
    out = np.full(x.shape, fill, dtype=float)
    if 0 < lag < x.size:
        out[lag:] = x[:-lag]
    elif lag == 0:
        out[:] = x
    return out


def _trailing_mean(x: np.ndarray, seconds: float) -> np.ndarray:
    w = max(1, int(round(seconds * V.FS)))
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(x.size)
    lo = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def _filter(x: np.ndarray, kind: str, freq) -> np.ndarray:
    sos = signal.butter(2, freq, btype=kind, fs=V.FS, output="sos")
    return signal.sosfilt(sos, x - x[0])


class LaunchFeatures:
    """Precomputes causal series for one launch; ``vector(t_now)`` returns the features."""

    def __init__(self, tel: np.ndarray):
        x = _ffill(tel)
        c = V.CH
        self.x = x
        self.n_avail = tel.shape[1]
        pc, n, temp = x[c["PC_CORE"]], x[c["N_LOX_TP"]], x[c["T_TURB_IN"]]
        vib, yaw, pitch = x[c["VIB_ENG"]], x[c["TVC_YAW"]], x[c["TVC_PITCH"]]
        t = V.TIME[: self.n_avail]
        s = {}
        s["pc_tm2"] = _trailing_mean(pc, 2.0)
        s["n_tm2"] = _trailing_mean(n, 2.0)
        s["t_tm1"] = _trailing_mean(temp, 1.0)
        t_tm02 = _trailing_mean(temp, 0.2)
        lag = 10
        s["t_step"] = np.abs(t_tm02 - np.concatenate([np.full(lag, t_tm02[0]), t_tm02[:-lag]]))
        s["t_spike"] = temp - s["t_tm1"]
        s["t_flat"] = np.concatenate([[0.0], (np.diff(temp) == 0.0).astype(float)])
        pc_hp = _filter(pc, "high", 1.5)
        n_hp = _filter(n, "high", 1.5)
        s["pc_hp"] = pc_hp
        s["n_hp"] = n_hp
        s["pc_hp_rms1"] = np.sqrt(_trailing_mean(pc_hp ** 2, 1.0))
        s["pc_hp_rms5"] = np.sqrt(_trailing_mean(pc_hp ** 2, 5.0))
        s["vib_tm1"] = _trailing_mean(vib, 1.0)
        ref_vib = V.reference_vibration(t)
        s["vib_excess"] = s["vib_tm1"] / ref_vib
        s["vib_ratio"] = vib / ref_vib
        lh2_bp = _filter(x[c["P_LH2_TANK"]], "band", (0.15, 0.5))
        s["lh2_bp2"] = lh2_bp ** 2
        s["lh2_rms5"] = np.sqrt(_trailing_mean(s["lh2_bp2"], 5.0))
        p_lc = _filter(pitch, "band", (0.9, 2.4))
        y_lc = _filter(yaw, "band", (0.9, 2.4))
        s["pitch_lc2"] = p_lc ** 2
        s["pitch_lc_rms10"] = np.sqrt(_trailing_mean(s["pitch_lc2"], 10.0))
        s["yaw_lc_rms10"] = np.sqrt(_trailing_mean(y_lc ** 2, 10.0))
        s["pitch_gust2"] = _filter(pitch, "band", (0.02, 0.5)) ** 2
        acc_bp = _filter(x[c["ACC_AX"]], "band", (10.0, 20.0))
        pcp_bp = _filter(pc, "band", (10.0, 20.0))
        s["acc_pogo2"] = acc_bp ** 2
        s["acc_pogo_rms5"] = np.sqrt(_trailing_mean(s["acc_pogo2"], 5.0))
        s["pc_pogo_rms5"] = np.sqrt(_trailing_mean(pcp_bp ** 2, 5.0))
        left, right = x[c["PC_SRB_L"]], x[c["PC_SRB_R"]]
        imb = 100.0 * (left - right) / np.maximum((left + right) / 2.0, 20.0)
        s["imb_abs_tm2"] = _trailing_mean(np.abs(imb), 2.0)
        s["imb_tm5"] = _trailing_mean(imb, 5.0)
        s["imb_tm3"] = _trailing_mean(imb, 3.0)
        s["imb_abs"] = np.abs(imb)
        s["yaw_tm3"] = _trailing_mean(yaw, 3.0)
        s["lox_deficit"] = V.reference_lox_ramp(t) - x[c["P_LOX_TANK"]]
        s["traj_tm1"] = _trailing_mean(x[c["TRAJ_DEV"]], 1.0)
        before = _delayed(s["traj_tm1"], int(round(TRAJ_RATE_S * V.FS)), s["traj_tm1"][0])
        s["traj_rate"] = (s["traj_tm1"] - before) / TRAJ_RATE_S
        s["traj_band"] = s["traj_tm1"] / V.trajectory_band(t)
        # The pointing error the deviation implies, in normal-spread units: net of the noise
        # floor, against the drift part of the band only. A constant pointing error drifts
        # exactly like that part, so a normal flight holds it flat; a failure makes it rise.
        # Trusted from H0 + 30 s, when the drift part is well above the noise.
        drift = V.trajectory_band(t) - V.trajectory_band(V.T_START)
        implied = np.maximum(s["traj_tm1"] - TRAJ_NOISE_KM, 0.0) / np.maximum(drift, 1e-6)
        implied[t < TRAJ_IMPLIED_FROM] = np.nan
        s["traj_band_rise"] = implied - _delayed(implied, int(round(TRAJ_TREND_S * V.FS)), np.nan)
        self.s = s

    # ---- window helpers --------------------------------------------------------------
    def _win(self, a: float, b: float, k_now: int, min_s: float = MIN_WINDOW_S
             ) -> Optional[Tuple[int, int]]:
        k0 = V.k_of(a)
        k1 = min(V.k_of(b), k_now + 1)
        if k1 - k0 < max(1, int(round(min_s * V.FS))):
            return None
        return k0, k1

    def _stat(self, arr: np.ndarray, a: float, b: float, k_now: int, fn: Callable,
              min_s: float = MIN_WINDOW_S) -> float:
        w = self._win(a, b, k_now, min_s)
        return float("nan") if w is None else float(fn(arr[w[0]:w[1]]))

    def _crossing(self, arr: np.ndarray, level: float, a: float, b: float, k_now: int,
                  t_origin: float) -> float:
        """Time from ``t_origin`` until ``arr`` first reaches ``level``; censored at now."""
        k0 = V.k_of(a)
        k1 = min(V.k_of(b), k_now + 1)
        if k1 <= k0:
            return float("nan")
        hit = np.nonzero(arr[k0:k1] >= level)[0]
        t_end = V.TIME[k1 - 1]
        return float((V.TIME[k0 + hit[0]] if hit.size else t_end) - t_origin)

    def _ref(self, arr: np.ndarray, k_now: int) -> float:
        return self._stat(arr, 4.0, 6.5, k_now, np.mean, min_s=2.5)

    # ---- the feature vector -------------------------------------------------------------
    def vector(self, t_now: float) -> np.ndarray:
        k = min(V.k_of(t_now), self.n_avail - 1)
        return np.array([fn(self, k) for fn in _FUNCS], dtype=np.float64)


_SPECS: List[FeatureSpec] = []
_FUNCS: List[Callable[[LaunchFeatures, int], float]] = []


def _add(name: str, description: str, unit: str, channels: Tuple[str, ...], phase: str,
         valid_from: float, fn: Callable[[LaunchFeatures, int], float]) -> None:
    _SPECS.append(FeatureSpec(name, description, unit, channels, phase, valid_from))
    _FUNCS.append(fn)


def _generic(phase: str, window: Tuple[float, float], keys: Tuple[str, ...]) -> None:
    stats = (("mean", np.mean, "mean"), ("std", np.std, "standard deviation"),
             ("min", np.min, "minimum"), ("max", np.max, "maximum"))
    names = {"CD": "countdown", "IGN": "engine start", "ASC": "boosted ascent"}
    for key in keys:
        ch = V.CHANNELS[V.CH[key]]
        for stat, fn, word in stats:
            _add(f"{phase}__{key}__{stat}", f"{ch.label}, {word} over the {names[phase]}",
                 ch.unit, (key,), phase, window[1],
                 lambda f, k, i=V.CH[key], a=window[0], b=window[1], fn=fn:
                 f._stat(f.x[i], a, b, k, fn))


def _build() -> None:
    c = V.CH
    _generic("CD", CD, ("P_LOX_TANK", "P_LH2_TANK", "BUS_28V"))
    _generic("IGN", IGN, ("PC_CORE", "N_LOX_TP", "T_TURB_IN", "VIB_ENG"))
    _generic("ASC", ASC, tuple(ch.key for ch in V.CHANNELS))

    def last10(i):
        def fn(f, k):
            t_end = min(CD[1], V.TIME[k])
            if t_end < -30.0:
                return float("nan")
            return f._stat(f.x[i], t_end - 10.0, t_end, k, np.mean)
        return fn

    # countdown
    _add("CD__P_LOX_TANK__last10", "LOX tank pressure, last 10 s of countdown", "bar",
         ("P_LOX_TANK",), "CD", -30.0, last10(c["P_LOX_TANK"]))
    _add("CD__P_LOX_TANK__ramp_deficit", "LOX tank pressure shortfall vs the reference ramp",
         "bar", ("P_LOX_TANK",), "CD", -54.0,
         lambda f, k: f._stat(f.s["lox_deficit"], -57.0, CD[1], k, np.mean))
    _add("CD__P_LOX_TANK__t_ready", f"Time for LOX tank pressure to reach {LOX_READY_BAR} bar",
         "s", ("P_LOX_TANK",), "CD", -57.0,
         lambda f, k: f._crossing(f.x[c["P_LOX_TANK"]], LOX_READY_BAR, -58.0, CD[1], k, -58.0))
    _add("CD__P_LH2_TANK__osc_rms", "LH2 tank pressure oscillation (0.15-0.5 Hz RMS)", "bar",
         ("P_LH2_TANK",), "CD", -25.0,
         lambda f, k: np.sqrt(f._stat(f.s["lh2_bp2"], -38.0, CD[1], k, np.mean, 5.0)))
    _add("CD__P_LH2_TANK__osc_rms_max5", "LH2 pressure oscillation, strongest 5 s", "bar",
         ("P_LH2_TANK",), "CD", -32.0,
         lambda f, k: f._stat(f.s["lh2_rms5"], -33.0, CD[1], k, np.max))
    _add("CD__P_LH2_TANK__last10", "LH2 tank pressure, last 10 s of countdown", "bar",
         ("P_LH2_TANK",), "CD", -30.0, last10(c["P_LH2_TANK"]))

    def transfer_dip(f, k):
        before = f._stat(f.x[c["BUS_28V"]], -33.0, -31.0, k, np.mean)
        after = f._stat(f.x[c["BUS_28V"]], -30.0, -20.0, k, np.min, 0.5)
        return before - after

    _add("CD__BUS_28V__transfer_dip", "Bus voltage dip at the switch to onboard power", "V",
         ("BUS_28V",), "CD", -29.5, transfer_dip)
    _add("CD__BUS_28V__post_transfer", "Bus voltage after the power transfer (mean from -25 s)",
         "V", ("BUS_28V",), "CD", -24.0,
         lambda f, k: f._stat(f.x[c["BUS_28V"]], -25.0, CD[1], k, np.mean))
    _add("CD__BUS_28V__min_post", "Lowest bus voltage from 2 s after the transfer", "V",
         ("BUS_28V",), "CD", -27.0,
         lambda f, k: f._stat(f.x[c["BUS_28V"]], -28.0, CD[1], k, np.min))

    # engine start
    _add("IGN__PC_CORE__t50", "Time for chamber pressure to reach 50 % of nominal", "s",
         ("PC_CORE",), "IGN", 0.5,
         lambda f, k: f._crossing(f.x[c["PC_CORE"]], PC_T50_BAR, 0.0, IGN[1], k, 0.0))
    _add("IGN__PC_CORE__t90", "Time for chamber pressure to reach 90 % of nominal", "s",
         ("PC_CORE",), "IGN", 0.5,
         lambda f, k: f._crossing(f.x[c["PC_CORE"]], PC_T90_BAR, 0.0, IGN[1], k, 0.0))
    _add("IGN__PC_CORE__at2p5", "Chamber pressure at H0 + 2.5 s", "bar", ("PC_CORE",), "IGN",
         2.6, lambda f, k: f._stat(f.x[c["PC_CORE"]], 2.4, 2.6, k, np.mean, 0.2))
    _add("IGN__N_LOX_TP__at2p5", "LOX pump speed at H0 + 2.5 s", "%", ("N_LOX_TP",), "IGN",
         2.6, lambda f, k: f._stat(f.x[c["N_LOX_TP"]], 2.4, 2.6, k, np.mean, 0.2))
    _add("IGN__PC_CORE__rough_max1", "Chamber pressure roughness (>1.5 Hz RMS), worst 1 s",
         "bar", ("PC_CORE",), "IGN", 3.5,
         lambda f, k: f._stat(f.s["pc_hp_rms1"], 3.0, IGN[1], k, np.max, 0.5))
    _add("IGN__PC_CORE__dip_count", "Chamber pressure dips deeper than 2 bar (samples)",
         "count", ("PC_CORE",), "IGN", 3.5,
         lambda f, k: f._stat(f.s["pc_hp"], 3.0, IGN[1], k,
                              lambda a: np.count_nonzero(a < -2.0), 0.5))
    _add("IGN__N_LOX_TP__spike_max", "Largest LOX pump speed spike (>1.5 Hz)", "%",
         ("N_LOX_TP",), "IGN", 3.0,
         lambda f, k: f._stat(f.s["n_hp"], 2.5, IGN[1], k, np.max, 0.5))
    _add("IGN__N_LOX_TP__spike_count", "LOX pump speed spikes above 1 % (samples)", "count",
         ("N_LOX_TP",), "IGN", 3.0,
         lambda f, k: f._stat(f.s["n_hp"], 2.5, IGN[1], k,
                              lambda a: np.count_nonzero(a > 1.0), 0.5))
    _add("IGN__T_TURB_IN__peak", "Turbine inlet temperature start peak", "K", ("T_TURB_IN",),
         "IGN", 3.5, lambda f, k: f._stat(f.x[c["T_TURB_IN"]], 0.0, 3.5, k, np.max))
    _add("IGN__T_TURB_IN__plateau", "Turbine inlet temperature, H0 + 4 to 6.5 s", "K",
         ("T_TURB_IN",), "IGN", 6.5,
         lambda f, k: f._stat(f.x[c["T_TURB_IN"]], 4.0, IGN[1], k, np.mean, 0.5))
    _add("IGN__VIB_ENG__max1", "Engine-bay vibration, worst 1 s of the start", "g rms",
         ("VIB_ENG",), "IGN", 1.5,
         lambda f, k: f._stat(f.s["vib_tm1"], 1.0, IGN[1], k, np.max, 0.5))
    _add("IGN__VIB_ENG__mean_late", "Engine-bay vibration, H0 + 3 to 6.5 s", "g rms",
         ("VIB_ENG",), "IGN", 6.5,
         lambda f, k: f._stat(f.x[c["VIB_ENG"]], 3.0, IGN[1], k, np.mean, 0.5))
    _add("IGN__P_LOX_TANK__mean", "LOX tank pressure during the engine start", "bar",
         ("P_LOX_TANK",), "IGN", 6.5,
         lambda f, k: f._stat(f.x[c["P_LOX_TANK"]], 0.0, IGN[1], k, np.mean))

    # boosted ascent: rises are relative to the engine plateau on the pad (H0 + 4 to 6.5 s)
    def rise_max(series: str, raw: int, a: float):
        def fn(f, k):
            ref = f._ref(f.x[raw], k)
            return f._stat(f.s[series] - ref, a, ASC[1], k, np.max, 0.5)
        return fn

    def unexplained(f, k):
        t_ref, n_ref = f._ref(f.x[c["T_TURB_IN"]], k), f._ref(f.x[c["N_LOX_TP"]], k)
        excess = (f.s["t_tm1"] - t_ref) - TEMP_PER_PUMP_PCT * (f.s["n_tm2"] - n_ref)
        return f._stat(excess, 9.0, ASC[1], k, np.max, 0.5)

    def corroborated(f, k):
        t_ref, n_ref = f._ref(f.x[c["T_TURB_IN"]], k), f._ref(f.x[c["N_LOX_TP"]], k)
        both = np.minimum(f.s["t_tm1"] - t_ref, TEMP_PER_PUMP_PCT * (f.s["n_tm2"] - n_ref))
        return f._stat(both, 9.0, ASC[1], k, np.max, 0.5)

    def yaw_imbalance_corr(f, k):
        w = f._win(12.0, 122.0, k, 10.0)
        if w is None:
            return float("nan")
        a, b = f.s["yaw_tm3"][w[0]:w[1]], f.s["imb_tm3"][w[0]:w[1]]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    _add("ASC__PC_CORE__rough", "Chamber pressure roughness (>1.5 Hz RMS) in ascent", "bar",
         ("PC_CORE",), "ASC", ASC[1],
         lambda f, k: np.sqrt(f._stat(f.s["pc_hp"] ** 2, 8.0, ASC[1], k, np.mean)))
    _add("ASC__PC_CORE__rough_max5", "Chamber pressure roughness, worst 5 s of ascent", "bar",
         ("PC_CORE",), "ASC", 9.5,
         lambda f, k: f._stat(f.s["pc_hp_rms5"], 9.0, ASC[1], k, np.max, 0.5))
    _add("ASC__PC_CORE__rise_max", "Chamber pressure rise above the pad plateau (max)", "bar",
         ("PC_CORE",), "ASC", 9.5, rise_max("pc_tm2", c["PC_CORE"], 9.0))
    _add("ASC__N_LOX_TP__rise_max", "LOX pump speed rise above the pad plateau (max)", "%",
         ("N_LOX_TP",), "ASC", 9.5, rise_max("n_tm2", c["N_LOX_TP"], 9.0))
    _add("ASC__T_TURB_IN__rise_max", "Turbine temperature rise above the pad plateau (max)", "K",
         ("T_TURB_IN",), "ASC", 8.5, rise_max("t_tm1", c["T_TURB_IN"], 8.0))
    _add("ASC__T_TURB_IN__step_max", "Largest turbine temperature jump within 0.2 s", "K",
         ("T_TURB_IN",), "ASC", 8.5,
         lambda f, k: f._stat(f.s["t_step"], 8.0, ASC[1], k, np.max, 0.5))
    _add("ASC__T_TURB_IN__spike_count", "Turbine temperature spikes above 40 K (samples)",
         "count", ("T_TURB_IN",), "ASC", 8.5,
         lambda f, k: f._stat(f.s["t_spike"], 8.0, ASC[1], k,
                              lambda a: np.count_nonzero(a > 40.0), 0.5))
    _add("ASC__T_TURB_IN__flat_frac", "Share of identical consecutive temperature samples",
         "fraction", ("T_TURB_IN",), "ASC", ASC[1],
         lambda f, k: f._stat(f.s["t_flat"], 8.0, ASC[1], k, np.mean))
    _add("XC__T_unexplained_max",
         f"Temperature rise not explained by pump speed ({TEMP_PER_PUMP_PCT:.0f} K per %), max",
         "K", ("T_TURB_IN", "N_LOX_TP"), "XC", 9.5, unexplained)
    _add("XC__T_corroborated_max", "Temperature rise confirmed by a pump speed rise, max", "K",
         ("T_TURB_IN", "N_LOX_TP"), "XC", 9.5, corroborated)
    _add("ASC__SRB__imbalance_mean", "Booster chamber pressure imbalance, mean", "%",
         ("PC_SRB_L", "PC_SRB_R"), "ASC", 122.0,
         lambda f, k: f._stat(f.s["imb_abs"], 8.0, 122.0, k, np.mean))
    _add("ASC__SRB__imbalance_max", "Booster chamber pressure imbalance, worst 2 s", "%",
         ("PC_SRB_L", "PC_SRB_R"), "ASC", 9.0,
         lambda f, k: f._stat(f.s["imb_abs_tm2"], 8.0, 122.0, k, np.max, 0.5))
    _add("ASC__SRB__imbalance_now", "Booster imbalance (L minus R), last 5 s", "%",
         ("PC_SRB_L", "PC_SRB_R"), "ASC", 12.0,
         lambda f, k: f._stat(f.s["imb_tm5"], 8.0, 122.0, k, lambda a: a[-1], 0.5))
    _add("ASC__TVC_YAW__bias_absmax", "TVC yaw offset (3 s mean), largest magnitude", "deg",
         ("TVC_YAW",), "ASC", 10.0,
         lambda f, k: f._stat(np.abs(f.s["yaw_tm3"]), 9.0, 134.0, k, np.max, 0.5))
    _add("ASC__TVC_YAW__bias_mean", "TVC yaw offset, mean over the ascent", "deg",
         ("TVC_YAW",), "ASC", 134.0,
         lambda f, k: f._stat(f.x[c["TVC_YAW"]], 9.0, 134.0, k, np.mean))
    _add("XC__yaw_imbalance_corr", "Correlation of TVC yaw offset with booster imbalance", "r",
         ("TVC_YAW", "PC_SRB_L", "PC_SRB_R"), "XC", 122.0, yaw_imbalance_corr)
    _add("ASC__TVC_PITCH__lc_rms", "TVC pitch oscillation 0.9-2.4 Hz (RMS)", "deg",
         ("TVC_PITCH",), "ASC", ASC[1],
         lambda f, k: np.sqrt(f._stat(f.s["pitch_lc2"], 10.0, ASC[1], k, np.mean)))
    _add("ASC__TVC_PITCH__lc_rms_max10", "TVC pitch oscillation 0.9-2.4 Hz, worst 10 s", "deg",
         ("TVC_PITCH",), "ASC", 12.0,
         lambda f, k: f._stat(f.s["pitch_lc_rms10"], 12.0, ASC[1], k, np.max, 0.5))
    _add("ASC__TVC_YAW__lc_rms_max10", "TVC yaw oscillation 0.9-2.4 Hz, worst 10 s", "deg",
         ("TVC_YAW",), "ASC", 12.0,
         lambda f, k: f._stat(f.s["yaw_lc_rms10"], 12.0, ASC[1], k, np.max, 0.5))
    _add("ASC__TVC_PITCH__gust_rms", "TVC pitch activity from wind, 0.02-0.5 Hz (RMS)", "deg",
         ("TVC_PITCH",), "ASC", ASC[1],
         lambda f, k: np.sqrt(f._stat(f.s["pitch_gust2"], 10.0, ASC[1], k, np.mean)))
    _add("ASC__ACC_AX__pogo_rms_max5", "Axial acceleration oscillation 10-20 Hz, worst 5 s",
         "g", ("ACC_AX",), "ASC", 10.5,
         lambda f, k: f._stat(f.s["acc_pogo_rms5"], 10.0, ASC[1], k, np.max, 0.5))
    _add("ASC__PC_CORE__pogo_rms_max5", "Chamber pressure oscillation 10-20 Hz, worst 5 s",
         "bar", ("PC_CORE",), "ASC", 10.5,
         lambda f, k: f._stat(f.s["pc_pogo_rms5"], 10.0, ASC[1], k, np.max, 0.5))
    _add("LATE__ACC_AX__pogo_rms", "Axial acceleration oscillation 10-20 Hz, late ascent",
         "g", ("ACC_AX",), "LATE", LATE[1],
         lambda f, k: np.sqrt(f._stat(f.s["acc_pogo2"], LATE[0], LATE[1], k, np.mean)))
    _add("ASC__VIB_ENG__excess_max", "Engine-bay vibration vs the nominal profile, worst 1 s",
         "ratio", ("VIB_ENG",), "ASC", 8.5,
         lambda f, k: f._stat(f.s["vib_excess"], 8.0, ASC[1], k, np.max, 0.5))
    _add("ASC__VIB_ENG__excess_mean", "Engine-bay vibration vs the nominal profile, mean",
         "ratio", ("VIB_ENG",), "ASC", ASC[1],
         lambda f, k: f._stat(f.s["vib_ratio"], 8.0, ASC[1], k, np.mean))
    _add("ASC__N_LOX_TP__spike_count", "LOX pump speed spikes above 1 % in ascent (samples)",
         "count", ("N_LOX_TP",), "ASC", 8.5,
         lambda f, k: f._stat(f.s["n_hp"], 8.0, ASC[1], k,
                              lambda a: np.count_nonzero(a > 1.0), 0.5))
    _add("ASC__P_LH2_TANK__osc_rms", "LH2 tank pressure oscillation in ascent (RMS)", "bar",
         ("P_LH2_TANK",), "ASC", ASC[1],
         lambda f, k: np.sqrt(f._stat(f.s["lh2_bp2"], 8.0, ASC[1], k, np.mean)))
    # trajectory: how far off the plan, how fast it grows, against the normal spread
    def after(key, delay):
        return lambda f, k: f.s[key][k] if V.TIME[k] >= FLT[0] + delay else float("nan")

    _add("FLT__TRAJ_DEV__now", "Trajectory deviation, last 1 s", "km",
         ("TRAJ_DEV",), "FLT", FLT[0] + 1.0, after("traj_tm1", 1.0))
    _add("FLT__TRAJ_DEV__max", "Trajectory deviation, largest so far in flight", "km",
         ("TRAJ_DEV",), "FLT", FLT[0] + 1.0,
         lambda f, k: f._stat(f.s["traj_tm1"], FLT[0], FLT[1], k, np.max, 0.5))
    _add("FLT__TRAJ_DEV__rate_now", f"Trajectory deviation growth over the last {TRAJ_RATE_S:.0f} s",
         "km/s", ("TRAJ_DEV",), "FLT", FLT[0] + TRAJ_RATE_S, after("traj_rate", TRAJ_RATE_S))
    _add("FLT__TRAJ_DEV__rate_max", f"Trajectory deviation growth, fastest {TRAJ_RATE_S:.0f} s so far",
         "km/s", ("TRAJ_DEV",), "FLT", FLT[0] + TRAJ_RATE_S,
         lambda f, k: f._stat(f.s["traj_rate"], FLT[0] + TRAJ_RATE_S, FLT[1], k, np.max, 0.5))
    _add("FLT__TRAJ_DEV__band_max", "Trajectory deviation against the normal 3-sigma spread, "
         "largest so far", "ratio", ("TRAJ_DEV",), "FLT", FLT[0] + 1.0,
         lambda f, k: f._stat(f.s["traj_band"], FLT[0] + 1.0, FLT[1], k, np.max, 0.5))
    _add("FLT__TRAJ_DEV__band_rise_now", "Rise of the trajectory deviation against the normal "
         f"spread over the last {TRAJ_TREND_S:.0f} s", "ratio", ("TRAJ_DEV",), "FLT",
         TRAJ_IMPLIED_FROM + TRAJ_TREND_S, lambda f, k: f.s["traj_band_rise"][k])
    _add("FLT__TRAJ_DEV__band_rise_max", "Rise of the trajectory deviation against the normal "
         f"spread over {TRAJ_TREND_S:.0f} s, largest so far", "ratio", ("TRAJ_DEV",), "FLT",
         TRAJ_IMPLIED_FROM + TRAJ_TREND_S + 0.5,
         lambda f, k: f._stat(f.s["traj_band_rise"], TRAJ_IMPLIED_FROM + TRAJ_TREND_S, FLT[1], k,
                              np.max, 0.5))
    _add("CTX__t_now", "Time of evaluation", "s", (), "CTX", V.T_START,
         lambda f, k: float(V.TIME[k]))


_build()
FEATURE_SPECS: Tuple[FeatureSpec, ...] = tuple(_SPECS)
FEATURE_NAMES: Tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)
FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}


def feature_matrix(tel: np.ndarray, times) -> np.ndarray:
    """Feature vectors of one launch at several evaluation times, shape (len(times), n_features)."""
    lf = LaunchFeatures(tel)
    return np.vstack([lf.vector(t) for t in times])
