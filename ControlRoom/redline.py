"""Fixed-limit (redline) monitor: the baseline the agent is compared with.

These limits are our own assumption of a simple limit checker. They are not real
launch-commit criteria. A limit trips after 5 consecutive samples (0.1 s) so that
single-sample glitches do not fire it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

DOMAIN_DIR = Path(__file__).resolve().parents[1]
if str(DOMAIN_DIR) not in sys.path:
    sys.path.insert(0, str(DOMAIN_DIR))

from launch_sim import vehicle as V  # noqa: E402
from launch_sim.features import _ffill  # noqa: E402

CONSECUTIVE = 5


@dataclass(frozen=True)
class Redline:
    text: str
    call: str                       # HOLD | ABORT | ALARM
    t_from: float
    t_to: float
    breach: Callable[[np.ndarray], np.ndarray]   # ffilled telemetry -> bool per sample


def _ch(key: str) -> int:
    return V.CH[key]


REDLINES: List[Redline] = [
    Redline("LOX tank pressure below 3.30 bar", "HOLD", -40.0, 0.0,
            lambda x: x[_ch("P_LOX_TANK")] < 3.30),
    Redline("LH2 tank pressure outside 2.15-2.45 bar", "HOLD", -35.0, 0.0,
            lambda x: (x[_ch("P_LH2_TANK")] < 2.15) | (x[_ch("P_LH2_TANK")] > 2.45)),
    Redline("Bus voltage below 27.0 V", "HOLD", -30.0, 0.0,
            lambda x: x[_ch("BUS_28V")] < 27.0),
    Redline("Chamber pressure below 100 bar after H0 + 3.5 s", "ABORT", 3.5, 7.0,
            lambda x: x[_ch("PC_CORE")] < 100.0),
    Redline("Turbine inlet temperature above 950 K", "ABORT", 0.0, 7.0,
            lambda x: x[_ch("T_TURB_IN")] > 950.0),
    Redline("LOX pump speed above 103.5 %", "ABORT", 0.0, 7.0,
            lambda x: x[_ch("N_LOX_TP")] > 103.5),
    Redline("Engine-bay vibration above 2 g rms", "ABORT", 0.0, 7.0,
            lambda x: x[_ch("VIB_ENG")] > 2.0),
    Redline("Turbine inlet temperature above 950 K", "ALARM", 7.0, 150.0,
            lambda x: x[_ch("T_TURB_IN")] > 950.0),
    Redline("Chamber pressure below 105 bar", "ALARM", 7.5, 134.0,
            lambda x: x[_ch("PC_CORE")] < 105.0),
    Redline("TVC deflection above 4 deg", "ALARM", 7.0, 150.0,
            lambda x: (np.abs(x[_ch("TVC_PITCH")]) > 4.0) | (np.abs(x[_ch("TVC_YAW")]) > 4.0)),
    Redline("Booster pressure difference above 8 bar", "ALARM", 8.0, 122.0,
            lambda x: np.abs(x[_ch("PC_SRB_L")] - x[_ch("PC_SRB_R")]) > 8.0),
    Redline("Engine-bay vibration above 6 g rms", "ALARM", 7.0, 150.0,
            lambda x: x[_ch("VIB_ENG")] > 6.0),
    # Our illustrative corridor: 1 km + 2.5 x the normal spread (1.2 km at H0 + 30 s, 6.6 km at
    # booster separation). Real corridors are set per mission by flight safety and not public.
    Redline("Trajectory deviation beyond the flight-safety corridor", "ALARM", 7.0, 150.0,
            lambda x: x[_ch("TRAJ_DEV")] > V.trajectory_corridor(V.TIME[: x.shape[1]])),
]


def redline_events(tel: np.ndarray) -> List[dict]:
    """First trip time of every redline (at most one event per redline)."""
    x = _ffill(tel)
    t = V.TIME[: x.shape[1]]
    events = []
    for i, rl in enumerate(REDLINES):
        window = (t >= rl.t_from) & (t < rl.t_to)
        hit = rl.breach(x) & window
        run = np.convolve(hit.astype(int), np.ones(CONSECUTIVE, dtype=int), "full")[: hit.size]
        trips = np.nonzero(run >= CONSECUTIVE)[0]
        if trips.size:
            events.append({"t": float(t[trips[0]]), "id": i, "text": rl.text, "call": rl.call})
    events.sort(key=lambda e: e["t"])
    return events
