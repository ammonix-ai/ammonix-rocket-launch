"""Streaming alert policy shared by the factory's threshold calibration and the agent.

The agent evaluates the features every ``TICK_S`` seconds using only telemetry received
so far. An alert is confirmed when a class probability stays at or above its threshold
for ``DEBOUNCE_TICKS`` consecutive ticks. Thresholds are calibrated on exactly this
policy, so the false-alarm budget chosen in the factory is the one the agent runs at.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from launch_sim import vehicle as V

TICK_S = 0.5
DEBOUNCE_TICKS = 2


def stream_ticks(t_end: float = V.T_END) -> np.ndarray:
    return np.round(np.arange(V.T_START, t_end + 1e-9, TICK_S), 3)


TICKS = stream_ticks()


def debounced(p: np.ndarray, axis: int = -2) -> np.ndarray:
    """Probability that has held for DEBOUNCE_TICKS ticks: rolling minimum along time."""
    p = np.moveaxis(np.asarray(p, dtype=np.float64), axis, 0)
    out = p.copy()
    for lag in range(1, DEBOUNCE_TICKS):
        shifted = np.concatenate([np.zeros_like(p[:lag]), p[:-lag]], axis=0)
        out = np.minimum(out, shifted)
    return np.moveaxis(out, 0, axis)


def first_alert_index(held: np.ndarray, threshold: float) -> Optional[int]:
    """First tick at which a debounced probability series reaches ``threshold``."""
    hit = np.nonzero(held >= threshold)[0]
    return int(hit[0]) if hit.size else None
