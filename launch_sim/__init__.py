"""Launch-vehicle domain package: notional launcher simulator and causal telemetry features.

It turns raw telemetry into the
feature table the Ammonix platform code consumes (discovery/, cross_validation/).
"""

from launch_sim.anomalies import BY_KEY, FAILURE_KEYS, FAILURE_MODES, NOMINAL_KEY
from launch_sim.simulate import Launch, launch_seed, simulate_launch
from launch_sim.vehicle import CH, CHANNELS, EVENTS, FS, N_SAMPLES, TIME

__all__ = [
    "BY_KEY", "CH", "CHANNELS", "EVENTS", "FAILURE_KEYS", "FAILURE_MODES", "FS", "Launch",
    "N_SAMPLES", "NOMINAL_KEY", "TIME", "launch_seed", "simulate_launch",
]
