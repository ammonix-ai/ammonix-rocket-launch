"""Showcase launches for the dashboard: one story per demo moment.

Each story forces specific failures (no random ones) on a fixed seed. The agent then
runs on it exactly as on any other launch; nothing about the story is scripted.
"""

from __future__ import annotations

from typing import Any, Dict, List

SHOWCASE: List[Dict[str, Any]] = [
    {
        "id": "nominal",
        "title": "Nominal launch",
        "summary": "Everything within normal variation. The agent should stay quiet from "
                   "countdown to booster separation.",
        "lay": "A normal launch. Every reading stays in its usual range, and the agent "
               "should stay quiet from the countdown until the boosters drop away.",
        "seed": 9_100_001,
        "forced": {},
    },
    {
        "id": "lox-hold",
        "title": "LOX tank under-pressure",
        "summary": "LOX tank pressurization is slow and settles low. The agent calls a HOLD "
                   "in the countdown, long before a fixed limit at 3.30 bar would.",
        "lay": "The oxygen tank never quite reaches its pressure. The agent notices the slow "
               "build-up long before a simple limit alarm would, and stops the countdown so "
               "the team can fix it on the ground.",
        "seed": 9_100_002,
        "forced": {"LOX_TANK_UNDERPRESSURE": {"severity": 0.3}},
    },
    {
        "id": "cavitation-abort",
        "title": "Pump cavitation at engine start",
        "summary": "LOX pump cavitation right after ignition. The agent calls ABORT before "
                   "the boosters light, the last point where stopping is possible.",
        "lay": "Seconds after the main engine lights, its oxygen pump starts to stutter. The "
               "agent shuts the engine down before the boosters are lit, the last moment "
               "when stopping is still possible.",
        "seed": 9_100_003,
        "forced": {"LOX_PUMP_CAVITATION": {"severity": 0.45, "onset": 1.6}},
    },
    {
        "id": "slow-start-abort",
        "title": "Slow engine start",
        "summary": "Chamber pressure builds up too slowly. The agent sees it while the "
                   "start is still in progress and calls ABORT.",
        "lay": "The main engine wakes up too slowly. The agent sees the sluggish start while "
               "it is still happening and shuts the engine down before the boosters ignite.",
        "seed": 9_100_004,
        "forced": {"SLOW_ENGINE_START": {"severity": 0.35}},
    },
    {
        "id": "sensor-fault",
        "title": "Temperature sensor fault",
        "summary": "The turbine temperature reading jumps, but pump speed and chamber "
                   "pressure do not move. The fixed 950 K limit alarms; the agent flags a "
                   "failed sensor instead.",
        "lay": "A temperature gauge suddenly reads very high, but pump speed and engine "
               "pressure stay calm. A simple limit alarm cries fire; the agent recognizes a "
               "broken sensor.",
        "seed": 9_100_005,
        "forced": {"TEMP_SENSOR_FAULT": {"severity": 0.4, "onset": 62.0, "mode": "step"}},
    },
    {
        "id": "overtemp",
        "title": "Real turbine over-temperature",
        "summary": "The turbine really runs hot: temperature, pump speed and chamber pressure "
                   "rise together. The agent calls it critical; the fixed limit never trips.",
        "lay": "This time the engine really runs hot: temperature, pump speed and pressure "
               "all rise together. The simple limit never trips; the agent calls it "
               "critical.",
        "seed": 9_100_006,
        "forced": {"TURBINE_OVERTEMP": {"severity": 0.3, "onset": 70.0, "ramp": 8.0}},
    },
    {
        "id": "booster-imbalance",
        "title": "Booster thrust imbalance",
        "summary": "The left booster loses pressure; the core engine steers against the "
                   "imbalance with a yaw offset.",
        "lay": "One of the two side boosters starts pushing a little less than the other. "
               "The main engine tilts its nozzle to keep the rocket straight, and the agent "
               "names the cause.",
        "seed": 9_100_007,
        "forced": {"SRB_THRUST_ASYMMETRY": {"severity": 0.35, "onset": 40.0, "side": "L",
                                            "sign": -1.0, "ramp": 6.0}},
    },
    {
        "id": "pogo",
        "title": "POGO onset",
        "summary": "A growing 15 Hz oscillation in acceleration and chamber pressure late in "
                   "the boosted phase.",
        "lay": "Late in the climb the rocket starts to shake lengthwise, fifteen times a "
               "second, like a pogo stick. The agent spots the growing vibration.",
        "seed": 9_100_008,
        "forced": {"POGO_ONSET": {"severity": 0.45, "onset": 104.0, "freq": 15.0}},
    },
    {
        "id": "instability-flight",
        "title": "Combustion instability in flight",
        "summary": "Rough combustion starts after liftoff: chamber pressure ripple and a jump "
                   "in engine-bay vibration.",
        "lay": "During the climb, the flame inside the main engine starts to burn unevenly "
               "and the engine bay shakes. The agent flags it as critical.",
        "seed": 9_100_009,
        "forced": {"COMBUSTION_INSTABILITY": {"severity": 0.3, "onset": 80.0}},
    },
    {
        "id": "trajectory-drift",
        "title": "Trajectory drift",
        "summary": "From H0 + 38 s the guidance follows a slowly drifting direction reference, "
                   "and the vehicle leaves its planned trajectory. Engines and steering look "
                   "normal; only the tracked trajectory shows it. The agent calls CRITICAL at "
                   "H0 + 59 s, as the drift leaves the normal scatter; the fixed flight-safety "
                   "corridor trips 37 s later.",
        "lay": "The rocket's sense of direction slowly goes wrong, and it faithfully flies "
               "the wrong way. The engines work normally; only the tracking of its position "
               "shows that it is leaving its planned path.",
        "seed": 9_100_010,
        "forced": {"TRAJECTORY_DEVIATION": {"severity": 0.45, "onset": 38.0, "mode": "drift",
                                            "phi": 1.1}},
    },
]

# Plain-language explanations for the hover cards and the ground-truth reveal.
TEST_LAY = ("A launch the agent has never seen, taken from the 500 held back for testing. It "
            "may be perfectly normal, or hide one of the twelve failure types, mild or strong. "
            "Play it, watch what the agent says, then reveal the truth.")

MODE_LAY: Dict[str, str] = {
    "LOX_TANK_UNDERPRESSURE": "The liquid-oxygen tank is not pressurized enough. Low tank "
                              "pressure can starve the engine's pump at start, like a straw "
                              "sucking air.",
    "LH2_PRESS_OSCILLATION": "The valve that holds the hydrogen tank at its pressure keeps "
                             "overshooting, so the pressure swings slowly up and down.",
    "BUS_VOLTAGE_SAG": "When the rocket switches from ground power to its own batteries, the "
                       "voltage drops too far and recovers too slowly: a weak battery or a "
                       "bad connection.",
    "SLOW_ENGINE_START": "The main engine takes too long to reach full power after ignition.",
    "LOX_PUMP_CAVITATION": "Vapor bubbles form in the oxygen pump, so it briefly loses its "
                           "grip on the liquid: pump speed jumps, engine pressure dips and "
                           "the engine shakes.",
    "COMBUSTION_INSTABILITY": "The flame in the combustion chamber burns unevenly and starts "
                              "to pulse, which shakes the engine.",
    "TURBINE_OVERTEMP": "The turbine that drives the pumps really runs too hot, and with the "
                        "extra power the pump speed and engine pressure rise too.",
    "TEMP_SENSOR_FAULT": "A temperature sensor breaks and shows nonsense while everything "
                         "else says the engine is fine: a broken gauge, not a fire.",
    "SRB_THRUST_ASYMMETRY": "One side booster pushes less than the other. The main engine "
                            "tilts its nozzle to keep the rocket straight, like steering "
                            "against a crosswind.",
    "TVC_ACTUATOR_DEGRADATION": "The actuator that steers the main engine's nozzle starts to "
                                "wobble back and forth on its own.",
    "POGO_ONSET": "The rocket starts to vibrate lengthwise like a pogo stick: thrust and fuel "
                  "flow feed each other into a growing oscillation.",
    "TRAJECTORY_DEVIATION": "The rocket's sense of direction is wrong, so it flies away from "
                            "its planned path while every engine reading looks fine. Only the "
                            "tracking of its position shows it.",
}
