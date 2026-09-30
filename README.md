# Rocket Launch: an Ammonix control-room agent for a launch vehicle

An agent that watches the telemetry of a rocket launch from the countdown to booster
separation, calls HOLD, ABORT or a flight warning when something goes wrong, and explains
every call with a clear-text rule and a measured value. It was built from 3,000 simulated
launches with the Ammonix platform code, the same rule discovery, splitting and evaluation
code our other agents are built with.

**All telemetry is synthetic.** It comes from a physics-lite model of a notional heavy
launcher with a cryogenic core stage and two solid boosters. It is not data from any real
vehicle, and every value is illustrative.

## Results

500 held-out launches the agent never saw in training, replayed at 2 Hz with only the
telemetry received so far, against a set of fixed limits (our own simple limit set, not real
launch-commit criteria):

| | Agent | Fixed limits |
|---|---:|---:|
| Engine-start failures stopped before booster ignition | 86 % | 39 % |
| Countdown failures held before ignition | 85 % | 56 % |
| Real turbine over-temperatures detected | 81 % | 34 % |
| Trajectory deviations detected | 100 % | 61 % |
| Sensor faults mistaken for an over-temperature | 0 % | 68 % |
| False holds per 1,000 nominal launches | 4 | 0 |
| False aborts per 1,000 nominal launches | 0 | 0 |

- Classifier snapshot at H0 + 150 s: macro F1 0.90, macro AUROC 0.98. The acceptance target
  of 0.80 was fixed in `spec.yaml` before any result existed.
- The rule discovery produced 42 clear-text rules graded A or B. For two failure modes (pump
  cavitation, TVC actuator degradation) no single-feature rule reaches grade B; they are
  flagged as gaps, and the agent says so when it calls them.
- Among the 237 normal held-out launches the agent raised one false HOLD and one false flight
  WARNING, and no false abort. The alarm threshold is a budget we set (0.15 % of launches per
  failure mode), not a hidden constant.

Every number above is in `ControlRoom/agent_evaluation.json` and `layer-4/evaluation.json`,
and the steps below regenerate them.

## Run it

Python 3.11 or newer. On Windows, double-click `Rocket Launch.bat`: it creates `.venv`,
builds the trained models on the first start (about 10 minutes), starts the demo and opens
http://localhost:8765. By hand, on any platform:

```text
python -m venv .venv                                then activate it and:
pip install -r requirements.txt
python run_factory.py                               simulation, Layers 1-4, trained models (about 10 min)
python ControlRoom/serve.py                         the demo at http://localhost:8765
```

`ControlRoom/dashboard.html` also opens by double-click, offline and without any install. It
then plays the showcase stories and one held-out launch per failure type, and lists every
other launch with its result; playing any of the 500 and the analyst chat need `serve.py`.

To regenerate every committed result and run the tests:

```text
python run_factory.py                               Layers 1-4 and layer-4/model.pkl
python ControlRoom/evaluate.py                      streaming evaluation, agent vs fixed limits
python ControlRoom/build_universe.py                knowledge universe, about 30 s
python ControlRoom/build_dashboard.py               dashboard.html
python -m pytest tests -q          
```

The simulation is deterministic (seed base 2026), so with the versions pinned in
`requirements.txt` the numbers reproduce. The trained models (`layer-4/model.pkl`) and every
launch's per-tick scores (`layer-4/streaming_scores.npz`) are not committed, because
scikit-learn pickles depend on its version; `run_factory.py` regenerates them. Edit
`ControlRoom/dashboard_template.html`, never `dashboard.html`: `build_dashboard.py` writes the
page from the template and the data, and a test checks that the two agree.

## What is where

| Level | Path | Contents |
|---|---|---|
| Platform | `discovery/`, `cross_validation/`, `spec_schema.py` | Rule discovery and grading, stratified splits, metrics, the domain specification schema |
| Domain | `launch_sim/` | Vehicle model, failure modes, simulator, causal features, streaming policy |
| Domain | `run_factory.py`, `spec.md`, `spec.yaml` | Runs Layers 1-4 on the platform code; the specification and its acceptance targets |
| Domain | `layer-1/` … `layer-4/` | Taxonomy, graded rules, knowledge graph, evaluation |
| Application | `ControlRoom/` | Agent, fixed-limit baseline, streaming evaluation, knowledge universe, dashboard, local server |

The platform code was written first for clinical data, so some of its field names still say
patients and diagnoses; here a patient is a launch and a diagnosis is a failure mode.
`DESIGN.md` records the design and every assumption behind the simulation.

## What the factory produced

| Layer | Artifact | Result |
|---|---|---|
| 1 | `layer-1/taxonomy.json` | 12 failure modes + nominal, 146-208 launches each (≥ 50 required) |
| 2 | `layer-2/rules.json` | 42 clear-text rules graded A/B, 1 exclusion rule; 2 modes flagged as Layer 4 gaps |
| 3 | `layer-3/knowledge-graph.json` | 143 nodes, 285 edges; found the planted LOX under-pressure → cavitation coupling (lift 4.0) |
| 4 | `layer-4/evaluation.json` | Holdout macro F1 0.90 at H0 + 150 s (target 0.80), macro AUROC 0.98 |

## The dashboard

- **Control room**: replay a launch against the clock, with the agent's call, the fixed
  limits, 13 telemetry strips and the event log. A picture of the vehicle follows the replay
  from the pad to H0 + 150 s. It is drawn from the telemetry: flame sizes follow the chamber
  pressures, the nozzle tilt follows the steering data, and a colored ring marks the part of
  the vehicle an alert is about. Hovering over a story in the list explains it in plain words.
- **Unseen test launches**: all 500 held-out launches, chosen in two steps. First a failure
  type (the twelve, plus *Normal (no failure)* and *Two or more failures*), then one of its
  launches, sorted from mild to strong, each with its severity and the agent's result: caught
  (with its call and the time), missed, or a false alarm. *Surprise me* plays a random one
  with the truth hidden until you reveal it. A link can open one directly:
  `http://localhost:8765/#room/case-202601459`.
- **Analyst chat** (served by `serve.py`): an LLM explains each new alert as the replay
  reaches it and answers questions about the launch. See below.
- **What the factory learned** and **Performance**: the layer artifacts and the streaming
  evaluation.
- **Knowledge universe**: every action state of all 3,000 launches as a 3D star map.

Four stories to start with: the nominal launch (quiet from countdown to booster separation);
LOX tank under-pressure (HOLD at H0 − 55.5 s, while the tank is still pressurizing); pump
cavitation at engine start (ABORT at H0 + 3.5 s, before the boosters ignite at H0 + 7 s); and
the sensor fault next to the real over-temperature, where the fixed 950 K limit alarms on the
failed sensor and never trips on the real, moderate over-temperature, while the agent does the
opposite because it checks whether pump speed and chamber pressure moved with the temperature.

## The knowledge universe

An action state is what the agent held at one 0.5 s check of one launch: one probability per
failure mode, the call it made, and the time. `ControlRoom/build_universe.py` collects them for
all 3,000 launches (the 2,500 training launches scored out of fold, the 500 held-out launches
by the final models) and cuts each launch where the agent stopped it: **1,004,715 action
states**, on 9,356 distinct probability vectors.

Positions are a 3D t-SNE of those vectors: action space, never feature space. Color is the
most probable action (the failure mode with the highest probability when it reaches 0.5, else
nominal). A launch is the chain of its states: hover a star for its properties, click to keep
the launch and see its timeline. The universe is embedded in `dashboard.html`, so it also
works when the page is opened as a file; it needs WebGL 2.

## Trajectory deviation

The 13th channel, `TRAJ_DEV`: how far (km) tracking puts the vehicle from where the planned
trajectory has it at the same moment. A normal flight points 0.1-0.3° off its plan (wind,
thrust misalignment), and that error times the acceleration, integrated twice, is the
deviation. The limits are our own estimates, not values of any real launcher or range: real
flight-safety corridors are set per mission and are not public.

| Moment | Normal spread (3σ) | Flight-safety corridor (fixed limit) |
|---|---|---|
| H0 + 30 s | 0.09 km | 1.2 km |
| Max-Q, H0 + 60 s | 0.35 km | 1.9 km |
| Booster separation, H0 + 136 s | 2.2 km | 6.6 km |
| H0 + 150 s | 2.8 km | 7.9 km |

The corridor is 1 km plus 2.5 × the normal spread; 400 simulated normal flights stay inside
the band and never come near the corridor. The failure **trajectory deviation** (CRITICAL) is
a guidance error the vehicle follows faithfully: a slowly drifting direction reference, or a
wrong one (as on Ariane 5 flight VA241 in 2018), from H0 + 12 s to H0 + 80 s. Engines and
steering look normal; only the trajectory shows it. On the 31 held-out launches with it, the
agent alerted on all 31, as the drift left the normal scatter; the fixed corridor limit
tripped on 19, always later (median 38 s, at least 27 s).

## The analyst chat

`ControlRoom/serve.py` serves the dashboard at http://localhost:8765, replays held-out
launches for the selector, and relays the chat (standard library only) to any
OpenAI-compatible chat server named in `llm_config.toml`. The default is a llama.cpp
`llama-server` on this machine with Qwen3.8 27B:

```text
llama-server -m <Qwen3.8-27B GGUF> --alias qwen3.8-27b --host 127.0.0.1 --port 8001 -c 32768 -ngl 99 --jinja
python ControlRoom/serve.py
```

vLLM, Ollama or a hosted endpoint work the same way: set `base_url` and `model` in
`llm_config.toml`, or the environment variables it lists (a hosted endpoint takes its key from
`ANALYST_API_KEY`). A smaller model works too; the answers get weaker. On Windows,
`Rocket Launch.bat` starts `llama-server` itself when the environment variables `LLAMA_SERVER`
and `LLAMA_MODEL` name the server and the GGUF file. llama.cpp answers any model name with
whatever model it has loaded, so the server asks a local model server which model is loaded and
refuses to answer under the wrong name.

With every question the page sends a situation report for the current replay time: the call
and its meaning, the fixed-limit status, each alert with its evidence, rules and
knowledge-graph notes, all twelve probabilities against their thresholds, the 13 channels
against the nominal band, the most unusual features, the event log, the failure-mode catalog
and the fixed limits. Ground truth is included only for showcase stories and for held-out
launches picked by failure type or revealed by the viewer. The analyst explains the call; it
never makes or changes one.

## Design choices

- **Causal features.** Every feature uses only data up to the evaluation time, so one model
  serves both the batch factory run and the live agent. A test proves it.
- **Thresholds from a false-alarm budget.** Each failure mode's alert threshold is the lowest
  value that fires on at most 0.15 % of launches without that failure, measured on
  out-of-fold streaming replays with the agent's own debounce.
- **No LLM in the decision loop.** Calls come from the classifiers; the agent's own
  explanations are templates filled with the fired rules and measured values. The analyst
  chat is a separate layer on top: it narrates those facts and answers questions.
- **Counterfactual continuation.** Training launches run to H0 + 150 s even when a real
  sequence would have stopped; the evaluation stops counting at the first HOLD or ABORT.
- **A replay ends where the vehicle stops.** A HOLD shows 12 s more of the stopped countdown.
  After an ABORT the call stays ABORT and neither the agent nor the fixed limits raise new
  alarms: the engine is shut down, and the models never saw a shutdown.

## Limits

- **Synthetic data.** The simulator contains our assumptions about failure signatures. The
  agent learning them back proves the method end to end, not real-world performance. Real
  value needs real data: engine test-bench firings, countdown and hold records, flight
  telemetry archives.
- **Advisory only.** Certified limits and the automatic sequence remain the safety net. The
  agent adds earlier warning, a diagnosis, and fewer false alarms from failed sensors.
- **The trade-off is visible.** The agent is more sensitive than the fixed limits, so it costs
  false alarms, reported above.
- **Background.** On launchers of this architecture the liquid core engine lights first and
  the solid boosters ignite only once it is confirmed healthy, about 7 s later; that is the
  last point a stop is possible, and the demo models it at H0 + 7 s. The sensor-fault story
  echoes STS-51F (1985), when faulty turbine temperature sensors shut down a healthy Space
  Shuttle main engine and a flight controller prevented a second shutdown by inhibiting the
  limits.

## License

This software is released under the Ammonix Research License (see `LICENSE.md`).
Research, educational, and evaluation use is free, including evaluation by a commercial
organization deciding whether to seek a commercial license. Any commercial use requires a
separate commercial license from Ammonix: contact licensing@ammonix.ai.

## Citation

If you use this software, please cite it as described in `CITATION.cff`.
