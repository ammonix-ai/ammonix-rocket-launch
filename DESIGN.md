# Design and simulation assumptions

The design the agent was built from: the simulator, the failure modes, the features, the
factory run, the agent's logic and the showcase stories, with every assumption. Where
the build departs from the design, the departure is listed first.

## Where the build departs from the design

- **Thresholds** are not F1-optimal on snapshots. Each is calibrated on out-of-fold
  *streaming* replays to a false-alarm budget of 0.15 % of launches per failure mode.
  The first version, which used F1-optimal thresholds, raised about 30 false alerts per
  1,000 nominal launches in streaming.
- **Training rows** are the two gates, the end of the window and 18 random times per
  launch, not a fixed 17-point grid. The fixed grid left the model unsure between grid
  times.
- **Streaming evaluation and the fixed-limit comparison** live in `ControlRoom/evaluate.py`
  (application level) rather than in `layer-4/`. Layer 4 keeps the per-class streaming
  metrics in `evaluation.json`.
- **Severities** follow Beta(1, 1.8) with lower minimums than section 2.4, so mild,
  hard-to-see cases dominate; the first calibration separated almost every class
  perfectly.
- **Mutex discovery** uses a 0.5 % co-occurrence cut-off instead of 2 %, and
  knowledge-graph couplings need a Bonferroni-corrected Poisson test. Both keep sampling
  noise out of the knowledge layer.
- **Rule confirmations:** the agent can call a failure before any graded rule can be
  checked, for example the LOX HOLD at H0 − 55.5 s. It then logs each rule as it starts
  to fire.
- **Added after the first review:** a local server (`ControlRoom/serve.py`) with an
  analyst chat on an LLM (`llm_config.toml`); a picture of the vehicle drawn from the
  telemetry; and plain-language texts for every story and failure mode. The analyst
  explains calls from a situation report and never makes one.

## 2. Simulator (`launch_sim/`)

Physics-lite: each channel follows a nominal profile, with physically consistent couplings. It is not
an engine or flight simulation. All values are notional orders of magnitude.

### 2.1 Time base and events

- 50 Hz; `t` from **H0 − 60 s to H0 + 150 s** → 10,501 samples per channel.
- Events: −58 s LOX tank pressurization starts · −55 s LH2 pressurization starts · −30 s transfer to
  onboard power · **0 s core engine ignition (H0)** · 6.5 s booster-commit gate · **7.0 s booster
  ignition** · liftoff when total thrust > weight (≈ 7.3 s) · max-Q ≈ 60 s (booster thrust bucket
  50–70 s) · booster tail-off 124–134 s · 136 s booster separation.
- Decision gates: **H0 − 1 s** ignition GO/HOLD; **H0 + 6.5 s** booster commit/abort (the last point
  an abort is possible: on launchers of this architecture the boosters light only after the liquid engine is healthy).

### 2.2 Vehicle-to-vehicle variation (per launch)

`Pc_nom ~ N(115, 0.8) bar`, `N_nom ~ N(100, 0.5) %`, `T_nom ~ N(880, 6) K`,
`P_LOX_set ~ N(3.50, 0.025) bar`, `P_LH2_set ~ N(2.30, 0.018) bar`, start midpoint
`m ~ N(1.60, 0.05) s`, start width `w ~ N(0.30, 0.02) s`, booster scale shared `N(1, 0.008)` ×
per-booster `N(1, 0.006)`, battery `N(28.6, 0.12) V`, wind factor `lognormal(0, 0.35)`.

### 2.3 The 12 channels, plus the trajectory deviation added later

Engine power state `e(t) = 1/(1 + exp(−(t − m)/w))` for t ≥ 0, else 0 (t90 ≈ 2.3 s), plus a +3 %
overshoot bump ≈ 1.2 s after `m`. Anomalies modify `e(t)` before the channels are derived.

| Channel | Meaning | Unit | Nominal model | Noise σ |
|---|---|---|---|---|
| `PC_CORE` | core chamber pressure | bar | 1.0 before H0; `1 + (Pc_nom − 1)·e(t)` | 0.25 |
| `N_LOX_TP` | LOX turbopump speed | % nominal | same logistic, 0.15 s earlier, 10 % steeper; `N_nom·e` | 0.12 |
| `T_TURB_IN` | gas-generator turbine inlet temp | K | 285 before H0; rises (0.3 s ahead of `e`) to `T_nom`, +35 K start overshoot peak | 1.5 |
| `P_LOX_TANK` | LOX tank ullage pressure | bar | 1.35; from −58 s first-order ramp τ = 4 s to set; ±0.015 regulation sawtooth (8 s); −0.05 dip at engine start | 0.008 |
| `P_LH2_TANK` | LH2 tank ullage pressure | bar | 1.25; from −55 s τ = 5 s to set; ±0.01 sawtooth | 0.006 |
| `PC_SRB_L`, `PC_SRB_R` | booster chamber pressures | bar | 1.0 before 7.0 s; rise τ ≈ 0.12 s; knots (7.3, 88) (20, 92) (35, 90) (50, 72) (70, 72) (85, 84) (110, 82) (124, 76), tail-off to ≈ 1 at 134 s; × booster scale | 0.35 |
| `TVC_PITCH`, `TVC_YAW` | core-engine TVC deflection | deg | ≈ 0 on pad; after liftoff gust response (low-pass noise, fc 0.5 Hz, ≈ 0.6 deg × wind × (0.15 + 0.85·q(t))); pitch program +0.35 deg bump over 10–30 s; **yaw += 12 deg × (Pc_L − Pc_R)/mean(Pc)** (TVC compensates booster imbalance); actuator lag 0.05 s | 0.012 |
| `ACC_AX` | sensed axial acceleration | g | 1.0 on pad; after liftoff `(F_core + F_srb − drag)/(m·g0)`; `F_core = e·(960 + 400·clip((t−7)/100, 0, 1))` kN; `F_srb = 40 kN/bar·(Pc − 1)` per booster; `drag = 180 kN·q(t)`, `q(t) = exp(−((t − 60)/22)²)`; `m0 = 530 t`, core 0.32 t/s, each booster burns 142 t in proportion to its Pc, −22 t casings at separation | 0.008 |
| `VIB_ENG` | engine-bay vibration | g rms | 0.03 before H0; `0.9·e(t)` engine; `+2.6·exp(−(t−7.2)/6)` liftoff acoustics; `+0.7·exp(−((t−52)/10)²)` transonic buffet; `+2.0·exp(−(t−136)/0.8)` separation shock | 6 % multiplicative |
| `BUS_28V` | avionics bus voltage | V | 28.0 on ground power; at −30 s battery with a 0.35 V / 0.3 s dip; −0.05 V/min discharge; `−0.15·e(t)` load at engine start | 0.03 |

Sanity targets: liftoff ≈ 7.3 s at ≈ 1.5 g; ≈ 3.5 g near booster burnout; ≈ 0.7 g after separation.

**Added later: `TRAJ_DEV`, trajectory deviation [km]** (subsystem guidance): the distance between
where tracking puts the vehicle and where the planned trajectory has it at the same moment.
Lateral acceleration = axial acceleration × pointing error, integrated twice per tracking axis.
Normal pointing errors: a fixed bias per axis N(0, 0.22°) × (0.75 + 0.25·wind), 5 % of the gust the
TVC answers, 3 % of a booster-imbalance correction; tracking noise 15 m per axis. These are our
estimates (real flight-safety corridors are mission-specific and not public): normal 3σ =
0.03 km + 2.2 km × the double integral of the mean acceleration from liftoff (1 at separation),
the shape a constant pointing error drifts in: 0.09 km at H0 + 30 s, 0.35 km at max-Q, 2.2 km at
separation. Flight-safety corridor = 1 km + 2.5 × that (1.2, 1.9, 6.6 km), a fixed limit in
`redline.py`. Features compare the deviation with this shape: a normal flight keeps a flat ratio,
a failure makes it rise, which is what keeps windy normal flights from looking like failures.
400 simulated normal flights stay inside the band (0.25 % above it) and never reach the corridor.
The channel and its failure draw from two random streams of their own, so every other channel,
label and lost frame of every launch stayed exactly as it was (checked on 300 launches).

### 2.4 Failure modes (Layer 1 labels)

Severity `s ~ U(0, 1)` scales each signature from barely visible to strong. Base prevalence **6 %**
per class, so about half the launches are fully nominal (failures oversampled on purpose).
`manifest` = the time from which the anomaly is visible in the data
(used for streaming labels).

| # | Class | Phase | Call | Signature | Onset / manifest |
|---|---|---|---|---|---|
| 1 | `LOX_TANK_UNDERPRESSURE` | countdown | HOLD | τ × (1.5 + 2s); setpoint − (0.10 + 0.45s) bar; persists into flight | manifest −52 s |
| 2 | `LH2_PRESS_OSCILLATION` | countdown | HOLD | sinusoid f ~ U(0.2, 0.4) Hz, A = 0.03 + 0.12s bar, 5 s ramp-in, continues | onset U(−50, −15) s; manifest onset + 1/f |
| 3 | `BUS_VOLTAGE_SAG` | countdown | HOLD | dip −(1.2 + 3.3s) V at −30 s, recovery τ = 2 + 8s s, residual −(0.2 + 0.8s) V | manifest −30 s |
| 4 | `SLOW_ENGINE_START` | engine start | ABORT | `m += 0.5 + 1.3s`, `w × (1 + 0.6s)` → t90 ≈ 2.9–4.4 s | manifest 2.0 s |
| 5 | `LOX_PUMP_CAVITATION` | engine start | ABORT | bursts (Poisson 0.8 + 1.5s per s): N +(2 + 4s) % spikes of 0.1–0.3 s, Pc −(2 + 6s) bar dips, VIB +(0.4 + 0.8s) g; episode ends ≈ 3 s after liftoff (acceleration head raises pump inlet pressure) | onset U(0.8, 4.0) s; manifest first burst |
| 6 | `COMBUSTION_INSTABILITY` | start (60 %) or flight (40 %) | ABORT before 7 s, else CRITICAL | Pc ripple σ = 0.8 + 2.5s bar; engine vibration × (1.6 + 2.4s); 0.5 s ramp | onset U(1, 5) or U(10, 120) s; manifest onset + 0.5 |
| 7 | `TURBINE_OVERTEMP` | flight | CRITICAL | ΔT = 40 + 120s K over ramp U(3, 15) s; engine power × (1 + 0.22·ΔT/T_nom) → **pump speed and Pc rise with it** | onset U(10, 110) s; manifest onset + 0.3·ramp |
| 8 | `TEMP_SENSOR_FAULT` | flight | ADVISORY | `T_TURB_IN` only: step +(80 + 220s) K (50 %), intermittent spikes +200–500 K (30 %), rail to 1,300 K and stuck (20 %); **nothing else moves**. Mutually exclusive with #7 | onset U(8, 130) s; manifest onset |
| 9 | `SRB_THRUST_ASYMMETRY` | flight | WARNING | one booster Pc × (1 ± (0.03 + 0.07s)) (70 % low / 30 % high), ramp U(2, 10) s → yaw offset through the TVC coupling | onset U(15, 90) s; manifest onset + 0.5·ramp |
| 10 | `TVC_ACTUATOR_DEGRADATION` | flight | WARNING | pitch limit cycle f ~ U(0.8, 2.0) Hz, A = 0.08 + 0.4s deg, 3 s ramp-in, + 0.03 deg extra noise | onset U(15, 110) s; manifest onset + 2 |
| 11 | `POGO_ONSET` | flight | WARNING | `ACC_AX` oscillation f ~ U(12, 18) Hz growing to A = 0.03 + 0.22s g (τ = U(4, 10) s), coupled into Pc at 12 bar/g, 0.8 rad lag; dies at tail-off | onset U(95, 125) s; manifest onset + τ |
| 12 | `TRAJECTORY_DEVIATION` | flight | CRITICAL | the guidance follows a wrong direction reference, uncorrected: drift 0.1 + 0.5s deg/s (60 %) or offset 1.5 + 6s deg with a TVC kick of min(0.4·offset, 3) deg decaying over 1.5 s (40 %), random direction, capped at 25° | onset U(12, 80) s; manifest when its own deviation exceeds max(0.15 km, band / 3); own random stream |

Plus `NOMINAL` (no labeled anomaly; `is_default: true` in the taxonomy).

**Planted tests of the factory** (the demo's key moments — do not tell the discovery code):

- **Hidden coupling:** P(cavitation) = 0.035 + 0.55·s₁ when #1 is present, else 0.035. Layer 3 should
  find the `CO_OCCURS` link (lift ≈ 5–7) on its own.
- **Sensor fault vs real fault (#7 vs #8):** a fixed temperature redline fires on both; the agent must
  separate them by cross-channel consistency. Historical parallel: on STS-51F (1985) faulty turbine
  temperature sensors shut down a healthy Space Shuttle main engine.

### 2.5 Realistic mess

- **Confusers** (unlabeled, each ≈ 12 % of launches): LH2 mini-oscillation 0.004–0.015 bar; LOX
  setpoint −0.02 to −0.06 bar; start midpoint +0.05 to +0.25 s; T drift +5–25 K with proportional
  corroboration; single-sample T glitch +40–120 K; extra booster mismatch 0.5–2 %; TVC mini limit
  cycle 0.01–0.05 deg; bus dip 0.2–0.9 V; vibration +5–20 %; POGO-like 0.005–0.02 g.
- **Telemetry dropouts:** 35 % of launches get 1–3 all-channel NaN gaps of 0.1–0.8 s at random times;
  50 % get one more gap between 7 and 12 s (plume attenuation at liftoff).

### 2.6 API

- `simulate_launch(seed, forced=None, random_anomalies=True, confusers=True, abort_at=None) -> Launch`
  with `Launch(launch_id, seed, tel: float32 (12, 10501), labels: {class: {severity, onset, manifest,
  params}}, meta: {vehicle params, wind, liftoff_time, dropouts, confusers})`.
- `forced={"TEMP_SENSOR_FAULT": {"severity": 0.7, "onset": 62.0, "mode": "step"}}` builds showcase
  stories; `random_anomalies=False` keeps them clean.
- `abort_at`: engine shutdown (`e` decays with τ = 0.4 s from `abort_at + 0.3`), no booster ignition,
  no liftoff — used to replay ABORT stories realistically.
- Training data lets every launch run to H0 + 150 s even when a real sequence would have stopped
  (counterfactual continuation). State this in the spec and the dashboard notes.
- Campaign: **3,000 launches**, launch seed `2026·100000 + i`, 4 worker processes. Do not hold all
  raw telemetry (≈ 1.5 GB): compute feature rows in the workers; re-simulate holdout launches from
  their seeds for streaming evaluation.

## 3. Features (`launch_sim/features.py`)

- **Causal only.** Per launch, precompute once: forward-filled channels (hold last value over
  dropouts), trailing moving averages (cumsum), causal IIR band-pass signals (`scipy.signal.butter` +
  `lfilter`), running max/min/count arrays. The vector at `t_now` is then a set of reductions over
  `[window start, min(window end, t_now)]` — identical to computing on truncated data, cheap, and with
  no future leakage. Add a test that proves the equality.
- **Names:** `PHASE__CHANNEL__stat`, with phases `CD` [−60, −1), `IGN` [0, 6.5), `ASC` [7, 136),
  `LATE` [100, 136), `XC` (cross-channel), `CTX`. Keep a `FEATURE_CATALOG` with a plain-English
  description, unit, channel(s), phase and `valid_from` for each feature.
- Windows not started, or with < 1 s of data → NaN (HistGradientBoosting handles NaN).
- `CTX__t_now` is a model input.
- **`valid_from`** (used only when displaying fired rules in streaming): running max/min/counts and
  censored crossing times are valid from window start + 1 s; mean/std/level features at window end.
- **Generic** (≈ 76): mean/std/min/max for CD: `P_LOX_TANK`, `P_LH2_TANK`, `BUS_28V`; IGN: `PC_CORE`,
  `N_LOX_TP`, `T_TURB_IN`, `VIB_ENG`; ASC: all 12 channels.
- **Engineered** (≈ 45), including:
  - CD: LOX pressure last 10 s; LOX ramp deficit vs the reference ramp; time to reach 3.3 bar
    (censored at `t_now`); LH2 band RMS 0.15–0.5 Hz; LH2 last 10 s; bus dip at transfer; bus mean
    after −25 s.
  - IGN: Pc t50 and t90 (censored at `t_now`, so a slow start is visible *while it happens*); Pc at
    2.5 s; Pc ripple after 3 s; Pc dip count; pump-speed spike count and max above the trailing mean;
    turbine temp start peak and 4–7 s plateau; vibration mean/max after 3 s; LOX pressure during
    start.
  - ASC: Pc ripple (overall and max over 5 s sub-windows); Pc / pump-speed / turbine-temp rise vs the
    4–7 s reference (running max); temperature step max and spike count; **`XC__T_unexplained_max` =
    max over time of (ΔT − 40 K per % × ΔN)** — the physics-informed consistency check that separates
    a sensor fault from a real over-temperature; booster imbalance mean/max/signed-last (%); yaw bias
    mean and |max|; correlation of yaw bias with imbalance; pitch band RMS 0.7–2.2 Hz (limit cycle,
    overall and max over 10 s); gust band RMS 0.02–0.5 Hz; yaw band RMS 0.7–2.2 Hz; `ACC_AX` and
    `PC_CORE` band RMS 10–20 Hz (max over 5 s, POGO); vibration excess vs the reference profile
    (max, mean); pump-speed spike count in flight; LH2 band RMS in flight.

## 4. Factory run (`run_factory.py`)

### Layer 1 — `layer-1/taxonomy.json`

Shape: `meta` (domain, version, canonical_count, generated_by, launch_count),
`diagnoses` (canonical, display, category, phase, action, severity, subsystem, channels, description,
positive_count, prevalence, is_default), `categories` (propulsion pressurization, electrical, core
engine, instrumentation, boosters, flight control, structural dynamics, nominal).
Acceptance (Layer 1 → 2): every class ≥ 50 positive launches.

### Layer 2 — `layer-2/rules.json` (platform `discovery/` code, unchanged)

Work set only; the holdout stays clean.

```python
X = work-set features at t_now = 150 (raw engineering units; impute any NaN with column median)
Y = 11 anomaly columns — exclude NOMINAL (it is the complement; mutex discovery would pair it with
    every class and emit overlapping groups)
corr  = compute_all_correlations(X, Y, feat_names, dx_names, min_positive=10, top_k=20)
rules = generate_threshold_rules(X, Y, feat_names, dx_names, corr, top_k=5, n_thresholds=50); grade_rules(rules)
mutex = discover_mutex_groups(Y, dx_names, threshold=0.02); grade_mutex(mutex)
equiv = discover_equivalence_groups(StandardScaler().fit_transform(X), Y, ...); grade_equivalence(equiv)
        # z-score first: raw units make every cosine ≈ 1
val   = validate_rules(rules, X_hold, Y_hold, feat_names, dx_names)
rules_json = assemble_rules_json("launch-vehicle", rules, mutex, equiv, grade_cutoff="B")
check_consistency(rules_json); classify_handoff(DiscoveryReport(...))
```

Add a readable sentence per rule from the feature catalog, e.g. "IF LOX tank pressure over the last
10 s of countdown < 3.31 bar THEN LOX tank under-pressure". Acceptance (Layer 2 → 3): ≥ 1 rule with
grade B+ per class; list classes without one as `layer4_gap`.

### Layer 3 — `layer-3/knowledge-graph.json`

Same `meta` block as the taxonomy. Nodes: failure modes, channels, subsystems, features used by rules
or top correlations, rules, actions, phases. Edges: `SUPPORTS` (rule → mode, with sens/spec/grade),
`DISCRIMINATED_BY` (mode → feature, from the correlations), `EXCLUDES`, `EQUIVALENT`, `CO_OCCURS`
(label lift ≥ 2 with ≥ 10 co-occurrences), `MEASURED_ON` (feature → channel), `PART_OF` (channel →
subsystem), `TRIGGERS` (mode → action per phase), `OCCURS_IN` (mode → phase).

### Layer 4 — model and evaluation (`layer-4/`)

- **Rows:** (launch, `t_now`) for `T_GRID = [−45, −30, −15, −1, 2, 3, 4, 5, 6.5, 15, 30, 45, 60, 80,
  100, 120, 150]`. Row label for class k = 1 iff k is labeled for the launch and `manifest_k ≤ t_now`.
- **Split at launch level** with the platform:
  `holdout = resolve_holdout_size(n_launches, n_folds=5)` (= 500 of 3,000) →
  `stratified_holdout_split(Y_launch, holdout)` → `stratified_cv_folds(Y_launch[work], n_folds=5)`;
  expand to rows so all rows of a launch stay together.
- **Classifier:** one `HistGradientBoostingClassifier` per class (max_iter 300, learning_rate 0.06,
  max_leaf_nodes 31, l2_regularization 1.0). XGBoost only if already installed.
- **Thresholds** per class from out-of-fold rows: maximize F1 subject to a row-level false-positive
  rate ≤ 0.3 % on nominal rows. Report streaming false alarms as measured; do not tune on the holdout.
- **Metrics** through `cross_validation.metrics` (`per_class_record`, `aggregate`
  macro/micro): holdout at `t_now = 150` (end-of-window view); gate views at −1 s (countdown classes)
  and 6.5 s (engine-start classes).
- **Streaming evaluation** on the 500 holdout launches at 2 Hz with the agent logic of section 5:
  per class detection rate and latency after `manifest`; share caught before their gate; false HOLD
  and false ABORT per 1,000 nominal launches; false CRITICAL on sensor-fault launches; the same
  numbers for the redline monitor.
- Acceptance (Layer 3 → deploy, in `spec.yaml`): holdout macro F1 at `t_now = 150` ≥ 0.80. Report
  whatever the result is.

## 5. Control-room agent (`ControlRoom/agent.py`, `redline.py`)

- Tick every 0.5 s: features at `t_now` → probabilities → alert when `p ≥ threshold` on **two
  consecutive ticks**.
- Mutex correction from `rules.json` (e.g. over-temperature vs sensor fault → keep the higher).
- **Calls by phase:**
  - countdown (t < 0): HOLD if any alert, else GO (gate at −1 s);
  - engine start (0 ≤ t < 7): ABORT (engine shutdown) if any alert, else COMMIT at 6.5 s;
  - flight: CRITICAL (combustion instability, turbine over-temperature, cavitation), WARNING (booster
    imbalance, TVC degradation, POGO), ADVISORY (sensor fault: "treat T_TURB_IN as a failed sensor; do
    not act on its redline without corroboration — human decision"), else NOMINAL. Flight calls
    inform the mission director and flight safety; an uncrewed launcher cannot be aborted in flight,
    so the agent commands nothing.
- **Evidence:** B+ rules supporting an alerting class whose feature is valid at `t_now`; top 3
  deviating features (z-score vs nominal launches at the nearest grid time ≤ `t_now`); knowledge-graph
  links (`CO_OCCURS`, `TRIGGERS`).
- **Template explanation**, e.g.: "HOLD — LOX tank under-pressure (p = 0.97). LOX tank pressure
  3.18 bar over the last 10 s (nominal 3.50 ± 0.03). Rule R-07 (grade A; sens 0.93, spec 0.98): IF …
  THEN …. Knowledge graph: raises LOX pump cavitation risk at engine start (lift 5.8×)."
- **Redline monitor** (baseline — our assumption, not a real launch-commit criterion): countdown:
  `P_LOX_TANK` < 3.30 bar (after −40 s), `P_LH2_TANK` outside [2.15, 2.45] bar (after −35 s),
  `BUS_28V` < 26.5 V (after −25 s) → HOLD; engine start: Pc < 90 bar at t ≥ 4.0 s, `T_TURB_IN` >
  950 K, `N_LOX_TP` > 106 %, `VIB_ENG` > 3 g → ABORT; flight: `T_TURB_IN` > 950 K, Pc < 105 bar (7–134 s),
  |TVC| > 2.5 deg, |Pc_L − Pc_R| > 8 bar, `VIB_ENG` > 6 g → ALARM.

## 6. Showcase launches (forced, `random_anomalies=False`)

1. **Nominal** — GO → COMMIT → NOMINAL, no alerts.
2. **LOX tank under-pressure** (s ≈ 0.6) → HOLD around H0 − 50 s; the replay ends at the hold, with the
   knowledge-graph note on cavitation risk.
3. **Cavitation at engine start** (mild under-pressure s ≈ 0.2 that the redline misses + cavitation
   s ≈ 0.6) → ABORT around H0 + 2–3 s; re-simulate with `abort_at` so the vehicle stays on the pad.
4. **Slow engine start** (s ≈ 0.5) → ABORT around H0 + 3 s (`abort_at` re-simulation).
5. **Sensor fault** (step, onset 62 s) → redline ALARM; agent ADVISORY "sensor fault — no
   corroboration from pump speed or chamber pressure".
6. **Real turbine over-temperature** (onset 70 s, s ≈ 0.5) → agent CRITICAL early (corroborated);
   redline later or never.
7. **Booster thrust imbalance** (onset 40 s) → WARNING, explained through the TVC yaw offset.
8. **POGO onset** (onset 110 s) → WARNING.

Plus all 500 holdout launches as "unseen test launches" (shows the demo is not cherry-picked):
the viewer picks a failure type, then one of its launches, sorted mild to strong with the
agent's result; or a random one with the ground truth hidden until revealed. The page embeds
one replay per type; `serve.py` replays any other on demand.
