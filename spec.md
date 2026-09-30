# Launch Vehicle — Domain Specification
# Ammonix platform, launch-vehicle domain
# Version: 0.1.0 | Date: 2026-09-26

All data in this domain is synthetic. The vehicle is notional, a heavy launcher with a cryogenic
core stage and two solid boosters, with illustrative, order-of-magnitude values. It is not data
from a real vehicle.

## 1. Database

### Source
A simulated campaign of 3,000 launches from `launch_sim/simulate.py`, seed base 2026.
Launches are regenerated from their seeds; raw telemetry is not stored.

### Launch population
- 12 failure modes injected independently at a 6 % base rate each (oversampled on purpose),
  severity drawn from Beta(1, 1.8) so that mild cases dominate.
- One planted coupling: LOX tank under-pressure raises the probability of LOX pump
  cavitation. Over-temperature and a temperature sensor fault never occur together.
- Unlabeled confusers (small disturbances inside normal variation) in about 12 % of
  launches each, plus lost telemetry frames.
- 1,436 of 3,000 launches are fully nominal; 505 carry two or more failures.

### Data format
Each launch: 13 channels × 10,501 samples (50 Hz, H0 − 60 s to H0 + 150 s), float32,
NaN for lost frames, plus labels (severity, onset, time from which it is visible).

### Splits
Launch-level stratified holdout of 1/(n_folds + 1) = 500 launches, and 5 folds on the
remaining 2,500 (`cross_validation/splitting.py`). All rows of a launch
stay in the same split.

## 2. Signal Processing Framework

### Method
Causal, phase-windowed telemetry features (`launch_sim/features.py`).
Every feature at time t uses only samples up to t, so the same features serve the
batch factory run and the live agent.

### Feature families (137 total)
- `CD` countdown [−60, −1) s, `IGN` engine start [0, 6.5) s, `ASC` boosted ascent
  [7, 136) s, `LATE` late ascent [100, 136) s: levels, variability, extrema, rise times,
  band-limited oscillation power, spike counts.
- `XC` cross-channel consistency, e.g. turbine-temperature rise not explained by pump
  speed (40 K per %), correlation of TVC yaw with booster imbalance.
- Trajectory, over the whole flight [7, 150] s: the deviation from the planned trajectory
  now, its largest value, its growth over 5 s, its ratio to the normal 3-sigma spread, and
  how fast the pointing error it implies rises (flat for a normal flight, rising in a failure).
- `CTX__t_now`, the evaluation time.

## 3. Building Blocks

### Canonical classes: 12 failure modes + NOMINAL
Countdown (HOLD): LOX tank under-pressure, LH2 pressure regulator oscillation, bus voltage
sag. Engine start (ABORT): slow start, LOX pump cavitation, combustion instability (also
in flight). Flight: turbine over-temperature (CRITICAL), temperature sensor fault
(ADVISORY), booster thrust imbalance, TVC actuator degradation, POGO onset (WARNING),
trajectory deviation (CRITICAL: the guidance follows a wrong direction reference).

### Classifier
One `HistGradientBoostingClassifier` per failure mode (one-vs-rest, ammonix-style),
trained on time-truncated rows: every launch contributes its two decision gates, the end of
the window, and 18 random evaluation times.

### Inference pipeline
Features every 0.5 s → class probabilities → 2-tick debounce → exclusion rule
(over-temperature vs sensor fault: keep the more probable) → call for the phase.

### Error control
Per-mode thresholds calibrated on out-of-fold streaming replays to a false-alarm budget
of 0.15 % of launches without that failure.

## 4. Product Vision

### Core product
A control-room agent that advises the launch team: GO/HOLD in the countdown,
COMMIT/ABORT during the engine start, and ADVISORY/WARNING/CRITICAL in flight for the
mission director and flight safety. It commands nothing.

### Explanations
Template text filled with the fired Layer 2 rules (with grade, sensitivity, specificity),
the measured values against nominal, and knowledge-graph couplings. No LLM in the loop.

### Visualization
`ControlRoom/dashboard.html`: a self-contained, offline page with a launch replay,
12 strip charts, the agent's and the fixed-limit monitor's calls side by side, and the
factory's layers and performance.

## 5. Acceptance Criteria

Fixed in the design (`DESIGN.md`) before any result existed.

### Performance targets
- Layer 1 → 2: every class ≥ 50 positive launches.
- Layer 2 → 3: at least one rule graded B or better per class; classes without one are
  reported as Layer 4 gaps.
- Layer 3 → deploy: holdout macro F1 ≥ 0.80 at H0 + 150 s.

### Auditability
- Every alert names its failure mode, probability, confirmation time and supporting rules.
- Every rule is clear text with sensitivity, specificity, evidence grade and holdout F1.
- Every reported metric is measured on launches the models never saw.
