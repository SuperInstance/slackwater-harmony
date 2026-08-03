# slackwater-harmony

![tests](https://img.shields.io/badge/tests-102%20passed-brightgreen)
![version](https://img.shields.io/badge/version-0.1.0-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

Cognitive friction monitoring and FEP-driven improvisation. This package implements a triadic architecture for measuring system-wide alignment: a Harmony Governor that tracks Φ (cognitive friction) per agent, an Executive that improvises when friction exceeds the deadband, a Groove Detector that spots system-wide harmony, and a Flow State layer that detects and protects the deepest state of player alignment.

## Installation

```bash
pip install slackwater-harmony
```

## Architecture

```
Layer 1 — Sandbox:        Forward simulation, hypothesis testing
Layer 2 — Governor:       Friction measurement, deadband enforcement
Layer 3 — Executive:      Improvisation when friction alarms fire

Flow layer extends all three with player-centric signals.
```

**Φ (phi) — Cognitive Friction:**

```
Φ(t) = α · H(prediction_error) + β · L(compute) + γ · Δ(state)
```

Default weights: α=0.50, β=0.30, γ=0.20. When Φ exceeds an agent's deadband, the Governor fires a `FrictionAlarm`. The Executive wakes and improvises.

## API Reference

### HarmonyGovernor

```python
from slackwater_harmony import HarmonyGovernor, FrictionAlarm, AlarmSeverity

HarmonyGovernor(
    alpha: float = 0.50,   # prediction error weight
    beta: float = 0.30,    # compute load weight
    gamma: float = 0.20,   # state delta weight
)
```

Measures cognitive friction across all registered agents. Does not act — only observes and alarms.

**Registration:**

```python
gov.register_agent(agent_id: str, base_deadband: float = 1.0) -> AgentFrictionProfile
gov.set_game_state(state: str) -> None
```

Game state multipliers adjust all agents' deadbands:

| Game State | Multiplier | Meaning |
|---|---|---|
| `tutorial` | 2.0× | Friction is learning |
| `stage_1` | 1.8× | Early game |
| `stage_3` | 1.2× | Mid game |
| `stage_5` | 0.7× | Expert — friction is trouble |
| `creative` | 3.0× | Exploration expected |

**Measurement:**

```python
phi: float = gov.measure_friction(
    agent_id: str,
    prediction: dict | float | list,
    actual: dict | float | list,
    compute_load: float = 0.0,
    state_delta: float = 0.0,
)
```

Prediction error is computed differently by type:
- **Scalars:** `|prediction − actual|`
- **Lists:** mean absolute difference elementwise
- **Dicts:** mean absolute difference over shared keys

**Deadband enforcement:**

```python
gov.check_and_alarm(
    agent_id: str,
    phi: float,
    context: dict | None = None,
    timestamp: int = 0,
) -> FrictionAlarm | None
```

Severity is based on the overshoot ratio (φ / deadband):

| Severity | Ratio Range | Response |
|---|---|---|
| `NONE` | ≤ 1.0 | Do nothing |
| `GENTLE` | 1.0 – 1.5 | Nudge |
| `MODERATE` | 1.5 – 2.0 | Adapt |
| `CRITICAL` | ≥ 2.0 | Intervene |

**Properties:**

```python
gov.total_friction -> float       # sum of all agents' average Φ
gov.max_friction -> float         # worst agent's average Φ
gov.is_harmonized -> bool         # all agents within deadbands
gov.agent_phi(agent_id) -> float  # most recent Φ for an agent
```

Deadbands are **adaptive**: they widen when an agent is struggling (alarm-heavy) and narrow after a calm streak (10+ clean readings).

### ExecutiveAgent

```python
from slackwater_harmony import ExecutiveAgent, Improvisation, ImprovisationType

ExecutiveAgent(
    sandbox: HypothesisSandbox | None = None,
    novelty_threshold: float = 0.15,
    max_rewrites_per_episode: int = 3,
)
```

Layer 3: improvises when friction alarms fire. The most common response is `NONE` — the agent stays quiet. Severity-based decision tree:

| Severity | Response | Description |
|---|---|---|
| `GENTLE` | `SIMPLIFY` | Reduce task complexity |
| `MODERATE` | `REWRITE` | Rewrite constraints (budget-limited) |
| `CRITICAL` | `TAKE_OVER` | Agent takes the lead |
| `CRITICAL` (cascading) | `RESET` | Reset context entirely |
| Any (probability 0.15) | `CROSS_WIRE` | Try something unexpected |

```python
improvisation = exec.handle_alarm(alarm) -> Improvisation
improvisations = exec.handle_alarms(alarms) -> list[Improvisation]  # batch, sorted by severity
```

`Improvisation` has `type`, `reason`, `constraint_rewrites`, `dialogue`, `novelty`, `confidence`. Apply to a plan dict via `improvisation.apply_to(plan)`.

### GrooveDetector

```python
from slackwater_harmony import GrooveDetector, GrooveState

GrooveDetector(
    governor: HarmonyGovernor,
    min_sustained_beats: int = 8,
    phi_variance_threshold: float = 0.15,
)
```

Watches the Governor for groove states. A groove requires all agents below deadband AND low Φ variance, sustained for `min_sustained_beats`.

**State machine:**

```
SEARCHING → SETTLING → IN_POCKET → DISRUPTED → SEARCHING
```

```python
detector.update(beat: int | None = None) -> GrooveState
detector.in_groove -> bool
detector.groove_quality() -> float    # 0.0–1.0
detector.in_pocket_percentage -> float
```

### FlowStateDetector

```python
from slackwater_harmony import FlowStateDetector, FlowPhase, FlowReading

FlowStateDetector(
    governor: HarmonyGovernor,
    flow_threshold: float = 0.72,
    deep_flow_threshold: float = 0.88,
    pre_flow_threshold: float = 0.45,
    min_flow_sustained: int = 5,
)
```

Extends GrooveDetector with four player-centric signals:

| Signal | What It Measures | Flow Signature |
|---|---|---|
| **Action entropy** | Shannon entropy of recent actions | Low (focused) |
| **Cadence regularity** | 1 − CV of inter-action intervals | High (steady rhythm) |
| **Hurst exponent** | R/S analysis of value time series | > 0.5 (persistent trending) |
| **Micro-timing** | Inverse normalized MAD of deltas | High (consistent intervals) |

**Composite flow score:**

```
score = 0.25·(1−entropy) + 0.30·cadence + 0.20·hurst + 0.25·micro_timing
```

Modified by groove state: ×1.1 if in pocket, ×0.8 if disrupted.

**Flow state machine:**

```
PRE_FLOW → FLOW → DEEP_FLOW → POST_FLOW → RECOVERY → (PRE_FLOW...)
```

Flow must sustain for `min_flow_sustained` readings before declaring. Transitions from DEEP_FLOW drop to FLOW first (not directly to POST_FLOW).

```python
detector.record_action(action: str, timestamp: float | None = None, value: float | None = None)
detector.update_flow(timestamp: float | None = None) -> FlowPhase
detector.in_flow -> bool
detector.flow_duration -> float    # seconds in current flow
detector.flow_score -> float       # most recent composite
```

### FlowStateProtector

```python
from slackwater_harmony import FlowStateProtector, ProtectiveAdjustment, TempoMap

FlowStateProtector(
    detector: FlowStateDetector,
    tempo: TempoMap,
    friction_window: int = 5,
    rising_friction_threshold: float = 0.15,
)
```

Makes imperceptible adjustments to protect flow. When rising friction is detected (slope of Φ over recent readings), applies tiny adjustments: −2 BPM tempo nudge, 10% chatter reduction, 5% ambient dim, 5% deadband widening. All adjustments are verified gentle (|Δbpm| ≤ 3, chatter_reduction ≤ 0.5, ambient_dim ≤ 0.3).

```python
protector.engage() -> ProtectiveAdjustment      # lock tempo, reduce noise
protector.gentle_adjust() -> ProtectiveAdjustment | None  # one tiny correction
protector.tick() -> ProtectiveAdjustment | None  # call each beat
protector.disengage() -> ProtectiveAdjustment    # restore normal
```

### FlowStateJournal

```python
from slackwater_harmony import FlowStateJournal, FlowSession

FlowStateJournal()
```

Long-term memory of flow sessions. Records when flow happened, what triggered it, how long it lasted, and what broke it.

```python
journal.record_flow_start(timestamp, conditions, player_state) -> FlowSession
journal.record_flow_score(score: float) -> None
journal.record_flow_end(timestamp, trigger, phase_reached) -> FlowSession | None
journal.get_patterns() -> dict       # aggregated analysis
journal.export_session() -> dict     # musical timeline export
journal.total_flow_time -> float
journal.longest_flow -> float
```

### TempoMap (harmony-local)

```python
from slackwater_harmony import TempoMap

TempoMap(bpm=120.0, min_bpm=60.0, max_bpm=180.0, adapt_rate=0.1)
```

Adaptive tempo with flow-lock. When `locked`, tempo does not change. Methods: `set_target()`, `update()`, `lock(reason)`, `unlock()`, `nudge(delta)`.

## Examples

### Basic friction monitoring

```python
from slackwater_harmony import HarmonyGovernor, AlarmSeverity

gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
gov.register_agent("lucineer", base_deadband=0.5)

phi = gov.measure_friction("lucineer", prediction=0.0, actual=0.8)
alarm = gov.check_and_alarm("lucineer", phi)

if alarm:
    print(f"{alarm.severity.name}: Φ={alarm.phi:.3f} vs deadband={alarm.deadband:.3f}")
```

### Executive improvisation

```python
from slackwater_harmony import ExecutiveAgent

exec = ExecutiveAgent(novelty_threshold=0.0)  # deterministic for testing
imp = exec.handle_alarm(alarm)
print(imp.type)   # e.g. SIMPLIFY, REWRITE, TAKE_OVER
print(imp.reason)
new_plan = imp.apply_to(current_plan)
```

### Flow detection

```python
from slackwater_harmony import HarmonyGovernor, FlowStateDetector, FlowPhase

gov = HarmonyGovernor()
gov.register_agent("a", base_deadband=5.0)
detector = FlowStateDetector(governor=gov, flow_threshold=0.5, min_flow_sustained=3)

for beat in range(20):
    gov.measure_friction("a", prediction=0, actual=0)  # zero friction
    detector.update(beat=beat)
    detector.record_action("place", timestamp=float(beat) * 1.0, value=0.9)
    phase = detector.update_flow()
    if detector.in_flow:
        print(f"IN FLOW at beat {beat} (score={detector.flow_score:.2f})")
```

## License

MIT
