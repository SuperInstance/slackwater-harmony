"""
Flow State Detector — detects and protects the deepest state of alignment.

Flow is what happens when the groove goes unconscious. The player stops
thinking about the instrument and becomes the music. Prediction and
action merge into a single continuous thread. Time distorts. Self-
consciousness drops away.

The GrooveDetector knows when the system is harmonized. The
FlowStateDetector knows when a human player has gone *past* harmony —
into the zone where the tool disappears and only the work remains.

This module extends the harmony architecture with:

    FlowStateDetector   — measures the depth and persistence of flow
    FlowStateProtector  — makes imperceptible adjustments to protect flow
    FlowStateJournal    — remembers when flow happened and what caused it
    TempoMap            — tempo adaptation that locks when flow is active

The detection is based on four signals:
    1. Action entropy     — low entropy = focused, not scattered
    2. Cadence regularity — steady rhythm = in the pocket
    3. Hurst exponent     — >0.5 = persistent trending (flow sustains)
    4. Micro-timing       — consistent intervals between actions

The state machine:

    PRE_FLOW → FLOW → DEEP_FLOW → POST_FLOW → RECOVERY → (PRE_FLOW...)

Flow is not declared on a single reading. It must sustain. And once
detected, it is protected: tempo locks, chatter reduces, ambient dims.
Never hard corrections. Flow is a soap bubble. You don't grab it.
You hold still and let the air do the work.

"When you're vibing with someone — really vibing — you're not
 triggering each other. You're simulating together."
    — Vibing Is Flow: Simulation-Confirmation Thinking
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from slackwater_harmony.governor import HarmonyGovernor
from slackwater_harmony.groove_detector import GrooveDetector, GrooveState


# ────────────────────────────────────────────────────────────
# State Machine
# ────────────────────────────────────────────────────────────

class FlowPhase(IntEnum):
    """
    The phases of the flow lifecycle.

    Flow is not binary. It has an approach, a deepening, a natural
    conclusion, and a recovery period. Each phase demands different
    behavior from the system.

    PRE_FLOW:   Signals are aligning. The player is warming into it.
                The system should be quiet and observant.
    FLOW:       Flow has been detected and sustained. The player is
                in the zone. Tempo locks. Chatter minimizes.
    DEEP_FLOW:  Flow has persisted well past the threshold. The player
                may be losing track of time. The system becomes nearly
                invisible.
    POST_FLOW:  Flow is breaking — signals are diverging. The system
                prepares for re-entry without jarring the player.
    RECOVERY:   Flow has ended. The player needs a breath. The system
                is gentle, reflective, does not immediately push for
                re-entry.
    """
    PRE_FLOW = 0
    FLOW = 1
    DEEP_FLOW = 2
    POST_FLOW = 3
    RECOVERY = 4


# ────────────────────────────────────────────────────────────
# Tempo Map
# ────────────────────────────────────────────────────────────

@dataclass
class TempoMap:
    """
    Adaptive tempo with flow-lock.

    The TempoMap manages BPM (beats per minute) for the system's
    internal clock. During normal operation, tempo adapts based on
    friction and activity. During flow, tempo LOCKS — no adjustments,
    because the groove is the groove and you don't mess with it.

    Attributes:
        bpm: Current tempo.
        min_bpm: Floor.
        max_bpm: Ceiling.
        target_bpm: Where the tempo is heading.
        adapt_rate: How fast BPM moves toward target (0-1 per update).
        locked: When True, tempo does not change.
        lock_reason: Why tempo is locked.
        _history: BPM values over time for analytics.
    """

    bpm: float = 120.0
    min_bpm: float = 60.0
    max_bpm: float = 180.0
    target_bpm: float = 120.0
    adapt_rate: float = 0.1
    locked: bool = False
    lock_reason: str = ""
    _history: list[float] = field(default_factory=list)

    def set_target(self, bpm: float) -> None:
        """Set a new target BPM. Will not apply if locked."""
        if self.locked:
            return
        self.target_bpm = max(self.min_bpm, min(self.max_bpm, bpm))

    def update(self) -> None:
        """Move BPM toward target. Frozen when locked."""
        if self.locked:
            return
        diff = self.target_bpm - self.bpm
        self.bpm += diff * self.adapt_rate
        self.bpm = max(self.min_bpm, min(self.max_bpm, self.bpm))
        self._history.append(self.bpm)
        if len(self._history) > 200:
            self._history.pop(0)

    def lock(self, reason: str = "flow") -> None:
        """Lock tempo — no further adjustments."""
        self.locked = True
        self.lock_reason = reason

    def unlock(self) -> None:
        """Release the tempo lock."""
        self.locked = False
        self.lock_reason = ""

    def nudge(self, delta: float) -> None:
        """Gently adjust target BPM by delta. Respects lock."""
        if self.locked:
            return
        self.set_target(self.target_bpm + delta)

    @property
    def beat_interval(self) -> float:
        """Seconds per beat at current BPM."""
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    @property
    def average_bpm(self) -> float:
        if not self._history:
            return self.bpm
        return sum(self._history) / len(self._history)


# ────────────────────────────────────────────────────────────
# Flow State Detector
# ────────────────────────────────────────────────────────────

@dataclass
class FlowReading:
    """
    A single flow measurement snapshot.

    Captures all four signals plus the composite score at one moment.
    """
    entropy: float = 0.0
    cadence: float = 0.0
    hurst: float = 0.5
    micro_timing: float = 0.0
    composite: float = 0.0
    phase: FlowPhase = FlowPhase.PRE_FLOW
    timestamp: float = 0.0


class FlowStateDetector(GrooveDetector):
    """
    Detects flow state by measuring action patterns.

    Extends GrooveDetector (which watches system-wide Φ) with
    player-centric signals: entropy of their actions, cadence
    regularity, Hurst exponent of their time series, and
    micro-timing consistency.

    The detector requires all four signals to converge before
    declaring flow. No single signal is sufficient. This prevents
    false positives from repetitive-but-unfocused behavior.

    Attributes:
        flow_threshold: Composite score above which flow is declared.
        deep_flow_threshold: Score for deep flow.
        pre_flow_threshold: Score to enter PRE_FLOW.
        min_flow_sustained: Readings needed to confirm flow.
        phase: Current flow phase.
        action_history: Recent actions for entropy calculation.
        timestamp_history: Recent action timestamps for cadence.
        time_series: Recent scalar values for Hurst computation.
        delta_history: Recent inter-action deltas for micro-timing.
        readings: History of FlowReading snapshots.
    """

    def __init__(
        self,
        governor: HarmonyGovernor,
        *,
        flow_threshold: float = 0.72,
        deep_flow_threshold: float = 0.88,
        pre_flow_threshold: float = 0.45,
        min_flow_sustained: int = 5,
        min_sustained_beats: int = 8,
        phi_variance_threshold: float = 0.15,
        max_history: int = 100,
    ):
        super().__init__(
            governor=governor,
            min_sustained_beats=min_sustained_beats,
            phi_variance_threshold=phi_variance_threshold,
        )
        self.flow_threshold = flow_threshold
        self.deep_flow_threshold = deep_flow_threshold
        self.pre_flow_threshold = pre_flow_threshold
        self.min_flow_sustained = min_flow_sustained
        self.max_history = max_history

        # Flow state machine
        self.phase: FlowPhase = FlowPhase.PRE_FLOW
        self._flow_sustained_count: int = 0
        self._flow_started_at: Optional[float] = None
        self._flow_ended_at: Optional[float] = None
        self._post_flow_grace: int = 0

        # Signal histories
        self.action_history: list[str] = []
        self.timestamp_history: list[float] = []
        self.time_series: list[float] = []
        self.delta_history: list[float] = []

        # Readings
        self.readings: list[FlowReading] = []

        # Last computed score
        self._last_score: float = 0.0
        self._last_reading: Optional[FlowReading] = None

    # ── Data Ingestion ───────────────────────────────────

    def record_action(
        self,
        action: str,
        timestamp: Optional[float] = None,
        value: Optional[float] = None,
    ) -> None:
        """
        Record a player action for flow analysis.

        Args:
            action: Action identifier (e.g., "place_block", "paint", "move").
            timestamp: When the action occurred (time.time() if None).
            value: Optional scalar value associated with the action
                   (e.g., quality score, placement accuracy).
        """
        if timestamp is None:
            timestamp = time.time()

        self.action_history.append(action)
        self.timestamp_history.append(timestamp)
        if value is not None:
            self.time_series.append(value)

        # Compute delta from previous action
        if len(self.timestamp_history) >= 2:
            delta = self.timestamp_history[-1] - self.timestamp_history[-2]
            self.delta_history.append(delta)

        # Trim histories
        for hist in (self.action_history, self.timestamp_history,
                     self.time_series, self.delta_history):
            if len(hist) > self.max_history:
                hist.pop(0)

    # ── Signal 1: Action Entropy ─────────────────────────

    def measure_action_entropy(self, recent_actions: Optional[list[str]] = None) -> float:
        """
        Measure Shannon entropy of recent actions.

        Low entropy = focused, repetitive-but-purposeful behavior.
        High entropy = scattered, switching between many things.

        Flow is characterized by LOW entropy. The player has converged
        on a task and is executing it with minimal context-switching.

        Returns:
            Normalized entropy 0.0-1.0 (0 = totally focused, 1 = chaotic).
        """
        actions = recent_actions if recent_actions is not None else self.action_history
        if len(actions) < 2:
            return 1.0  # Not enough data — assume unfocused

        # Count action frequencies
        counts: dict[str, int] = {}
        for a in actions:
            counts[a] = counts.get(a, 0) + 1

        n = len(actions)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by max possible entropy (log2 of unique actions)
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        if max_entropy == 0:
            return 0.0

        normalized = entropy / max_entropy
        return normalized

    # ── Signal 2: Cadence Regularity ─────────────────────

    def measure_cadence_regularity(self, action_timestamps: Optional[list[float]] = None) -> float:
        """
        Measure how regular the player's action cadence is.

        High regularity = the player has found their rhythm.
        Low regularity = timing is erratic, searching.

        Uses coefficient of variation (CV) of inter-action intervals.
        CV = std / mean. Lower CV = more regular.
        Returns 1.0 - CV (clamped to 0-1), so high = regular.

        Returns:
            Regularity score 0.0-1.0 (1 = perfectly metronomic).
        """
        timestamps = action_timestamps if action_timestamps is not None else self.timestamp_history
        if len(timestamps) < 3:
            return 0.0

        # Compute inter-action intervals
        intervals = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
        ]

        if not intervals or all(i == 0 for i in intervals):
            return 0.0

        mean = sum(intervals) / len(intervals)
        if mean == 0:
            return 0.0

        variance = sum((i - mean) ** 2 for i in intervals) / len(intervals)
        std = math.sqrt(variance)
        cv = std / mean

        # Convert CV to regularity: CV=0 → regularity=1, CV=1+ → regularity=0
        regularity = max(0.0, min(1.0, 1.0 - cv))
        return regularity

    # ── Signal 3: Hurst Exponent ─────────────────────────

    def measure_hurst_exponent(self, time_series: Optional[list[float]] = None) -> float:
        """
        Estimate the Hurst exponent via rescaled range analysis.

        H > 0.5: Persistent/trending — the player is building momentum.
        H ≈ 0.5: Random walk — no clear direction.
        H < 0.5: Mean-reverting — oscillating, not sustaining.

        Flow is characterized by H > 0.5: the player's quality or
        output is trending upward and sustaining. Each action builds
        on the last.

        Uses the simplified R/S method over the full series.
        For small series, returns 0.5 (neutral).

        Returns:
            Hurst exponent, approximately 0.0-1.0.
        """
        series = time_series if time_series is not None else self.time_series
        n = len(series)

        if n < 8:
            return 0.5  # Insufficient data — neutral

        # Compute mean
        mean = sum(series) / n

        # Cumulative deviation from mean
        cumdev = []
        running = 0.0
        for val in series:
            running += (val - mean)
            cumdev.append(running)

        # Range R = max(cumdev) - min(cumdev)
        r = max(cumdev) - min(cumdev)
        if r == 0:
            return 0.5  # No variation — neutral

        # Standard deviation S
        variance = sum((v - mean) ** 2 for v in series) / n
        s = math.sqrt(variance)
        if s == 0:
            return 0.5

        # Rescaled range
        rs = r / s

        # Estimate Hurst: R/S ≈ c * n^H, so H ≈ log(R/S) / log(n)
        if rs <= 0 or n <= 1:
            return 0.5

        hurst = math.log(rs) / math.log(n)

        # Clamp to plausible range
        return max(0.0, min(1.0, hurst))

    # ── Signal 4: Micro-Timing ───────────────────────────

    def measure_micro_timing(self, action_deltas: Optional[list[float]] = None) -> float:
        """
        Measure consistency of inter-action timing.

        In music, micro-timing is the tiny variations in when notes
        are played. Consistent micro-timing (slightly behind, slightly
        ahead, but consistently) is the signature of a player who
        has internalized the beat.

        We measure this as the inverse of the normalized mean absolute
        deviation of inter-action deltas. Consistent deltas → high score.

        Returns:
            Timing consistency 0.0-1.0 (1 = perfectly consistent).
        """
        deltas = action_deltas if action_deltas is not None else self.delta_history
        if len(deltas) < 3:
            return 0.0

        mean_delta = sum(deltas) / len(deltas)
        if mean_delta == 0:
            return 0.0

        # Mean absolute deviation normalized by mean
        mad = sum(abs(d - mean_delta) for d in deltas) / len(deltas)
        nmad = mad / mean_delta

        # Convert: nmad=0 → consistency=1, nmad>=1 → consistency=0
        consistency = max(0.0, min(1.0, 1.0 - nmad))
        return consistency

    # ── Composite Flow Score ─────────────────────────────

    def compute_flow_score(self) -> float:
        """
        Compute the composite flow score from all four signals.

        The score is a weighted combination:
            - Action entropy (inverted): 25%
            - Cadence regularity:        30%
            - Hurst exponent (>0.5 bias): 20%
            - Micro-timing:              25%

        Additionally, the system-wide groove state from the underlying
        GrooveDetector provides a multiplicative bonus/penalty: if the
        system is in the pocket, flow score gets a 1.1× multiplier
        (clamped to 1.0). If the system is disrupted, a 0.8× penalty.

        Returns:
            Flow score 0.0-1.0.
        """
        # Compute raw signals
        entropy = self.measure_action_entropy()
        cadence = self.measure_cadence_regularity()
        hurst = self.measure_hurst_exponent()
        micro = self.measure_micro_timing()

        # Invert entropy: low entropy = high focus
        focus = 1.0 - entropy

        # Hurst: bias toward >0.5 being good
        hurst_score = max(0.0, min(1.0, hurst))

        # Weighted combination
        score = (
            0.25 * focus
            + 0.30 * cadence
            + 0.20 * hurst_score
            + 0.25 * micro
        )

        # System groove modulation
        if self.in_groove:
            score = min(1.0, score * 1.1)
        elif self._state == GrooveState.DISRUPTED:
            score *= 0.8

        score = max(0.0, min(1.0, score))
        self._last_score = score

        # Store reading
        reading = FlowReading(
            entropy=entropy,
            cadence=cadence,
            hurst=hurst,
            micro_timing=micro,
            composite=score,
            phase=self.phase,
            timestamp=time.time(),
        )
        self._last_reading = reading
        self.readings.append(reading)
        if len(self.readings) > self.max_history:
            self.readings.pop(0)

        return score

    # ── State Machine Update ─────────────────────────────

    def update_flow(self, timestamp: Optional[float] = None) -> FlowPhase:
        """
        Advance the flow state machine.

        Call this after recording actions and updating the groove
        detector. This reads the composite flow score and transitions
        the phase accordingly.

        Transitions:
            PRE_FLOW → FLOW:        score ≥ flow_threshold, sustained
            FLOW → DEEP_FLOW:       score ≥ deep_flow_threshold, sustained
            FLOW/DEEP → POST_FLOW:  score drops below threshold
            POST_FLOW → RECOVERY:   after grace period or score drops further
            RECOVERY → PRE_FLOW:    score rising again

        Returns:
            Current FlowPhase.
        """
        if timestamp is None:
            timestamp = time.time()

        score = self.compute_flow_score()
        prev_phase = self.phase

        if prev_phase == FlowPhase.RECOVERY:
            # Wait for score to climb back
            if score >= self.pre_flow_threshold:
                self.phase = FlowPhase.PRE_FLOW
                self._flow_sustained_count = 0

        elif prev_phase == FlowPhase.POST_FLOW:
            self._post_flow_grace += 1
            if score >= self.flow_threshold:
                # Recovered back into flow
                self.phase = FlowPhase.FLOW
                self._flow_sustained_count = 1
                self._post_flow_grace = 0
            elif self._post_flow_grace > self.min_flow_sustained * 2:
                # Grace period expired
                self.phase = FlowPhase.RECOVERY
                self._flow_ended_at = timestamp
                self._post_flow_grace = 0

        elif prev_phase in (FlowPhase.PRE_FLOW,):
            if score >= self.flow_threshold:
                self._flow_sustained_count += 1
                if self._flow_sustained_count >= self.min_flow_sustained:
                    self.phase = FlowPhase.FLOW
                    self._flow_started_at = timestamp
                    self._flow_sustained_count = 0
            elif score >= self.pre_flow_threshold:
                # Still warming up
                pass
            else:
                self._flow_sustained_count = 0

        elif prev_phase == FlowPhase.FLOW:
            if score >= self.deep_flow_threshold:
                self._flow_sustained_count += 1
                if self._flow_sustained_count >= self.min_flow_sustained:
                    self.phase = FlowPhase.DEEP_FLOW
                    self._flow_sustained_count = 0
            elif score < self.flow_threshold:
                self.phase = FlowPhase.POST_FLOW
                self._post_flow_grace = 0
                if self._flow_started_at is not None:
                    self._flow_ended_at = timestamp
            else:
                self._flow_sustained_count = 0

        elif prev_phase == FlowPhase.DEEP_FLOW:
            if score < self.deep_flow_threshold:
                # Drop back to FLOW (not immediately to POST_FLOW)
                self.phase = FlowPhase.FLOW
                self._flow_sustained_count = 0
            if score < self.pre_flow_threshold:
                # Hard break from deep flow
                self.phase = FlowPhase.POST_FLOW
                self._post_flow_grace = 0
                if self._flow_started_at is not None:
                    self._flow_ended_at = timestamp

        return self.phase

    # ── Queries ──────────────────────────────────────────

    @property
    def in_flow(self) -> bool:
        """True if currently in FLOW or DEEP_FLOW."""
        return self.phase in (FlowPhase.FLOW, FlowPhase.DEEP_FLOW)

    @property
    def in_deep_flow(self) -> bool:
        return self.phase == FlowPhase.DEEP_FLOW

    @property
    def flow_duration(self) -> float:
        """Duration of current flow in seconds (0 if not flowing)."""
        if self._flow_started_at is None or not self.in_flow:
            return 0.0
        return time.time() - self._flow_started_at

    @property
    def last_flow_duration(self) -> float:
        """Duration of the most recent completed flow period."""
        if self._flow_started_at is not None and self._flow_ended_at is not None:
            return self._flow_ended_at - self._flow_started_at
        return 0.0

    @property
    def flow_score(self) -> float:
        """Most recently computed flow score."""
        return self._last_score

    @property
    def last_reading(self) -> Optional[FlowReading]:
        return self._last_reading

    def phase_distribution(self) -> dict[FlowPhase, float]:
        """Fraction of readings spent in each phase."""
        if not self.readings:
            return {FlowPhase.PRE_FLOW: 1.0}
        counts: dict[FlowPhase, int] = {}
        for r in self.readings:
            counts[r.phase] = counts.get(r.phase, 0) + 1
        total = len(self.readings)
        return {p: c / total for p, c in counts.items()}

    @property
    def flow_percentage(self) -> float:
        """Percentage of readings where the player was in flow."""
        dist = self.phase_distribution()
        return dist.get(FlowPhase.FLOW, 0.0) + dist.get(FlowPhase.DEEP_FLOW, 0.0)


# ────────────────────────────────────────────────────────────
# Flow State Protector
# ────────────────────────────────────────────────────────────

@dataclass
class ProtectiveAdjustment:
    """
    A single gentle adjustment to protect flow.

    These are imperceptible by design. The player should never
    notice them happening. They are the system holding its breath.
    """
    description: str
    bpm_delta: float = 0.0       # Tempo change (tiny)
    chatter_reduction: float = 0.0  # 0-1, how much to reduce agent dialogue
    ambient_dim: float = 0.0    # 0-1, how much to dim ambient elements
    friction_tolerance: float = 0.0  # How much to widen the deadband

    @property
    def is_gentle(self) -> bool:
        """Verify this adjustment is imperceptible."""
        return (
            abs(self.bpm_delta) <= 3.0
            and self.chatter_reduction <= 0.5
            and self.ambient_dim <= 0.3
            and self.friction_tolerance <= 0.3
        )


class FlowStateProtector:
    """
    Protects flow state with imperceptible adjustments.

    When flow is detected, the protector watches for rising friction
    that could break it. Instead of hard corrections (which would
    jar the player out of flow), it makes tiny, invisible adjustments:

    - Slow tempo by 2 BPM (widens the pocket slightly)
    - Reduce agent chatter (less noise in the channel)
    - Dim ambient elements slightly (less visual load)
    - Widen friction tolerance (let small bumps pass)

    The key principle: flow is a soap bubble. You don't grab it.
    You hold still and make the air gentler around it.

    Attributes:
        detector: The FlowStateDetector to monitor.
        tempo: The TempoMap to adjust.
        active: Whether protection is currently active.
        adjustments_made: History of applied adjustments.
        friction_window: Number of recent readings to watch for rising friction.
    """

    def __init__(
        self,
        detector: FlowStateDetector,
        tempo: TempoMap,
        *,
        friction_window: int = 5,
        rising_friction_threshold: float = 0.15,
    ):
        self.detector = detector
        self.tempo = tempo
        self.active = False
        self.protect_until: Optional[float] = None
        self.adjustments_made: list[ProtectiveAdjustment] = []
        self.friction_window = friction_window
        self.rising_friction_threshold = rising_friction_threshold

        # Pre-flow state for restoration
        self._pre_lock_bpm: Optional[float] = None
        self._pre_lock_target: Optional[float] = None
        self._original_chatter_level: float = 1.0
        self._original_ambient_level: float = 1.0

        # Current applied adjustments (cumulative)
        self._current = ProtectiveAdjustment(description="none")

    def detect_rising_friction(self) -> bool:
        """
        Early warning: is friction rising in a way that could break flow?

        Checks the slope of the governor's total friction over the
        last `friction_window` readings. If friction is trending up
        faster than `rising_friction_threshold` per reading, returns True.

        This is the canary. When it sings, the protector makes
        gentle adjustments before the friction gets bad enough to
        trigger the governor's deadband alarm.
        """
        profiles = list(self.detector.governor.profiles.values())
        if not profiles:
            return False

        # Get recent phi history from all agents
        recent_phis: list[float] = []
        for profile in profiles:
            recent = profile.phi_history[-self.friction_window:]
            if len(recent) >= 2:
                # Compute slope for this agent
                n = len(recent)
                # Simple linear regression slope
                x_mean = (n - 1) / 2
                y_mean = sum(recent) / n
                numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                if denominator > 0:
                    slope = numerator / denominator
                    recent_phis.append(slope)

        if not recent_phis:
            return False

        # If average slope is above threshold, friction is rising
        avg_slope = sum(recent_phis) / len(recent_phis)
        return avg_slope > self.rising_friction_threshold

    def engage(self) -> Optional[ProtectiveAdjustment]:
        """
        Engage flow protection.

        Locks tempo and applies gentle adjustments. This should be
        called when flow is first detected.
        """
        if self.active:
            return None

        # Save pre-flow state
        self._pre_lock_bpm = self.tempo.bpm
        self._pre_lock_target = self.tempo.target_bpm

        # Lock tempo — don't break the groove
        self.tempo.lock(reason="flow_protection")

        adjustment = ProtectiveAdjustment(
            description="Flow protection engaged: tempo locked",
            bpm_delta=0.0,
            chatter_reduction=0.3,
            ambient_dim=0.15,
            friction_tolerance=0.1,
        )

        self._current = adjustment
        self.active = True
        self.adjustments_made.append(adjustment)
        return adjustment

    def gentle_adjust(self) -> Optional[ProtectiveAdjustment]:
        """
        Make a single imperceptible adjustment to protect flow.

        Called when rising friction is detected but flow hasn't
        broken yet. The adjustment is tiny — the player should
        never notice it happened.

        Returns the adjustment made, or None if no adjustment was needed.
        """
        if not self.active:
            return None

        if not self.detect_rising_friction():
            return None

        # Make a tiny tempo adjustment: slow by ~2 BPM
        adj = ProtectiveAdjustment(
            description="Gentle friction response: slowing tempo by 2 BPM",
            bpm_delta=-2.0,
            chatter_reduction=0.1,
            ambient_dim=0.05,
            friction_tolerance=0.05,
        )

        # Apply (nudge tempo — but since it's locked, we adjust the lock point)
        # We unlock briefly, nudge, re-lock
        self.tempo.unlock()
        self.tempo.nudge(adj.bpm_delta)
        self.tempo.lock(reason="flow_protection_adjusted")

        # Widen deadbands slightly
        for profile in self.detector.governor.profiles.values():
            profile.current_deadband *= (1.0 + adj.friction_tolerance)

        # Accumulate
        self._current = ProtectiveAdjustment(
            description=f"Cumulative: {self._current.description} + {adj.description}",
            bpm_delta=self._current.bpm_delta + adj.bpm_delta,
            chatter_reduction=min(0.5, self._current.chatter_reduction + adj.chatter_reduction),
            ambient_dim=min(0.3, self._current.ambient_dim + adj.ambient_dim),
            friction_tolerance=min(0.3, self._current.friction_tolerance + adj.friction_tolerance),
        )

        self.adjustments_made.append(adj)
        return adj

    def protect_for(self, seconds: float) -> ProtectiveAdjustment:
        """
        Maintain protective conditions for a given duration.

        During this time, the protector will:
        1. Keep tempo locked
        2. Check for rising friction each tick
        3. Apply gentle adjustments as needed
        4. Automatically disengage after the duration expires

        Args:
            seconds: How long to maintain protection.

        Returns:
            The initial protective adjustment.
        """
        self.protect_until = time.time() + seconds
        initial = self.engage()
        return initial if initial else self._current

    def tick(self) -> Optional[ProtectiveAdjustment]:
        """
        Called each beat/frame to maintain protection.

        Checks if protection should still be active, and makes
        gentle adjustments if friction is rising.

        Returns any adjustment made, or None.
        """
        if not self.active:
            return None

        # Check if protection period has expired
        if self.protect_until is not None and time.time() >= self.protect_until:
            self.disengage()
            return None

        # Check if flow has broken
        if not self.detector.in_flow and self.detector.phase != FlowPhase.POST_FLOW:
            self.disengage()
            return None

        # Check for rising friction and adjust
        return self.gentle_adjust()

    def disengage(self) -> ProtectiveAdjustment:
        """
        Release flow protection.

        Unlocks tempo and restores pre-flow conditions gradually.
        The restoration is also gentle — don't snap back.
        """
        self.tempo.unlock()

        # Restore tempo target gradually (don't snap)
        if self._pre_lock_target is not None:
            # Move halfway back this tick
            current = self.tempo.target_bpm
            target = self._pre_lock_target
            restored = current + (target - current) * 0.5
            self.tempo.set_target(restored)

        adjustment = ProtectiveAdjustment(
            description="Flow protection disengaged: restoring normal tempo",
            bpm_delta=0.0,
            chatter_reduction=0.0,
            ambient_dim=0.0,
            friction_tolerance=0.0,
        )

        self._current = adjustment
        self.active = False
        self.protect_until = None
        self.adjustments_made.append(adjustment)
        return adjustment

    @property
    def adjustment_count(self) -> int:
        """Total adjustments made in this protection session."""
        return len([a for a in self.adjustments_made if a.description != "none"])

    @property
    def total_tempo_change(self) -> float:
        """Total BPM delta applied during protection."""
        return sum(a.bpm_delta for a in self.adjustments_made)


# ────────────────────────────────────────────────────────────
# Flow State Journal
# ────────────────────────────────────────────────────────────

@dataclass
class FlowSession:
    """
    A single recorded flow session.
    """
    started_at: float
    ended_at: Optional[float] = None
    duration: float = 0.0
    peak_score: float = 0.0
    avg_score: float = 0.0
    phase_reached: FlowPhase = FlowPhase.FLOW
    trigger_conditions: dict = field(default_factory=dict)
    player_state: dict = field(default_factory=dict)
    end_trigger: str = ""

    @property
    def is_complete(self) -> bool:
        return self.ended_at is not None

    @property
    def duration_minutes(self) -> float:
        return self.duration / 60.0


class FlowStateJournal:
    """
    Remembers flow — what triggered it, how long it lasted, what broke it.

    The journal is the system's long-term memory of flow. Over time,
    patterns emerge: this player flows best in the morning, with ambient
    audio at 40%, after a 5-minute warm-up of repetitive placement tasks.

    The journal does not optimize for flow. It witnesses it. The patterns
    it finds are offered to the player, not imposed on them.

    "The best infrastructure is the infrastructure you stop noticing."

    Attributes:
        sessions: All recorded flow sessions.
        current_session: The in-progress session, if any.
    """

    def __init__(self):
        self.sessions: list[FlowSession] = []
        self.current_session: Optional[FlowSession] = None
        self._score_history: list[float] = []

    def record_flow_start(
        self,
        timestamp: float,
        conditions: dict,
        player_state: dict,
    ) -> FlowSession:
        """
        Record the beginning of a flow session.

        Args:
            timestamp: When flow started.
            conditions: System conditions at flow start (tempo, friction, etc.).
            player_state: What the player was doing (action type, position, etc.).

        Returns:
            The created FlowSession.
        """
        session = FlowSession(
            started_at=timestamp,
            trigger_conditions=conditions,
            player_state=player_state,
        )
        self.current_session = session
        self._score_history = []
        return session

    def record_flow_score(self, score: float) -> None:
        """Record a flow score during an active session."""
        if self.current_session is not None:
            self._score_history.append(score)
            self.current_session.peak_score = max(self.current_session.peak_score, score)

    def record_flow_end(
        self,
        timestamp: float,
        trigger: str,
        phase_reached: FlowPhase = FlowPhase.FLOW,
    ) -> Optional[FlowSession]:
        """
        Record the end of a flow session.

        Args:
            timestamp: When flow ended.
            trigger: What broke the flow ("friction_spike", "interruption",
                     "natural_decay", "unknown").
            phase_reached: The deepest phase reached during this session.

        Returns:
            The completed FlowSession, or None if no session was active.
        """
        if self.current_session is None:
            return None

        session = self.current_session
        session.ended_at = timestamp
        session.duration = timestamp - session.started_at
        session.end_trigger = trigger
        session.phase_reached = phase_reached

        if self._score_history:
            session.avg_score = sum(self._score_history) / len(self._score_history)

        self.sessions.append(session)
        self.current_session = None
        self._score_history = []
        return session

    def get_patterns(self) -> dict:
        """
        Analyze flow patterns across all recorded sessions.

        Returns what conditions commonly precede flow for this player:
        - Most common trigger actions
        - Average flow duration
        - Time-of-day patterns
        - Friction level at flow onset
        - Most common flow-breaking triggers

        Returns:
            Dict of pattern analysis.
        """
        if not self.sessions:
            return {
                "total_sessions": 0,
                "avg_duration_minutes": 0.0,
                "flow_rate": 0.0,
            }

        total = len(self.sessions)
        durations = [s.duration for s in self.sessions if s.is_complete]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # Most common trigger actions
        action_counts: dict[str, int] = {}
        for session in self.sessions:
            action = session.player_state.get("primary_action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

        # Most common end triggers
        end_counts: dict[str, int] = {}
        for session in self.sessions:
            if session.end_trigger:
                end_counts[session.end_trigger] = end_counts.get(session.end_trigger, 0) + 1

        # Peak scores
        peak_scores = [s.peak_score for s in self.sessions]
        avg_peak = sum(peak_scores) / len(peak_scores) if peak_scores else 0.0

        # Phase distribution
        phase_counts: dict[str, int] = {}
        for session in self.sessions:
            phase_name = session.phase_reached.name
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1

        # Common conditions
        condition_keys: set[str] = set()
        for session in self.sessions:
            condition_keys.update(session.trigger_conditions.keys())

        common_conditions: dict[str, float] = {}
        for key in condition_keys:
            values = [
                session.trigger_conditions[key]
                for session in self.sessions
                if key in session.trigger_conditions
                and isinstance(session.trigger_conditions[key], (int, float))
            ]
            if values:
                common_conditions[key] = sum(values) / len(values)

        return {
            "total_sessions": total,
            "avg_duration_minutes": avg_duration / 60.0,
            "avg_peak_score": avg_peak,
            "most_common_actions": sorted(
                action_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "most_common_end_triggers": sorted(
                end_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "phase_distribution": {
                k: v / total for k, v in phase_counts.items()
            },
            "common_conditions": common_conditions,
        }

    def export_session(self) -> dict:
        """
        Export the flow moments of all sessions as a musical timeline.

        Each flow session becomes a "movement" in the timeline, with:
        - Start time (as offset from first session)
        - Duration (in beats, assuming 120 BPM default)
        - Peak intensity (0-1, mapped to dynamics)
        - Phase reached (mapped to movement character)

        The result can be rendered as a visual timeline or even
        played back as a musical approximation of the session's
        emotional arc.

        Returns:
            Dict with timeline data.
        """
        if not self.sessions:
            return {
                "movements": [],
                "total_duration_beats": 0,
                "bpm": 120,
            }

        bpm = 120  # Default rendering tempo
        beat_duration = 60.0 / bpm

        if not self.sessions:
            base_time = 0.0
        else:
            base_time = self.sessions[0].started_at

        movements = []
        for i, session in enumerate(self.sessions):
            start_offset = (session.started_at - base_time) / beat_duration
            duration_beats = session.duration / beat_duration

            # Map phase to character
            phase_character = {
                FlowPhase.FLOW: "allegro",
                FlowPhase.DEEP_FLOW: "andante",
                FlowPhase.POST_FLOW: "decrescendo",
            }.get(session.phase_reached, "moderato")

            # Map peak score to dynamic marking
            if session.peak_score >= 0.9:
                dynamic = "fortissimo"
            elif session.peak_score >= 0.75:
                dynamic = "forte"
            elif session.peak_score >= 0.6:
                dynamic = "mezzo-forte"
            else:
                dynamic = "piano"

            movements.append({
                "index": i,
                "start_beat": round(start_offset, 2),
                "duration_beats": round(duration_beats, 2),
                "peak_intensity": round(session.peak_score, 3),
                "avg_intensity": round(session.avg_score, 3),
                "phase": session.phase_reached.name,
                "character": phase_character,
                "dynamic": dynamic,
                "end_trigger": session.end_trigger,
                "primary_action": session.player_state.get("primary_action", "unknown"),
            })

        total_beats = max(
            (m["start_beat"] + m["duration_beats"]) for m in movements
        ) if movements else 0

        return {
            "movements": movements,
            "total_duration_beats": round(total_beats, 2),
            "bpm": bpm,
            "session_count": len(movements),
        }

    @property
    def total_flow_time(self) -> float:
        """Total time spent in flow across all sessions (seconds)."""
        return sum(s.duration for s in self.sessions if s.is_complete)

    @property
    def longest_flow(self) -> float:
        """Longest single flow session (seconds)."""
        if not self.sessions:
            return 0.0
        return max(s.duration for s in self.sessions if s.is_complete)

    @property
    def average_peak_score(self) -> float:
        """Average peak flow score across sessions."""
        if not self.sessions:
            return 0.0
        return sum(s.peak_score for s in self.sessions) / len(self.sessions)
