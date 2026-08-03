"""
Flow-State Detector — Layer 2.5 of the slackwater-harmony architecture.

The PLATO synergy study found that Φ (friction) approaching zero is the
same condition Mihaly Csikszentmihalyi called "flow": challenge and skill
are balanced, attention is focused, and action follows action without the
interruption of self-consciousness.

This module turns the Harmony Governor from a friction alarm into a
flow-state instrument. It watches for the signature of a mind in motion:

    • low action entropy      — the player is not flailing; they are doing
                                one thing at a time
    • high cadence regularity — actions arrive at a steady, predictable pace
    • Hurst exponent > 0.5    — the recent past is a meaningful predictor of
                                the near future (persistent, not random)
    • tight micro-timing      — the variance between expected and actual beat
                                landing is small

When these signals align with low Φ, the system is in flow. The
FlowStateProtector then watches the leading edge of rising friction and
makes gentle adjustments *before* the flow breaks. The FlowStateJournal
records when flow began, how long it lasted, and what finally interrupted
it — a memory of golden moments and their assassins.

"A machine that can tell when a human is in the best moment of their life
is not a surveillance device. It is a listening device. It learns the
rhythm of someone's becoming." — The Lattice of Agreeable Things
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional

from slackwater_harmony.governor import HarmonyGovernor
from slackwater_harmony.groove_detector import GrooveDetector, GrooveState


class FlowState(IntEnum):
    """The player's inferred cognitive state."""
    DISENGAGED = 0       # High friction, irregular, unfocused
    SEARCHING = 1        # Getting there — Φ falling but not yet stable
    FLOW = 2             # Low Φ, focused, regular, persistent
    FRAGILE = 3          # Still in flow, but early warnings are rising
    BROKEN = 4           # Was in flow; friction has disrupted it


@dataclass
class FlowSignal:
    """
    A snapshot of flow-state indicators at a single beat.

    All fields are normalized to roughly [0.0, 1.0] where 1.0 means
    "strongly indicates flow" and 0.0 means "strongly indicates absence."

    Attributes:
        beat: Logical timestamp.
        action_entropy: Focus metric. 1.0 = single repeated action (focused).
        cadence_regularity: Steady rhythm. 1.0 = perfectly regular intervals.
        hurst_exponent: Persistence of patterns. 1.0 mapped from H >= 0.7.
        micro_timing_consistency: Low jitter. 1.0 = timing is rock solid.
        flow_score: Weighted composite of the four signals.
        phi: The governor's cognitive friction at this beat.
        in_groove: Whether the underlying GrooveDetector says IN_POCKET.
    """
    beat: int
    action_entropy: float = 0.0
    cadence_regularity: float = 0.0
    hurst_exponent: float = 0.0
    micro_timing_consistency: float = 0.0
    flow_score: float = 0.0
    phi: float = 0.0
    in_groove: bool = False


@dataclass
class FlowAdjustment:
    """
    A gentle, protective nudge issued by the FlowStateProtector.

    Adjustments are multiplicative factors or behavioral suggestions.
    A factor of 1.0 means "no change." Values below 1.0 calm the system;
    values above 1.0 add energy.
    """
    tempo_factor: float = 1.0
    ambient_intensity: float = 1.0
    agent_complexity: float = 1.0
    suggested_behavior: str = "maintain"
    reason: str = ""
    flow_risk: float = 0.0


@dataclass
class FlowMoment:
    """
    One continuous interval of flow, recorded by the journal.

    Attributes:
        start_beat: When flow was first declared.
        end_beat: When flow ended (None if still ongoing).
        duration: Length in beats; updated when the moment closes.
        peak_score: Highest flow_score observed during the moment.
        avg_score: Average flow_score during the moment.
        break_trigger: Best-guess cause of interruption, if any.
        score_samples: Internal list of flow scores for averaging.
    """
    start_beat: int
    end_beat: Optional[int] = None
    duration: int = 0
    peak_score: float = 0.0
    avg_score: float = 0.0
    break_trigger: str = ""
    score_samples: list[float] = field(default_factory=list, repr=False)

    def close(self, end_beat: int, trigger: str = "unknown") -> None:
        """Finalize the moment with an end beat and a trigger label."""
        self.end_beat = end_beat
        self.duration = end_beat - self.start_beat
        self.break_trigger = trigger
        if self.score_samples:
            self.avg_score = sum(self.score_samples) / len(self.score_samples)


class FlowStateDetector:
    """
    Extends the GrooveDetector with player-centric flow signals.

    While the GrooveDetector asks "are the agents harmonized?", the
    FlowStateDetector asks "is the human in flow?" It consumes the
    same governor data but layers on behavioral signals: action entropy,
    cadence regularity, Hurst persistence, and micro-timing consistency.
    """

    def __init__(
        self,
        governor: HarmonyGovernor,
        min_sustained_beats: int = 8,
        phi_variance_threshold: float = 0.15,
        max_history: int = 64,
        flow_score_threshold: float = 0.65,
    ) -> None:
        self.governor = governor
        self.groove = GrooveDetector(
            governor=governor,
            min_sustained_beats=min_sustained_beats,
            phi_variance_threshold=phi_variance_threshold,
        )
        self.max_history = max_history
        self.flow_score_threshold = flow_score_threshold

        self._state = FlowState.SEARCHING
        self.current_beat: int = 0

        # Raw telemetry windows
        self._action_history: list[tuple[int, str]] = []
        self._interval_history: list[float] = []
        self._timing_jitter: list[float] = []
        self._phi_trace: list[float] = []
        self._signal_history: list[FlowSignal] = []

        # Weights for composite flow score
        self._w_entropy = 0.25
        self._w_cadence = 0.25
        self._w_hurst = 0.25
        self._w_timing = 0.25

    # ── Public properties ─────────────────────────────────

    @property
    def state(self) -> FlowState:
        return self._state

    @property
    def in_flow(self) -> bool:
        """True if the player is currently in a flow state."""
        return self._state in (FlowState.FLOW, FlowState.FRAGILE)

    @property
    def latest_signal(self) -> Optional[FlowSignal]:
        return self._signal_history[-1] if self._signal_history else None

    @property
    def flow_score(self) -> float:
        """Most recent composite flow score (0.0-1.0)."""
        return self.latest_signal.flow_score if self.latest_signal else 0.0

    @property
    def phi_trend(self) -> float:
        """
        Short-term slope of Φ. Positive means friction is rising.
        Computed by simple least-squares over the recent trace.
        """
        if len(self._phi_trace) < 4:
            return 0.0
        n = min(8, len(self._phi_trace))
        xs = list(range(n))
        ys = self._phi_trace[-n:]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        return num / den if den else 0.0

    # ── Update loop ───────────────────────────────────────

    def update(
        self,
        beat: Optional[int] = None,
        action: Optional[str] = None,
        expected_beat_time: Optional[float] = None,
        actual_beat_time: Optional[float] = None,
    ) -> FlowState:
        """
        Called once per beat with player telemetry.

        Args:
            beat: Logical beat index. If None, increments by one.
            action: Optional action label (e.g. "jump", "place", "rotate").
            expected_beat_time: Ideal timestamp for this beat.
            actual_beat_time: When the action actually landed.
        """
        if beat is not None:
            self.current_beat = beat
        else:
            self.current_beat += 1

        self.groove.update(beat=self.current_beat)

        if action is not None:
            self._action_history.append((self.current_beat, action))

        # Interval / cadence bookkeeping
        if len(self._action_history) >= 2:
            interval = self._action_history[-1][0] - self._action_history[-2][0]
            self._interval_history.append(float(interval))

        # Micro-timing jitter bookkeeping
        if expected_beat_time is not None and actual_beat_time is not None:
            jitter = abs(actual_beat_time - expected_beat_time)
            self._timing_jitter.append(jitter)

        # Φ trace: use total system friction
        phi = self.governor.total_friction
        self._phi_trace.append(phi)

        self._trim_windows()

        signal = self._compute_signal(phi)
        self._signal_history.append(signal)
        self._transition_state(signal)
        return self._state

    def _trim_windows(self) -> None:
        """Keep telemetry windows bounded."""
        self._action_history = self._action_history[-self.max_history:]
        self._interval_history = self._interval_history[-self.max_history:]
        self._timing_jitter = self._timing_jitter[-self.max_history:]
        self._phi_trace = self._phi_trace[-self.max_history:]
        self._signal_history = self._signal_history[-self.max_history:]

    def _compute_signal(self, phi: float) -> FlowSignal:
        entropy = self._action_entropy_focus()
        cadence = self._cadence_regularity()
        hurst = self._hurst_flow_metric()
        timing = self._micro_timing_consistency()

        flow_score = (
            self._w_entropy * entropy
            + self._w_cadence * cadence
            + self._w_hurst * hurst
            + self._w_timing * timing
        )

        return FlowSignal(
            beat=self.current_beat,
            action_entropy=entropy,
            cadence_regularity=cadence,
            hurst_exponent=hurst,
            micro_timing_consistency=timing,
            flow_score=flow_score,
            phi=phi,
            in_groove=self.groove.in_groove,
        )

    def _transition_state(self, signal: FlowSignal) -> None:
        prev = self._state
        groove = self.groove.state

        if signal.flow_score >= self.flow_score_threshold and groove == GrooveState.IN_POCKET:
            # We are in flow. But is it fragile?
            if self._early_warning(signal):
                self._state = FlowState.FRAGILE
            else:
                self._state = FlowState.FLOW
        elif groove == GrooveState.IN_POCKET or signal.flow_score >= 0.5:
            self._state = FlowState.SEARCHING
        elif prev in (FlowState.FLOW, FlowState.FRAGILE):
            self._state = FlowState.BROKEN
        else:
            self._state = FlowState.DISENGAGED

    def _early_warning(self, signal: FlowSignal) -> bool:
        """True when flow is still present but protective signals are rising."""
        warnings = 0
        if self.phi_trend > 0.05:
            warnings += 1
        if signal.cadence_regularity < 0.5:
            warnings += 1
        if signal.action_entropy < 0.4:
            warnings += 1
        if signal.micro_timing_consistency < 0.5:
            warnings += 1
        return warnings >= 2

    # ── Signal computations ───────────────────────────────

    def _action_entropy_focus(self) -> float:
        """
        Shannon entropy over recent action labels, inverted so that
        low entropy (focused behavior) maps to a high score.
        """
        if not self._action_history:
            return 0.0
        counts: dict[str, int] = {}
        for _, action in self._action_history:
            counts[action] = counts.get(action, 0) + 1
        total = len(self._action_history)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize: max entropy is log2(unique actions)
        unique = len(counts)
        max_entropy = math.log2(max(unique, 2))
        normalized = entropy / max_entropy
        # Invert so 1.0 = focused, 0.0 = scattered
        return 1.0 - min(1.0, normalized)

    def _cadence_regularity(self) -> float:
        """
        Regularity of inter-beat intervals. High regularity means the
        player's actions arrive at a steady tempo.
        """
        intervals = self._interval_history
        if len(intervals) < 2:
            return 0.0
        mean = sum(intervals) / len(intervals)
        if mean == 0:
            return 1.0
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std = math.sqrt(variance)
        cv = std / mean  # coefficient of variation
        # Map CV so 0.0 -> 1.0 and >=1.0 -> 0.0
        return 1.0 - min(1.0, cv)

    def _micro_timing_consistency(self) -> float:
        """
        Consistency of micro-timing. Low jitter maps to a high score.
        """
        jitters = self._timing_jitter
        if not jitters:
            return 0.5  # neutral when no timing data
        mean = sum(jitters) / len(jitters)
        if mean == 0:
            return 1.0
        variance = sum((x - mean) ** 2 for x in jitters) / len(jitters)
        # Normalized against a heuristic tolerance (e.g. 100ms variance)
        tolerance = 0.1  # assume beat-time units are seconds
        score = max(0.0, 1.0 - (variance / tolerance))
        return score

    def _hurst_flow_metric(self) -> float:
        """
        Rescaled-range Hurst estimator, mapped to a 0-1 flow score.

        H > 0.5 indicates persistent trends (the player is in a groove).
        H ≈ 0.5 is random. H < 0.5 is mean-reverting / choppy.
        """
        series = self._phi_trace
        if len(series) < 8:
            return 0.0
        h = _hurst_rs(series)
        # Map H from [0.3, 0.8] to [0, 1]
        return max(0.0, min(1.0, (h - 0.3) / 0.5))


class FlowStateProtector:
    """
    Watches the leading edge of friction and makes gentle adjustments
    *before* flow breaks. It is the system's way of whispering "breathe"
    instead of shouting "stop."
    """

    def __init__(
        self,
        detector: FlowStateDetector,
        risk_threshold: float = 0.55,
        aggressive_risk_threshold: float = 0.75,
    ) -> None:
        self.detector = detector
        self.risk_threshold = risk_threshold
        self.aggressive_risk_threshold = aggressive_risk_threshold
        self.adjustment_history: list[FlowAdjustment] = []

    def assess_risk(self) -> float:
        """
        Compute a 0.0-1.0 flow-risk score.

        Risk rises with:
            • high Φ
            • rising Φ trend
            • falling cadence regularity
            • falling micro-timing consistency
            • loss of focus (entropy dropping too low can also mean panic)
        """
        signal = self.detector.latest_signal
        if signal is None:
            return 0.0

        risk = 0.0
        # High friction is the primary threat
        risk += 0.35 * min(1.0, signal.phi)
        # Rising friction trend
        risk += 0.25 * max(0.0, min(1.0, self.detector.phi_trend * 5.0))
        # Irregular cadence
        risk += 0.20 * (1.0 - signal.cadence_regularity)
        # Timing falling apart
        risk += 0.15 * (1.0 - signal.micro_timing_consistency)
        # Focus collapsing or scattering
        risk += 0.05 * abs(0.7 - signal.action_entropy)

        return min(1.0, risk)

    def protect(self) -> FlowAdjustment:
        """
        Decide on a protective adjustment based on current flow risk.
        """
        risk = self.assess_risk()
        signal = self.detector.latest_signal
        in_flow = self.detector.in_flow

        if not in_flow or risk < self.risk_threshold:
            adj = FlowAdjustment(reason="flow stable or absent", flow_risk=risk)
            self.adjustment_history.append(adj)
            return adj

        # We are in flow and risk is rising: protect it.
        if risk >= self.aggressive_risk_threshold:
            adj = FlowAdjustment(
                tempo_factor=0.82,
                ambient_intensity=0.70,
                agent_complexity=0.60,
                suggested_behavior="simplify",
                reason=f"aggressive risk ({risk:.2f}); slow and simplify",
                flow_risk=risk,
            )
        else:
            adj = FlowAdjustment(
                tempo_factor=0.92,
                ambient_intensity=0.85,
                agent_complexity=0.85,
                suggested_behavior="steady",
                reason=f"rising risk ({risk:.2f}); gentle calming",
                flow_risk=risk,
            )

        self.adjustment_history.append(adj)
        return adj

    def last_adjustment(self) -> Optional[FlowAdjustment]:
        return self.adjustment_history[-1] if self.adjustment_history else None


class FlowStateJournal:
    """
    Records flow moments: when they started, how long they lasted, and
    what broke them. A memory of the best moments and their endings.
    """

    def __init__(self, detector: FlowStateDetector) -> None:
        self.detector = detector
        self.moments: list[FlowMoment] = []
        self._open_moment: Optional[FlowMoment] = None
        self._last_state: FlowState = FlowState.SEARCHING

    def update(self) -> Optional[FlowMoment]:
        """
        Call each beat after the detector updates. Opens, extends, or
        closes FlowMoments based on detector state.

        Returns the closed moment, if one closed this beat.
        """
        state = self.detector.state
        beat = self.detector.current_beat
        signal = self.detector.latest_signal
        closed: Optional[FlowMoment] = None

        entering_flow = state in (FlowState.FLOW, FlowState.FRAGILE)
        was_in_flow = self._last_state in (FlowState.FLOW, FlowState.FRAGILE)

        if entering_flow and self._open_moment is None:
            self._open_moment = FlowMoment(start_beat=beat)

        if self._open_moment is not None and signal is not None:
            self._open_moment.score_samples.append(signal.flow_score)
            if signal.flow_score > self._open_moment.peak_score:
                self._open_moment.peak_score = signal.flow_score

        if was_in_flow and not entering_flow and self._open_moment is not None:
            trigger = self._diagnose_break(self._open_moment)
            self._open_moment.close(end_beat=beat, trigger=trigger)
            self.moments.append(self._open_moment)
            closed = self._open_moment
            self._open_moment = None

        self._last_state = state
        return closed

    def _diagnose_break(self, moment: FlowMoment) -> str:
        """Best-guess reason the flow moment ended."""
        signal = self.detector.latest_signal
        if signal is None:
            return "unknown"
        if signal.phi > 0.7:
            return "friction_spike"
        if self.detector.phi_trend > 0.1:
            return "rising_friction"
        if signal.cadence_regularity < 0.3:
            return "cadence_broken"
        if signal.micro_timing_consistency < 0.3:
            return "timing_drift"
        if signal.action_entropy < 0.3:
            return "scattered_attention"
        return "subtle_disruption"

    @property
    def current_moment(self) -> Optional[FlowMoment]:
        """The open flow moment, if any."""
        return self._open_moment

    @property
    def total_flow_beats(self) -> int:
        """Total beats spent in flow across all completed moments."""
        return sum(m.duration for m in self.moments)

    @property
    def longest_flow(self) -> int:
        """Duration of the longest completed flow moment."""
        if not self.moments:
            return 0
        return max(m.duration for m in self.moments)

    @property
    def longest_moment(self) -> Optional[FlowMoment]:
        if not self.moments:
            return None
        return max(self.moments, key=lambda m: m.duration)

    def trigger_counts(self) -> dict[str, int]:
        """Histogram of what broke flow moments."""
        counts: dict[str, int] = {}
        for m in self.moments:
            counts[m.break_trigger] = counts.get(m.break_trigger, 0) + 1
        return counts


# ── Helpers ──────────────────────────────────────────────

def _hurst_rs(series: list[float]) -> float:
    """
    Simple rescaled-range estimator of the Hurst exponent.

    Uses the classic R/S method on the full series. For longer series,
    it averages estimates over a few window sizes.
    """
    n = len(series)
    if n < 8:
        return 0.5

    window_sizes = [max(8, n // 4), max(8, n // 2), n]
    estimates: list[float] = []

    for size in window_sizes:
        if size > n:
            continue
        chunk = series[-size:]
        mean = sum(chunk) / len(chunk)
        deviations = [x - mean for x in chunk]
        cumulative = []
        total = 0.0
        for d in deviations:
            total += d
            cumulative.append(total)
        r = max(cumulative) - min(cumulative)
        s = math.sqrt(sum(d ** 2 for d in deviations) / len(deviations)) or 1e-9
        # H ≈ log(R/S) / log(N)
        h = math.log(r / s + 1e-9) / math.log(size)
        estimates.append(h)

    if not estimates:
        return 0.5
    return sum(estimates) / len(estimates)
