"""
Groove Detector — detects when the system is "in the pocket."

When Φ is low across ALL agents simultaneously — not just one, but all
of them — the system has found a groove. Predictions match reality.
The harmony governor is quiet. The executive sleeps.

This is the moment in music where the band locks in. Nobody calls it.
Nobody announces it. The rhythm section breathes together and the
tune plays itself.

The groove detector watches for this state and can trigger rewards,
narrative beats, or simply let the silence be golden.

"In the pocket" is not the absence of friction. It is the presence
of alignment. The agents are not idle — they are working, but their
work flows. The intervals between them are consonant. The lattice
has found its minimum-energy configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from slackwater_harmony.governor import HarmonyGovernor


class GrooveState(IntEnum):
    """The system's current groove status."""
    SEARCHING = 0    # Φ is high or variable — system is looking for it
    SETTLING = 1     # Φ is dropping — getting close
    IN_POCKET = 2    # Φ is low and stable across all agents — locked in
    DISRUPTED = 3    # Was in pocket, now broken — friction spike


@dataclass
class GrooveDetector:
    """
    Watches the Harmony Governor for groove states.

    A groove requires:
    1. ALL agents below their deadbands (harmonized)
    2. Sustained for `min_sustained_beats` consecutive observations
    3. Low variance in Φ (not oscillating near the threshold)

    Attributes:
        governor: The HarmonyGovernor to monitor.
        min_sustained_beats: How long Φ must stay low before declaring groove.
        phi_variance_threshold: Max Φ variance allowed during groove.
        groove_started_at: Beat when groove started (None if not grooving).
        longest_groove: Longest sustained groove in beats.
    """

    governor: HarmonyGovernor
    min_sustained_beats: int = 8
    phi_variance_threshold: float = 0.15
    groove_started_at: Optional[int] = None
    longest_groove: int = 0
    current_beat: int = 0

    _state: GrooveState = GrooveState.SEARCHING
    _sustained_count: int = 0
    _state_history: list[GrooveState] = field(default_factory=list)

    @property
    def state(self) -> GrooveState:
        return self._state

    @property
    def in_groove(self) -> bool:
        """True if the system is currently in the pocket."""
        return self._state == GrooveState.IN_POCKET

    @property
    def groove_duration(self) -> int:
        """How many beats the current groove has lasted (0 if not grooving)."""
        if self.groove_started_at is None:
            return 0
        return self.current_beat - self.groove_started_at

    # ── Update Loop ──────────────────────────────────────

    def update(self, beat: Optional[int] = None) -> GrooveState:
        """
        Called once per beat to assess groove status.

        Reads the governor's current state and transitions accordingly.

        State transitions:
            SEARCHING → SETTLING (all agents below deadband, not yet sustained)
            SETTLING → IN_POCKET (sustained for min_sustained_beats)
            IN_POCKET → DISRUPTED (any agent exceeds deadband)
            DISRUPTED → SEARCHING (after acknowledging the disruption)
            * → SEARCHING (friction too high)
        """
        if beat is not None:
            self.current_beat = beat
        else:
            self.current_beat += 1

        all_harmonized = self.governor.is_harmonized
        phi_var = self._system_phi_variance()

        prev_state = self._state

        if all_harmonized:
            # Check variance — need stability, not just low Φ
            if phi_var <= self.phi_variance_threshold:
                self._sustained_count += 1
            else:
                # High variance — reset
                self._sustained_count = 0

            if self._sustained_count >= self.min_sustained_beats:
                if prev_state != GrooveState.IN_POCKET:
                    self.groove_started_at = self.current_beat
                self._state = GrooveState.IN_POCKET
                # Track longest groove
                if self.groove_duration > self.longest_groove:
                    self.longest_groove = self.groove_duration
            elif self._sustained_count > 0:
                self._state = GrooveState.SETTLING
            else:
                self._state = GrooveState.SEARCHING
        else:
            # Not harmonized
            if prev_state == GrooveState.IN_POCKET:
                self._state = GrooveState.DISRUPTED
            else:
                self._state = GrooveState.SEARCHING
            self._sustained_count = 0
            self.groove_started_at = None

        self._state_history.append(self._state)
        return self._state

    def _system_phi_variance(self) -> float:
        """
        Compute variance of Φ across all agents.
        Low variance = agents are aligned. High variance = some agents
        are struggling while others are fine.
        """
        profiles = list(self.governor.profiles.values())
        if len(profiles) < 2:
            return 0.0
        avg_phis = [p.average_phi for p in profiles]
        mean = sum(avg_phis) / len(avg_phis)
        variance = sum((p - mean) ** 2 for p in avg_phis) / len(avg_phis)
        return variance

    # ── Queries ──────────────────────────────────────────

    def groove_quality(self) -> float:
        """
        A 0.0-1.0 metric of how deeply in the groove the system is.

        1.0 = perfect groove (all agents zero Φ, zero variance, long duration)
        0.0 = no groove at all
        """
        if not self.in_groove:
            return 0.0

        # Duration factor: longer groove = higher quality
        duration_factor = min(1.0, self.groove_duration / 32.0)

        # Friction factor: lower Φ = higher quality
        friction_factor = 1.0 - min(1.0, self.governor.total_friction)

        # Variance factor: lower variance = higher quality
        variance = self._system_phi_variance()
        variance_factor = 1.0 - min(1.0, variance / (self.phi_variance_threshold * 2))

        return (0.4 * friction_factor + 0.35 * variance_factor + 0.25 * duration_factor)

    def state_distribution(self) -> dict[GrooveState, float]:
        """
        Fraction of time spent in each state.
        Useful for analytics: "the system was in the pocket 23% of the time."
        """
        if not self._state_history:
            return {GrooveState.SEARCHING: 1.0}
        counts: dict[GrooveState, int] = {}
        for s in self._state_history:
            counts[s] = counts.get(s, 0) + 1
        total = len(self._state_history)
        return {s: c / total for s, c in counts.items()}

    @property
    def in_pocket_percentage(self) -> float:
        """Percentage of observations where the system was in the pocket."""
        dist = self.state_distribution()
        return dist.get(GrooveState.IN_POCKET, 0.0)
