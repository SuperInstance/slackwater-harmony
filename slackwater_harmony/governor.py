"""
Harmony Governor — Layer 2 of the snapkit triadic architecture.

Measures Φ (cognitive friction): the gap between prediction and reality.
When Φ exceeds a deadband threshold, the governor fires an alarm that
wakes the Executive layer.

Φ(t) = α · H(prediction_error) + β · L(compute) + γ · Δ(state)

The deadband is not fixed. It adapts per agent, per game state.
A learning player has a wide deadband (friction is expected).
An expert player has a narrow deadband (friction means something is wrong).

"The system does not minimize friction instantly. It uses friction
 as fuel for exploration." — The Lattice of Agreeable Things
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class AlarmSeverity(IntEnum):
    """How far Φ has exceeded the deadband."""
    NONE = 0       # Φ within deadband — do nothing
    GENTLE = 1     # Slightly above — nudge
    MODERATE = 2   # Clearly above — adapt
    CRITICAL = 3   # Far above — intervene immediately


@dataclass
class FrictionAlarm:
    """
    Fired when Φ exceeds the deadband for a given agent.

    Attributes:
        agent_id: Which agent's prediction failed.
        phi: The friction value that triggered the alarm.
        deadband: The threshold that was exceeded.
        severity: How bad it is.
        context: Arbitrary metadata about the situation.
        timestamp: Logical beat when the alarm fired.
    """
    agent_id: str
    phi: float
    deadband: float
    severity: AlarmSeverity
    context: dict = field(default_factory=dict)
    timestamp: int = 0

    @property
    def overshoot(self) -> float:
        """How much Φ exceeded the deadband."""
        return max(0.0, self.phi - self.deadband)

    @property
    def is_active(self) -> bool:
        """True if this alarm represents actual friction (severity > NONE)."""
        return self.severity != AlarmSeverity.NONE

    def __repr__(self) -> str:
        return (
            f"FrictionAlarm(agent={self.agent_id}, Φ={self.phi:.3f}, "
            f"deadband={self.deadband:.3f}, severity={self.severity.name})"
        )


@dataclass
class AgentFrictionProfile:
    """
    Per-agent friction tracking with adaptive deadband.

    The deadband widens or narrows based on recent history:
    - If the agent has been alarm-free, narrow the deadband (expect more)
    - If the agent has been alarming frequently, widen it (give room)
    """
    agent_id: str
    base_deadband: float = 1.0
    current_deadband: float = 1.0
    phi_history: list[float] = field(default_factory=list)
    alarm_count: int = 0
    calm_streak: int = 0  # consecutive checks without alarm
    max_history: int = 50

    def record_phi(self, phi: float) -> None:
        """Record a Φ observation and adapt the deadband."""
        self.phi_history.append(phi)
        if len(self.phi_history) > self.max_history:
            self.phi_history.pop(0)

        if phi > self.current_deadband:
            self.alarm_count += 1
            self.calm_streak = 0
            # Widen deadband slightly — agent is struggling
            self.current_deadband = min(
                self.base_deadband * 2.0,
                self.current_deadband * 1.1,
            )
        else:
            self.calm_streak += 1
            # After a long calm streak, narrow the deadband
            if self.calm_streak >= 10:
                self.current_deadband = max(
                    self.base_deadband * 0.5,
                    self.current_deadband * 0.95,
                )

    @property
    def average_phi(self) -> float:
        if not self.phi_history:
            return 0.0
        return sum(self.phi_history) / len(self.phi_history)

    @property
    def phi_variance(self) -> float:
        if len(self.phi_history) < 2:
            return 0.0
        avg = self.average_phi
        return sum((p - avg) ** 2 for p in self.phi_history) / len(self.phi_history)


@dataclass
class HarmonyGovernor:
    """
    Layer 2: Measures cognitive friction across all agents.

    The governor does not act. It observes and alarms.
    The Executive layer acts on the alarms.

    Φ = α·|prediction - actual| + β·compute_load + γ·state_delta

    Attributes:
        alpha: Weight for prediction error (primary friction source).
        beta: Weight for computational load (secondary).
        gamma: Weight for state change rate (tertiary).
        profiles: Per-agent friction profiles with adaptive deadbands.
        game_state: Current game state key (affects base deadband).
    """

    alpha: float = 0.50
    beta: float = 0.30
    gamma: float = 0.20
    profiles: dict[str, AgentFrictionProfile] = field(default_factory=dict)
    game_state: str = "default"

    # Game-state-dependent deadband multipliers
    # Stage 1 (tutorial): wide deadband — friction is learning
    # Stage 5 (expert): narrow deadband — friction is trouble
    _state_multipliers: dict[str, float] = field(default_factory=lambda: {
        "tutorial": 2.0,
        "stage_1": 1.8,
        "stage_2": 1.5,
        "stage_3": 1.2,
        "stage_4": 0.9,
        "stage_5": 0.7,
        "default": 1.0,
        "creative": 3.0,  # Creative mode: very wide — exploration expected
    })

    def __post_init__(self) -> None:
        self._alarms_fired: list[FrictionAlarm] = []

    # ── Registration ─────────────────────────────────────

    def register_agent(
        self,
        agent_id: str,
        base_deadband: float = 1.0,
    ) -> AgentFrictionProfile:
        """Register a new agent for friction monitoring."""
        profile = AgentFrictionProfile(
            agent_id=agent_id,
            base_deadband=base_deadband,
            current_deadband=base_deadband * self._state_multipliers.get(self.game_state, 1.0),
        )
        self.profiles[agent_id] = profile
        return profile

    def set_game_state(self, state: str) -> None:
        """
        Change the game state, which adjusts all agents' base deadbands.

        Example: transitioning from "tutorial" to "stage_2" narrows
        everyone's deadband — the system expects more from the agents now.
        """
        self.game_state = state
        mult = self._state_multipliers.get(state, 1.0)
        for profile in self.profiles.values():
            profile.current_deadband = profile.base_deadband * mult

    # ── Friction Measurement ─────────────────────────────

    def measure_friction(
        self,
        agent_id: str,
        prediction: dict | float | list,
        actual: dict | float | list,
        compute_load: float = 0.0,
        state_delta: float = 0.0,
    ) -> float:
        """
        Compute Φ (cognitive friction) for an agent.

        Φ = α · prediction_error + β · compute_load + γ · state_delta

        The prediction error is the primary component: how wrong was
        the agent's model of the world? Compute load and state delta
        are secondary factors that modulate sensitivity.

        Args:
            agent_id: Which agent to measure.
            prediction: What the agent expected (any comparable structure).
            actual: What actually happened.
            compute_load: Normalized 0-1 computational effort (0=idle, 1=maxed).
            state_delta: Normalized 0-1 rate of world-state change (0=static, 1=chaotic).

        Returns:
            Φ value (float). Higher = more friction.
        """
        error = self._prediction_error(prediction, actual)
        phi = self.alpha * error + self.beta * compute_load + self.gamma * state_delta

        # Record for this agent
        if agent_id not in self.profiles:
            self.register_agent(agent_id)
        self.profiles[agent_id].record_phi(phi)

        return phi

    def _prediction_error(
        self,
        prediction: dict | float | list,
        actual: dict | float | list,
    ) -> float:
        """
        Compute normalized prediction error between prediction and actual.

        For scalars: absolute difference.
        For lists: mean absolute difference elementwise.
        For dicts: mean absolute difference over shared keys.
        """
        if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
            return float(abs(prediction - actual))

        if isinstance(prediction, list) and isinstance(actual, list):
            if not prediction and not actual:
                return 0.0
            if len(prediction) != len(actual):
                return 1.0  # Structural mismatch = max friction
            diffs = [abs(p - a) for p, a in zip(prediction, actual)
                     if isinstance(p, (int, float)) and isinstance(a, (int, float))]
            return sum(diffs) / len(diffs) if diffs else 0.0

        if isinstance(prediction, dict) and isinstance(actual, dict):
            if not prediction and not actual:
                return 0.0
            keys = set(prediction.keys()) | set(actual.keys())
            if not keys:
                return 0.0
            total_error = 0.0
            for k in keys:
                p = prediction.get(k, 0)
                a = actual.get(k, 0)
                if isinstance(p, (int, float)) and isinstance(a, (int, float)):
                    total_error += abs(p - a)
                elif p != a:
                    total_error += 1.0
            return total_error / len(keys)

        # Type mismatch or incomparable
        return 1.0 if prediction != actual else 0.0

    # ── Deadband Enforcement ─────────────────────────────

    def check_deadband(self, agent_id: str, phi: float) -> bool:
        """
        Return True if Φ exceeds the deadband for this agent.

        The deadband is adaptive — it changes based on game state
        and recent friction history.
        """
        profile = self.profiles.get(agent_id)
        if profile is None:
            return False
        return phi > profile.current_deadband

    def check_and_alarm(
        self,
        agent_id: str,
        phi: float,
        context: Optional[dict] = None,
        timestamp: int = 0,
    ) -> Optional[FrictionAlarm]:
        """
        Check Φ against deadband and fire an alarm if exceeded.

        This is the main entry point for the governor's monitoring loop.
        Returns a FrictionAlarm if Φ is too high, None otherwise.
        """
        profile = self.profiles.get(agent_id)
        if profile is None:
            return None

        if phi <= profile.current_deadband:
            return None

        # Compute severity based on overshoot ratio
        ratio = phi / profile.current_deadband if profile.current_deadband > 0 else 99
        if ratio >= 2.0:
            severity = AlarmSeverity.CRITICAL
        elif ratio >= 1.5:
            severity = AlarmSeverity.MODERATE
        else:
            severity = AlarmSeverity.GENTLE

        alarm = FrictionAlarm(
            agent_id=agent_id,
            phi=phi,
            deadband=profile.current_deadband,
            severity=severity,
            context=context or {},
            timestamp=timestamp,
        )
        self._alarms_fired.append(alarm)
        return alarm

    # ── Queries ──────────────────────────────────────────

    @property
    def total_friction(self) -> float:
        """Sum of all agents' average Φ. A system-wide friction metric."""
        return sum(p.average_phi for p in self.profiles.values())

    @property
    def max_friction(self) -> float:
        """The worst agent's average Φ."""
        if not self.profiles:
            return 0.0
        return max(p.average_phi for p in self.profiles.values())

    @property
    def is_harmonized(self) -> bool:
        """True when ALL agents are within their deadbands (latest Φ)."""
        for profile in self.profiles.values():
            if not profile.phi_history:
                continue
            latest_phi = profile.phi_history[-1]
            if latest_phi > profile.current_deadband:
                return False
        return True

    def agent_phi(self, agent_id: str) -> float:
        """Get the most recent Φ for an agent."""
        profile = self.profiles.get(agent_id)
        if profile is None or not profile.phi_history:
            return 0.0
        return profile.phi_history[-1]

    def recent_alarms(self, count: int = 10) -> list[FrictionAlarm]:
        """Get the most recent alarms."""
        return self._alarms_fired[-count:]

    def alarm_rate(self, window: int = 50) -> float:
        """Fraction of recent Φ measurements that triggered alarms."""
        total = sum(len(p.phi_history) for p in self.profiles.values())
        if total == 0:
            return 0.0
        return len(self._alarms_fired) / min(total, window * len(self.profiles))
