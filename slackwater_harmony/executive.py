"""
Executive Agent — Layer 3 of the snapkit triadic architecture.

When the Harmony Governor fires an alarm (Φ > deadband), the Executive wakes.
It improvises: rewrites constraints, crosses wires, injects novelty, or
decides that the best action is to do nothing.

The Executive is the jazz musician hearing the dissonance and choosing
how to resolve it. Not by returning to the old key — by finding a new
configuration that makes the dissonance meaningful.

"This is not consensus. This is harmony. And harmony, it turns out,
 has a geometry." — The Lattice of Agreeable Things
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from slackwater_harmony.governor import FrictionAlarm, AlarmSeverity
from slackwater_harmony.sandbox import HypothesisSandbox, SandboxResult


class ImprovisationType(IntEnum):
    """What the Executive decided to do."""
    NONE = 0           # Do nothing — Φ was within deadband
    SIMPLIFY = 1       # Reduce task complexity
    ASSIST = 2         # Offer targeted help
    TAKE_OVER = 3      # Agent takes the lead temporarily
    REWRITE = 4        # Rewrite the plan/constraints
    CROSS_WIRE = 5     # Try something unexpected
    RESET = 6          # Reset context entirely
    DEFER = 7          # Defer to later


@dataclass
class Improvisation:
    """
    The Executive's response to a friction alarm.

    Attributes:
        type: What kind of improvisation.
        reason: Human-readable explanation.
        constraint_rewrites: Changes to apply to the current plan.
        dialogue: What the agent should say (if any).
        novelty: Whether this introduces novelty (unexpected behavior).
        confidence: How confident the Executive is in this improvisation.
    """
    type: ImprovisationType = ImprovisationType.NONE
    reason: str = ""
    constraint_rewrites: dict[str, Any] = field(default_factory=dict)
    dialogue: Optional[str] = None
    novelty: bool = False
    confidence: float = 1.0

    @property
    def is_active(self) -> bool:
        """True if the Executive chose to act (type != NONE)."""
        return self.type != ImprovisationType.NONE

    def apply_to(self, plan: dict) -> dict:
        """
        Apply constraint rewrites to a plan dictionary.
        Returns a new plan with the rewrites applied.
        """
        new_plan = dict(plan)
        new_plan.update(self.constraint_rewrites)
        return new_plan

    def __repr__(self) -> str:
        if not self.is_active:
            return "Improvisation(none)"
        return f"Improvisation({self.type.name}: {self.reason})"


@dataclass
class ExecutiveAgent:
    """
    Layer 3: Improvises when friction exceeds deadband.

    The Executive is the last line of defense. When the Governor
    says "something is wrong" and the Sandbox can't fix it through
    simulation, the Executive improvises a response.

    Design principles:
    1. The most common response is NONE — the agent stays quiet.
    2. Simpler improvisations are preferred over complex ones.
    3. Novelty (cross-wiring) is rare but important — it breaks loops.
    4. The Executive can escalate: GENTLE → MODERATE → CRITICAL.

    Attributes:
        sandbox: Reference to the hypothesis sandbox for testing alternatives.
        improvisation_history: Record of past improvisations.
        novelty_threshold: Probability of choosing a cross-wire response.
        max_rewrites_per_episode: Don't rewrite endlessly.
    """

    sandbox: Optional[HypothesisSandbox] = None
    improvisation_history: list[Improvisation] = field(default_factory=list)
    novelty_threshold: float = 0.15
    max_rewrites_per_episode: int = 3
    _rewrites_this_episode: int = 0

    def reset_episode(self) -> None:
        """Call at the start of a new episode/turn to reset counters."""
        self._rewrites_this_episode = 0

    # ── Alarm Handling ───────────────────────────────────

    def handle_alarm(self, alarm: FrictionAlarm) -> Improvisation:
        """
        The main entry point: the Governor fired, now what?

        Decision tree:
        1. GENTLE → nudge (simplify or assist)
        2. MODERATE → adapt (rewrite constraints)
        3. CRITICAL → intervene (take over or reset)

        At any level, cross-wire (novelty) may trigger with small probability.
        """
        if not alarm.is_active and alarm.severity == AlarmSeverity.NONE:
            imp = Improvisation(type=ImprovisationType.NONE)
            self.improvisation_history.append(imp)
            return imp

        # Roll for novelty first — it can override any severity
        if random.random() < self.novelty_threshold:
            imp = self.cross_wire(alarm)
            self.improvisation_history.append(imp)
            return imp

        # Severity-based response
        if alarm.severity == AlarmSeverity.GENTLE:
            imp = self._handle_gentle(alarm)
        elif alarm.severity == AlarmSeverity.MODERATE:
            imp = self._handle_moderate(alarm)
        elif alarm.severity == AlarmSeverity.CRITICAL:
            imp = self._handle_critical(alarm)
        else:
            imp = Improvisation(type=ImprovisationType.NONE)

        self.improvisation_history.append(imp)
        return imp

    def _handle_gentle(self, alarm: FrictionAlarm) -> Improvisation:
        """Gentle friction: nudge, don't shove."""
        return Improvisation(
            type=ImprovisationType.SIMPLIFY,
            reason=f"Gentle friction for {alarm.agent_id}: reducing complexity",
            constraint_rewrites={"max_complexity": 3},
            dialogue="Let's keep it simple for now.",
            confidence=0.8,
        )

    def _handle_moderate(self, alarm: FrictionAlarm) -> Improvisation:
        """Moderate friction: rewrite constraints if we haven't exhausted the budget."""
        if self._rewrites_this_episode >= self.max_rewrites_per_episode:
            return Improvisation(
                type=ImprovisationType.ASSIST,
                reason=f"Rewrite budget exhausted; offering targeted help to {alarm.agent_id}",
                constraint_rewrites={"assist_level": "physical"},
                dialogue="I'll help with this part.",
                confidence=0.7,
            )

        self._rewrites_this_episode += 1
        imp = Improvisation(
            type=ImprovisationType.REWRITE,
            reason=f"Moderate friction: rewriting constraints for {alarm.agent_id}",
            constraint_rewrites=self.rewrite_constraints(alarm.context),
            dialogue="Let me rethink this.",
            confidence=0.6,
        )

        # Test the rewrite in the sandbox if available
        if self.sandbox:
            test_result = self.sandbox.simulate(imp.constraint_rewrites)
            if not test_result.passed:
                imp.notes = "Sandbox rejected the rewrite, applying anyway"
                imp.confidence *= 0.5

        return imp

    def _handle_critical(self, alarm: FrictionAlarm) -> Improvisation:
        """Critical friction: take over or reset."""
        context = alarm.context
        repeated_failures = context.get("repeated_failures", 0)

        if repeated_failures > 5:
            return Improvisation(
                type=ImprovisationType.RESET,
                reason=f"Cascading failure ({repeated_failures} repeats): resetting context",
                constraint_rewrites={"reset": True},
                dialogue="Give me a minute. I'll sort it.",
                confidence=0.5,
            )

        return Improvisation(
            type=ImprovisationType.TAKE_OVER,
            reason=f"Critical friction: agent taking lead for {alarm.agent_id}",
            constraint_rewrites={"agent_takes_lead": True},
            dialogue="Let me handle this one.",
            confidence=0.7,
        )

    # ── Constraint Rewriting ─────────────────────────────

    def rewrite_constraints(self, context: dict) -> dict:
        """
        Generate new constraints based on the alarm context.

        This is where the Executive's "creativity" lives. It reads the
        situation and proposes changes. The changes are domain-specific;
        the default implementation provides generic heuristics.
        """
        rewrites: dict[str, Any] = {}

        # If there's a complexity metric, reduce it
        if "complexity" in context:
            rewrites["max_complexity"] = max(1, context["complexity"] - 2)

        # If there's a deadline, extend it
        if "deadline" in context:
            rewrites["deadline"] = context["deadline"] * 1.5

        # If there's a resource limit, relax it
        if "resource_budget" in context:
            rewrites["resource_budget"] = context["resource_budget"] * 1.3

        # If there's an era/stage gate, preview the next era
        if "era" in context and isinstance(context["era"], int):
            rewrites["era_unlock"] = "preview"

        # Always add a fresh timestamp
        rewrites["rewritten_at"] = context.get("timestamp", 0)

        return rewrites

    # ── Cross-Wiring ─────────────────────────────────────

    def cross_wire(self, alarm: FrictionAlarm) -> Improvisation:
        """
        Try something unexpected.

        Cross-wiring is the Executive's creative escape hatch. When the
        system is stuck in a loop — same friction, same response — the
        Executive can break the pattern by doing something novel.

        In music: this is the moment the pianist plays a note from
        outside the key — a chromatic passing tone that opens a
        new harmonic path nobody expected.

        Novelty should be rare (novelty_threshold ~0.15) but when it
        fires, it should be surprising enough to break the degenerative
        loop but not so random that it destroys the build.
        """
        strategies = [
            # Strategy 1: Flip a constraint
            {
                "type": ImprovisationType.CROSS_WIRE,
                "reason": f"Cross-wire: flipping a constraint to escape loop for {alarm.agent_id}",
                "constraint_rewrites": {"flip_axis": True, "novelty_injected": True},
                "dialogue": "What if we try it the other way?",
                "novelty": True,
                "confidence": 0.3,
            },
            # Strategy 2: Introduce new material/element
            {
                "type": ImprovisationType.CROSS_WIRE,
                "reason": f"Cross-wire: introducing new material for {alarm.agent_id}",
                "constraint_rewrites": {"introduce_new_material": True},
                "dialogue": "I've got an idea — different approach entirely.",
                "novelty": True,
                "confidence": 0.4,
            },
            # Strategy 3: Defer and observe
            {
                "type": ImprovisationType.DEFER,
                "reason": f"Cross-wire: deferring action to observe for {alarm.agent_id}",
                "constraint_rewrites": {"observe_mode": True},
                "dialogue": "Hold on — let me watch this for a second.",
                "novelty": True,
                "confidence": 0.5,
            },
        ]

        choice = random.choice(strategies)
        return Improvisation(**choice)

    # ── Batch Handling ───────────────────────────────────

    def handle_alarms(self, alarms: list[FrictionAlarm]) -> list[Improvisation]:
        """
        Handle multiple alarms. Sorts by severity (highest first)
        so critical issues get attention before gentle ones.
        """
        sorted_alarms = sorted(alarms, key=lambda a: a.severity, reverse=True)
        return [self.handle_alarm(alarm) for alarm in sorted_alarms]

    # ── Statistics ───────────────────────────────────────

    @property
    def total_improvisations(self) -> int:
        return len(self.improvisation_history)

    @property
    def active_improvisations(self) -> int:
        """Count of non-NONE improvisations."""
        return sum(1 for i in self.improvisation_history if i.is_active)

    @property
    def novelty_rate(self) -> float:
        """Fraction of improvisations that introduced novelty."""
        if not self.improvisation_history:
            return 0.0
        return sum(1 for i in self.improvisation_history if i.novelty) / len(self.improvisation_history)

    @property
    def most_common_response(self) -> ImprovisationType:
        """The most frequently chosen improvisation type."""
        if not self.improvisation_history:
            return ImprovisationType.NONE
        counts: dict[ImprovisationType, int] = {}
        for imp in self.improvisation_history:
            counts[imp.type] = counts.get(imp.type, 0) + 1
        return max(counts, key=counts.get)
