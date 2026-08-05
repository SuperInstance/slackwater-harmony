"""
Hypothesis Sandbox — Layer 1 of the snapkit triadic architecture.

Forward simulation: "If I apply action X, what should happen?"

The sandbox runs the agent's proposed action in a lightweight simulation
and scores the result against reality. This is the FEP prediction step:
the agent generates hypotheses, tests them in the sandbox, and only
commits actions that survive simulation.

    predict(action) → expected outcome
    score(result) → quality metric
    reconcile(predicted, actual) → friction contribution

The sandbox does not modify the real world. It is a thought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class SandboxResult:
    """
    Outcome of a simulated action.

    Attributes:
        action: What was simulated.
        predicted_outcome: What the sandbox thinks will happen.
        confidence: 0.0-1.0, how sure the sandbox is.
        collisions: List of detected conflicts.
        valid: Whether the action passes all checks.
        quality_score: Overall quality metric (0.0-1.0).
        notes: Free-form annotations.
    """
    action: Any
    predicted_outcome: Any = None
    confidence: float = 1.0
    collisions: list[str] = field(default_factory=list)
    valid: bool = True
    quality_score: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if the simulation result is viable (valid and no collisions)."""
        return self.valid and not self.collisions


@dataclass
class HypothesisSandbox:
    """
    Layer 1: Forward simulation and hypothesis testing.

    The sandbox maintains a simulated world state that can be:
    1. Updated with a proposed action (simulate)
    2. Scored against reality (reconcile)
    3. Reset (rollback)

    Simulators are pluggable: register domain-specific simulators
    for collision detection, physics, era-appropriateness, etc.

    Attributes:
        world_state: The simulated world (mutable, does not affect reality).
        simulators: Registered simulation functions.
        pass_through_rate: Fraction of results to let through even if they fail
            (for intentional fallibility — Lucineer sometimes misreads).
    """

    world_state: dict = field(default_factory=dict)
    simulators: list[Callable[[Any, dict], SandboxResult]] = field(default_factory=list)
    pass_through_rate: float = 0.0  # 0 = strict, >0 = intentionally fallible
    history: list[SandboxResult] = field(default_factory=list)

    def register_simulator(
        self,
        simulator: Callable[[Any, dict], SandboxResult],
    ) -> None:
        """
        Register a domain-specific simulator.

        Each simulator takes (action, world_state) and returns a SandboxResult.
        Multiple simulators are aggregated: the overall result is the
        intersection of all simulators' checks.
        """
        self.simulators.append(simulator)

    # ── Simulation ───────────────────────────────────────

    def simulate(self, action: Any) -> SandboxResult:
        """
        Run a proposed action through all registered simulators.

        This is the prediction step: "If I do X, what happens?"
        Returns the aggregate result.
        """
        # Aggregate results from all simulators
        aggregate = SandboxResult(
            action=action,
            predicted_outcome=None,
            confidence=1.0,
            valid=True,
            quality_score=1.0,
        )

        for sim in self.simulators:
            result = sim(action, dict(self.world_state))

            # Merge: any failure fails the aggregate
            if not result.valid:
                aggregate.valid = False
            aggregate.collisions.extend(result.collisions)
            aggregate.notes.extend(result.notes)

            # Confidence is the product of individual confidences
            aggregate.confidence *= result.confidence

            # Quality is the minimum
            aggregate.quality_score = min(aggregate.quality_score, result.quality_score)

            if result.predicted_outcome is not None:
                aggregate.predicted_outcome = result.predicted_outcome

        # Pass-through: intentionally let some failures through
        if not aggregate.passed and self.pass_through_rate > 0:
            import random
            if random.random() < self.pass_through_rate:
                aggregate.notes.append("Intentional pass-through (fallibility)")
                aggregate.valid = True
                aggregate.collisions = []  # Forgive the collisions

        self.history.append(aggregate)
        return aggregate

    def simulate_batch(self, actions: list[Any]) -> list[SandboxResult]:
        """Simulate a batch of proposed actions. Returns one result per action."""
        return [self.simulate(action) for action in actions]

    # ── Scoring ──────────────────────────────────────────

    def score(self, result: Any) -> float:
        """
        Score a result on a 0.0-1.0 quality metric.

        Accepts either a SandboxResult or a raw outcome dict.
        The score reflects how good the outcome is — not whether it
        was predicted, but whether it is desirable.
        """
        if isinstance(result, SandboxResult):
            return result.quality_score

        # Score a raw outcome dict
        if isinstance(result, dict):
            score = 1.0
            if "errors" in result and result["errors"]:
                score -= 0.3 * len(result["errors"])
            if "warnings" in result and result["warnings"]:
                score -= 0.1 * len(result["warnings"])
            if "quality" in result:
                score = min(score, float(result["quality"]))
            return max(0.0, min(1.0, score))

        return 0.5  # Unknown → neutral

    def reconcile(
        self,
        predicted: Any,
        actual: Any,
    ) -> float:
        """
        Compare predicted vs actual outcome.

        Returns the prediction error (0.0 = perfect prediction, 1.0 = totally wrong).
        This feeds back into the HarmonyGovernor's Φ computation.
        """
        if isinstance(predicted, dict) and isinstance(actual, dict):
            if not predicted and not actual:
                return 0.0
            keys = set(predicted.keys()) | set(actual.keys())
            if not keys:
                return 0.0
            error_sum = 0.0
            for k in keys:
                p = predicted.get(k)
                a = actual.get(k)
                if isinstance(p, (int, float)) and isinstance(a, (int, float)):
                    error_sum += abs(p - a)
                elif p != a:
                    error_sum += 1.0
            return min(1.0, error_sum / max(1, len(keys)))

        if isinstance(predicted, list) and isinstance(actual, list):
            if not predicted and not actual:
                return 0.0
            if len(predicted) != len(actual):
                return 1.0
            diffs = [abs(p - a) for p, a in zip(predicted, actual)
                     if isinstance(p, (int, float)) and isinstance(a, (int, float))]
            return min(1.0, sum(diffs) / len(diffs)) if diffs else 0.0

        if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
            return min(1.0, abs(predicted - actual))

        return 0.0 if predicted == actual else 1.0

    # ── World State ──────────────────────────────────────

    def update_state(self, key: str, value: Any) -> None:
        """Update the simulated world state."""
        self.world_state[key] = value

    def checkpoint(self) -> dict:
        """Return a snapshot of the current world state for rollback."""
        return dict(self.world_state)

    def restore(self, snapshot: dict) -> None:
        """Restore world state from a checkpoint."""
        self.world_state = dict(snapshot)

    def reset(self) -> None:
        """Clear the simulated world state and history."""
        self.world_state.clear()
        self.history.clear()

    # ── Statistics ───────────────────────────────────────

    @property
    def pass_rate(self) -> float:
        """Fraction of simulated actions that passed."""
        if not self.history:
            return 1.0
        return sum(1 for r in self.history if r.passed) / len(self.history)

    @property
    def average_confidence(self) -> float:
        """Average confidence across simulation history."""
        if not self.history:
            return 1.0
        return sum(r.confidence for r in self.history) / len(self.history)


# ── Built-in Simulators ───────────────────────────────────

def collision_simulator(
    occupied_positions: set[tuple[float, float]],
) -> Callable[[Any, dict], SandboxResult]:
    """
    Create a collision-detection simulator.

    Usage:
        sandbox.register_simulator(collision_simulator({(1,2), (3,4)}))
    """
    def sim(action: Any, world_state: dict) -> SandboxResult:
        result = SandboxResult(action=action)

        # Try to extract position from action
        pos = None
        if isinstance(action, dict):
            pos = action.get("pos") or action.get("position")
        elif hasattr(action, "pos"):
            pos = action.pos
        elif isinstance(action, (list, tuple)) and len(action) >= 2:
            pos = (action[0], action[1])

        if pos is not None:
            pos_tuple = (pos[0], pos[1]) if not isinstance(pos, tuple) else pos
            if pos_tuple in occupied_positions:
                result.valid = False
                result.collisions.append(f"Collision at {pos_tuple}")
                result.confidence = 0.0
                result.quality_score = 0.0

        return result

    return sim


def threshold_simulator(
    key: str,
    min_val: float,
    max_val: float,
) -> Callable[[Any, dict], SandboxResult]:
    """
    Create a range-checking simulator for a numeric field.
    """
    def sim(action: Any, world_state: dict) -> SandboxResult:
        result = SandboxResult(action=action)

        val = None
        if isinstance(action, dict):
            val = action.get(key)
        elif isinstance(world_state, dict):
            val = world_state.get(key)

        if val is not None and isinstance(val, (int, float)):
            if val < min_val or val > max_val:
                result.valid = False
                result.collisions.append(f"{key}={val} out of range [{min_val}, {max_val}]")
                result.quality_score = 0.1

        return result

    return sim
