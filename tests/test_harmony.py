"""
Tests for slackwater-harmony.

Exercises the Harmony Governor, Hypothesis Sandbox, Executive Agent,
and Groove Detector.
"""

import pytest

from slackwater_harmony import (
    HarmonyGovernor,
    FrictionAlarm,
    HypothesisSandbox,
    SandboxResult,
    ExecutiveAgent,
    Improvisation,
    GrooveDetector,
    GrooveState,
)
from slackwater_harmony.governor import (
    AgentFrictionProfile,
    AlarmSeverity,
)
from slackwater_harmony.executive import ImprovisationType


class TestHarmonyGovernor:
    """Test cognitive friction measurement and deadband enforcement."""

    def test_register_agent(self):
        gov = HarmonyGovernor()
        profile = gov.register_agent("lucineer", base_deadband=0.8)
        assert profile.agent_id == "lucineer"
        assert profile.base_deadband == 0.8
        assert "lucineer" in gov.profiles

    def test_measure_friction_zero(self):
        """Perfect prediction = zero friction."""
        gov = HarmonyGovernor()
        phi = gov.measure_friction("agent_a", prediction=1.0, actual=1.0)
        assert phi == 0.0

    def test_measure_friction_nonzero(self):
        """Prediction error produces friction."""
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        phi = gov.measure_friction("agent_a", prediction=0.0, actual=1.0)
        assert phi == pytest.approx(1.0)

    def test_measure_friction_partial(self):
        gov = HarmonyGovernor(alpha=0.5, beta=0.0, gamma=0.0)
        phi = gov.measure_friction("agent_a", prediction=0.0, actual=1.0)
        assert phi == pytest.approx(0.5)

    def test_measure_friction_compute_load(self):
        gov = HarmonyGovernor(alpha=0.0, beta=1.0, gamma=0.0)
        phi = gov.measure_friction("a", prediction=0, actual=0, compute_load=0.5)
        assert phi == pytest.approx(0.5)

    def test_measure_friction_state_delta(self):
        gov = HarmonyGovernor(alpha=0.0, beta=0.0, gamma=1.0)
        phi = gov.measure_friction("a", prediction=0, actual=0, state_delta=0.3)
        assert phi == pytest.approx(0.3)

    def test_friction_with_dict_prediction(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        phi = gov.measure_friction(
            "a",
            prediction={"x": 1.0, "y": 2.0},
            actual={"x": 1.0, "y": 3.0},
        )
        assert phi > 0.0
        assert phi == pytest.approx(0.5)  # avg(|1-1|, |2-3|) = 0.5

    def test_friction_with_list_prediction(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        phi = gov.measure_friction(
            "a",
            prediction=[1.0, 2.0, 3.0],
            actual=[1.0, 2.0, 4.0],
        )
        assert phi == pytest.approx(1.0 / 3.0)

    def test_deadband_not_exceeded(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=2.0)
        gov.measure_friction("a", prediction=0, actual=0.5)
        assert not gov.check_deadband("a", 0.5)

    def test_deadband_exceeded(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=0.3)
        phi = gov.measure_friction("a", prediction=0, actual=1.0)
        assert gov.check_deadband("a", phi)

    def test_check_and_alarm_fires(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=0.3)
        alarm = gov.check_and_alarm("a", phi=1.0, timestamp=42)
        assert alarm is not None
        assert alarm.agent_id == "a"
        assert alarm.phi == 1.0
        assert alarm.timestamp == 42
        assert alarm.severity == AlarmSeverity.CRITICAL  # 1.0 / 0.3 > 2.0

    def test_check_and_alarm_no_fire(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=2.0)
        alarm = gov.check_and_alarm("a", phi=0.5)
        assert alarm is None

    def test_alarm_severity_gentle(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=1.0)
        # phi = 1.1, ratio = 1.1 → GENTLE
        alarm = gov.check_and_alarm("a", phi=1.1)
        assert alarm is not None
        assert alarm.severity == AlarmSeverity.GENTLE

    def test_alarm_severity_moderate(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=1.0)
        alarm = gov.check_and_alarm("a", phi=1.6)
        assert alarm is not None
        assert alarm.severity == AlarmSeverity.MODERATE

    def test_alarm_severity_critical(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=1.0)
        alarm = gov.check_and_alarm("a", phi=3.0)
        assert alarm is not None
        assert alarm.severity == AlarmSeverity.CRITICAL

    def test_game_state_changes_deadband(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=1.0)
        # Tutorial: deadband = 1.0 * 2.0 = 2.0
        gov.set_game_state("tutorial")
        assert gov.profiles["a"].current_deadband == pytest.approx(2.0)

        # Stage 5: deadband = 1.0 * 0.7 = 0.7
        gov.set_game_state("stage_5")
        assert gov.profiles["a"].current_deadband == pytest.approx(0.7)

    def test_adaptive_deadband_widens(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=1.0)
        original = gov.profiles["a"].current_deadband
        # Fire several alarms → deadband should widen
        for _ in range(5):
            gov.measure_friction("a", prediction=0, actual=5.0)
        assert gov.profiles["a"].current_deadband > original

    def test_adaptive_deadband_narrows(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=1.0)
        # Long calm streak → deadband narrows
        for _ in range(15):
            gov.measure_friction("a", prediction=0, actual=0.0)
        assert gov.profiles["a"].current_deadband < 1.0

    def test_is_harmonized(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=2.0)
        gov.register_agent("b", base_deadband=2.0)
        gov.measure_friction("a", prediction=0, actual=0)
        gov.measure_friction("b", prediction=0, actual=0)
        assert gov.is_harmonized

    def test_not_harmonized(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=0.1)
        gov.measure_friction("a", prediction=0, actual=1.0)
        assert not gov.is_harmonized

    def test_total_friction(self):
        gov = HarmonyGovernor()
        gov.register_agent("a")
        gov.register_agent("b")
        gov.measure_friction("a", prediction=0, actual=1.0)
        gov.measure_friction("b", prediction=0, actual=2.0)
        assert gov.total_friction > 0.0

    def test_recent_alarms(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=0.1)
        for i in range(5):
            gov.check_and_alarm("a", phi=1.0 + i * 0.1)
        recent = gov.recent_alarms(count=3)
        assert len(recent) == 3

    def test_friction_alarm_overshoot(self):
        alarm = FrictionAlarm(
            agent_id="a",
            phi=1.5,
            deadband=1.0,
            severity=AlarmSeverity.MODERATE,
        )
        assert alarm.overshoot == pytest.approx(0.5)

    def test_unknown_agent_no_alarm(self):
        gov = HarmonyGovernor()
        alarm = gov.check_and_alarm("ghost", phi=999.0)
        assert alarm is None


class TestHypothesisSandbox:
    """Test forward simulation and hypothesis testing."""

    def test_simulate_pass(self):
        sandbox = HypothesisSandbox()
        result = sandbox.simulate({"type": "place", "pos": [0, 0]})
        assert result.passed
        assert result.confidence == 1.0

    def test_simulate_with_collision_detector(self):
        from slackwater_harmony.sandbox import collision_simulator
        sandbox = HypothesisSandbox()
        sandbox.register_simulator(collision_simulator({(0, 0), (1, 1)}))

        # Collision at origin
        result = sandbox.simulate({"pos": [0, 0]})
        assert not result.passed
        assert len(result.collisions) > 0

        # Free position
        result = sandbox.simulate({"pos": [5, 5]})
        assert result.passed

    def test_simulate_with_threshold(self):
        from slackwater_harmony.sandbox import threshold_simulator
        sandbox = HypothesisSandbox()
        sandbox.register_simulator(threshold_simulator("height", 0, 10))

        # In range
        assert sandbox.simulate({"height": 5}).passed
        # Out of range
        assert not sandbox.simulate({"height": 15}).passed

    def test_multiple_simulators_aggregate(self):
        from slackwater_harmony.sandbox import collision_simulator, threshold_simulator
        sandbox = HypothesisSandbox()
        sandbox.register_simulator(collision_simulator({(0, 0)}))
        sandbox.register_simulator(threshold_simulator("height", 0, 10))

        # Fails both
        result = sandbox.simulate({"pos": [0, 0], "height": 99})
        assert not result.passed
        assert len(result.collisions) >= 2

    def test_score_sandbox_result(self):
        sandbox = HypothesisSandbox()
        result = SandboxResult(action="test", quality_score=0.7)
        assert sandbox.score(result) == pytest.approx(0.7)

    def test_score_dict(self):
        sandbox = HypothesisSandbox()
        score = sandbox.score({"errors": ["bad"], "quality": 0.9})
        assert score < 1.0  # Penalized for error

    def test_reconcile_perfect(self):
        sandbox = HypothesisSandbox()
        error = sandbox.reconcile({"x": 1}, {"x": 1})
        assert error == 0.0

    def test_reconcile_mismatch(self):
        sandbox = HypothesisSandbox()
        error = sandbox.reconcile({"x": 0}, {"x": 1})
        assert error > 0.0

    def test_reconcile_scalar(self):
        sandbox = HypothesisSandbox()
        assert sandbox.reconcile(1.0, 1.0) == 0.0
        assert sandbox.reconcile(0.0, 1.0) == pytest.approx(1.0)

    def test_checkpoint_and_restore(self):
        sandbox = HypothesisSandbox(world_state={"a": 1})
        checkpoint = sandbox.checkpoint()
        sandbox.update_state("a", 99)
        sandbox.restore(checkpoint)
        assert sandbox.world_state["a"] == 1

    def test_pass_through_rate(self):
        from slackwater_harmony.sandbox import collision_simulator
        sandbox = HypothesisSandbox(pass_through_rate=1.0)  # Always forgive
        sandbox.register_simulator(collision_simulator({(0, 0)}))
        result = sandbox.simulate({"pos": [0, 0]})
        assert result.passed  # Collision forgiven

    def test_pass_rate_statistic(self):
        from slackwater_harmony.sandbox import collision_simulator
        sandbox = HypothesisSandbox()
        sandbox.register_simulator(collision_simulator({(0, 0)}))
        sandbox.simulate({"pos": [0, 0]})  # fail
        sandbox.simulate({"pos": [5, 5]})  # pass
        assert sandbox.pass_rate == pytest.approx(0.5)

    def test_reset(self):
        sandbox = HypothesisSandbox(world_state={"x": 1})
        sandbox.simulate("test")
        sandbox.reset()
        assert len(sandbox.world_state) == 0
        assert len(sandbox.history) == 0


class TestExecutiveAgent:
    """Test improvisation when friction alarms fire."""

    def test_handle_no_alarm(self):
        exec = ExecutiveAgent()
        alarm = FrictionAlarm(
            agent_id="a", phi=0, deadband=1.0, severity=AlarmSeverity.NONE
        )
        imp = exec.handle_alarm(alarm)
        assert imp.type == ImprovisationType.NONE

    def test_handle_gentle_alarm(self):
        exec = ExecutiveAgent(novelty_threshold=0.0)  # Disable randomness
        alarm = FrictionAlarm(
            agent_id="a", phi=1.1, deadband=1.0, severity=AlarmSeverity.GENTLE
        )
        imp = exec.handle_alarm(alarm)
        assert imp.type == ImprovisationType.SIMPLIFY
        assert imp.is_active

    def test_handle_moderate_alarm(self):
        exec = ExecutiveAgent(novelty_threshold=0.0)
        alarm = FrictionAlarm(
            agent_id="a", phi=1.6, deadband=1.0, severity=AlarmSeverity.MODERATE,
            context={"complexity": 5, "era": 2},
        )
        imp = exec.handle_alarm(alarm)
        assert imp.type == ImprovisationType.REWRITE
        assert "max_complexity" in imp.constraint_rewrites

    def test_handle_critical_alarm(self):
        exec = ExecutiveAgent(novelty_threshold=0.0)
        alarm = FrictionAlarm(
            agent_id="a", phi=3.0, deadband=1.0, severity=AlarmSeverity.CRITICAL,
            context={"repeated_failures": 2},
        )
        imp = exec.handle_alarm(alarm)
        assert imp.type == ImprovisationType.TAKE_OVER

    def test_handle_critical_cascading(self):
        exec = ExecutiveAgent(novelty_threshold=0.0)
        alarm = FrictionAlarm(
            agent_id="a", phi=5.0, deadband=1.0, severity=AlarmSeverity.CRITICAL,
            context={"repeated_failures": 10},
        )
        imp = exec.handle_alarm(alarm)
        assert imp.type == ImprovisationType.RESET

    def test_cross_wire(self):
        exec = ExecutiveAgent(novelty_threshold=1.0)  # Always cross-wire
        alarm = FrictionAlarm(
            agent_id="a", phi=1.5, deadband=1.0, severity=AlarmSeverity.MODERATE,
        )
        imp = exec.handle_alarm(alarm)
        assert imp.novelty

    def test_rewrite_budget(self):
        exec = ExecutiveAgent(novelty_threshold=0.0, max_rewrites_per_episode=2)
        alarm = FrictionAlarm(
            agent_id="a", phi=1.5, deadband=1.0, severity=AlarmSeverity.MODERATE,
            context={"complexity": 5},
        )
        # First two rewrites
        imp1 = exec.handle_alarm(alarm)
        imp2 = exec.handle_alarm(alarm)
        assert imp1.type == ImprovisationType.REWRITE
        assert imp2.type == ImprovisationType.REWRITE
        # Third should fall back to ASSIST
        imp3 = exec.handle_alarm(alarm)
        assert imp3.type == ImprovisationType.ASSIST

    def test_reset_episode(self):
        exec = ExecutiveAgent(novelty_threshold=0.0, max_rewrites_per_episode=1)
        alarm = FrictionAlarm(
            agent_id="a", phi=1.5, deadband=1.0, severity=AlarmSeverity.MODERATE,
            context={"complexity": 5},
        )
        exec.handle_alarm(alarm)
        exec.reset_episode()
        # Budget should be reset
        assert exec._rewrites_this_episode == 0

    def test_improvisation_apply_to(self):
        imp = Improvisation(
            type=ImprovisationType.REWRITE,
            constraint_rewrites={"a": 1, "b": 2},
        )
        new_plan = imp.apply_to({"a": 0, "c": 3})
        assert new_plan == {"a": 1, "b": 2, "c": 3}

    def test_handle_alarms_batch(self):
        exec = ExecutiveAgent(novelty_threshold=0.0)
        alarms = [
            FrictionAlarm(agent_id="a", phi=3.0, deadband=1.0, severity=AlarmSeverity.CRITICAL),
            FrictionAlarm(agent_id="b", phi=1.1, deadband=1.0, severity=AlarmSeverity.GENTLE),
        ]
        results = exec.handle_alarms(alarms)
        assert len(results) == 2
        # Critical should be handled first
        assert results[0].type == ImprovisationType.TAKE_OVER

    def test_statistics(self):
        exec = ExecutiveAgent(novelty_threshold=0.0)
        for _ in range(3):
            exec.improvisation_history.append(Improvisation(type=ImprovisationType.SIMPLIFY))
        exec.improvisation_history.append(Improvisation(type=ImprovisationType.NONE))
        assert exec.total_improvisations == 4
        assert exec.active_improvisations == 3
        assert exec.most_common_response == ImprovisationType.SIMPLIFY


class TestGrooveDetector:
    """Test groove detection across the agent system."""

    def test_initial_state_searching(self):
        gov = HarmonyGovernor()
        gov.register_agent("a")
        detector = GrooveDetector(governor=gov)
        assert detector.state == GrooveState.SEARCHING

    def test_transitions_to_settling(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=2.0)
        gov.measure_friction("a", prediction=0, actual=0)
        detector = GrooveDetector(governor=gov, min_sustained_beats=3)
        state = detector.update(beat=1)
        assert state == GrooveState.SETTLING

    def test_transitions_to_in_pocket(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=2.0)
        detector = GrooveDetector(governor=gov, min_sustained_beats=3)
        # Generate enough zero-friction observations
        for beat in range(5):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=beat)
        assert detector.state == GrooveState.IN_POCKET
        assert detector.in_groove

    def test_disruption_breaks_groove(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=2.0)
        detector = GrooveDetector(governor=gov, min_sustained_beats=2)
        # Get into groove
        for beat in range(5):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=beat)
        assert detector.in_groove
        # Spike friction
        gov.measure_friction("a", prediction=0, actual=5.0)
        state = detector.update(beat=6)
        assert state == GrooveState.DISRUPTED

    def test_groove_duration(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=5.0)
        detector = GrooveDetector(governor=gov, min_sustained_beats=2)
        for beat in range(10):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=beat)
        assert detector.groove_duration > 0

    def test_groove_quality(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=5.0)
        gov.register_agent("b", base_deadband=5.0)
        detector = GrooveDetector(governor=gov, min_sustained_beats=2)
        for beat in range(10):
            gov.measure_friction("a", prediction=0, actual=0)
            gov.measure_friction("b", prediction=0, actual=0)
            detector.update(beat=beat)
        assert detector.in_groove
        q = detector.groove_quality()
        assert q > 0.5  # Should be high quality groove

    def test_not_in_groove_quality_zero(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=0.01)
        detector = GrooveDetector(governor=gov)
        gov.measure_friction("a", prediction=0, actual=1.0)
        detector.update(beat=1)
        assert not detector.in_groove
        assert detector.groove_quality() == 0.0

    def test_state_distribution(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=5.0)
        detector = GrooveDetector(governor=gov, min_sustained_beats=2)
        for beat in range(5):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=beat)
        dist = detector.state_distribution()
        assert sum(dist.values()) == pytest.approx(1.0)
        assert GrooveState.IN_POCKET in dist

    def test_longest_groove_tracked(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=5.0)
        detector = GrooveDetector(governor=gov, min_sustained_beats=2)
        # Build a groove
        for beat in range(10):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=beat)
        assert detector.longest_groove > 0

    def test_in_pocket_percentage(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=5.0)
        detector = GrooveDetector(governor=gov, min_sustained_beats=1)
        for beat in range(10):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=beat)
        assert detector.in_pocket_percentage > 0.0
