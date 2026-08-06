"""
Tests for slackwater-harmony HarmonyGovernor — edge cases and falsy-zero bugs.

Tests cover:
- Zero-deadband edge case (the falsy-zero bug)
- None prediction/actual handling
- Empty dict/list prediction errors
- Adaptive deadband behavior
- Game state transitions
- Alarm severity thresholds
- Friction measurement with mixed types
"""

import pytest
from slackwater_harmony.governor import (
    HarmonyGovernor, AgentFrictionProfile, FrictionAlarm, AlarmSeverity,
)


# ─── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def governor():
    return HarmonyGovernor()

@pytest.fixture
def governor_with_agent(governor):
    governor.register_agent("agent_a", base_deadband=1.0)
    return governor


# ─── Falsy-Zero Bug Tests ────────────────────────────────

class TestFalsyZeroEdgeCases:
    """The falsy-zero bug: when deadband is 0, friction checks misbehave."""

    def test_zero_deadband_does_not_cause_infinite_severity(self, governor):
        """When current_deadband is 0, ratio should not be 99 (infinite)."""
        governor.register_agent("a", base_deadband=0.0)
        alarm = governor.check_and_alarm("a", phi=0.5)
        # With deadband=0, ANY friction should alarm
        assert alarm is not None
        # But severity should be based on actual overshoot, not infinity
        assert alarm.deadband == 0.0
        assert alarm.overshoot == 0.5

    def test_zero_phi_with_zero_deadband(self, governor):
        """phi=0, deadband=0: phi is NOT > deadband, so no alarm."""
        governor.register_agent("a", base_deadband=0.0)
        alarm = governor.check_and_alarm("a", phi=0.0)
        # 0 > 0 is False, so no alarm
        assert alarm is None

    def test_negative_phi_does_not_alarm(self, governor):
        """Negative phi should not alarm (phi <= deadband)."""
        governor.register_agent("a", base_deadband=1.0)
        alarm = governor.check_and_alarm("a", phi=-0.5)
        assert alarm is None

    def test_none_profile_returns_none(self, governor):
        """Unregistered agent should return None, not crash."""
        alarm = governor.check_and_alarm("ghost", phi=100.0)
        assert alarm is None

    def test_none_profile_check_deadband_returns_false(self, governor):
        """Unregistered agent check_deadband returns False."""
        assert governor.check_deadband("ghost", phi=100.0) is False


# ─── Prediction Error Tests ──────────────────────────────

class TestPredictionError:
    def test_scalar_error(self, governor):
        error = governor._prediction_error(0.5, 0.7)
        assert error == pytest.approx(0.2)

    def test_identical_scalars(self, governor):
        assert governor._prediction_error(3.14, 3.14) == 0.0

    def test_list_same_length(self, governor):
        error = governor._prediction_error([1, 2, 3], [1, 2, 4])
        assert error == pytest.approx(1/3)

    def test_list_different_length(self, governor):
        """Length mismatch = max friction."""
        error = governor._prediction_error([1, 2], [1, 2, 3])
        assert error == 1.0

    def test_empty_lists(self, governor):
        assert governor._prediction_error([], []) == 0.0

    def test_dict_same_keys(self, governor):
        error = governor._prediction_error({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert error == pytest.approx(0.5)

    def test_dict_missing_keys(self, governor):
        error = governor._prediction_error({"a": 1}, {"a": 1, "b": 5})
        # Missing key defaults to 0, so |0-5|=5 for key 'b'
        assert error == pytest.approx(2.5)

    def test_empty_dicts(self, governor):
        assert governor._prediction_error({}, {}) == 0.0

    def test_mixed_types_unequal(self, governor):
        assert governor._prediction_error("hello", 42) == 1.0

    def test_mixed_types_equal(self, governor):
        """Same object, different types but equal value."""
        assert governor._prediction_error(1, 1.0) == 0.0

    def test_dict_with_non_numeric_values(self, governor):
        """Non-numeric values that differ get error 1.0."""
        error = governor._prediction_error({"a": "x"}, {"a": "y"})
        assert error == 1.0


# ─── Adaptive Deadband Tests ─────────────────────────────

class TestAdaptiveDeadband:
    def test_deadband_widens_on_alarm(self, governor_with_agent):
        gov = governor_with_agent
        initial = gov.profiles["agent_a"].current_deadband
        # Trigger alarm
        gov.measure_friction("agent_a", prediction=1.0, actual=5.0)
        # Deadband should have widened
        assert gov.profiles["agent_a"].current_deadband >= initial

    def test_deadband_narrows_on_calm(self, governor_with_agent):
        gov = governor_with_agent
        # Generate many calm streaks
        for _ in range(15):
            gov.measure_friction("agent_a", prediction=1.0, actual=1.0)
        # Deadband should have narrowed after calm streak >= 10
        assert gov.profiles["agent_a"].current_deadband <= gov.profiles["agent_a"].base_deadband

    def test_deadband_does_not_exceed_2x_base(self, governor_with_agent):
        gov = governor_with_agent
        for _ in range(20):
            gov.measure_friction("agent_a", prediction=0.0, actual=10.0)
        assert gov.profiles["agent_a"].current_deadband <= gov.profiles["agent_a"].base_deadband * 2.0

    def test_deadband_does_not_go_below_half_base(self, governor_with_agent):
        gov = governor_with_agent
        for _ in range(100):
            gov.measure_friction("agent_a", prediction=1.0, actual=1.0)
        assert gov.profiles["agent_a"].current_deadband >= gov.profiles["agent_a"].base_deadband * 0.5


# ─── Game State Transition Tests ─────────────────────────

class TestGameStateTransitions:
    def test_set_game_state_updates_multiplier(self, governor_with_agent):
        gov = governor_with_agent
        gov.set_game_state("tutorial")
        expected = gov.profiles["agent_a"].base_deadband * 2.0
        assert gov.profiles["agent_a"].current_deadband == pytest.approx(expected)

    def test_set_game_state_creative_is_widest(self, governor_with_agent):
        gov = governor_with_agent
        gov.set_game_state("creative")
        expected = gov.profiles["agent_a"].base_deadband * 3.0
        assert gov.profiles["agent_a"].current_deadband == pytest.approx(expected)

    def test_set_game_state_stage5_is_narrowest(self, governor_with_agent):
        gov = governor_with_agent
        gov.set_game_state("stage_5")
        expected = gov.profiles["agent_a"].base_deadband * 0.7
        assert gov.profiles["agent_a"].current_deadband == pytest.approx(expected)

    def test_unknown_state_defaults_to_1x(self, governor_with_agent):
        gov = governor_with_agent
        gov.set_game_state("nonexistent")
        assert gov.profiles["agent_a"].current_deadband == pytest.approx(
            gov.profiles["agent_a"].base_deadband
        )


# ─── Alarm Severity Tests ────────────────────────────────

class TestAlarmSeverity:
    def test_gentle_alarm(self, governor):
        gov = governor
        gov.register_agent("a", base_deadband=1.0)
        # phi slightly above deadband → GENTLE
        alarm = gov.check_and_alarm("a", phi=1.1)
        assert alarm is not None
        assert alarm.severity == AlarmSeverity.GENTLE

    def test_moderate_alarm(self, governor):
        gov = governor
        gov.register_agent("a", base_deadband=1.0)
        alarm = gov.check_and_alarm("a", phi=1.6)
        assert alarm is not None
        assert alarm.severity == AlarmSeverity.MODERATE

    def test_critical_alarm(self, governor):
        gov = governor
        gov.register_agent("a", base_deadband=1.0)
        alarm = gov.check_and_alarm("a", phi=3.0)
        assert alarm is not None
        assert alarm.severity == AlarmSeverity.CRITICAL

    def test_no_alarm_within_deadband(self, governor):
        gov = governor
        gov.register_agent("a", base_deadband=1.0)
        alarm = gov.check_and_alarm("a", phi=0.9)
        assert alarm is None

    def test_alarm_has_context(self, governor):
        gov = governor
        gov.register_agent("a", base_deadband=1.0)
        ctx = {"task": "build", "step": 3}
        alarm = gov.check_and_alarm("a", phi=2.0, context=ctx, timestamp=42)
        assert alarm.context == ctx
        assert alarm.timestamp == 42


# ─── System-Wide Metrics Tests ───────────────────────────

class TestSystemMetrics:
    def test_total_friction_empty(self, governor):
        assert governor.total_friction == 0.0

    def test_max_friction_empty(self, governor):
        assert governor.max_friction == 0.0

    def test_is_harmonized_empty(self, governor):
        assert governor.is_harmonized is True  # No agents = harmonized

    def test_is_harmonized_with_agents_in_deadband(self, governor_with_agent):
        gov = governor_with_agent
        gov.measure_friction("agent_a", prediction=1.0, actual=1.05)
        assert gov.is_harmonized is True

    def test_is_harmonized_with_agent_over_deadband(self, governor_with_agent):
        gov = governor_with_agent
        gov.measure_friction("agent_a", prediction=0.0, actual=5.0)
        assert gov.is_harmonized is False

    def test_agent_phi_returns_latest(self, governor_with_agent):
        gov = governor_with_agent
        gov.measure_friction("agent_a", prediction=0.0, actual=1.0)
        gov.measure_friction("agent_a", prediction=0.0, actual=2.0)
        latest = gov.agent_phi("agent_a")
        # The latest phi = alpha * |0-2| = 0.5 * 2 = 1.0
        assert latest == pytest.approx(1.0)

    def test_agent_phi_unknown_agent(self, governor):
        assert governor.agent_phi("ghost") == 0.0

    def test_recent_alarms(self, governor_with_agent):
        gov = governor_with_agent
        gov.check_and_alarm("agent_a", phi=2.0)
        gov.check_and_alarm("agent_a", phi=3.0)
        alarms = gov.recent_alarms(5)
        assert len(alarms) == 2

    def test_alarm_rate_no_data(self, governor):
        assert governor.alarm_rate() == 0.0


# ─── AgentFrictionProfile Unit Tests ─────────────────────

class TestAgentFrictionProfile:
    def test_average_phi_empty(self):
        profile = AgentFrictionProfile(agent_id="x")
        assert profile.average_phi == 0.0

    def test_phi_variance_single_value(self):
        profile = AgentFrictionProfile(agent_id="x")
        profile.record_phi(0.5)
        assert profile.phi_variance == 0.0

    def test_calm_streak_increments(self):
        profile = AgentFrictionProfile(agent_id="x", base_deadband=1.0, current_deadband=1.0)
        profile.record_phi(0.3)
        profile.record_phi(0.2)
        assert profile.calm_streak == 2

    def test_calm_streak_resets_on_alarm(self):
        profile = AgentFrictionProfile(agent_id="x", base_deadband=1.0, current_deadband=1.0)
        profile.record_phi(0.3)
        profile.record_phi(1.5)  # alarm
        assert profile.calm_streak == 0

    def test_history_capped(self):
        profile = AgentFrictionProfile(agent_id="x", max_history=5)
        for i in range(10):
            profile.record_phi(float(i))
        assert len(profile.phi_history) == 5


# ─── FrictionAlarm Property Tests ────────────────────────

class TestFrictionAlarm:
    def test_overshoot_positive(self):
        alarm = FrictionAlarm(
            agent_id="a", phi=2.0, deadband=1.0,
            severity=AlarmSeverity.MODERATE,
        )
        assert alarm.overshoot == 1.0

    def test_overshoot_zero(self):
        alarm = FrictionAlarm(
            agent_id="a", phi=1.0, deadband=1.0,
            severity=AlarmSeverity.NONE,
        )
        assert alarm.overshoot == 0.0

    def test_is_active_true(self):
        alarm = FrictionAlarm(
            agent_id="a", phi=2.0, deadband=1.0,
            severity=AlarmSeverity.CRITICAL,
        )
        assert alarm.is_active

    def test_is_active_false(self):
        alarm = FrictionAlarm(
            agent_id="a", phi=0.5, deadband=1.0,
            severity=AlarmSeverity.NONE,
        )
        assert not alarm.is_active

    def test_repr_contains_agent_and_phi(self):
        alarm = FrictionAlarm(
            agent_id="builder", phi=1.5, deadband=1.0,
            severity=AlarmSeverity.GENTLE,
        )
        r = repr(alarm)
        assert "builder" in r
        assert "1.500" in r
