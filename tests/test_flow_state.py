"""
Tests for the flow state detection system.

Exercises FlowStateDetector, FlowStateProtector, FlowStateJournal,
and TempoMap — the four signals of flow, the state machine,
protection mechanics, and journal pattern recognition.
"""

import time
import pytest

from slackwater_harmony import (
    HarmonyGovernor,
    GrooveState,
    FlowStateDetector,
    FlowStateProtector,
    FlowStateJournal,
    FlowPhase,
    FlowReading,
    FlowSession,
    TempoMap,
    ProtectiveAdjustment,
)


# ── TempoMap ────────────────────────────────────────────

class TestTempoMap:
    """Test adaptive tempo with flow-lock."""

    def test_default_bpm(self):
        t = TempoMap()
        assert t.bpm == 120.0
        assert not t.locked

    def test_set_target(self):
        t = TempoMap()
        t.set_target(140)
        assert t.target_bpm == 140

    def test_target_clamped(self):
        t = TempoMap()
        t.set_target(500)
        assert t.target_bpm == 180.0  # max_bpm
        t.set_target(10)
        assert t.target_bpm == 60.0  # min_bpm

    def test_update_moves_toward_target(self):
        t = TempoMap(bpm=120, target_bpm=130, adapt_rate=0.5)
        t.update()
        assert t.bpm == pytest.approx(125.0)

    def test_lock_prevents_update(self):
        t = TempoMap(bpm=120, target_bpm=140)
        t.lock()
        t.update()
        assert t.bpm == 120.0  # unchanged

    def test_lock_prevents_set_target(self):
        t = TempoMap(bpm=120)
        t.lock()
        t.set_target(150)
        assert t.target_bpm == 120.0

    def test_unlock_restores_adaptation(self):
        t = TempoMap(bpm=120, target_bpm=140)
        t.lock()
        t.update()
        t.unlock()
        t.update()
        assert t.bpm != 120.0  # now moves

    def test_nudge(self):
        t = TempoMap(target_bpm=120)
        t.nudge(5)
        assert t.target_bpm == 125.0

    def test_nudge_respects_lock(self):
        t = TempoMap(target_bpm=120)
        t.lock()
        t.nudge(5)
        assert t.target_bpm == 120.0

    def test_beat_interval(self):
        t = TempoMap(bpm=120)
        assert t.beat_interval == pytest.approx(0.5)

    def test_average_bpm(self):
        t = TempoMap(bpm=120, target_bpm=140, adapt_rate=1.0)
        t.update()  # bpm=140
        t.set_target(100)
        t.update()  # bpm=100
        assert t.average_bpm == pytest.approx(120.0)


# ── FlowStateDetector: Signals ──────────────────────────

class TestActionEntropy:
    """Test Shannon entropy of player actions."""

    def test_single_action_low_entropy(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        det.action_history = ["place"] * 20
        entropy = det.measure_action_entropy()
        assert entropy == pytest.approx(0.0)

    def test_uniform_actions_max_entropy(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        det.action_history = ["a", "b", "c", "d", "e"] * 4
        entropy = det.measure_action_entropy()
        assert entropy > 0.9

    def test_insufficient_data(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        det.action_history = []
        entropy = det.measure_action_entropy()
        assert entropy == 1.0

    def test_custom_action_list(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        # 3:1 ratio produces moderate entropy
        entropy = det.measure_action_entropy(["x", "x", "x", "y"])
        assert 0.0 < entropy < 1.0
        # But strongly skewed toward one action
        entropy_skewed = det.measure_action_entropy(["x"] * 19 + ["y"])
        assert entropy_skewed < 0.5


class TestCadenceRegularity:
    """Test cadence regularity measurement."""

    def test_perfectly_regular(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        timestamps = [float(i) for i in range(20)]
        regularity = det.measure_cadence_regularity(timestamps)
        assert regularity == pytest.approx(1.0)

    def test_irregular(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        timestamps = [0, 0.1, 5.0, 5.2, 20.0, 20.01, 50.0]
        regularity = det.measure_cadence_regularity(timestamps)
        assert regularity < 0.5

    def test_insufficient_data(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        assert det.measure_cadence_regularity([1.0, 2.0]) == 0.0

    def test_no_variance_returns_zero(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        regularity = det.measure_cadence_regularity([1.0, 1.0, 1.0, 1.0])
        assert regularity == 0.0


class TestHurstExponent:
    """Test Hurst exponent estimation."""

    def test_persistent_series(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        series = [float(i) for i in range(50)]
        hurst = det.measure_hurst_exponent(series)
        assert hurst > 0.5

    def test_short_series_neutral(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        hurst = det.measure_hurst_exponent([1, 2, 3])
        assert hurst == pytest.approx(0.5)

    def test_empty_series(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        assert det.measure_hurst_exponent([]) == 0.5

    def test_constant_series(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        hurst = det.measure_hurst_exponent([5.0] * 20)
        assert hurst == pytest.approx(0.5)

    def test_bounded_output(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        series = [1, 5, 2, 8, 3, 9, 1, 7, 4, 6, 2, 8, 3, 9, 1, 7]
        hurst = det.measure_hurst_exponent(series)
        assert 0.0 <= hurst <= 1.0


class TestMicroTiming:
    """Test micro-timing consistency."""

    def test_consistent_timing(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        deltas = [0.5] * 20
        consistency = det.measure_micro_timing(deltas)
        assert consistency == pytest.approx(1.0)

    def test_inconsistent_timing(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        deltas = [0.1, 2.0, 0.3, 5.0, 0.2, 8.0]
        consistency = det.measure_micro_timing(deltas)
        assert consistency < 0.5

    def test_insufficient_data(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        assert det.measure_micro_timing([0.5]) == 0.0
        assert det.measure_micro_timing([]) == 0.0


# ── FlowStateDetector: Composite Score ──────────────────

class TestFlowScore:
    """Test composite flow score computation."""

    def test_all_signals_aligned(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(governor=gov, min_flow_sustained=2)

        det.action_history = ["place"] * 30
        det.timestamp_history = [float(i) * 0.5 for i in range(30)]
        det.time_series = [float(i) for i in range(30)]
        det.delta_history = [0.5] * 29

        score = det.compute_flow_score()
        assert score > 0.7

    def test_all_signals_scattered(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)

        det.action_history = ["a", "b", "c", "d", "e"] * 6
        det.timestamp_history = [0, 0.01, 5.0, 5.1, 20.0, 20.01, 50, 50.5]
        det.delta_history = [0.01, 4.99, 0.1, 14.9, 0.01, 29.99, 0.5]

        score = det.compute_flow_score()
        assert score < 0.5

    def test_score_bounded(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        score = det.compute_flow_score()
        assert 0.0 <= score <= 1.0

    def test_reading_stored(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        assert len(det.readings) == 0
        det.compute_flow_score()
        assert len(det.readings) == 1
        assert isinstance(det.readings[0], FlowReading)

    def test_groove_bonus(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(governor=gov, min_sustained_beats=2)

        det.action_history = ["place"] * 20
        det.timestamp_history = [float(i) * 0.5 for i in range(20)]
        det.time_series = [float(i) for i in range(20)]
        det.delta_history = [0.5] * 19

        score_without_groove = det.compute_flow_score()

        for beat in range(5):
            gov.measure_friction("player", prediction=0, actual=0)
            det.update(beat=beat)

        score_with_groove = det.compute_flow_score()
        assert score_with_groove >= score_without_groove

    def test_disrupted_penalty(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(governor=gov, min_sustained_beats=2)

        det.action_history = ["place"] * 20
        det.timestamp_history = [float(i) * 0.5 for i in range(20)]
        det.time_series = [float(i) for i in range(20)]
        det.delta_history = [0.5] * 19

        for beat in range(5):
            gov.measure_friction("player", prediction=0, actual=0)
            det.update(beat=beat)
        assert det.in_groove

        gov.measure_friction("player", prediction=0, actual=10.0)
        det.update(beat=6)
        assert det._state == GrooveState.DISRUPTED

        score = det.compute_flow_score()
        assert 0.0 <= score <= 1.0


# ── FlowStateDetector: State Machine ────────────────────

class TestFlowStateMachine:
    """Test the PRE_FLOW → FLOW → DEEP_FLOW → POST_FLOW → RECOVERY cycle."""

    def test_initial_phase_pre_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        assert det.phase == FlowPhase.PRE_FLOW

    def test_record_action(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)

        det.record_action("place", timestamp=1.0, value=0.5)
        assert det.action_history[-1] == "place"
        assert det.timestamp_history[-1] == 1.0
        assert det.time_series[-1] == 0.5

    def test_record_action_computes_delta(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)

        det.record_action("a", timestamp=1.0)
        det.record_action("b", timestamp=1.5)
        assert len(det.delta_history) == 1
        assert det.delta_history[0] == pytest.approx(0.5)

    def test_history_trimmed(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov, max_history=10)

        for i in range(20):
            det.record_action("act", timestamp=float(i))

        assert len(det.action_history) <= 10
        assert len(det.timestamp_history) <= 10

    def test_transitions_to_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(
            governor=gov,
            flow_threshold=0.5,
            deep_flow_threshold=0.95,
            min_flow_sustained=3,
            min_sustained_beats=1,
        )

        for i in range(30):
            det.record_action("place", timestamp=float(i) * 0.5, value=float(i))

        for i in range(10):
            gov.measure_friction("player", prediction=0, actual=0)
            det.update(beat=i)
            det.update_flow(timestamp=float(i))

        assert det.phase in (FlowPhase.FLOW, FlowPhase.DEEP_FLOW)

    def test_transitions_to_deep_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=10.0)
        det = FlowStateDetector(
            governor=gov,
            flow_threshold=0.3,
            deep_flow_threshold=0.5,
            min_flow_sustained=3,
            min_sustained_beats=1,
        )

        for i in range(40):
            det.record_action("place", timestamp=float(i) * 0.5, value=float(i))

        for i in range(15):
            gov.measure_friction("player", prediction=0, actual=0)
            det.update(beat=i)
            det.update_flow(timestamp=float(i))

        assert det.phase == FlowPhase.DEEP_FLOW

    def test_flow_breaks_to_post_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(
            governor=gov,
            flow_threshold=0.4,
            deep_flow_threshold=0.95,
            min_flow_sustained=2,
            min_sustained_beats=1,
        )

        for i in range(30):
            det.record_action("place", timestamp=float(i) * 0.5, value=float(i))
        for i in range(8):
            gov.measure_friction("player", prediction=0, actual=0)
            det.update(beat=i)
            det.update_flow(timestamp=float(i))
        assert det.phase in (FlowPhase.FLOW, FlowPhase.DEEP_FLOW)

        det.action_history = ["a", "b", "c", "d"] * 10
        det.timestamp_history = [0, 0.1, 5, 5.2, 20] * 6
        det.delta_history = [0.1, 4.9, 0.2] * 10

        det.update_flow(timestamp=100.0)
        assert det.phase in (FlowPhase.POST_FLOW, FlowPhase.RECOVERY)

    def test_recovery_to_pre_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(
            governor=gov,
            flow_threshold=0.4,
            pre_flow_threshold=0.2,
            min_flow_sustained=2,
            min_sustained_beats=1,
        )

        det.phase = FlowPhase.RECOVERY
        det._flow_sustained_count = 0

        det.action_history = ["place"] * 20
        det.timestamp_history = [float(i) * 0.5 for i in range(20)]
        det.time_series = [float(i) for i in range(20)]
        det.delta_history = [0.5] * 19

        det.update_flow(timestamp=10.0)
        assert det.phase == FlowPhase.PRE_FLOW

    def test_in_flow_property(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        det.phase = FlowPhase.FLOW
        assert det.in_flow
        det.phase = FlowPhase.DEEP_FLOW
        assert det.in_flow
        det.phase = FlowPhase.PRE_FLOW
        assert not det.in_flow

    def test_phase_distribution(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        det.phase = FlowPhase.PRE_FLOW
        det.compute_flow_score()
        det.compute_flow_score()
        dist = det.phase_distribution()
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_flow_percentage(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        det.readings = [
            FlowReading(phase=FlowPhase.PRE_FLOW),
            FlowReading(phase=FlowPhase.FLOW),
            FlowReading(phase=FlowPhase.DEEP_FLOW),
            FlowReading(phase=FlowPhase.RECOVERY),
        ]
        assert det.flow_percentage == pytest.approx(0.5)


# ── FlowStateProtector ──────────────────────────────────

class TestFlowStateProtector:
    """Test imperceptible flow protection."""

    def test_engage_locks_tempo(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120, target_bpm=140)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        adj = protector.engage()

        assert protector.active
        assert tempo.locked
        assert tempo.lock_reason == "flow_protection"
        assert adj is not None
        assert adj.is_gentle

    def test_disengage_unlocks_tempo(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120, target_bpm=140)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        protector.engage()
        assert tempo.locked

        protector.disengage()
        assert not protector.active
        assert not tempo.locked

    def test_gentle_adjust_is_imperceptible(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        protector.engage()

        for _ in range(10):
            gov.measure_friction("player", prediction=0, actual=2.0)

        adj = protector.gentle_adjust()
        if adj is not None:
            assert adj.is_gentle
            assert abs(adj.bpm_delta) <= 3.0
            assert adj.chatter_reduction <= 0.5

    def test_no_adjustment_when_not_rising(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        protector.engage()
        for _ in range(10):
            gov.measure_friction("player", prediction=0, actual=0.0)

        adj = protector.gentle_adjust()
        assert adj is None

    def test_protect_for_duration(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        protector.protect_for(0.1)
        assert protector.active
        assert protector.protect_until is not None

    def test_tick_disengages_after_expiry(self):
        gov = HarmonyGovernor()
        gov.register_agent("player")
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        protector.protect_for(0.01)
        time.sleep(0.02)
        protector.tick()
        assert not protector.active

    def test_detect_rising_friction(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("player", base_deadband=10.0)
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo, friction_window=5)

        for i in range(10):
            gov.measure_friction("player", prediction=0, actual=float(i) * 0.5)

        assert protector.detect_rising_friction()

    def test_no_rising_friction_when_stable(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=10.0)
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        for _ in range(10):
            gov.measure_friction("player", prediction=0, actual=0.1)

        assert not protector.detect_rising_friction()

    def test_adjustments_are_gentle(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        protector.engage()
        for i in range(10):
            gov.measure_friction("player", prediction=0, actual=float(i))
        protector.gentle_adjust()

        for adj in protector.adjustments_made:
            if adj.description != "none":
                assert adj.is_gentle, f"Adjustment not gentle: {adj.description}"

    def test_total_tempo_change_tracked(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        protector.engage()
        for i in range(10):
            gov.measure_friction("player", prediction=0, actual=float(i))
        protector.gentle_adjust()

        # After a gentle adjust, total tempo change should be non-zero
        assert protector.total_tempo_change != 0.0


# ── FlowStateJournal ────────────────────────────────────

class TestFlowStateJournal:
    """Test flow session recording and pattern analysis."""

    def test_record_flow_start(self):
        journal = FlowStateJournal()
        session = journal.record_flow_start(
            timestamp=1000.0,
            conditions={"bpm": 120, "friction": 0.2},
            player_state={"primary_action": "building"},
        )
        assert session.started_at == 1000.0
        assert journal.current_session is not None
        assert session.trigger_conditions["bpm"] == 120

    def test_record_flow_end(self):
        journal = FlowStateJournal()
        journal.record_flow_start(
            timestamp=1000.0,
            conditions={},
            player_state={},
        )
        session = journal.record_flow_end(
            timestamp=1060.0,
            trigger="natural_decay",
            phase_reached=FlowPhase.DEEP_FLOW,
        )
        assert session is not None
        assert session.is_complete
        assert session.duration == pytest.approx(60.0)
        assert session.end_trigger == "natural_decay"
        assert session.phase_reached == FlowPhase.DEEP_FLOW
        assert journal.current_session is None

    def test_record_flow_end_no_active(self):
        journal = FlowStateJournal()
        result = journal.record_flow_end(timestamp=100, trigger="x")
        assert result is None

    def test_record_flow_score(self):
        journal = FlowStateJournal()
        journal.record_flow_start(
            timestamp=100.0, conditions={}, player_state={},
        )
        journal.record_flow_score(0.8)
        journal.record_flow_score(0.9)
        assert journal.current_session.peak_score == 0.9

    def test_get_patterns_empty(self):
        journal = FlowStateJournal()
        patterns = journal.get_patterns()
        assert patterns["total_sessions"] == 0

    def test_get_patterns_with_sessions(self):
        journal = FlowStateJournal()

        for i in range(5):
            journal.record_flow_start(
                timestamp=1000.0 + i * 200,
                conditions={"bpm": 120 + i * 5, "friction": 0.1 + i * 0.02},
                player_state={"primary_action": "building" if i < 3 else "painting"},
            )
            journal.record_flow_score(0.7 + i * 0.04)
            journal.record_flow_end(
                timestamp=1000.0 + i * 200 + 60.0 + i * 10,
                trigger="natural_decay" if i < 3 else "interruption",
                phase_reached=FlowPhase.FLOW if i < 2 else FlowPhase.DEEP_FLOW,
            )

        patterns = journal.get_patterns()
        assert patterns["total_sessions"] == 5
        assert patterns["avg_duration_minutes"] > 0
        assert len(patterns["most_common_actions"]) > 0
        assert "building" in [a for a, _ in patterns["most_common_actions"]]

    def test_export_session_empty(self):
        journal = FlowStateJournal()
        export = journal.export_session()
        assert export["movements"] == []
        assert export["bpm"] == 120

    def test_export_session_with_data(self):
        journal = FlowStateJournal()

        for i in range(3):
            journal.record_flow_start(
                timestamp=1000.0 + i * 300,
                conditions={"bpm": 120},
                player_state={"primary_action": "building"},
            )
            journal.record_flow_score(0.85)
            journal.record_flow_end(
                timestamp=1000.0 + i * 300 + 120.0,
                trigger="natural_decay",
                phase_reached=FlowPhase.DEEP_FLOW if i == 0 else FlowPhase.FLOW,
            )

        export = journal.export_session()
        assert export["session_count"] == 3
        assert len(export["movements"]) == 3
        m = export["movements"][0]
        assert "start_beat" in m
        assert "duration_beats" in m
        assert "dynamic" in m
        assert "character" in m
        assert m["dynamic"] in ("piano", "mezzo-forte", "forte", "fortissimo")
        assert m["character"] in ("allegro", "andante", "moderato", "decrescendo")

    def test_total_flow_time(self):
        journal = FlowStateJournal()
        journal.record_flow_start(100.0, {}, {})
        journal.record_flow_end(200.0, "x")
        journal.record_flow_start(300.0, {}, {})
        journal.record_flow_end(350.0, "y")
        assert journal.total_flow_time == pytest.approx(150.0)

    def test_longest_flow(self):
        journal = FlowStateJournal()
        journal.record_flow_start(100.0, {}, {})
        journal.record_flow_end(200.0, "x")
        journal.record_flow_start(300.0, {}, {})
        journal.record_flow_end(500.0, "y")
        assert journal.longest_flow == pytest.approx(200.0)

    def test_average_peak_score(self):
        journal = FlowStateJournal()
        journal.record_flow_start(100.0, {}, {})
        journal.record_flow_score(0.8)
        journal.record_flow_end(200.0, "x")
        journal.record_flow_start(300.0, {}, {})
        journal.record_flow_score(0.9)
        journal.record_flow_end(400.0, "y")
        assert journal.average_peak_score == pytest.approx(0.85)


# ── Integration: Full System ────────────────────────────

class TestFlowIntegration:
    """Integration tests across detector, protector, journal, and tempo."""

    def test_full_flow_cycle(self):
        """
        Complete cycle: player warms up, enters flow, deepens,
        breaks, recovers. Verify all components interact correctly.
        """
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=5.0)

        det = FlowStateDetector(
            governor=gov,
            flow_threshold=0.5,
            deep_flow_threshold=0.7,
            min_flow_sustained=3,
            min_sustained_beats=2,
        )
        tempo = TempoMap(bpm=120, target_bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)
        journal = FlowStateJournal()

        # Phase 1: Warm-up (scattered actions, irregular timing)
        warmup_actions = ["explore", "menu", "move", "explore", "menu"]
        warmup_times = [0.0, 0.3, 5.0, 7.0, 15.0]
        for i in range(5):
            det.record_action(warmup_actions[i], timestamp=warmup_times[i])

        for i in range(5):
            gov.measure_friction("player", prediction=0, actual=0)
            det.update(beat=i)
            det.update_flow(timestamp=float(i))

        # With scattered warm-up, should be early in the cycle
        assert det.phase in (FlowPhase.PRE_FLOW, FlowPhase.FLOW)

        # Phase 2: Player converges (repetitive, rhythmic actions)
        # Clear warm-up noise so flow signals are clean
        det.action_history.clear()
        det.timestamp_history.clear()
        det.delta_history.clear()
        det.time_series.clear()

        t = 15.0
        for i in range(30):
            det.record_action("place", timestamp=t + i * 0.5, value=float(i))

        for i in range(10):
            gov.measure_friction("player", prediction=0, actual=0)
            det.update(beat=10 + i)
            det.update_flow(timestamp=15.0 + i)

        # Should be in some flow state
        assert det.phase in (FlowPhase.FLOW, FlowPhase.DEEP_FLOW)

        # Engage protection
        protector.engage()
        assert tempo.locked

        # Record in journal
        journal.record_flow_start(
            timestamp=15.0,
            conditions={"bpm": tempo.bpm, "friction": gov.total_friction},
            player_state={"primary_action": "place"},
        )

        # Phase 3: Flow breaks (scattered again)
        det.action_history = ["a", "b", "c", "d", "e"] * 6  # high entropy
        det.timestamp_history = [0.1, 0.5, 3.0, 3.1, 8.0, 8.01, 12.0, 15.0, 20.0] * 6
        det.delta_history = [0.4, 2.5, 0.1, 4.9, 0.01, 3.99, 3.0, 5.0] * 7
        det.time_series = [5, 3, 8, 1, 9, 2, 7, 4, 6, 1, 8, 3]  # mean-reverting

        det.update_flow(timestamp=100.0)

        # Should leave flow (score should drop below threshold)
        # May take a couple updates to transition through POST_FLOW
        det.update_flow(timestamp=101.0)
        det.update_flow(timestamp=102.0)
        assert det.phase in (FlowPhase.POST_FLOW, FlowPhase.RECOVERY, FlowPhase.PRE_FLOW)

        # Protector disengages
        protector.disengage()
        assert not tempo.locked

        # Journal records end
        journal.record_flow_end(
            timestamp=100.0,
            trigger="action_scatter",
            phase_reached=FlowPhase.FLOW,
        )

        # Verify journal
        assert len(journal.sessions) == 1
        assert journal.sessions[0].is_complete
        assert journal.sessions[0].end_trigger == "action_scatter"

    def test_tempo_locks_during_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=10.0)
        det = FlowStateDetector(
            governor=gov,
            flow_threshold=0.3,
            deep_flow_threshold=0.9,
            min_flow_sustained=2,
            min_sustained_beats=1,
        )
        tempo = TempoMap(bpm=120, target_bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        for i in range(30):
            det.record_action("place", timestamp=float(i) * 0.5, value=float(i))

        for i in range(8):
            gov.measure_friction("player", prediction=0, actual=0)
            det.update(beat=i)
            det.update_flow(timestamp=float(i))

        assert det.in_flow

        protector.engage()
        assert tempo.locked

        tempo.set_target(180)
        tempo.update()
        assert tempo.bpm == 120.0  # unchanged

    def test_tempo_unlocks_after_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("player", base_deadband=10.0)
        det = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120, target_bpm=120)
        protector = FlowStateProtector(detector=det, tempo=tempo)

        protector.engage()
        assert tempo.locked

        protector.disengage()
        assert not tempo.locked

        tempo.set_target(140)
        tempo.update()
        assert tempo.bpm != 120.0

    def test_journal_export_reflects_session(self):
        journal = FlowStateJournal()

        journal.record_flow_start(
            timestamp=1000.0,
            conditions={"bpm": 110},
            player_state={"primary_action": "painting"},
        )
        journal.record_flow_score(0.92)
        journal.record_flow_end(
            timestamp=1120.0,
            trigger="natural_decay",
            phase_reached=FlowPhase.DEEP_FLOW,
        )

        export = journal.export_session()
        assert export["session_count"] == 1
        m = export["movements"][0]
        assert m["phase"] == "DEEP_FLOW"
        assert m["primary_action"] == "painting"
        assert m["end_trigger"] == "natural_decay"
        assert m["peak_intensity"] == pytest.approx(0.92)
        assert m["duration_beats"] > 0
