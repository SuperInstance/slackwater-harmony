"""
Tests for the flow-state detector, protector, and journal.

These tests exercise the idea that Φ → 0 is Csikszentmihalyi flow:
low friction, steady cadence, focused actions, and persistent patterns
combine into a detectable flow state.
"""

import time

import pytest

from slackwater_harmony import (
    HarmonyGovernor,
    GrooveState,
    FlowPhase,
    FlowReading,
    FlowSession,
    TempoMap,
    ProtectiveAdjustment,
    FlowStateDetector,
    FlowStateProtector,
    FlowStateJournal,
)


class TestTempoMap:
    """Test adaptive tempo with flow-lock."""

    def test_initial_bpm(self):
        tempo = TempoMap()
        assert tempo.bpm == pytest.approx(120.0)
        assert not tempo.locked

    def test_set_target_clamps(self):
        tempo = TempoMap()
        tempo.set_target(200.0)
        assert tempo.target_bpm == pytest.approx(180.0)
        tempo.set_target(30.0)
        assert tempo.target_bpm == pytest.approx(60.0)

    def test_update_moves_toward_target(self):
        tempo = TempoMap(bpm=120.0, target_bpm=130.0, adapt_rate=0.5)
        tempo.update()
        assert tempo.bpm > 120.0
        assert tempo.bpm < 130.0

    def test_lock_prevents_change(self):
        tempo = TempoMap(bpm=120.0, target_bpm=140.0)
        tempo.lock(reason="flow")
        tempo.update()
        assert tempo.bpm == pytest.approx(120.0)
        assert tempo.locked
        assert tempo.lock_reason == "flow"

    def test_nudge_respects_lock(self):
        tempo = TempoMap(bpm=120.0)
        tempo.lock()
        tempo.nudge(10.0)
        assert tempo.target_bpm == pytest.approx(120.0)

    def test_beat_interval(self):
        tempo = TempoMap(bpm=120.0)
        assert tempo.beat_interval == pytest.approx(0.5)

    def test_average_bpm(self):
        tempo = TempoMap(bpm=100.0, target_bpm=100.0)
        for _ in range(3):
            tempo.update()
        assert tempo.average_bpm == pytest.approx(100.0)


class TestFlowStateDetector:
    """Test flow-signal computation and phase transitions."""

    def test_initial_phase(self):
        gov = HarmonyGovernor()
        gov.register_agent("a")
        detector = FlowStateDetector(governor=gov)
        assert detector.phase == FlowPhase.PRE_FLOW
        assert not detector.in_flow

    def test_record_action(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        detector.record_action("place", timestamp=0.0)
        detector.record_action("place", timestamp=1.0)
        assert detector.action_history == ["place", "place"]
        assert len(detector.delta_history) == 1
        assert detector.delta_history[0] == pytest.approx(1.0)

    def test_action_entropy_focused(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        for i in range(10):
            detector.record_action("place", timestamp=float(i))
        entropy = detector.measure_action_entropy()
        assert entropy < 0.2  # focused

    def test_action_entropy_scattered(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        actions = ["jump", "place", "rotate", "dash", "select", "move"]
        for i, action in enumerate(actions):
            detector.record_action(action, timestamp=float(i))
        entropy = detector.measure_action_entropy()
        assert entropy > 0.7  # scattered

    def test_cadence_regularity(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        for i in range(10):
            detector.record_action("tap", timestamp=float(i) * 1.0)
        assert detector.measure_cadence_regularity() > 0.9

    def test_cadence_irregular(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        timestamps = [0.0, 1.0, 1.2, 2.5, 3.0, 5.0]
        for t in timestamps:
            detector.record_action("tap", timestamp=t)
        assert detector.measure_cadence_regularity() < 0.5

    def test_micro_timing_consistent(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        for i in range(10):
            detector.record_action("tap", timestamp=float(i) * 1.0 + 0.01)
        assert detector.measure_micro_timing() > 0.9

    def test_micro_timing_inconsistent(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        timestamps = [0.0, 1.0, 2.8, 3.2, 5.5, 6.0]
        for t in timestamps:
            detector.record_action("tap", timestamp=t)
        assert detector.measure_micro_timing() < 0.5

    def test_hurst_random_walk(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        # Short series returns neutral
        assert detector.measure_hurst_exponent() == pytest.approx(0.5)

    def test_hurst_trending_series(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        for i in range(16):
            detector.record_action("place", timestamp=float(i), value=float(i) * 0.1)
        h = detector.measure_hurst_exponent()
        assert h > 0.5

    def test_compute_flow_score_increases_with_focus(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(governor=gov)
        # Focused, regular, consistent
        for i in range(12):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=i)
            detector.record_action("place", timestamp=float(i) * 1.0, value=0.9)
        score = detector.compute_flow_score()
        assert score > 0.0
        assert detector.last_reading is not None
        assert detector.last_reading.composite == pytest.approx(score)

    def test_update_flow_enters_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(
            governor=gov,
            flow_threshold=0.5,
            min_flow_sustained=2,
        )
        # Build focused, low-friction state
        for i in range(10):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=i)
            detector.record_action("place", timestamp=float(i) * 1.0, value=0.9)
            detector.update_flow(timestamp=float(i))
        assert detector.in_flow

    def test_update_flow_enters_deep_flow(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(
            governor=gov,
            flow_threshold=0.4,
            deep_flow_threshold=0.6,
            min_flow_sustained=2,
        )
        for i in range(20):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=i)
            detector.record_action("place", timestamp=float(i) * 1.0, value=0.95)
            detector.update_flow(timestamp=float(i))
        assert detector.phase == FlowPhase.DEEP_FLOW
        assert detector.in_deep_flow

    def test_flow_breaks_into_post_flow(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(
            governor=gov,
            flow_threshold=0.5,
            min_flow_sustained=2,
        )
        # Enter flow
        for i in range(10):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=i)
            detector.record_action("place", timestamp=float(i) * 1.0, value=0.9)
            detector.update_flow(timestamp=float(i))
        assert detector.in_flow

        # Break it with multiple beats of high friction and scattered behavior
        for i in range(3):
            gov.measure_friction("a", prediction=0, actual=10.0)
            detector.update(beat=10 + i)
            detector.record_action("panic", timestamp=10.0 + i, value=0.1)
            detector.record_action("menu", timestamp=10.1 + i, value=0.0)
            detector.record_action("die", timestamp=10.2 + i, value=0.0)
            detector.update_flow(timestamp=10.0 + i)
        assert not detector.in_flow

    def test_phase_distribution(self):
        gov = HarmonyGovernor()
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(
            governor=gov,
            flow_threshold=0.5,
            min_flow_sustained=2,
        )
        for i in range(8):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=i)
            detector.record_action("place", timestamp=float(i) * 1.0, value=0.9)
            detector.update_flow(timestamp=float(i))
        dist = detector.phase_distribution()
        assert sum(dist.values()) == pytest.approx(1.0)


class TestFlowStateProtector:
    """Test protective adjustments before flow breaks."""

    def test_detect_rising_friction(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(governor=gov)
        tempo = TempoMap()
        protector = FlowStateProtector(
            detector=detector,
            tempo=tempo,
            rising_friction_threshold=0.05,
        )

        # Build rising friction: 0 -> 0.5 over 6 readings
        for i in range(6):
            gov.measure_friction("a", prediction=0, actual=float(i) * 0.1)
            detector.update(beat=i)
        assert protector.detect_rising_friction()

    def test_detect_flat_friction(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(governor=gov)
        tempo = TempoMap()
        protector = FlowStateProtector(detector=detector, tempo=tempo)

        for i in range(6):
            gov.measure_friction("a", prediction=0, actual=0.1)
            detector.update(beat=i)
        assert not protector.detect_rising_friction()

    def test_engage_locks_tempo(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120.0, target_bpm=140.0)
        protector = FlowStateProtector(detector=detector, tempo=tempo)
        adj = protector.engage()
        assert adj is not None
        assert tempo.locked
        assert adj.is_gentle
        assert adj.chatter_reduction > 0

    def test_gentle_adjust_applies_when_active(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120.0)
        protector = FlowStateProtector(
            detector=detector,
            tempo=tempo,
            rising_friction_threshold=0.05,
        )

        protector.engage()
        for i in range(6):
            gov.measure_friction("a", prediction=0, actual=float(i) * 0.1)
            detector.update(beat=i)

        adj = protector.gentle_adjust()
        assert adj is not None
        assert adj.bpm_delta < 0
        assert adj.is_gentle
        assert protector.adjustment_count > 0

    def test_gentle_adjust_no_op_when_friction_flat(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(governor=gov)
        tempo = TempoMap()
        protector = FlowStateProtector(detector=detector, tempo=tempo)
        protector.engage()
        for i in range(6):
            gov.measure_friction("a", prediction=0, actual=0.1)
            detector.update(beat=i)
        assert protector.gentle_adjust() is None

    def test_disengage_restores_tempo(self):
        gov = HarmonyGovernor()
        detector = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120.0, target_bpm=120.0)
        protector = FlowStateProtector(detector=detector, tempo=tempo)
        protector.engage()
        assert tempo.locked
        protector.disengage()
        assert not tempo.locked
        assert not protector.active

    def test_tick_disengages_when_flow_breaks(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(
            governor=gov,
            flow_threshold=0.5,
            min_flow_sustained=2,
        )
        tempo = TempoMap()
        protector = FlowStateProtector(detector=detector, tempo=tempo)

        # Get into flow
        for i in range(10):
            gov.measure_friction("a", prediction=0, actual=0)
            detector.update(beat=i)
            detector.record_action("place", timestamp=float(i) * 1.0, value=0.9)
            detector.update_flow(timestamp=float(i))
        protector.engage()
        assert protector.active

        # Break flow and keep it broken long enough to enter RECOVERY
        for i in range(8):
            gov.measure_friction("a", prediction=0, actual=10.0)
            detector.update(beat=10 + i)
            detector.record_action("panic", timestamp=10.0 + i, value=0.1)
            detector.record_action("menu", timestamp=10.1 + i, value=0.0)
            detector.record_action("die", timestamp=10.2 + i, value=0.0)
            detector.update_flow(timestamp=10.0 + i)
            protector.tick()
        assert not protector.active

    def test_total_tempo_change_tracks_adjustments(self):
        gov = HarmonyGovernor(alpha=1.0, beta=0.0, gamma=0.0)
        gov.register_agent("a", base_deadband=5.0)
        detector = FlowStateDetector(governor=gov)
        tempo = TempoMap(bpm=120.0)
        protector = FlowStateProtector(
            detector=detector,
            tempo=tempo,
            rising_friction_threshold=0.05,
        )
        protector.engage()
        for i in range(6):
            gov.measure_friction("a", prediction=0, actual=float(i) * 0.1)
            detector.update(beat=i)
        protector.gentle_adjust()
        assert protector.total_tempo_change < 0


class TestFlowStateJournal:
    """Test recording of flow sessions."""

    def test_record_flow_start(self):
        journal = FlowStateJournal()
        session = journal.record_flow_start(
            timestamp=100.0,
            conditions={"bpm": 120.0, "friction": 0.1},
            player_state={"primary_action": "place"},
        )
        assert session is journal.current_session
        assert session.started_at == pytest.approx(100.0)

    def test_record_flow_score_updates_peak(self):
        journal = FlowStateJournal()
        journal.record_flow_start(
            timestamp=100.0,
            conditions={},
            player_state={},
        )
        journal.record_flow_score(0.7)
        journal.record_flow_score(0.9)
        journal.record_flow_score(0.8)
        assert journal.current_session.peak_score == pytest.approx(0.9)

    def test_record_flow_end_completes_session(self):
        journal = FlowStateJournal()
        journal.record_flow_start(
            timestamp=100.0,
            conditions={},
            player_state={},
        )
        journal.record_flow_score(0.8)
        session = journal.record_flow_end(
            timestamp=160.0,
            trigger="friction_spike",
            phase_reached=FlowPhase.FLOW,
        )
        assert session is not None
        assert session.is_complete
        assert session.duration == pytest.approx(60.0)
        assert session.avg_score == pytest.approx(0.8)
        assert session.end_trigger == "friction_spike"
        assert journal.current_session is None

    def test_no_end_without_start(self):
        journal = FlowStateJournal()
        assert journal.record_flow_end(timestamp=100.0, trigger="x") is None

    def test_total_flow_time(self):
        journal = FlowStateJournal()
        journal.record_flow_start(0.0, {}, {})
        journal.record_flow_end(30.0, "x")
        journal.record_flow_start(40.0, {}, {})
        journal.record_flow_end(100.0, "y")
        assert journal.total_flow_time == pytest.approx(90.0)

    def test_longest_flow(self):
        journal = FlowStateJournal()
        journal.record_flow_start(0.0, {}, {})
        journal.record_flow_end(10.0, "x")
        journal.record_flow_start(20.0, {}, {})
        journal.record_flow_end(50.0, "y")
        assert journal.longest_flow == pytest.approx(30.0)

    def test_average_peak_score(self):
        journal = FlowStateJournal()
        journal.record_flow_start(0.0, {}, {})
        journal.record_flow_score(0.7)
        journal.record_flow_end(10.0, "x")
        journal.record_flow_start(20.0, {}, {})
        journal.record_flow_score(0.9)
        journal.record_flow_end(30.0, "y")
        assert journal.average_peak_score == pytest.approx(0.8)

    def test_get_patterns_empty(self):
        journal = FlowStateJournal()
        patterns = journal.get_patterns()
        assert patterns["total_sessions"] == 0

    def test_get_patterns_with_sessions(self):
        journal = FlowStateJournal()
        journal.record_flow_start(
            0.0,
            {"bpm": 120.0, "friction": 0.1},
            {"primary_action": "place"},
        )
        journal.record_flow_score(0.8)
        journal.record_flow_end(60.0, "interruption", FlowPhase.FLOW)

        journal.record_flow_start(
            120.0,
            {"bpm": 125.0, "friction": 0.15},
            {"primary_action": "place"},
        )
        journal.record_flow_score(0.9)
        journal.record_flow_end(240.0, "friction_spike", FlowPhase.DEEP_FLOW)

        patterns = journal.get_patterns()
        assert patterns["total_sessions"] == 2
        assert patterns["avg_duration_minutes"] == pytest.approx(1.5)
        assert patterns["avg_peak_score"] == pytest.approx(0.85)
        assert patterns["most_common_actions"][0][0] == "place"
        assert "friction_spike" in dict(patterns["most_common_end_triggers"])

    def test_export_session_timeline(self):
        journal = FlowStateJournal()
        journal.record_flow_start(0.0, {}, {"primary_action": "place"})
        journal.record_flow_score(0.95)
        journal.record_flow_end(60.0, "natural_decay", FlowPhase.DEEP_FLOW)

        timeline = journal.export_session()
        assert timeline["session_count"] == 1
        assert len(timeline["movements"]) == 1
        movement = timeline["movements"][0]
        assert movement["dynamic"] == "fortissimo"
        assert movement["character"] == "andante"


class TestProtectiveAdjustment:
    """Test the protective adjustment dataclass."""

    def test_gentle_thresholds(self):
        adj = ProtectiveAdjustment(
            description="gentle",
            bpm_delta=-2.0,
            chatter_reduction=0.2,
            ambient_dim=0.1,
            friction_tolerance=0.1,
        )
        assert adj.is_gentle

    def test_not_gentle_when_too_large(self):
        adj = ProtectiveAdjustment(
            description="harsh",
            bpm_delta=-5.0,
            chatter_reduction=0.2,
            ambient_dim=0.1,
            friction_tolerance=0.1,
        )
        assert not adj.is_gentle


class TestFlowReading:
    """Test the flow reading dataclass."""

    def test_default_reading(self):
        reading = FlowReading()
        assert reading.entropy == 0.0
        assert reading.hurst == 0.5


class TestFlowSession:
    """Test the flow session dataclass."""

    def test_session_completion(self):
        session = FlowSession(started_at=0.0)
        assert not session.is_complete
        session.ended_at = 60.0
        session.duration = 60.0
        assert session.is_complete
        assert session.duration_minutes == pytest.approx(1.0)
