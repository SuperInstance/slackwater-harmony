"""
slackwater-harmony: Cognitive friction monitoring and FEP-driven improvisation.

The Harmony Governor measures Φ — cognitive friction — the gap between
what an agent expects and what actually happens. When Φ exceeds the
deadband, the Executive wakes and improvises.

Three layers, following the snapkit-v2 architecture:
    Layer 1 — Sandbox:     Forward simulation, hypothesis testing
    Layer 2 — Governor:     Friction measurement, deadband enforcement
    Layer 3 — Executive:    Improvisation when friction alarms fire

When Φ is low across all agents, the system is "in the pocket."
The groove detector sees it. The agents are harmonized.

    >>> from slackwater_harmony import HarmonyGovernor, HypothesisSandbox, ExecutiveAgent
"""

from slackwater_harmony.governor import HarmonyGovernor, FrictionAlarm
from slackwater_harmony.sandbox import HypothesisSandbox, SandboxResult
from slackwater_harmony.executive import ExecutiveAgent, Improvisation
from slackwater_harmony.groove_detector import GrooveDetector, GrooveState
from slackwater_harmony.flow_state import (
    FlowPhase,
    FlowReading,
    FlowSession,
    TempoMap,
    ProtectiveAdjustment,
    FlowStateDetector,
    FlowStateProtector,
    FlowStateJournal,
)

__all__ = [
    "HarmonyGovernor",
    "FrictionAlarm",
    "HypothesisSandbox",
    "SandboxResult",
    "ExecutiveAgent",
    "Improvisation",
    "GrooveDetector",
    "GrooveState",
    "FlowPhase",
    "FlowReading",
    "FlowSession",
    "TempoMap",
    "ProtectiveAdjustment",
    "FlowStateDetector",
    "FlowStateProtector",
    "FlowStateJournal",
]

__version__ = "0.1.0"
