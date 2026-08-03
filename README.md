# Slackwater Harmony

*The triadic cognitive architecture for game agents. Sandbox → Governor → Executive. Based on the Free Energy Principle.*

A Python package that gives game agents a triadic cognitive architecture adapted from Snapkit v2. Agents don't just respond — they simulate, monitor friction, and improvise.

## The Three Layers

### Layer 1: Sandbox (Forward Simulation)
Before an agent acts, it simulates the action in a headless sandbox. "If I place this wall here, does it overlap the dock? Does it block the path to the forge?" The sandbox runs the forward model and scores the result.

### Layer 2: Harmony Governor (Friction Monitoring)
The governor measures Φ (phi) — the cognitive friction between what the agent expected and what actually happened. Low Φ: the agent's model is accurate. High Φ: something surprised it. When Φ exceeds the deadband, the Executive wakes.

For game agents, friction comes from:
- Player behavior that doesn't match the agent's prediction
- Build outcomes that don't match the plan
- Environmental changes (storms, tide shifts) that break the model

### Layer 3: Executive (Improvisation)
When friction exceeds the deadband, the Executive improvises. It rewrites constraints, cross-wires I/O, tries something the agent has never done before. In game terms: Lucineer adapts. He changes the plan. He tries a different approach. He says something he's never said before.

## What this enables

- Agents that NOTICE when things aren't going as planned
- Agents that IMPROVISE instead of repeating the same error
- Agents that ADAPT to the player's skill level and energy
- The "in the pocket" measurement — when Φ is low across all agents, the system knows the groove is happening

## Related

- [Snapkit v2](https://github.com/SuperInstance/snapkit-v2) — the original triadic architecture
- [Slackwater Tempo](https://github.com/SuperInstance/slackwater-tempo) — the shared tempo that feeds the governor
