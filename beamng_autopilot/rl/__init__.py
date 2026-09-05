"""M4 decision layer: DQN over the FSD stack's perception observation.

The decision layer sits ABOVE the planner: it does not steer - it caps
the plan's target speed with discrete decisions (cruise / ease / slow),
exactly the cruise/decelerate action set M4 designed.  Everything here
runs in two modes:

* offline - a game-free car-following simulator (``DecisionSpeedEnv``
  with ``mode="offline"``) so training + evaluation close the loop
  without BeamNG;
* sim - the same env driven by ``BeamNGConnector`` + ``FSDStack``
  perception for training against the real stack.
"""
from beamng_autopilot.rl.obs import decision_observation, DECISION_OBS_SIZE
from beamng_autopilot.rl.env import DecisionSpeedEnv
from beamng_autopilot.rl.dqn_runtime import DQNRuntime, action_to_target

__all__ = [
    "decision_observation", "DECISION_OBS_SIZE",
    "DecisionSpeedEnv", "DQNRuntime", "action_to_target",
]
