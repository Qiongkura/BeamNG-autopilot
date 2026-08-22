"""FSD-style layered planner.

Tesla FSD plans in layers over a vector-space/occupancy scene: it does
not steer straight from a single tracked path but evaluates many
candidate trajectories against the fused environment (occupancy, lane
geometry, rules) and picks the feasible, comfortable one.  This package
gives the project the same *shape* on top of the existing rule planner:

* ``scene.py`` - the Scene: a snapshot of occupancy grid + lane/route
  reference + ego state that every planner stage reads (the FSD
  "vector space" the planner consumes).
* ``trajectory.py`` - CandidateSet: generates a fan of candidate
  trajectories (arc/lane-shift samples + the legacy planner's path as a
  prior) parameterised by path.
* ``constraints.py`` - feasibility / cost evaluation: collision with
  the occupancy grid, lane alignment, curvature/comfort, speed.
* ``selector.py`` - picks the best feasible trajectory.
* ``arbiter.py`` - FSD trajectory vs rule fallback arbitration (the car
  never stops dead when the layered planner declines).

The existing ``LocalPlanner``/``PurePursuit`` stay the execution layer;
this package *prepends* visualised/model-level planning rather than
replacing it, so the 94.6% driving result is untouched until this stack
is proven and switched over.
"""

from .scene import Scene
from .trajectory import CandidateSet, sample_arc, sample_lane_shift
from .constraints import (
    lane_cross_dist_m,
    Constraints,
    corridor_free_band,
    cost_collision,
    cost_curvature,
    cost_lane_align,
)
from .selector import select_trajectory
from .arbiter import (
    ArbiterOutcome,
    anchored_rule_ref,
    arbitrate,
    choose_plan_route,
)
from .speed_profile import speed_profile_for_path
from .intent import (
    RoutingIntent,
    infer_route_intent,
)

__all__ = [
    "Scene",
    "CandidateSet",
    "sample_arc",
    "sample_lane_shift",
    "Constraints",
    "corridor_free_band",
    "cost_collision",
    "cost_curvature",
    "lane_cross_dist_m",
    "cost_lane_align",
    "select_trajectory",
    "ArbiterOutcome",
    "anchored_rule_ref",
    "arbitrate",
    "choose_plan_route",
    "speed_profile_for_path",
]