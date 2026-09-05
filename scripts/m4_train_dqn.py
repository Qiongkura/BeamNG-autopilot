"""M4 DQN training + evaluation (offline closed loop, no game needed).

Trains a Stable-Baselines3 DQN on ``DecisionSpeedEnv`` (mode="offline":
a game-free car-following simulator with the same decision observation
the live FSD drive feeds the policy) and evaluates it against the
always-cruise baseline:

    .venv\Scripts\python.exe scripts\m4_train_dqn.py --steps 60000
    .venv\Scripts\python.exe scripts\m4_train_dqn.py --eval-only \
        --weights logs\m4_dqn\dqn_decision.zip

Outputs: ``logs/m4_dqn/dqn_decision.zip`` + ``report.json`` (mean
episode reward, collision rate, mean speed - policy vs baseline).
``--sim`` switches the env to the live BeamNG connector for training
against the real stack (needs the game running; not part of the
offline loop).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.rl.env import DecisionSpeedEnv


def evaluate(model, env, episodes: int = 12) -> dict:
    """Roll the policy (or the always-cruise baseline) over episodes."""
    totals, collisions, speeds = [], 0, []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        vs = []
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = 0                  # baseline: always cruise
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            vs.append(float(info.get("speed", 0.0)))
            done = terminated or truncated
        totals.append(total)
        speeds.append(sum(vs) / max(1, len(vs)))
        collisions += 1 if info.get("collided") else 0
    return {
        "episodes": episodes,
        "mean_reward": round(sum(totals) / max(1, len(totals)), 2),
        "collision_rate": round(collisions / max(1, episodes), 3),
        "mean_speed": round(sum(speeds) / max(1, len(speeds)), 2),
    }


def _train_sim(args, out) -> int:
    """Online training against the live stack: the layered planner
    steers, the DQN decides the longitudinal level.  Guarded by the
    input watchdog; every episode ends with a full brake."""
    import math
    import time as _time

    import numpy as np
    from stable_baselines3 import DQN

    from beamng_autopilot.connector import BeamNGConnector
    from beamng_autopilot.fsd_stack import FSDStack
    from beamng_autopilot.roadnet import RoadNetwork
    from beamng_autopilot.vision.heads import (
        LaneTopologyHead, ObjectHead, SemanticHead, TrafficSignalHead,
    )
    from beamng_autopilot.watchdog import arm as wd_arm
    from beamng_autopilot.watchdog import disarm as wd_disarm

    conn = BeamNGConnector(
        "italy", "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    conn.open(launch=False)
    conn.attach_vehicle(already_open=True)
    stack = FSDStack(conn, args.runtime,
                     heads=[SemanticHead(), TrafficSignalHead(),
                            ObjectHead(), LaneTopologyHead()],
                     ring_roles=("front_main",),
                     cam_w=400, cam_h=300,
                     range_every_n=3, semantic_every_n=2,
                     object_every_n=2)
    stack.target_speed = 4.0
    nav_route = [None]

    def route_provider(pos, heading):
        if args.sim_goal is None:
            xs = np.linspace(0, 40, 41)
            return np.column_stack([pos[0] + xs * math.cos(heading),
                                    pos[1] + xs * math.sin(heading)])
        if nav_route[0] is None:
            rn = RoadNetwork()
            t0 = _time.time()
            while not rn.ready and _time.time() - t0 < 90.0:
                try:
                    if rn.build(conn.bng):
                        break
                except Exception:
                    pass
                _time.sleep(1.0)
            if rn.ready:
                n0 = rn.nodes[rn._nearest(pos[:2])]
                rwe = rn.route_with_edges(
                    n0, np.asarray(args.sim_goal, dtype=float))
                if rwe[0] is not None and len(rwe[0]) >= 4:
                    nav_route[0] = np.asarray(rwe[0][:, :2], dtype=float)
                    print(f"[m4-sim] route: {len(nav_route[0])} pts")
        r = nav_route[0]
        if r is None:
            xs = np.linspace(0, 40, 41)
            return np.column_stack([pos[0] + xs * math.cos(heading),
                                    pos[1] + xs * math.sin(heading)])
        from beamng_autopilot.planning.local_route import local_route
        return local_route(pos, heading, r)

    env = DecisionSpeedEnv(
        mode="sim", seed=args.seed, conn=conn, stack=stack,
        route_provider=route_provider, cruise_speed=4.0,
        episode_s=args.sim_decisions * 0.25)
    try:
        wd_arm(conn)
        model = DQN("MlpPolicy", env, seed=args.seed, verbose=0,
                    learning_rate=1e-3, buffer_size=20000,
                    batch_size=64, train_freq=2,
                    exploration_fraction=0.2, exploration_final_eps=0.1)
        print(f"[m4-sim] training {args.steps} decisions live "
              f"({args.sim_episodes} x {args.sim_decisions} steps)")
        model.learn(total_timesteps=int(args.steps),
                    reset_num_timesteps=False)
        model.save(str(out))
        print(f"[m4-sim] saved -> {out}")
    finally:
        try:
            conn.control(throttle=0.0, brake=1.0, steering=0.0)
            conn.step(5)
        except Exception:
            pass
        try:
            wd_disarm(conn)
        except Exception:
            pass
    print("[m4-sim] done (car braked, watchdog disarmed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="M4 DQN train + evaluate")
    ap.add_argument("--steps", type=int, default=60000,
                    help="training timesteps (offline env)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=None,
                    help="output zip path (default logs/m4_dqn/"
                         "dqn_decision.zip)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, evaluate existing weights")
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--sim", action="store_true",
                    help="train against the LIVE BeamNG stack (needs the "
                         "game running; the layered stack steers, the DQN "
                         "only caps the target speed)")
    ap.add_argument("--sim-goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="road-graph goal for the sim route (else "
                         "straight-ahead reference)")
    ap.add_argument("--sim-decisions", type=int, default=60,
                    help="decision steps per sim episode (0.25 s each)")
    ap.add_argument("--sim-episodes", type=int, default=1)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    args = ap.parse_args()

    from stable_baselines3 import DQN
    out = Path(args.out) if args.out else (
        config.LOGS_DIR / "m4_dqn" / "dqn_decision.zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.sim:
        return _train_sim(args, out)
    env = DecisionSpeedEnv(mode="offline", seed=args.seed)
    if not args.eval_only:
        model = DQN(
            "MlpPolicy", env, seed=args.seed, verbose=0,
            learning_rate=1e-3, buffer_size=60000,
            batch_size=128, train_freq=4, target_update_interval=500,
            exploration_fraction=0.3, exploration_final_eps=0.05)
        t0 = time.time()
        model.learn(total_timesteps=int(args.steps), progress_bar=False)
        train_s = time.time() - t0
        model.save(str(out))
        print(f"[m4] trained {args.steps} steps in {train_s:.0f}s "
              f"-> {out}")
    else:
        if not out.exists():
            print(f"[m4] weights not found: {out}")
            return 1
        model = DQN.load(str(out))
        print(f"[m4] loaded {out}")

    if args.no_eval:
        return 0
    report = {
        "weights": str(out),
        "policy": evaluate(model, DecisionSpeedEnv(
            mode="offline", seed=args.seed + 1),
            episodes=args.eval_episodes),
        "baseline_cruise": evaluate(None, DecisionSpeedEnv(
            mode="offline", seed=args.seed + 1),
            episodes=args.eval_episodes),
    }
    report_path = out.parent / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[m4] policy   : {report['policy']}")
    print(f"[m4] baseline : {report['baseline_cruise']}")
    print(f"[m4] report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
