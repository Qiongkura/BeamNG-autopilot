"""M5-L 横向残差 DQN 训练 + 评估（offline 程序化道路，无游戏）.

在 ``LateralResidualEnv``（mode="offline"）上训练 Stable-Baselines3 DQN，
学"车道内何时压多大横移残差"；对照基线是永远零残差（纯基控）：

    .venv\\Scripts\\python.exe scripts\\m5_train_lateral_dqn.py --steps 40000
    .venv\\Scripts\\python.exe scripts\\m5_train_lateral_dqn.py --eval-only \\
        --weights logs\\m5_rl\\dqn_lateral.zip

输出：``logs/m5_rl/dqn_lateral.zip`` + ``report.json``（平均回合奖励、
|横向误差|均值、越界率——策略 vs 零残差基线）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beamng_autopilot import config
from beamng_autopilot.rl.lateral_env import LateralResidualEnv, ROAD_HALF_M


def _eval_policy(model, env, episodes: int = 20) -> dict:
    """评估：平均奖励、|横向误差|、越界率。"""
    totals, lat_abs, off = [], [], 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_r, ep_lat, ep_n = 0.0, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            ep_r += r
            ep_n += 1
            lat = float(env.lat)
            ep_lat += abs(lat)
            if abs(lat) > ROAD_HALF_M:
                off += 1
            done = term or trunc
        totals.append(ep_r)
        lat_abs.append(ep_lat / max(ep_n, 1))
    return {"mean_reward": float(np.mean(totals)),
            "mean_abs_lat": float(np.mean(lat_abs)),
            "offroad_rate": off / max(episodes * env.episode_steps, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="横向残差 DQN 训练")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str,
                    default=str(config.LOGS_DIR / "m5_rl" / "dqn_lateral.zip"))
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    from stable_baselines3 import DQN

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        model = DQN.load(str(out))
    else:
        env = LateralResidualEnv(mode="offline", seed=args.seed)
        model = DQN("MlpPolicy", env, seed=args.seed,
                    learning_rate=3e-4, buffer_size=60000,
                    batch_size=128, gamma=0.99, train_freq=4,
                    exploration_fraction=0.25, verbose=0)
        model.learn(total_timesteps=args.steps, progress_bar=False)
        model.save(str(out))

    env = LateralResidualEnv(mode="offline", seed=args.seed + 1)
    policy = _eval_policy(model, env)

    base = LateralResidualEnv(mode="offline", seed=args.seed + 1)

    class _Zero:
        def predict(self, obs, deterministic=True):
            return np.int64(2), None      # 动作 2 = 零残差

    baseline = _eval_policy(_Zero(), base)
    report = {"policy": policy, "baseline_zero_residual": baseline,
              "steps": args.steps, "seed": args.seed}
    (out.with_suffix(".report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
