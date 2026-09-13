"""横向残差决策 env：RL 只学"车道内有界横移修正"，转向主干仍归基控.

``LateralResidualEnv`` 是纵向 ``DecisionSpeedEnv`` 的横向姊妹篇：

* 动作（Discrete(5)）：有界横向偏移残差 {-0.5, -0.25, 0, +0.25, +0.5} m，
  叠加在基控（PurePursuit/FSD 轨迹）瞄准的车道中心上；残差每步向 0 衰减
  （自回中），RL 永远无法把车一把甩出路面。
* 两种模式共用一个契约：
  - ``mode="offline"``（默认）：无游戏的程序化道路——曲率随机游走，
    弯道把车往外推（drift），基控把车往目标拉（corr），观测里只有
    带噪的横向/航向误差。训的就是"何时压多大残差"这个决策形状。
  - ``mode="sim"``：live FSDStack/connector，残差写进规划目标的横向
    偏移，reward 的横向误差来自 ``painted_line_lane_center``/BEV。
* Reward：前进 (v·dt) − 4·|横向误差|·dt − 越界大罚 − 残差动作成本。

观测（5 维，全部归一）：横向误差、航向误差、道路曲率、速度、当前残差。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_OFFSETS = (-0.5, -0.25, 0.0, 0.25, 0.5)
DECAY = 0.92               # 残差每步自回中
DT = 0.1
EPISODE_STEPS = 300
ROAD_HALF_M = 1.8          # 半车道宽：|lat| 超过即越界
OFF_ROAD_PENALTY = 8.0
LAT_GAIN = 4.0
ACTION_COST = 0.02
DELTA_COST = 0.1
CTRL_GAIN = 2.0            # 基控回拉速率（一阶）
DRIFT_GAIN = 0.5           # 弯道外推：curv*v^2*DRIFT_GAIN*dt
LAT_NOISE = 0.01
CURV_MAX = 0.02            # 最小转弯半径 50 m


class LateralResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mode: str = "offline", seed: int | None = None,
                 episode_steps: int = EPISODE_STEPS):
        super().__init__()
        assert mode in ("offline", "sim")
        self.mode = mode
        self.episode_steps = int(episode_steps)
        self.action_space = spaces.Discrete(len(ACTION_OFFSETS))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._reset_state()

    # -- 状态 ----------------------------------------------------------------
    def _reset_state(self) -> None:
        self.lat = float(self.rng.uniform(-0.4, 0.4))
        self.heading = float(self.rng.uniform(-0.15, 0.15))
        self.curv = 0.0
        self.v = float(self.rng.uniform(7.0, 13.0))
        self.applied = 0.0
        self.t = 0

    def _obs(self) -> np.ndarray:
        return np.array([self.lat / ROAD_HALF_M,
                         self.heading / 0.5,
                         self.curv / CURV_MAX,
                         self.v / 14.0,
                         self.applied / 0.5], dtype=np.float32)

    # -- gym 接口 ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        applied_target = ACTION_OFFSETS[int(action)]
        self.applied = self.applied * DECAY + applied_target * (1 - DECAY)
        reward, terminated, truncated = 0.0, False, False
        if self.mode == "offline":
            reward = self._step_offline()
        else:
            reward, terminated, truncated = self._step_sim()
        self.t += 1
        if self.t >= self.episode_steps:
            truncated = True
        return self._obs(), float(reward), terminated, truncated, {}

    # -- offline 程序化道路 --------------------------------------------------
    def _step_offline(self) -> float:
        r = self.rng
        # 曲率 OU 随机游走 + 速度慢随机游走
        self.curv = float(np.clip(
            self.curv + r.normal(0, 0.002) - 0.05 * self.curv,
            -CURV_MAX, CURV_MAX))
        self.v = float(np.clip(self.v + r.normal(0, 0.15), 6.0, 14.0))
        # 弯道外推 + 基控回拉（RL 残差平移目标） + 噪声
        drift = self.curv * self.v * self.v * DRIFT_GAIN * DT
        corr = CTRL_GAIN * (self.lat + self.applied) * DT
        self.lat += drift - corr + float(r.normal(0, LAT_NOISE))
        self.heading += (self.curv - CTRL_GAIN * 0.3 * self.heading) * self.v * DT

        reward = self.v * DT                       # 前进进度
        reward -= LAT_GAIN * abs(self.lat) * DT    # 压横向误差
        if abs(self.lat) > ROAD_HALF_M:
            reward -= OFF_ROAD_PENALTY * DT        # 越出半车道
        reward -= ACTION_COST * abs(self.applied)
        reward -= DELTA_COST * abs(self.applied)
        return reward

    # -- sim 接口（live FSDStack，后续接入） ---------------------------------
    def _step_sim(self):
        raise NotImplementedError(
            "sim 模式需要 live connector/FSDStack 注入，见 rl/lateral_env "
            "后续版本；离线训练 + sim 评估的分工与 DecisionSpeedEnv 一致")
