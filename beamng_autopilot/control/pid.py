"""简易 PID 控制器（带积分限幅与输出限幅）。"""

from __future__ import annotations


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits=(-1.0, 1.0),
        integral_limit: float = 1.0,
        dt: float = 1.0 / 60.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limit = integral_limit
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = 0.0

    def update(self, error: float, dt: float | None = None) -> float:
        if dt is None:
            dt = self.dt
        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral + error * dt),
        )
        derivative = (error - self._last_error) / max(dt, 1e-6)
        self._last_error = error
        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.output_limits[0], min(self.output_limits[1], out))
