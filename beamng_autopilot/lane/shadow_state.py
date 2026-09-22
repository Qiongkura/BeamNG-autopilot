"""Shadow lateral state posterior: (e, e_dot, theta, theta_dot) + covariance.

T06 of the round-5 plan.  This is the "simple model first" rung of the
ladder (baseline -> identity-gated robust/one-model filter -> EKF -> maybe
IMM), so it is deliberately a linear KF with an explicit, documented
measurement model - not an IMM, and not a second tracker competing with
``lane/tracking.py`` (which owns lane-frame freshness) or
``lane/stability.py`` (which owns reference authority):

* ``tracking.py``   - is the lane FRAME usable / fresh?
* ``stability.py``  - does this reference deserve steering authority?
* ``shadow_state.py`` - where is the car relative to the accepted reference,
  how fast is that error changing, and how well is any of it known?

State (all in the car's lateral frame, metres / radians):

    x = [e, e_dot, theta, theta_dot]

``e``     : signed lateral offset of the EGO from the accepted reference
            (+ = the ego is LEFT of the reference), metres.
``theta`` : heading error against the reference TANGENT
            (+ = heading left of the reference direction), radians.

Predict (low-slip kinematics, the same check the plan asks for):

    e_dot    = v * sin(theta)                     (consistency residual)
    theta_dot = yaw_rate - v * kappa / (1 + e * kappa)

``yaw_rate`` is the VEHICLE's measured rotation rate; the steering command
is a different quantity and is never used here.  When no yaw rate is
supplied it is derived by finite difference from consecutive headings and
labelled as such - it is an estimate, not a measurement.

Shadow only: nothing in this module can move the steering.  It publishes
its verdict for comparison against the current behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Measurement noise: a two-sided, freshly observed reference is the best
# case; a single-sided read adds the width prior's uncertainty to the
# lateral offset (the missing edge could be anywhere inside that prior).
SIGMA_E_TWO_SIDED_M = 0.10
SIGMA_E_INFERRED_BASE_M = 0.25
WIDTH_SIGMA_M = 0.30
SIGMA_THETA_RAD = 0.02

# Process noise per second (position, rate, heading, heading rate).
Q_DIAG_PER_S = (0.02, 0.30, 0.004, 0.20)
# Extra heading-rate process noise when no yaw rate is known.
Q_EXTRA_NO_YAW_RATE = 0.60

# A measurement this far from the prediction is an IDENTITY change (a
# different lane / the other boundary), not a wobble: reset instead of
# differentiating across two lanes.
IDENTITY_JUMP_M = 1.5
# Reference stations used for the near-field read.
NEAR_STATION_M = 1.0
FAR_STATION_M = 20.0
CURVATURE_SPAN_M = 6.0


def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ReferenceRead:
    """One tick's measurement of the ego relative to a reference."""

    e_m: float
    theta_rad: float
    kappa_1pm: float          # reference curvature (1/m), + = curves left
    ok: bool
    reason: str = ""


def read_reference(ref, pos, heading: float, *,
                   near_m: float = NEAR_STATION_M,
                   far_m: float = FAR_STATION_M) -> ReferenceRead:
    """Measure (e, theta, kappa) of the ego against a reference polyline.

    The reference is resampled to the station nearest the ego, the lateral
    offset is measured in the REFERENCE's frame (so a bend does not leak
    into ``e`` the way a car-frame median does), and the curvature comes
    from the tangent turn over ``CURVATURE_SPAN_M``.  Purely geometric:
    no map, no nav line.
    """
    if ref is None:
        return ReferenceRead(0.0, 0.0, 0.0, False, "no reference")
    arr = np.asarray(ref, dtype=float)
    if arr.ndim != 2 or len(arr) < 3 or arr.shape[1] < 2:
        return ReferenceRead(0.0, 0.0, 0.0, False, "reference too short")
    pts = arr[:, :2]
    if not np.isfinite(pts).all():
        return ReferenceRead(0.0, 0.0, 0.0, False, "reference not finite")
    p = np.asarray(pos, dtype=float)[:2]
    d = np.linalg.norm(pts - p, axis=1)
    i = int(np.argmin(d))
    if float(d[i]) > far_m:
        return ReferenceRead(0.0, 0.0, 0.0, False, "reference too far")
    # local tangent from the two points bracketing the nearest station
    i0, i1 = max(0, i - 1), min(len(pts) - 1, i + 1)
    if i1 <= i0:
        return ReferenceRead(0.0, 0.0, 0.0, False, "no tangent")
    seg = pts[i1] - pts[i0]
    n = float(np.linalg.norm(seg))
    if n < 1e-6:
        return ReferenceRead(0.0, 0.0, 0.0, False, "degenerate tangent")
    u = seg / n
    left = np.array([-u[1], u[0]])
    e = float((p - pts[i]) @ left)
    ref_heading = math.atan2(float(u[1]), float(u[0]))
    theta = wrap_pi(float(heading) - ref_heading)
    # curvature: tangent turn over a span, in the reference's own stations
    j0 = int(np.argmin(np.linalg.norm(pts - (pts[i] - u * CURVATURE_SPAN_M),
                                      axis=1)))
    j1 = int(np.argmin(np.linalg.norm(pts - (pts[i] + u * CURVATURE_SPAN_M),
                                      axis=1)))
    kappa = 0.0
    if j1 > j0 + 1:
        s0 = pts[j0 + 1] - pts[j0]
        s1 = pts[j1] - pts[j1 - 1]
        n0, n1 = float(np.linalg.norm(s0)), float(np.linalg.norm(s1))
        if n0 > 1e-6 and n1 > 1e-6:
            a0 = math.atan2(float(s0[1]), float(s0[0]))
            a1 = math.atan2(float(s1[1]), float(s1[0]))
            span = float(np.linalg.norm(pts[j1] - pts[j0]))
            if span > 1.0:
                kappa = wrap_pi(a1 - a0) / span
    return ReferenceRead(e, theta, kappa, True)


@dataclass
class ShadowState:
    """One tick's shadow posterior (JSON-safe for telemetry)."""

    e_m: float = 0.0
    e_dot_mps: float = 0.0
    theta_rad: float = 0.0
    theta_dot_radps: float = 0.0
    sigma_e_m: float | None = None
    sigma_theta_rad: float | None = None
    cov: list = field(default_factory=list)
    mode: str = "init"                 # measured | predicted | reset
    identity: str = ""
    identity_event: str | None = None
    n_updates: int = 0
    age_since_update_s: float | None = None
    consistency_residual_mps: float | None = None
    reason: str = ""
    yaw_rate_source: str = "none"
    kappa_1pm: float = 0.0

    def as_dict(self) -> dict:
        out = {
            "e_m": round(self.e_m, 4),
            "e_dot_mps": round(self.e_dot_mps, 4),
            "theta_rad": round(self.theta_rad, 5),
            "theta_dot_radps": round(self.theta_dot_radps, 5),
            "mode": self.mode, "identity": self.identity,
            "n_updates": int(self.n_updates),
            "kappa_1pm": round(self.kappa_1pm, 5),
            "yaw_rate_source": self.yaw_rate_source,
        }
        if self.sigma_e_m is not None:
            out["sigma_e_m"] = round(self.sigma_e_m, 4)
        if self.sigma_theta_rad is not None:
            out["sigma_theta_rad"] = round(self.sigma_theta_rad, 5)
        if self.cov:
            out["cov"] = [[round(float(v), 6) for v in row]
                          for row in self.cov]
        if self.identity_event:
            out["identity_event"] = self.identity_event
        if self.age_since_update_s is not None:
            out["age_since_update_s"] = round(self.age_since_update_s, 3)
        if self.consistency_residual_mps is not None:
            out["consistency_residual_mps"] = round(
                self.consistency_residual_mps, 4)
        if self.reason:
            out["reason"] = self.reason
        return out


class LateralShadowEstimator:
    """Identity-gated 4-state lateral KF, read-only by construction."""

    def __init__(self, *, identity_jump_m: float = IDENTITY_JUMP_M,
                 sigma_e_two_sided: float = SIGMA_E_TWO_SIDED_M,
                 sigma_e_inferred_base: float = SIGMA_E_INFERRED_BASE_M,
                 width_sigma_m: float = WIDTH_SIGMA_M,
                 sigma_theta: float = SIGMA_THETA_RAD) -> None:
        self.identity_jump_m = float(identity_jump_m)
        self.sigma_e_two_sided = float(sigma_e_two_sided)
        self.sigma_e_inferred_base = float(sigma_e_inferred_base)
        self.width_sigma_m = float(width_sigma_m)
        self.sigma_theta = float(sigma_theta)
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float) * 1.0
        self.started = False
        self.identity = ""
        self.n_updates = 0
        self.identity_events: list[dict] = []
        self._last_update_t: float | None = None
        self._last_t: float | None = None
        self._prev_heading: float | None = None
        self._prev_meas: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------
    def reset(self, reason: str, *, identity: str = "") -> ShadowState:
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float) * 1.0
        self.started = False
        self.n_updates = 0
        self._last_update_t = None
        if identity:
            self.identity = str(identity)
        st = ShadowState(mode="reset", identity=self.identity, reason=reason)
        st.cov = self.P.tolist()
        return st

    def _yaw_rate(self, heading: float, prev_t: float | None,
                  prev_h: float | None, now: float,
                  supplied: float | None) -> tuple[float, str]:
        if supplied is not None:
            try:
                return float(supplied), "supplied"
            except (TypeError, ValueError):
                pass
        if prev_h is not None and prev_t is not None and now > prev_t + 1e-6:
            dt = float(now) - float(prev_t)
            if dt <= 1.0:                    # a gap is not a rate
                return wrap_pi(float(heading) - float(prev_h)) / dt, \
                    "finite_difference"
        return 0.0, "none"

    def predict(self, *, speed_mps: float, heading: float, now: float,
                kappa: float = 0.0, yaw_rate: float | None = None) -> None:
        """One prediction step (also used when no measurement exists)."""
        _prev_t = self._last_t
        dt = 0.0 if _prev_t is None else max(0.0, float(now) - float(_prev_t))
        dt = min(dt, 1.0)
        # NOTE: ``_last_t`` is advanced AFTER the yaw-rate difference is
        # taken - advancing it first made every finite difference zero.
        _prev_heading = self._prev_heading
        self._last_t = float(now)
        if not self.started:
            self._prev_heading = float(heading)
            return
        e, e_dot, th, th_dot = (float(v) for v in self.x)
        v = float(speed_mps)
        self._last_speed = v
        yr, yr_src = self._yaw_rate(heading, _prev_t, _prev_heading,
                                    float(now), yaw_rate)
        # low-slip kinematics (the plan's consistency model)
        e_new = e + dt * (v * math.sin(th))
        th_new = th + dt * (yr - v * float(kappa) / (1.0 + e * float(kappa)))
        # Jacobian: d(e')/d(theta) dominates; the kappa/(1+e kappa)^2 term is
        # second order for the curvatures this car sees (<0.05 1/m) and is
        # left out, with the heading-rate process noise carrying it.
        F = np.eye(4, dtype=float)
        F[0, 2] = dt * v * math.cos(th)
        self.x = np.array([e_new, e_dot, th_new, th_dot], dtype=float)
        q = np.asarray(Q_DIAG_PER_S, dtype=float) * max(dt, 1e-3)
        if yr_src == "none":
            q[3] += Q_EXTRA_NO_YAW_RATE * max(dt, 1e-3)
        self.P = F @ self.P @ F.T + np.diag(q)
        self._prev_heading = float(heading)
        self._yaw_rate_src = yr_src

    def update(self, *, ref, pos, heading: float, speed_mps: float, now: float,
               two_sided: bool = False, inferred: bool = True,
               fresh_obs: bool = False, width_m: float = 0.0,
               yaw_rate: float | None = None,
               lane_id: str | None = None) -> ShadowState:
        """Predict + (when the evidence allows) measure-update this tick.

        A HOLD (``fresh_obs`` False) is not a measurement: the filter
        predicts and says so, so a remembered reference can never look like
        a fresh observation in the shadow telemetry either (T02/T03 rule).
        """
        read = read_reference(ref, pos, heading)
        ident = str(lane_id) if lane_id else self.identity
        if ident and self.identity and ident != self.identity:
            ev = {"reason": "identity", "from": self.identity, "to": ident}
            self.identity_events.append(ev)
            self.identity = ident
            st = self.reset("identity changed", identity=ident)
            st.identity_event = "identity"
            st.kappa_1pm = read.kappa_1pm
            # a reset still measures the new identity this tick
            if read.ok and fresh_obs:
                fresh = self.update(
                    ref=ref, pos=pos, heading=heading, speed_mps=speed_mps,
                    now=now, two_sided=two_sided, inferred=inferred,
                    fresh_obs=fresh_obs, width_m=width_m, yaw_rate=yaw_rate,
                    lane_id=ident)
                fresh.identity_event = "identity"
                return fresh
            return st
        if ident and not self.identity:
            self.identity = ident
        self.predict(speed_mps=speed_mps, heading=heading, now=now,
                     kappa=read.kappa_1pm if read.ok else 0.0,
                     yaw_rate=yaw_rate)
        st = ShadowState(identity=self.identity,
                         yaw_rate_source=getattr(self, "_yaw_rate_src", "none"),
                         kappa_1pm=read.kappa_1pm)
        if not read.ok:
            st.mode = "predicted" if self.started else "init"
            st.reason = read.reason
            return self._finish(st)
        if not fresh_obs:
            st.mode = "predicted" if self.started else "init"
            st.reason = "held reference is not a measurement"
            return self._finish(st)
        if self.started and abs(read.e_m - float(self.x[0])) \
                > self.identity_jump_m:
            # A jump this large is a different lane / the other boundary.
            # Resetting keeps the derivative from spanning two lanes, and
            # the event is recorded rather than filtered away.
            self.identity_events.append(
                {"reason": "jump", "measurement_m": round(read.e_m, 3),
                 "predicted_m": round(float(self.x[0]), 3)})
            self.reset("measurement jump")
            st.identity_event = "jump"
        # measurement noise: single-sided reads carry the width prior
        sig_e = self.sigma_e_two_sided if two_sided \
            else self.sigma_e_inferred_base
        if not two_sided:
            sig_e = math.hypot(sig_e, 0.5 * self.width_sigma_m)
        if not self.started:
            self.x = np.array([read.e_m, 0.0, read.theta_rad, 0.0],
                              dtype=float)
            self.P = np.diag([sig_e ** 2, 1.0, self.sigma_theta ** 2, 0.5])
            self.started = True
        else:
            H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
            R = np.diag([sig_e ** 2, self.sigma_theta ** 2])
            z = np.array([read.e_m, read.theta_rad], dtype=float)
            y = z - H @ self.x
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ H) @ self.P
        self.n_updates += 1
        self._last_update_t = float(now)
        st.mode = "measured"
        # Model-consistency residual from the MEASURED rate (the state's
        # rate is not observable from position+heading alone, so checking
        # the state against the model would flag the filter's own
        # observability, not a modelling error).
        if self._prev_meas is not None:
            t0, e0, th0 = self._prev_meas
            dt = float(now) - float(t0)
            if dt > 0.05:
                meas_rate = (read.e_m - e0) / dt
                st.consistency_residual_mps = float(
                    meas_rate - float(speed_mps) * math.sin(th0))
        self._prev_meas = (float(now), read.e_m, read.theta_rad)
        return self._finish(st)

    def _finish(self, st: ShadowState) -> ShadowState:
        st.e_m = float(self.x[0])
        st.e_dot_mps = float(self.x[1])
        st.theta_rad = float(self.x[2])
        st.theta_dot_radps = float(self.x[3])
        st.sigma_e_m = float(math.sqrt(max(0.0, float(self.P[0, 0]))))
        st.sigma_theta_rad = float(math.sqrt(max(0.0, float(self.P[2, 2]))))
        st.cov = self.P.tolist()
        st.n_updates = int(self.n_updates)
        if self._last_update_t is not None and self._last_t is not None:
            st.age_since_update_s = max(
                0.0, float(self._last_t) - float(self._last_update_t))
        # The plan's model-consistency check: with low slip, e_dot should
        # track v*sin(theta).  A persistent residual means the model (or the
        # reference identity) is wrong, and it is published rather than
        # smoothed over - a filter that hides it would look better and mean
        # less.
        if st.consistency_residual_mps is None:
            # No measurement this tick: fall back to the state-based check,
            # which is weaker (the rate state is only weakly observable
            # from position and heading).
            v = float(getattr(self, "_last_speed", 0.0))
            st.consistency_residual_mps = float(
                st.e_dot_mps - v * math.sin(st.theta_rad))
        return st
