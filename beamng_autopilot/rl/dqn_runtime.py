"""Serve a trained SB3 DQN decision policy inside the FSD drive loop.

``DQNRuntime`` wraps a Stable-Baselines3 DQN zip (M4 decision layer):
one ``decision_observation`` vector in, one discrete action out, mapped
onto a cap for the plan's target speed (``action_to_target``).  The
decision layer can only SLOW the plan down - the layered planner, the
safety monitor and the no-cross rules stay authoritative, so a bad DQN
action degrades comfort, never safety.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beamng_autopilot.rl.obs import (
    CLEARANCE_NORM_M, DECISION_OBS_SCHEMA, DECISION_OBS_SIZE,
    DECISION_OBS_VERSION, LANE_DEV_NORM_M, ROAD_OFF_NORM_M, TRACKS_NORM,
    decision_observation,
)

DEFAULT_DQN_WEIGHTS = "logs/m4_dqn/dqn_decision.zip"

ACTION_MULT = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.35, 4: 0.1}
DECISION_DT_S = 0.25       # must match DecisionSpeedEnv.DT


@dataclass(frozen=True)
class DQNContract:
    """Validated checkpoint contract (structural + optional sidecar)."""

    ok: bool
    reason: str
    meta: dict


def action_to_target(action: int, target_speed: float,
                     min_speed: float = 0.5) -> float:
    """Map a discrete decision onto a capped plan target speed.

    Five speed levels (1.0 / 0.8 / 0.6 / 0.35 / 0.1 of the plan target,
    mirroring the env's ACTION_MULT); unknown actions fall back to
    cruise.  The mapping can only SLOW the plan.
    """
    mult = ACTION_MULT.get(int(action), 1.0)
    return max(float(min_speed), float(target_speed) * mult)


class DQNRuntime:
    """Load + serve one trained SB3 DQN decision policy."""

    def __init__(self, weights=None, device: str | None = None) -> None:
        self.weights = Path(weights) if weights else None
        self.device = device
        self.model = None
        self._err: str | None = None
        self.contract: DQNContract | None = None
        self.meta: dict = {}
        self.meta_warning: str = ""
        if self.weights is not None and self.weights.exists():
            try:
                from stable_baselines3 import DQN
                model = DQN.load(str(self.weights),
                                 device=self.device or "auto")
                contract = self._validate_contract(model)
                self.contract = contract
                if not contract.ok:
                    raise ValueError(
                        f"checkpoint contract mismatch: {contract.reason}")
                self.meta = contract.meta
                self.model = model
            except Exception as exc:
                self._err = str(exc)
                self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def error(self) -> str | None:
        return self._err

    def _sidecar_path(self) -> Path:
        return self.weights.with_suffix(".meta.json") \
            if self.weights is not None else Path()

    def _validate_contract(self, model) -> DQNContract:
        """Fail closed when the checkpoint cannot serve live observations.

        Structural checks always run: the observation box must be the
        decision vector and the action count must match ``ACTION_MULT``.
        When training wrote a ``<weights>.meta.json`` sidecar, every
        normalization constant, the action mapping and the control dt
        must match the live contract too.  Legacy checkpoints without a
        sidecar stay loadable with a warning (structural check only).
        """
        obs_space = getattr(model, "observation_space", None)
        act_space = getattr(model, "action_space", None)
        shape = tuple(getattr(obs_space, "shape", ()) or ())
        if shape != (DECISION_OBS_SIZE,):
            return DQNContract(
                False,
                f"observation shape {shape} != ({DECISION_OBS_SIZE},)",
                {})
        low = getattr(obs_space, "low", None)
        high = getattr(obs_space, "high", None)
        if low is not None and high is not None:
            try:
                lo = np.asarray(low, dtype=float)
                hi = np.asarray(high, dtype=float)
                if (not np.isfinite(lo).all()
                        or not np.isfinite(hi).all()):
                    return DQNContract(
                        False, "invalid observation bounds", {})
                if float(np.max(lo)) > 0.0 or float(np.min(hi)) < 1.0:
                    return DQNContract(
                        False,
                        f"normalized observation bounds [{lo.min()}, "
                        f"{hi.max()}] do not contain [0, 1]", {})
            except (TypeError, ValueError):
                return DQNContract(False, "invalid observation bounds", {})
        n_actions = int(getattr(act_space, "n", -1))
        if n_actions != len(ACTION_MULT):
            return DQNContract(
                False, f"action count {n_actions} != {len(ACTION_MULT)}", {})

        meta_path = self._sidecar_path()
        if not meta_path.exists():
            self.meta_warning = "no checkpoint sidecar; structural check only"
            return DQNContract(True, "", {})
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return DQNContract(False, f"sidecar unreadable: {exc}", {})
        if not isinstance(meta, dict):
            return DQNContract(False, "sidecar is not an object", {})
        if meta.get("schema") != DECISION_OBS_SCHEMA:
            return DQNContract(
                False, f"sidecar schema {meta.get('schema')!r} "
                f"!= {DECISION_OBS_SCHEMA!r}", meta)
        if int(meta.get("schema_version", -1)) != DECISION_OBS_VERSION:
            return DQNContract(
                False, f"sidecar version {meta.get('schema_version')!r} "
                f"!= {DECISION_OBS_VERSION}", meta)
        if int(meta.get("obs_size", -1)) != DECISION_OBS_SIZE:
            return DQNContract(
                False, f"sidecar obs_size {meta.get('obs_size')!r} "
                f"!= {DECISION_OBS_SIZE}", meta)
        expected_norms = {
            "clearance_norm_m": CLEARANCE_NORM_M,
            "lane_dev_norm_m": LANE_DEV_NORM_M,
            "road_off_norm_m": ROAD_OFF_NORM_M,
            "tracks_norm": TRACKS_NORM,
        }
        for key, value in expected_norms.items():
            try:
                got = float(meta.get(key))
            except (TypeError, ValueError):
                return DQNContract(
                    False, f"sidecar {key} missing/invalid", meta)
            if abs(got - float(value)) > 1e-9:
                return DQNContract(
                    False, f"sidecar {key}={got} != {value}", meta)
        try:
            mult = {int(k): float(v)
                    for k, v in dict(meta.get("action_mult") or {}).items()}
        except (TypeError, ValueError):
            return DQNContract(False, "sidecar action_mult invalid", meta)
        if mult != ACTION_MULT:
            return DQNContract(
                False, f"sidecar action_mult {mult} != {ACTION_MULT}", meta)
        try:
            dt = float(meta.get("dt_s"))
        except (TypeError, ValueError):
            return DQNContract(False, "sidecar dt_s missing/invalid", meta)
        if abs(dt - DECISION_DT_S) > 1e-9:
            return DQNContract(
                False, f"sidecar dt_s={dt} != {DECISION_DT_S}", meta)
        return DQNContract(True, "", meta)

    def predict(self, speed: float, target_speed: float,
                fwd_clearance, closest_obs, lane_dev, road_off,
                n_tracks) -> tuple[int, float]:
        """Decision observation -> ``(action, inference ms)``."""
        if self.model is None:
            return 0, 0.0
        obs = decision_observation(
            speed=speed, target_speed=target_speed,
            fwd_clearance=fwd_clearance, closest_obs=closest_obs,
            lane_dev=lane_dev, road_off=road_off, n_tracks=n_tracks)
        t0 = time.time()
        action, _ = self.model.predict(obs, deterministic=True)
        ms = (time.time() - t0) * 1000.0
        return int(action), ms
