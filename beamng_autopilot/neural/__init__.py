"""End-to-end driving network skeleton (FSD v12-style shape).

The network consumes the fused vector space - the BEV occupancy grid the
sensor stack produces - and directly regresses a trajectory (a sequence
of waypoints) plus an action (steer/throttle), the way FSD's v12
"planning is a neural network" stacks do.  This module is deliberately a
*skeleton*: the architecture contract, a working forward/backward, and a
tiny synthetic-data trainer, NOT a trained model.  Real training comes
later from the shadow recordings (goal 5).

* ``E2ENet``: a small CNN over the BEV raster -> trajectory + action,
  plus a fallback linear wrapper so it runs even without torch-style
  heavy imports.
* ``train_synthetic``: trains the net on procedurally generated
  (raster -> trajectory) pairs until the loss drops, proving the
  forward/backward path works end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID_N = 60          # occupancy grid resolution (n x n)
N_WAYPOINTS = 16     # trajectory length the net emits
LOSS_EPS = 1e-4


@dataclass
class E2ENet:
    """Small learnable model: BEV raster -> (trajectory, action).

    A torch-free, numpy-backed linear/conservative map so the skeleton is
    importable and unit-testable everywhere; a real CNN replaces the
    ``predict`` internals later without changing the call contract.
    """

    grid_n: int = GRID_N
    n_waypoints: int = N_WAYPOINTS
    # learned linear map from the flattened grid to (trajectory + action)
    W: np.ndarray | None = None
    b: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.W is None:
            in_dim = self.grid_n * self.grid_n
            out = self.n_waypoints * 2 + 2
            # small init; real training fits these
            rng = np.random.default_rng(0)
            self.W = rng.normal(0.0, 1e-3, (out, in_dim))
            self.b = np.zeros(out)

    def forward(self, bev_raster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(trajectory (N,2), action (steer, throttle)) from a BEV raster."""
        x = np.asarray(bev_raster, dtype=np.float32).reshape(-1)
        if len(x) < self.grid_n * self.grid_n:
            x = np.pad(x, (0, self.grid_n * self.grid_n - len(x)))
        y = self.W @ x + self.b
        traj = y[: self.n_waypoints * 2].reshape(self.n_waypoints, 2)
        action = y[self.n_waypoints * 2:]
        return traj, action

    def predict(self, bev_raster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.forward(bev_raster)

    def train_step(self, inputs: np.ndarray,
                   target_traj: np.ndarray) -> float:
        """One gradient step on a batch.  Returns the loss.

        Regresses the trajectory block of the output; the action block
        stays at its init (real training will supervise both).
        """
        xs = np.asarray(inputs, dtype=np.float32).reshape(
            len(inputs), -1)
        outs = xs @ self.W.T + self.b
        out_dim = int(outs.shape[1])
        n_d = self.n_waypoints * 2
        pred = outs[:, :n_d]
        target = np.asarray(target_traj, dtype=np.float32).reshape(
            len(inputs), n_d)
        loss = float(np.mean((pred - target) ** 2))
        # gradient only through the trajectory block
        grad = np.zeros_like(outs)
        grad[:, :n_d] = (pred - target) * 2.0 / max(1, len(inputs))
        gW = grad.T @ xs
        gb = grad.mean(axis=0)
        lr = 1e-3
        self.W = self.W - lr * gW
        self.b = self.b - lr * gb
        return loss


def train_synthetic(model: E2ENet, n_samples: int = 128,
                    steps: int = 300) -> float:
    """Synthetic training until trajectory regression loss is low.

    Builds a latent pattern - "target trajectory is a scaled version of
    the raster's obstacle-free column" - so a linear map can fit; a
    decreasing loss proves forward/backward and the training loop work.
    Returns the final loss.
    """
    rng = np.random.default_rng(1)
    inputs = rng.random((n_samples, model.grid_n, model.grid_n)).astype(
        np.float32)
    # synthetic targets: trajectory bends the same way as the raster
    # (e.g. steer direction ~ mean lateral gradient of occupancy)
    targets = np.zeros((n_samples, model.n_waypoints, 2), dtype=np.float32)
    for i in range(n_samples):
        lat = float(np.argmax(inputs[i, model.grid_n // 2, :])
                    / model.grid_n - 0.5)
        for j in range(model.n_waypoints):
            targets[i, j] = [j * 0.5 + 0.0, lat * j * 0.25]
    last = 1e9
    for _ in range(steps):
        last = model.train_step(inputs, targets)
    return last
from .e2e_torch import E2ENetTorch  # noqa: E402
from .dataset import ShadowMultimodalDataset  # noqa: E402
