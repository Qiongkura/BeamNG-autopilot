"""Post-drive telemetry chart: continuous bar graphs of throttle / brake /
speed history, shown when autopilot ends.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_telemetry(history: dict, out_path: Path, block: bool = False,
                   show: bool = True) -> Path:
    """Plot throttle/brake/speed bar histories and save a PNG.

    `history` is a dict with 't', 'throttle', 'brake', 'speed' lists.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    t = np.asarray(history["t"], dtype=float)
    thr = np.asarray(history["throttle"], dtype=float)
    brk = np.asarray(history["brake"], dtype=float)
    spd = np.asarray(history["speed"], dtype=float)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"hspace": 0.35}, constrained_layout=True)
    fig.suptitle("Autopilot telemetry", fontsize=13, fontweight="bold")

    for ax, y, color, label, ymax in (
        (axes[0], thr, "#2ecc71", "Throttle", 1.0),
        (axes[1], brk, "#e74c3c", "Brake", 1.0),
        (axes[2], spd, "#3498db", "Speed (m/s)", None),
    ):
        ax.bar(t, y, width=max(np.median(np.diff(t)), 0.01),
               color=color, alpha=0.85, edgecolor="none")
        if ymax:
            ax.set_ylim(0, ymax)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.savefig(str(out_path), dpi=120)
    if show:
        plt.show(block=block)
    else:
        plt.close(fig)
    return out_path
