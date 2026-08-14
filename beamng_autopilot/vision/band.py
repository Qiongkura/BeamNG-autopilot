"""Rubber-track band pair detection on a single image row.

On the test track the driven racing line shows up as two dark rubber
strips roughly symmetric around the path the car follows.  The detector
finds the two strongest dark bands on one image row and returns the
midpoint, which tracks the drivable centreline.
"""

from __future__ import annotations

import cv2
import numpy as np


def _contrast_profile(row, lo, hi, contrast_k, smooth):
    prof = np.asarray(row[lo:hi], dtype=np.float32)
    prof = cv2.GaussianBlur(prof, (1, smooth), 0).ravel()
    k = contrast_k
    kernel = np.ones(2 * k + 1, np.float32) / (2 * k + 1)
    mean = np.convolve(np.pad(prof, k, mode="reflect"), kernel, mode="valid")
    return mean - prof, prof


def _band_runs(contrast, lo, thr):
    mask = contrast >= thr
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return []
    runs = []
    run = [idxs[0]]
    for a, b in zip(idxs, idxs[1:]):
        if b - a == 1:
            run.append(b)
        else:
            runs.append(run)
            run = [b]
    runs.append(run)
    return runs


def detect_pair_center(row, center, expected_px=140.0, window=180):
    """Find the two strongest dark bands and return their midpoint offset.

    The two dark rubber tracks sit roughly symmetrically around the path the
    car follows, so their midpoint tracks the drivable centreline.  Returns
    (offset_px, pair_width_px, confidence 0..1, n_bands).  offset_px is
    relative to the requested `center` (right-positive in the image).
    """
    contrast_k = int(np.clip(0.22 * expected_px, 7, 35))
    smooth = 5
    lo = max(0, int(center) - window)
    hi = min(len(row), int(center) + window)
    if hi - lo < 40:
        return None, 0.0, 0.0, 0
    contrast, _ = _contrast_profile(row, lo, hi, contrast_k, smooth)
    cmax = float(contrast.max())
    if cmax < 5.0:
        return None, 0.0, 0.0, 0
    runs = _band_runs(contrast, lo, 0.45 * cmax)
    # merge runs that are closer than ~1/3 of the expected pair spacing
    merged = []
    for run in runs:
        if not merged:
            merged.append(run)
            continue
        prev = merged[-1]
        gap = run[0] - prev[-1]
        if gap < max(4, 0.35 * expected_px):
            merged[-1] = prev + run
        else:
            merged.append(run)
    runs = merged
    centers = []
    widths = []
    strengths = []
    for run in runs:
        if len(run) < 2:
            continue
        sub = np.arange(lo, hi)[run]
        w = contrast[run]
        c = float(np.sum(sub * w) / np.sum(w))
        centers.append(c)
        widths.append(float(run[-1] - run[0]))
        strengths.append(float(w.max()))
    if not centers:
        return None, 0.0, 0.0, 0
    order = np.argsort(strengths)[::-1]
    c0 = centers[order[0]]
    # pick a second band at roughly the expected tire-track spacing
    c1 = None
    partner_strength = 0.0
    for i in order[1:]:
        gap = abs(centers[i] - c0)
        if 0.4 * expected_px < gap < 1.8 * expected_px:
            c1 = centers[i]
            partner_strength = strengths[i]
            break
    if c1 is None:
        # fall back: search for a symmetric partner at the expected offset
        for side in (-1, 1):
            target = c0 + side * expected_px
            near = np.arange(lo, hi)
            cand = near[np.abs(near - target) <= 0.3 * expected_px]
            if len(cand) == 0:
                continue
            c = cand[np.argmax(contrast[cand - lo])]
            if contrast[int(c - lo)] > 0.4 * cmax:
                c1 = float(c)
                partner_strength = float(contrast[int(c - lo)])
                break
    if c1 is None:
        return c0 - center, 0.0, min(1.0, strengths[order[0]] / 40.0), 1
    mid = 0.5 * (c0 + c1)
    pair_w = abs(c1 - c0)
    s1 = strengths[order[1]] if len(strengths) > 1 else 0.0
    conf = min(1.0, (strengths[order[0]] + max(s1, partner_strength)) / 60.0)
    return mid - center, pair_w, conf, 2
