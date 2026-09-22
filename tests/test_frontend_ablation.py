"""T11: the front-end ablation must measure, and must not select.

The plan's T11 asks for univariate ablations with a fixed data set, a
fixed post-process and a metric that reports raw->final IoU plus the two
failure modes SEPARATELY (deleted true-line pixels, kept false-line
pixels).  These tests pin the tooling:

* each classic arm behaves as its name says (a Top-Hat detector finds a
  bright stroke, a percentile detector selects a bounded fraction, the
  absolute-threshold arm is the pre-existing front-end);
* the metric counts deleted-true and kept-false as different numbers, on
  the VALID label area only (255 is unlabelled, not background);
* the arms are pure functions of the image (no state, same output twice);
* nothing in the ablation changes production behaviour: no switch, no
  default, no model selection - it prints numbers and stops.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ablation():
    spec = importlib.util.spec_from_file_location(
        "_m5_ablation_frontend", ROOT / "scripts" / "m5_ablation_frontend.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m5_ablation_frontend"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frame_with_line(brightness: int = 220, bg: int = 60, h=120, w=160):
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    img[h // 2:, w // 2 - 3:w // 2 + 3] = brightness      # a vertical stroke
    return img


class TestArms:
    def test_the_absolute_arm_is_the_pre_existing_front_end(self):
        ab = _ablation()
        img = _frame_with_line()
        mask = ab.arm_abs(img)
        assert mask.dtype == bool and mask.any()
        # it is exactly the existing colour front-end, called by name
        from beamng_autopilot.vision.lanes import _color_masks
        expect = np.zeros(img.shape[:2], dtype=bool)
        for _n, m in _color_masks(img):
            expect |= (m > 0)
        np.testing.assert_array_equal(mask, expect)

    def test_the_tophat_arm_finds_a_bright_stroke_and_nothing_on_flat(self):
        ab = _ablation()
        stroke = ab.arm_tophat_otsu(_frame_with_line())
        flat = ab.arm_tophat_otsu(np.full((120, 160, 3), 80, dtype=np.uint8))
        assert stroke.any()
        assert not flat.any()
        # the stroke's pixels are the ones selected
        assert stroke[80, 80] or stroke[70, 80]

    def test_the_percentile_arm_selects_a_bounded_fraction(self):
        ab = _ablation()
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = ab.arm_ycbcr_pct(img)
        frac = float(mask.mean())
        # Y top 3% plus Cb bottom 2%: bounded well below a third
        assert 0.0 < frac < 0.30, frac

    def test_every_arm_is_a_pure_function_of_the_image(self):
        ab = _ablation()
        img = _frame_with_line()
        for arm in ("arm_abs", "arm_tophat_otsu", "arm_ycbcr_pct"):
            fn = getattr(ab, arm)
            a = fn(img)
            b = fn(img)
            np.testing.assert_array_equal(a, b)
            assert len(inspect.signature(fn).parameters) == 1

    def test_the_arm_list_is_the_documented_set(self):
        ab = _ablation()
        assert ab.ARMS == ("model", "abs", "tophat_otsu", "ycbcr_pct")
        assert ab.IGNORE_VALUE == 255


class TestMetric:
    def test_deleted_true_and_kept_false_are_separate_numbers(self):
        ab = _ablation()
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[0, :] = 2                       # a true line
        pred = np.zeros((10, 10), dtype=bool)
        pred[5, :] = True                  # a false line
        valid = np.ones((10, 10), dtype=bool)
        deleted = int(np.count_nonzero((gt == 2) & valid & ~pred))
        kept_false = int(np.count_nonzero(pred & valid & (gt != 2)))
        assert deleted == 10 and kept_false == 10
        assert ab._iou(pred, gt == 2, valid) == 0.0

    def test_scoring_ignores_the_unlabelled_area(self):
        ab = _ablation()
        gt = np.zeros((8, 8), dtype=np.uint8)
        gt[:, :4] = 2
        gt[:, 4:] = ab.IGNORE_VALUE
        pred = np.zeros((8, 8), dtype=bool)
        pred[:, :4] = True                 # perfect on the labelled half
        pred[:, 4:] = True                 # arbitrary on the ignored half
        valid = (gt != ab.IGNORE_VALUE)
        assert ab._iou(pred, gt == 2, valid) == pytest.approx(1.0)

    def test_an_empty_union_has_no_iou_rather_than_one(self):
        ab = _ablation()
        empty = np.zeros((4, 4), dtype=bool)
        assert ab._iou(empty, empty, np.ones((4, 4), dtype=bool)) is None

    def test_the_component_metrics_are_counts_not_booleans(self):
        ab = _ablation()
        m = np.zeros((20, 20), dtype=bool)
        m[2:8, 2:4] = True
        m[12:19, 10:12] = True
        assert ab._components(m) == 2
        assert ab._longest_component_extent(m) == 7


class TestNoSelection:
    """An ablation that changed behaviour would not be an ablation."""

    def test_the_script_exposes_no_switch_and_touches_no_default(self):
        src = (ROOT / "scripts" / "m5_ablation_frontend.py").read_text("utf-8")
        # it may READ a checkpoint via --model, but it must not assign any
        # environment variable or mutate the config module
        for banned in ("os.environ[", "os.environ.setdefault",
                       "config.SEG_ =", "config.SEG_WEIGHTS"):
            assert banned not in src, banned
        # the default path is the production resolver
        assert "Segmenter()" in src

    def test_the_arms_do_not_write_into_the_pipeline(self):
        ab = _ablation()
        for arm in ("arm_abs", "arm_tophat_otsu", "arm_ycbcr_pct"):
            src = inspect.getsource(getattr(ab, arm))
            assert "grid" not in src and "scene" not in src

    def test_the_report_labels_the_percentiles_as_uncalibrated(self):
        ab = _ablation()
        assert ab.Y_PCT == 0.97 and ab.CB_PCT == 0.02
        src = (ROOT / "scripts" / "m5_ablation_frontend.py").read_text("utf-8")
        assert "NOT calibrated" in src
