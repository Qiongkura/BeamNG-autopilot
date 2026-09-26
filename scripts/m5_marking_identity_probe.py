"""Confirm line candidates against the ENGINE's own annotation (T08 prerequisite).

T08's acceptance needs candidates whose identity has been confirmed by an
independent source - "在人工/真值确认的候选身份上，减少错误关联".  The engine
renders an annotated frame from its own materials, so its line class is
independent of the perception model; this tool measures the perception's
line output against it in the IMAGE PLANE, per frame:

* precision = |perception_line AND engine_line| / |perception_line|
  (of what the model calls a line, how much the engine also calls a line);
* recall    = |AND| / |engine_line|
  (of the engine's line pixels, how much the model found);
* per-side coverage: engine line pixels left/right of the image centre,
  and how much of each side the model covers - a candidate set that only
  ever sees one side cannot support a two-sided reference.

Two limits are part of the result, not footnotes:

* the engine renders KNOWN line materials - worn or faded paint may be
  unlabelled, so low precision can mean "unlabelled", not "false";
* this is the image plane.  Per-CANDIDATE identity (which marking is the
  left boundary, which the divider) needs the camera model and a ground
  projection; the ring collector now saves the model per view, but a run
  recorded before that cannot answer the per-candidate question.
* a frame with no camera model is UNKNOWN - missing evidence, never a
  negative (plan v2 §3.4/T09).  Such frames are reported per frame under
  ``unknowns`` and are NOT counted as processed measurements, so a run with
  no camera is refused instead of looking "processed but empty".

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_marking_identity_probe.py \\
        --run logs/m5_seg/ident_probe_20260923/front_main \\
        --meta logs/m5_seg/ident_probe_20260923/meta.json \\
        --view front_main --json logs/goal_20260921/marking_identity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.hydra import FrameContext, HydraNet  # noqa: E402
from beamng_autopilot.vision.heads.semantic import SemanticHead  # noqa: E402
from beamng_autopilot.experiments import candidate_metrics as cm  # noqa: E402
from beamng_autopilot.vision.projection import CameraModel  # noqa: E402

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from m5_geometry_audit import oracle_ground_point  # noqa: E402

CLS_LINE = 2

# --- ego-relative identity (vehicle frame), no lane/map assumption --------
#: |lateral| below this means the car is ON the line, not beside it.
STRADDLE_M = 0.5
#: A line this close on either side bounds the ego's lane.
NEAR_M = 5.0
#: Forward window in which a ground projection is believable for identity.
FWD_MIN_M = 2.0
FWD_MAX_M = 30.0
LAT_MAX_M = 12.0
#: Two lines are the same physical line when their lateral offsets differ
#: by less than this (the plane model's own error is a separate matter).
MATCH_M = 0.8
LINE_BIN_M = 2.0
LINE_LAT_GAP_M = 0.5
#: A straight line OBLIQUE to the ego axis drifts laterally by
#: ``LINE_BIN_M * tan(theta)`` per bin, so a fixed chain step is really a
#: fixed ANGLE limit.  Measured on an urban junction: the engine's own
#: painted line drifts ~1 m per 2 m bin (the first version used 0.6 m and
#: broke that line into a stub).  The step therefore allows the geometry:
#: ``LINE_CHAIN_STEP_M + LINE_BIN_M * LINE_SLOPE_MAX`` (~35 deg).
LINE_CHAIN_STEP_M = 0.6
LINE_SLOPE_MAX = 0.7
LINE_MIN_BINS = 3


def camera_from_meta(meta: dict, view: str):
    """CameraModel for ``view`` from a collector's ``cameras`` block."""
    cam = (meta or {}).get("cameras", {}).get(view)
    if not cam:
        return None
    try:
        return CameraModel(
            offset=np.asarray(cam["offset"], dtype=float),
            fwd_local=np.asarray(cam["fwd"], dtype=float),
            up_local=np.asarray(cam["up"], dtype=float),
            fov_deg=float(cam["fov_deg"]),
            width=int(cam["width"]),
            height=int(cam["height"]))
    except (KeyError, TypeError, ValueError):
        return None


def compare_masks(perception_line, engine_line, *, centre_col=None) -> dict:
    """Image-plane precision / recall / side coverage of a line mask.

    Returns UNKNOWN-style fields (``None``) where the comparison is not
    defined: a frame with no engine line pixels has no recall to report,
    and reporting 1.0 there would be the classic "no data reads as pass".
    """
    perc = np.asarray(perception_line, dtype=bool)
    eng = np.asarray(engine_line, dtype=bool)
    if perc.shape != eng.shape:
        return {"reason": f"shape mismatch {perc.shape} vs {eng.shape}"}
    n_perc = int(perc.sum())
    n_eng = int(eng.sum())
    inter = int((perc & eng).sum())
    out = {"n_perception_px": n_perc, "n_engine_px": n_eng, "n_intersection": inter,
           "precision": (None if n_perc == 0 else round(inter / n_perc, 4)),
           "recall": (None if n_eng == 0 else round(inter / n_eng, 4)),
           "iou": (None if (n_perc + n_eng - inter) == 0 else
                   round(inter / (n_perc + n_eng - inter), 4))}
    if centre_col is None:
        centre_col = perc.shape[1] // 2
    c = int(centre_col)
    for side, sl in (("left", slice(0, c)), ("right", slice(c, None))):
        e = int(eng[:, sl].sum())
        p = int(perc[:, sl].sum())
        i = int((perc[:, sl] & eng[:, sl]).sum())
        out[f"engine_px_{side}"] = e
        out[f"perception_px_{side}"] = p
        out[f"recall_{side}"] = (None if e == 0 else round(i / e, 4))
    out["engine_sides"] = int(out["engine_px_left"] > 0) + int(
        out["engine_px_right"] > 0)
    out["perception_sides"] = int(out["perception_px_left"] > 0) + int(
        out["perception_px_right"] > 0)
    return out


def project_pixels(us, vs, cam, *, stride: int = 1,
                   fwd_min: float = FWD_MIN_M, fwd_max: float = FWD_MAX_M,
                   lat_max: float = LAT_MAX_M):
    """Pixel list -> vehicle-frame ground points via the T05 oracle.

    The oracle is imported, not re-derived: it is the independent closed
    form validated against the pipeline (p50 0.0000 m) in the geometry
    work.  Its flat-ground assumption is stated in the limits - a slope
    moves the FORWARD distance, while the lateral SIGN (which side of the
    car a line is on) is what identity needs.
    """
    pts = []
    for u, v in zip(np.asarray(us).ravel()[::stride],
                    np.asarray(vs).ravel()[::stride]):
        hit = oracle_ground_point(float(u), float(v), cam)
        if hit is None:
            continue
        fwd, lat = float(hit[0]), float(hit[1])
        if fwd < float(fwd_min) or fwd > float(fwd_max):
            continue
        if abs(lat) > float(lat_max):
            continue
        pts.append((fwd, lat))
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def role_of(lat_m: float) -> str:
    """Ego-relative role of ONE line, from its lateral offset alone."""
    if lat_m is None:
        return "unknown"
    if abs(float(lat_m)) <= STRADDLE_M:
        return "straddled"
    return "left" if float(lat_m) > 0.0 else "right"


def assign_roles(lines) -> list:
    """Role per line, with ``near_``/``far_`` for the same side.

    Only the nearest line on a side can bound the ego's lane, so the role
    vocabulary distinguishes it from the second line further out - the
    distinction a downstream consumer needs and a bare left/right cannot
    give.
    """
    out = [dict(ln) for ln in lines]          # never mutate the caller's view
    for side in ("left", "right"):
        same = [i for i, ln in enumerate(out) if ln["role"] == side]
        same.sort(key=lambda i: abs(float(out[i]["lat_m"])))
        for rank, i in enumerate(same):
            out[i]["role"] = f"{'near' if rank == 0 else 'far'}_{side}"
    return out


def engine_lines(engine_mask, cam, *, stride: int = 3) -> list:
    """The engine's own lines as vehicle-frame clusters with roles.

    Dense labelled pixels are binned forward, split laterally and chained
    across bins; a chain that does not cover ``LINE_MIN_BINS`` bins is not
    a line (it is a patch - the same discipline the perception side uses).
    """
    vs, us = np.nonzero(np.asarray(engine_mask, dtype=bool))
    if len(us) == 0:
        return []
    pts = project_pixels(us, vs, cam, stride=stride)
    if len(pts) == 0:
        return []
    bins: dict[int, list] = {}
    for k in range(len(pts)):
        bins.setdefault(int(pts[k, 0] // LINE_BIN_M), []).append(k)
    segments = []
    for b in sorted(bins):
        idx = np.asarray(bins[b], dtype=int)
        order = idx[np.argsort(pts[idx, 1])]
        cur = [int(order[0])]
        for j in order[1:]:
            if float(pts[j, 1]) - float(pts[cur[-1], 1]) <= LINE_LAT_GAP_M:
                cur.append(int(j))
                continue
            if len(cur) >= 2:
                segments.append((b, float(np.median(pts[cur, 1])), cur))
            cur = [int(j)]
        if len(cur) >= 2:
            segments.append((b, float(np.median(pts[cur, 1])), cur))
    segments.sort(key=lambda t: (t[0], t[1]))
    chains: list[dict] = []
    for b, lat, members in segments:
        best = None
        for ch in chains:
            if ch["last_bin"] != b - 1:
                continue
            jump = abs(lat - ch["lat"])
            step_max = LINE_CHAIN_STEP_M + LINE_BIN_M * LINE_SLOPE_MAX
            if jump <= step_max and (best is None or jump < best[0]):
                best = (jump, ch)
        if best is None:
            chains.append({"first_bin": b, "last_bin": b, "lat": lat,
                           "bins": 1, "members": list(members)})
        else:
            ch = best[1]
            ch["last_bin"] = b
            ch["lat"] = lat
            ch["bins"] += 1
            ch["members"].extend(members)
    lines = []
    for ch in chains:
        if ch["bins"] < LINE_MIN_BINS:
            continue
        sel = np.unique(np.asarray(ch["members"], dtype=int))
        fwd = pts[sel, 0]
        lat = float(np.median(pts[sel, 1]))
        lines.append({"lat_m": round(lat, 3), "role": role_of(lat),
                      "fwd_span_m": [round(float(fwd.min()), 2),
                                     round(float(fwd.max()), 2)],
                      "n_px": int(len(sel))})
    return assign_roles(lines)


def candidate_label_breakdown(pixels, label) -> dict:
    """Which ENGINE class does a candidate's own pixel set sit on?

    Three outcomes matter and they are not the same statement:
    ``on_line`` (confirmed paint), ``on_road`` (on the driving surface but
    not paint - a boundary-like edge, a seam, a shadow) and neither (off
    the road entirely).  Measured on an urban junction: every candidate was
    on_road and NONE was on_line, while the engine's paint sat 6-12 m away.
    """
    px = np.asarray(pixels)
    if px.ndim != 2 or len(px) == 0:
        return {}
    lab = np.asarray(label)
    ui = np.clip(np.round(px[:, 0]).astype(int), 0, lab.shape[1] - 1)
    vi = np.clip(np.round(px[:, 1]).astype(int), 0, lab.shape[0] - 1)
    vals = lab[vi, ui]
    n = int(len(vals))
    return {"n_px": n,
            "on_line_frac": round(float((vals == CLS_LINE).mean()), 4),
            "on_road_frac": round(float((vals == 1).mean()), 4),
            "off_road_frac": round(float((vals == 0).mean()), 4)}


# --- 候选来源分解（方案 v2 §3.3 末段：来源冻结，不许为提高分数删困难来源）----
#: ``learned_frac >= 此值`` = 候选自己的像素有一半以上落在模型自己的线掩码
#: 里，记 "learned" 臂；有值但低于它 = 只有经典 CV 亮带支持（门限要求细长
#: 且在路上），记 "cv_only"；``learned_frac`` 缺失 = 发布它的臂没有留下逐像素
#: 溯源（经典检测器 / HSV 黄色先验 / 虚线恢复链），记 "unattributed"——
#: **不是"没有来源"**，只是无法与 learned 掩码对账。判据只看已有字段，
#: 不重新跑管线、不丢弃任何候选。
LEARNED_FRAC_MIN = 0.5

#: 来源臂取值（冻结；``by_arm`` 的键）
SOURCE_ARMS = ("learned", "cv_only", "unattributed")

#: 不属于分解求和、只作交叉口径的两个子集键
SOURCE_CROSS_COUNTS = ("dashed_recovery", "yellow_classic")

#: 分解的固定字段（求和字段 + 交叉计数），缺键补 0 保证形状稳定
SOURCE_FIELDS = ("by_kind", "by_colour", "by_arm")


def candidate_source_of(cand: dict) -> str:
    """单个候选的来源臂：``learned`` / ``cv_only`` / ``unattributed``。

    只读候选已经带出来的 ``learned_frac``（见 ``gate_line_candidates``），
    所以任何来源都会被计数，不会因为"说不清是哪条臂"而被丢掉。
    """
    frac = (cand or {}).get("learned_frac")
    if frac is None:
        return "unattributed"
    try:
        return ("learned" if float(frac) >= LEARNED_FRAC_MIN else "cv_only")
    except (TypeError, ValueError):
        return "unattributed"


def candidate_source_breakdown(cands) -> dict:
    """一帧（或一个集合）内候选的来源/种类分解：**只计数，不删任何来源**。

    * ``by_kind``：head 发布的 ``kind``（"solid"/"dashed"/"thin"/"unknown"），
      虚线恢复链发布为 "dashed"；
    * ``by_colour``：``colour``（"white"/"yellow"/"unknown"）；
    * ``by_arm``：见 ``candidate_source_of``（"learned"/"cv_only"/
      "unattributed"）；
    * ``dashed_recovery``：``kind == "dashed"`` 且 arm == "unattributed" 的
      子集（虚线恢复链/经典检测器，无逐像素溯源）；
    * ``yellow_classic``：``colour == "yellow"`` 且 arm == "unattributed"
      的子集（HSV 黄色先验/经典黄色臂）。

    后两个是**交叉口径**（与 ``by_kind``/``by_arm`` 有重叠），不参与求和：
    ``total == sum(by_kind.values()) == sum(by_colour.values())
    == sum(by_arm.values())``。未知的 kind/colour 值原样计进对应字典，
    不静默丢弃——"查不到来源数"正是这一节要防的事。
    """
    out = {"total": 0, "by_kind": {}, "by_colour": {}, "by_arm": {},
           "dashed_recovery": 0, "yellow_classic": 0}
    for c in cands or ():
        if not isinstance(c, dict):
            continue
        kind = str(c.get("kind") or "unknown")
        colour = str(c.get("colour") or "unknown")
        arm = candidate_source_of(c)
        out["total"] += 1
        out["by_kind"][kind] = out["by_kind"].get(kind, 0) + 1
        out["by_colour"][colour] = out["by_colour"].get(colour, 0) + 1
        out["by_arm"][arm] = out["by_arm"].get(arm, 0) + 1
        if kind == "dashed" and arm == "unattributed":
            out["dashed_recovery"] += 1
        if colour == "yellow" and arm == "unattributed":
            out["yellow_classic"] += 1
    for arm in SOURCE_ARMS:
        out["by_arm"].setdefault(arm, 0)
    return out


def merge_source_breakdowns(parts) -> dict:
    """逐帧来源分解相加（整数相加，不做比率平均；缺失/空帧不参与）。"""
    out = {"total": 0, "by_kind": {}, "by_colour": {}, "by_arm": {},
           "dashed_recovery": 0, "yellow_classic": 0}
    for p in parts or ():
        if not p:
            continue
        out["total"] += int(p.get("total", 0) or 0)
        for field in SOURCE_FIELDS:
            for k, v in (p.get(field) or {}).items():
                out[field][k] = out[field].get(k, 0) + int(v)
        for k in SOURCE_CROSS_COUNTS:
            out[k] += int(p.get(k, 0) or 0)
    for arm in SOURCE_ARMS:
        out["by_arm"].setdefault(arm, 0)
    return out


#: 一"侧"至少要有这么多引擎线像素，才算这一侧存在参考。低于它的时候，
#: 候选"没配上"只说明**没得比**，不说明候选是错的（实测：三条开发路的引擎
#: line 类≈中央线+右边缘，左边缘漆线常常一个像素都没有）。
MIN_REF_PX = 50


def candidate_side_reference(row: dict, lat_m: float,
                            *, min_px: int = MIN_REF_PX) -> tuple:
    """``(side, ref_px, available)``：候选所在侧的引擎参考像素与是否可用。"""
    side = "left" if float(lat_m) > 0 else "right"
    ref = int(row.get(f"engine_px_{side}") or 0)
    return side, ref, ref >= int(min_px)


def frame_counts(cands, *, in_p: bool) -> dict:
    """一帧的整数计数（C/R/M/L/A；公式只此一份，方案 v2 §3.3）。

    ``in_p`` 由调用方按**逐帧真值**给出：该帧真值明确有漆线像素时为 True，
    全部候选进 ``C``（覆盖率分母）；否则候选进 ``C_outside_P``，不产生任何
    R/M/L/A（``R⊆C`` 是契约不变量，实测踩到过覆盖率 >1）。

    P 的判定**不在**这里做：同一个 ``P_frames`` 键不允许有两种含义。主口径
    （真值有漆线像素）与诊断口径（引擎线能链成线）由调用方各判一次、分别
    传入，分别落在 ``counts`` 与 ``counts_chained_engine_line`` 两个键上。
    """
    sel = list(cands) if in_p else []
    matched = [c for c in sel
               if c.get("reference_available") and c.get("matched")]
    return {
        "P_frames": 1 if in_p else 0,
        "C": len(sel),
        "C_outside_P": 0 if in_p else len(cands),
        "R": sum(1 for c in sel if c.get("reference_available")),
        "M": len(matched),
        "L": len(matched),
        "A": sum(1 for c in matched if c.get("role_agrees")),
    }


def match_rate_with_reference(rows: list, *, min_px: int = MIN_REF_PX) -> dict:
    """只在"候选所在侧有参考"的候选上算匹配率，并把覆盖一起报出来。

    为什么另起一个数而不是改 ``match_rate``：旧数已被多处引用，直接改会让
    历史结论不可比；但旧数的分母里混着"该侧没有参考"的候选（实测 38%），
    把它当"未确认/假线"是错的口径。所以两个数并排给，分母都写清楚。
    """
    n_ref = n_noref = n_match_ref = 0
    for r in rows:
        for c in (r.get("candidates") or []):
            _side, _px, ok = candidate_side_reference(r, c.get("lat_m") or 0.0,
                                                      min_px=min_px)
            if not ok:
                n_noref += 1
                continue
            n_ref += 1
            if c.get("matched"):
                n_match_ref += 1
    return {"n_candidates": n_ref + n_noref,
            "n_candidates_with_reference": n_ref,
            "n_candidates_no_reference": n_noref,
            "min_ref_px": int(min_px),
            "match_rate_with_reference": (None if not n_ref
                                          else round(n_match_ref / n_ref, 4))}


def match_candidate(cand_lat: float, engine: list, *,
                    tol_m: float = MATCH_M):
    """The engine line a candidate would be the SAME marking as."""
    best = None
    for ln in engine:
        d = abs(float(cand_lat) - float(ln["lat_m"]))
        if d <= float(tol_m) and (best is None or d < best[0]):
            best = (d, ln)
    return None if best is None else best[1]


def overlay_image(colour, label, cands, cand_masks) -> np.ndarray:
    """复核图：原图 + 引擎漆线（蓝）+ 候选（命中=绿 / 未命中=红）。

    方案要求"错误候选可回查原始帧、候选像素"。先画引擎线再画候选，保证候选
    像素一定可见（两者重叠时以候选色为准，但底下的蓝仍然露在边缘）。
    """
    ov = np.array(colour, dtype=np.uint8, copy=True)
    eng = np.asarray(label) == CLS_LINE
    if eng.any():
        ov[eng] = (0.35 * ov[eng] + 0.65 * np.array((40, 110, 255))
                   ).astype(np.uint8)
    for c, m in zip(cands, cand_masks):
        if m is None or not np.any(m):
            continue
        col = np.array((0, 220, 0) if c.get("matched") else (255, 45, 45))
        ov[m] = (0.30 * ov[m] + 0.70 * col).astype(np.uint8)
    return ov


def crop_window(mask, shape, *, half_w: int = 96, half_h: int = 72):
    """围绕候选像素质心的裁剪窗口 ``(y0, y1, x0, x1)``，边界处自动收窄。"""
    ys, xs = np.nonzero(np.asarray(mask))
    if len(xs) == 0:
        return None
    cy, cx = int(np.median(ys)), int(np.median(xs))
    h, w = int(shape[0]), int(shape[1])
    y0, y1 = max(0, cy - half_h), min(h, cy + half_h)
    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
    return (y0, y1, x0, x1)


def review_crop(image, mask, *, zoom: int = 2) -> np.ndarray:
    """把候选附近放大，便于判"这是不是真漆线"（整帧缩略图上判不出来）。"""
    import cv2
    win = crop_window(mask, image.shape)
    if win is None:
        return image
    y0, y1, x0, x1 = win
    sub = image[y0:y1, x0:x1]
    return cv2.resize(sub, (sub.shape[1] * zoom, sub.shape[0] * zoom),
                      interpolation=cv2.INTER_NEAREST)


def caption_for(frame_name: str, idx: int, cands, label) -> str:
    """复核图的题注：帧名、候选数、命中数、引擎漆线像素数、未命中候选的横向距。"""
    n_match = sum(1 for c in cands if c.get("matched"))
    n_eng = int((np.asarray(label) == CLS_LINE).sum())
    lats = ", ".join(f"{c['lat_m']:+.2f}" for c in cands
                     if not c.get("matched"))[:60]
    return (f"{frame_name} i={idx} cand={len(cands)} matched={n_match} "
            f"engine_px={n_eng} unmatched_lat_m=[{lats}]")


def write_overlay(path: Path, image, caption: str) -> Path:
    """把复核图写盘（题注画在最上面，黑描边保证任何底色都读得出）。"""
    import cv2
    out = image.copy()
    for i, line in enumerate(caption.split(" | ")):
        y = 16 + i * 16
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3)
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return path


def frame_error_entry(name, index: int, exc: Exception, *,
                      stage: str = "measure") -> dict:
    """单帧失败的**结构化**条目（方案 v2 §3.3：失败必须留痕）。

    ``stage="measure"``：测量阶段失败，该帧被跳过（不进 ``rows``、计数为
    "无"而非 0）；``stage="overlay"``：只写复核图失败，测量结果仍然有效
    （帧仍在 ``rows`` 里），只记账不丢数。
    """
    return {"frame": str(name), "index": int(index), "stage": str(stage),
            "error": type(exc).__name__, "message": str(exc)[:500],
            "skipped": str(stage) == "measure"}


def frame_unknown_entry(name, index: int, reason: str, *,
                        stage: str = "camera",
                        n_engine_px: int | None = None) -> dict:
    """单帧**缺证据（UNKNOWN）**的结构化条目（方案 v2 §3.4 / T09）。

    与 ``frame_error_entry`` 的区别：UNKNOWN 不是异常，也**不等于"无线"**——
    帧读得动、真值也可能有漆线，只是这个测量问题（逐候选身份）缺少证据
    （当前只有"缺相机模型"一种）。它**不进 ``rows``**（不是成功测量，不进
    任何计数或 p50），以 ``status="unknown"`` 进 ``unknowns``；``n_engine_px``
    照记，保证"有真漆线却缺相机"没有被改写成"无线"后排除（§3.3）。
    """
    entry = {"frame": str(name), "index": int(index), "stage": str(stage),
             "status": "unknown", "reason": str(reason)}
    if n_engine_px is not None:
        entry["n_engine_px"] = int(n_engine_px)
        entry["truth_has_line_px"] = int(n_engine_px) > 0
    return entry


def refusal(reason: str, **extra) -> dict:
    """结构化拒绝结果：``reason`` + **真实**帧数（T11：processed 不许缺省成 0
    后被当成成功）。调用方按实际进度覆盖 ``frames_*``/``errors``/``unknowns``。
    """
    out = {"reason": str(reason), "errors": [], "n_errors": 0,
           "unknowns": [], "frames_unknown": 0, "frames_processed": 0,
           "frames_skipped": 0, "frames_total": 0}
    out.update(extra)
    return out


def probe(run_dir: Path, meta: dict | None = None, *, view: str = "front_main",
          limit: int | None = None, model_path: str | None = None,
          frames: list | None = None,
          null_shift_m: float | None = None,
          overlay_dir: Path | None = None,
          overlay_limit: int | None = None,
          crop_limit: int | None = None) -> dict:
    """一帧一测，逐帧整数计数进 summary；失败留痕，来源不删。

    ``frames``：显式帧清单（方案 v2 §3.1）。标定/评价必须消费**审计后的唯一
    清单**，而不是各自 glob —— 目录复制（reviewed/ 与 reviewed_full/）会带回
    同一张图的多份拷贝，实测 159 输入里只有 136 张唯一图。

    失败可见性（方案 v2 §3.3 末段）：

    * 单帧处理抛异常 -> 结构化条目进 ``errors``/summary（``frame``/``index``/
      ``stage``/``error``/``message``/``skipped``），该帧**不进 ``rows``**，
      它的计数是"无"，绝不冒充 0；
    * summary 的 ``n_errors``/``frames_skipped``/``skipped_frames`` 保证任何
      被跳过的帧都查得到；``stage="overlay"`` 的写图失败只记账，不丢测量；
    * run 整目录读不到（不存在/读不了）与"一个 frame_*.npz 都没有"是两种不同
      的 ``reason``；全部帧失败时返回带 ``errors`` 的 ``reason``，不返回空
      summary 冒充"测过但没东西"。

    summary 的计数键（名称与语义冻结，勿改）：

    * ``counts``：见 ``experiments/candidate_metrics.py`` 的 ``COUNTERS``；
      ``counts["C"]`` 只含 P 帧候选；
    * ``candidate_sources``：**P 帧内**候选的来源分解，Σ各来源 ==
      ``counts["C"]``；
    * ``candidate_sources_all``：**全部已处理帧**候选的来源分解，Σ ==
      ``candidates_total``。两份都报：只看 P 帧口径会把"只在非 P 帧出现的
      困难来源"藏掉（正是"临时删来源"想达到的效果）。

    来源分解定义（``candidate_source_breakdown``，逐候选只计数不删）：
    ``by_kind``（head 发布的 kind，虚线恢复链为 "dashed"）、``by_colour``、
    ``by_arm``（learned / cv_only / unattributed，阈值 ``LEARNED_FRAC_MIN``）、
    交叉口径 ``dashed_recovery`` 与 ``yellow_classic``（不参与求和）。

    帧数口径：``frames_processed`` = ``len(rows)``（读成功并跑完测量）；
    ``frames_with_counts`` = 真的产出 counts 的帧（**缺失 ≠ 0**，绝不给没测的
    帧补 0）；``frames_no_line_mask`` = head 没给线掩码的帧（合法结果，不是
    错误）；``frames_skipped``/``skipped_frames`` = 被异常跳过的帧；
    ``frames_unknown``/``unknown_frames``/``unknowns`` = 缺证据的 UNKNOWN 帧
    （当前只有"缺相机模型"，§3.4/T09）——**不进 rows、不进任何计数与 p50**，
    也不冒充"无线"负例；``frames_total`` == processed + skipped + unknown，
    帧数守恒、不静默少帧。

    ``counts["P_frames"]`` 的**唯一**口径（方案 §3.3 表 / 协议 v5）：该帧真值
    **明确有漆线像素**，逐帧判断，= summary 的 ``frames_with_engine_line``；
    它的候选进覆盖率分母 ``C``——真值有漆线却投影/链线失败属于缺证据，不许当
    "无线"排除（§3.3）。"引擎线能**链成线**"是**另一个量**，不占 ``P_frames``：
    逐帧 ``P_frame_chained`` 与 ``counts_chained_engine_line``，summary 汇总为
    ``frames_with_chained_engine_line``、``frames_truth_line_unchained`` 与
    ``counts_chained_engine_line``（诊断口径，不进覆盖率主口径）。
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        # "读不到" 与 "没有帧" 必须分开报：目录不存在时 glob 会静默给空表，
        # 把"路径写错/盘没挂上"伪装成"这个 run 没采到帧"。
        return refusal(f"cannot read run dir {run_dir}: not a directory")
    if frames is not None:
        fs = [Path(f) for f in frames]
        if not fs:
            return refusal(f"no frame_*.npz in {run_dir}: "
                           f"explicit frame list is empty")
    else:
        try:
            fs = sorted(run_dir.glob("frame_*.npz"))
        except OSError as exc:
            return refusal(f"cannot read run dir {run_dir}: {exc}")
        if not fs:
            return refusal(f"no frame_*.npz in {run_dir}")
    if limit:
        fs = fs[:int(limit)]
    cam = camera_from_meta(meta or {}, view)
    try:
        net = HydraNet()
        if model_path:
            # the checkpoint is chosen on the SEGMENTER, not on the head: the
            # head is the pipeline, the segmenter owns the weights
            from beamng_autopilot.vision.segmentation import Segmenter
            net.add(SemanticHead(segmenter=Segmenter(model_path=model_path)))
        else:
            net.add(SemanticHead())
    except Exception as exc:                          # noqa: BLE001
        # T11：探针自己装配失败（坏权重/CUDA OOM/依赖缺失）也必须**显式可见**，
        # 不能抛出去被调用方的 except 静默吞掉后当成"这个 run 没东西"。返回
        # 结构化 reason + 真实已处理帧数（0），绝不返回 frames=0 的成功 summary。
        return refusal(f"probe setup failed before any frame was measured: "
                       f"{type(exc).__name__}: {exc}",
                       frames_total=len(fs), camera_model_used=cam is not None,
                       run=str(run_dir), view=view)
    rows = []
    errors: list = []
    unknowns: list = []
    n_overlays = 0
    n_crops = 0
    for i, f in enumerate(fs):
        # 单帧测量整体加护栏（方案 v2 §3.3）：抛异常 -> 结构化 errors 条目 +
        # 不进 rows（计数是"无"，不是 0），summary 用 frames_skipped/
        # skipped_frames 让它可见；绝不再有 except: continue 式的静默跳过。
        overlay_plan = None
        try:
            with np.load(f) as z:
                if "colour" not in z.files:
                    raise ValueError("npz has no 'colour' array")
                colour = np.asarray(z["colour"])
                if "label" not in z.files:
                    # 这不是"某一帧坏了"，而是这个 run 不是带标注的产物：
                    # 整体拒绝，但要写清是缺 label 数组，不是读不到。
                    return refusal(
                        f"{f.name}: no label array (not a labelled run)",
                        errors=errors, n_errors=len(errors),
                        unknowns=unknowns, frames_unknown=len(unknowns),
                        frames_processed=len(rows),
                        n_frames_processed_before_refusal=len(rows),
                        frames_skipped=sum(1 for e in errors
                                           if e.get("skipped")),
                        frames_total=len(fs))
                label = np.asarray(z["label"])
            if colour.ndim != 3 or label.ndim != 2:
                raise ValueError(f"bad array shapes colour={colour.shape} "
                                 f"label={label.shape}")
            if label.shape[:2] != colour.shape[:2]:
                raise ValueError(f"label {label.shape[:2]} does not match "
                                 f"colour {colour.shape[:2]}")
            if cam is None:
                # 方案 v2 §3.3/§3.4、T09：缺相机 = **缺证据（UNKNOWN）**，不是
                # "处理过但为空"，更不是"没有线"。该帧不进 rows（不算成功测量、
                # 不进任何计数/p50），逐帧结构化条目进 unknowns；真值有没有漆线
                # 像素照记，防止把缺相机写成"无线"后排除（§3.3）。
                # 不跑 head：没有相机模型就回答不了逐候选身份问题，为一个无法
                # 回答的问题推理只是浪费；图像面统计留给有相机的 run。
                unknowns.append(frame_unknown_entry(
                    f.name, i,
                    f"no camera model for view {view!r} (meta missing or "
                    f"malformed): per-candidate identity is UNKNOWN",
                    n_engine_px=int((np.asarray(label) == CLS_LINE).sum())))
                continue
            ctx = FrameContext(frame_rgb=colour, cam=cam, pos=np.zeros(3),
                               heading=0.0, ground_z=0.0, role=view)
            # HydraNet 把 head 异常吞进 net.errors（run 不抛），只看 out 会把
            # "head 失败"记成"没有输出"；snapshot 前后差异把失败捞回来。
            head_errors_before = dict(getattr(net, "errors", None) or {})
            out = net.run(ctx).get("semantic")
            head_failed = None
            for _name, _msg in (getattr(net, "errors", None) or {}).items():
                if head_errors_before.get(_name) != _msg:
                    head_failed = f"{_name}: {_msg}"
            line_mask = None
            if out is not None and "line" in out.masks:
                line_mask = np.asarray(out.masks["line"], dtype=bool)
            if line_mask is None:
                if head_failed:
                    raise RuntimeError(f"semantic head failed ({head_failed})")
                rows.append({"frame": i, "status": "no_line_mask",
                             "reason": "no line mask from the head"})
                continue
            if line_mask.shape != label.shape[:2]:
                raise ValueError(f"line mask {line_mask.shape} does not match "
                                 f"label {label.shape[:2]}")
            stats = compare_masks(line_mask, label == CLS_LINE)
            stats["frame"] = i
            # --- ground-projected identity (vehicle frame) ---------------
            if cam is not None:
                eng_lines = engine_lines(label == CLS_LINE, cam)
                cands = []
                cand_masks: list = []
                cand_px = np.zeros(label.shape, dtype=bool)
                for mk in ((out.meta.get("markings") if out is not None
                            else None) or []):
                    px = (np.asarray(mk.pixels)
                          if mk.pixels is not None else np.empty((0, 2)))
                    if px.size == 0:
                        continue
                    _ui = np.clip(np.round(px[:, 0]).astype(int), 0,
                                  label.shape[1] - 1)
                    _vi = np.clip(np.round(px[:, 1]).astype(int), 0,
                                  label.shape[0] - 1)
                    cand_px[_vi, _ui] = True
                    gp = project_pixels(px[:, 0], px[:, 1], cam, stride=2)
                    if len(gp) == 0:
                        continue
                    m = np.zeros(label.shape, dtype=bool)
                    m[_vi, _ui] = True
                    cand_masks.append(m)
                    lat = float(np.median(gp[:, 1]))
                    prov = dict(getattr(mk, "meta", None) or {})
                    cands.append({"kind": mk.kind, "colour": mk.color,
                                  "lat_m": round(lat, 3),
                                  "role": role_of(lat), "n_px": int(len(gp)),
                                  "learned_frac": prov.get("learned_frac"),
                                  "on_road_frac": prov.get("on_road_frac"),
                                  "aspect": prov.get("aspect"),
                                  **candidate_label_breakdown(px, label)})
                cands = assign_roles(cands)
                null_shift = float(null_shift_m or 0.0)
                eng_null = ([{**ln, "lat_m": ln["lat_m"] - null_shift}
                             for ln in eng_lines] if null_shift else [])
                matched = agree = n_null = 0
                by_role = {}
                for c in cands:
                    by_role.setdefault(c["role"], 0)
                    by_role[c["role"]] += 1
                    ln = match_candidate(c["lat_m"], eng_lines)
                    c["engine_lat_m"] = None if ln is None else ln["lat_m"]
                    c["engine_role"] = None if ln is None else ln["role"]
                    c["matched"] = ln is not None
                    _side, _ref, _ok = candidate_side_reference(stats,
                                                                c["lat_m"])
                    c["side"] = _side
                    c["side_ref_px"] = _ref
                    c["reference_available"] = bool(_ok)
                    c["role_agrees"] = bool(ln is not None
                                            and ln["role"] == c["role"])
                    if ln is not None:
                        matched += 1
                        if ln["role"] == c["role"]:
                            agree += 1
                    if (eng_null
                            and match_candidate(c["lat_m"], eng_null)
                            is not None):
                        n_null += 1
                stats["engine_lines"] = eng_lines
                stats["candidates"] = cands
                stats["n_candidates"] = len(cands)
                stats["n_candidates_matched"] = matched
                stats["n_role_agreement"] = agree
                stats["match_rate"] = (None if not cands
                                       else round(matched / len(cands), 4))
                stats["role_agreement_rate"] = (None if not matched
                                                else round(agree / matched, 4))
                stats["candidate_roles"] = by_role
                # 逐帧整数计数（方案 v2 §3.3）：C 全部候选 / R 该侧有参考 /
                # M 匹配（⊆R）/ L 可判角色（⊆M）/ A 一致（⊆L）。判据**复用**
                # 上面同一循环写下的逐候选标志，公式只此一份（frame_counts）。
                #
                # ``counts["P_frames"]`` 的**唯一主口径**（§3.3 表 + 协议 v5）：
                # 该帧真值**明确有漆线像素**，逐帧判断——与"能不能链成线"无关。
                # 真值有漆线却投影/链线失败属于**缺证据**，不许当"无线"排除
                # （§3.3），所以它的候选仍进覆盖率分母 C。
                # "引擎线能链成线"是**另一个量**，另立显式键、不占 P_frames：
                # 逐帧 ``P_frame_chained`` + ``counts_chained_engine_line``；
                # summary 汇总为 ``frames_with_chained_engine_line`` 与
                # ``counts_chained_engine_line``（诊断口径，不进覆盖率主口径）。
                _p_frame = int(stats.get("n_engine_px") or 0) > 0
                _p_chained = bool(eng_lines)
                stats["counts"] = frame_counts(cands, in_p=_p_frame)
                stats["counts_chained_engine_line"] = frame_counts(
                    cands, in_p=_p_chained)
                stats["P_frame"] = _p_frame
                stats["P_frame_chained"] = _p_chained
                stats["status"] = "measured"
                # 逐帧来源分解（方案 v2 §3.3 末段）：本帧**全部**候选按
                # kind/颜色/来源臂计数，不删任何来源；非 P 帧也记，否则"只在
                # 非 P 帧出现的困难来源"会在 summary 里消失。summary 再分两份
                # 汇总：P 帧口径（Σ == counts["C"]）与全帧口径。
                stats["candidate_sources"] = candidate_source_breakdown(cands)
                # Did the candidate SET cover the engine's paint at all?  This
                # is the question the per-candidate fractions cannot answer: a
                # set can be entirely off-paint even when one candidate
                # overlaps a few paint pixels.
                eng_mask = (label == CLS_LINE)
                n_eng = int(eng_mask.sum())
                stats["candidate_paint_recall"] = (
                    None if n_eng == 0 else
                    round(float((cand_px & eng_mask).sum()) / n_eng, 4))
                stats["candidates_learned_backed"] = sum(
                    1 for c in cands
                    if (c.get("learned_frac") or 0.0) >= LEARNED_FRAC_MIN)
                stats["candidates_cv_only"] = sum(
                    1 for c in cands
                    if (c.get("learned_frac") or 0.0) < LEARNED_FRAC_MIN)
                stats["line_candidate_gate"] = (out.meta.get("line_candidates")
                                                if out is not None else None)
                stats["n_candidates_matched_null"] = n_null
                stats["candidates_on_line"] = sum(
                    1 for c in cands if (c.get("on_line_frac") or 0.0) >= 0.5)
                stats["candidates_on_road_only"] = sum(
                    1 for c in cands
                    if (c.get("on_line_frac") or 0.0) < 0.5
                    and (c.get("on_road_frac") or 0.0) >= 0.5)
                stats["candidates_off_road"] = sum(
                    1 for c in cands
                    if (c.get("on_line_frac") or 0.0) < 0.5
                    and (c.get("on_road_frac") or 0.0) < 0.5)
                # 复核图只影响"看得到什么"，不影响测量范围：帧集由 --limit
                # 决定，overlay_limit / crop_limit 只限制写多少张。这里只登记
                # "要画"，绘图/写盘放到 try 外面的装饰阶段，失败不丢测量。
                if overlay_dir is not None and (
                        overlay_limit is None
                        or n_overlays < int(overlay_limit)):
                    overlay_plan = (colour, label, cands, cand_masks)
        except Exception as exc:                      # noqa: BLE001
            # 失败可见：结构化条目 + 该帧不进 rows（计数是"无"，不是 0）
            errors.append(frame_error_entry(f.name, i, exc))
            continue
        rows.append(stats)
        if overlay_plan is not None:
            ov_colour, ov_label, ov_cands, ov_masks = overlay_plan
            try:
                base = overlay_image(ov_colour, ov_label, ov_cands, ov_masks)
                write_overlay(Path(overlay_dir) / f"review_{i:03d}.png", base,
                              caption_for(f.name, i, ov_cands, ov_label))
                n_overlays += 1
                if crop_limit:
                    k = 0
                    for c, m in zip(ov_cands, ov_masks):
                        if c.get("matched") or k >= int(crop_limit):
                            continue
                        cap_c = (f"{f.name} i={i} lat={c['lat_m']:+.2f}m "
                                 f"on_line={(c.get('on_line_frac') or 0):.2f} "
                                 f"on_road={(c.get('on_road_frac') or 0):.2f} "
                                 f"off={(c.get('off_road_frac') or 0):.2f} "
                                 f"kind={c.get('kind')} "
                                 f"learned={(c.get('learned_frac') or 0):.2f}")
                        write_overlay(
                            Path(overlay_dir) / f"crop_{i:03d}_{k}.png",
                            review_crop(base, m), cap_c)
                        k += 1
                        n_crops += 1
            except Exception as exc:                  # noqa: BLE001
                # 装饰失败不丢测量：只记账（stage="overlay"），帧仍在 rows 里
                errors.append(frame_error_entry(f.name, i, exc,
                                                stage="overlay"))
    if not rows:
        # 全部帧都测不了（失败和/或缺证据 UNKNOWN）：返回"不可测"的 reason +
        # 结构化 errors/unknowns 与**真实**帧数，不返回空 summary 冒充"测过但
        # 没东西"（读不到/没有帧已在上面分开报过）。帧数缺测时也不默认成 0
        # 后继续报成功（方案 T11）。
        first = errors[0] if errors else None
        first_unknown = unknowns[0] if unknowns else None
        if first is not None:
            detail = f"{first['frame']}: {first['error']}: {first['message']}"
        elif first_unknown is not None:
            detail = f"{first_unknown['frame']}: {first_unknown['reason']}"
        else:
            detail = "no frame produced a measurement"
        return refusal(f"no frame in {run_dir} could be measured "
                       f"({len(errors)} frame error(s), {len(unknowns)} "
                       f"UNKNOWN frame(s)); first: {detail}",
                       errors=errors, n_errors=len(errors),
                       unknowns=unknowns, frames_unknown=len(unknowns),
                       frames_skipped=len(errors), frames_total=len(fs),
                       camera_model_used=cam is not None)
    def p50(key):
        vals = [r[key] for r in rows
                if isinstance(r.get(key), (int, float)) and r[key] is not None]
        return None if not vals else round(float(np.median(vals)), 4)
    # 有真值漆线像素的帧数（只可能出现在有 counts 的帧上：无掩码帧根本没算
    # n_engine_px；缺相机的帧已进 unknowns、不在 rows 里冒充）
    n_engine_seen = sum(1 for r in rows if (r.get("n_engine_px") or 0) > 0)

    def total(key):
        """Sum an integer counter across rows (bool is not a counter)."""
        vals = [r.get(key) for r in rows]
        return int(sum(v for v in vals if isinstance(v, int) and not
                       isinstance(v, bool)))

    def total_len(key):
        """Number of ENTRIES across rows for a per-frame list field."""
        return int(sum(len(v) for v in (r.get(key) for r in rows)
                       if isinstance(v, (list, tuple))))

    n_cand = total("n_candidates")
    n_match = total("n_candidates_matched")
    n_agree = total("n_role_agreement")
    # 失败可见性（方案 v2 §3.3）：被跳过的帧数与帧名必须在 summary 里可查
    n_skipped = sum(1 for e in errors if e.get("skipped"))
    # "引擎线能链成线"是诊断口径（另一个键），与 counts["P_frames"] 无关
    n_chained_p = sum(1 for r in rows if r.get("P_frame_chained"))
    # 来源分解：P 帧口径（Σ == counts["C"]）与全帧口径（Σ == candidates_total）
    src_pframe = merge_source_breakdowns(
        [r.get("candidate_sources") for r in rows
         if (r.get("counts") or {}).get("P_frames")])
    src_all = merge_source_breakdowns([r.get("candidate_sources")
                                       for r in rows])
    summary = {
        "frames": len(rows),
        "frames_with_engine_line": n_engine_seen,
        # ground-projected identity, vehicle frame
        "candidates_total": n_cand,
        "candidates_matched": n_match,
        # NULL control: the same measurement against engine lines shifted
        # sideways.  If the shifted rate is similar, "matches" are just
        # candidates landing inside the tolerance by chance.
        "null_shift_m": float(null_shift_m or 0.0),
        "candidates_matched_null": total("n_candidates_matched_null"),
        "match_rate_null": (None if not n_cand or not null_shift_m
                            else round(total("n_candidates_matched_null")
                                       / n_cand, 4)),
        "roles_agreeing": n_agree,
        "match_rate": (None if not n_cand else round(n_match / n_cand, 4)),
        "role_agreement_rate": (None if not n_match
                                else round(n_agree / n_match, 4)),
        "engine_lines_total": total_len("engine_lines"),
        # the three-valued statement per candidate, against the engine's
        # own classes: confirmed paint / on the road but unpainted / off it
        # 按侧判定：分母只含"该侧有参考"的候选（见 match_rate_with_reference）
        **match_rate_with_reference(rows),
        # 整数计数总计（先加总、再算比率；见 experiments/candidate_metrics.py）。
        # P_frames 的**唯一**口径 = 真值明确有漆线像素的帧（逐帧）；这里不做
        # 任何"覆盖成有像素帧数"的二次赋值——同一键只有一种含义。
        "counts": cm.totals([r.get("counts") or {} for r in rows]),
        # 同一批帧、同样公式，但 P 只取"引擎线能链成线"的帧（诊断口径，**不进**
        # 覆盖率主口径）：二者之差就是"有漆线像素却链不成线"的缺证据帧，方案
        # §3.3 不许把它们当"无线"排除，所以主口径 counts 仍含它们的候选。
        "counts_chained_engine_line": cm.totals(
            [r.get("counts_chained_engine_line") or {} for r in rows]),
        # 帧数口径（方案 v2 §3.3/§3.4）：processed = 读成功并跑完测量；
        # with_counts = 真的产出 counts（缺 counts 的帧**不补 0**，也不混进
        # 计数；缺失与"实测 0"因此可区分）；no_line_mask = head 没给线掩码
        # （合法结果，不是错误）；unknown = 缺证据的 UNKNOWN 帧（缺相机），
        # **不进 rows**、不进计数与 p50；skipped = 被异常跳过的帧，
        # skipped_frames 是帧名清单；overlay 阶段失败只进 errors 不跳帧。
        "frames_processed": len(rows),
        "frames_with_counts": sum(1 for r in rows
                                  if isinstance(r.get("counts"), dict)),
        "frames_no_line_mask": sum(1 for r in rows if r.get("reason")),
        "frames_unknown": len(unknowns),
        "unknown_frames": [u["frame"] for u in unknowns],
        "unknowns": unknowns,
        "frames_skipped": n_skipped,
        "skipped_frames": [e["frame"] for e in errors if e.get("skipped")],
        "n_errors": len(errors),
        "errors": errors,
        # 帧数守恒：frames_total == frames_processed + frames_skipped +
        # frames_unknown（UNKNOWN/失败都不进 rows，帧不会静默消失）。
        "frames_total": len(fs),
        # counts["P_frames"] 主口径 = 真值明确有漆线**像素**的帧（逐帧，
        # = frames_with_engine_line）；"引擎线能**链成线**"是另一个量，落在
        # frames_with_chained_engine_line / counts_chained_engine_line 两个
        # 独立键上，两者之差 = 有像素但链不成线的缺证据帧数。
        "frames_with_chained_engine_line": n_chained_p,
        "frames_truth_line_unchained": n_engine_seen - n_chained_p,
        # 候选来源分解（方案 v2 §3.3 末段，定义见 candidate_source_breakdown）：
        # candidate_sources 只含 **P 帧**候选（Σ == counts["C"]）；
        # candidate_sources_all 含全部已处理帧（Σ == candidates_total）。
        # 两份都报：只看 P 帧口径会让"只在非 P 帧出现的困难来源"消失。
        "candidate_sources": src_pframe,
        "candidate_sources_all": src_all,
        "role_compared": sum(1 for r in rows
                             for c in (r.get("candidates") or [])
                             if c.get("reference_available")
                             and c.get("matched")),
        "candidates_on_engine_line": total("candidates_on_line"),
        "candidates_on_road_only": total("candidates_on_road_only"),
        "candidates_off_road": total("candidates_off_road"),
        "candidates_learned_backed": total("candidates_learned_backed"),
        "candidates_cv_only": total("candidates_cv_only"),
        "candidate_paint_recall_p50": p50("candidate_paint_recall"),
        "precision_p50": p50("precision"),
        "recall_p50": p50("recall"),
        "iou_p50": p50("iou"),
        "recall_left_p50": p50("recall_left"),
        "recall_right_p50": p50("recall_right"),
        "frames_both_engine_sides": sum(1 for r in rows
                                        if r.get("engine_sides") == 2),
        "frames_both_perception_sides": sum(1 for r in rows
                                            if r.get("perception_sides") == 2),
        "camera_model_used": cam is not None,
        "limits": [
            "Engine labels render KNOWN line materials: worn/faded paint may "
            "be unlabelled, so low precision can mean 'unlabelled'.",
            "Ground projection uses the T05 oracle's flat-plane model: a "
            "slope moves the FORWARD distance (0.9 m p50 at 8 deg); the "
            "lateral SIGN identity relies on is not affected by it.",
            "A candidate with no engine line nearby is 'unconfirmed', not "
            "'false': the engine renders known materials only, so faded or "
            "worn paint can be unlabelled.",
            "match_rate_with_reference conditions on the candidate's own "
            "side having >=50 engine px: the raw match_rate also counts "
            "candidates whose side has no reference at all, and those are "
            "UNKNOWN rather than wrong (measured: the engine's line class "
            "here is roughly centre+right edge, left edge often unlabelled).",
        ],
    }
    summary["model"] = str(model_path) if model_path else "default_model_path"
    out = {"run": str(run_dir), "view": view, "summary": summary, "rows": rows,
           "errors": errors, "unknowns": unknowns,
           "n_unknown": len(unknowns)}
    if overlay_dir is not None:
        out["overlay_dir"] = str(overlay_dir)
        out["n_overlays"] = n_overlays
        out["n_crops"] = n_crops
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir with frame_*.npz")
    ap.add_argument("--meta", default=None, help="collector meta.json")
    ap.add_argument("--view", default="front_main")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--null-shift-m", type=float, default=None,
                    help="lateral shift applied to the engine lines as a "
                         "NULL control for the match rate")
    ap.add_argument("--model", default=None,
                    help="segmentation checkpoint (default: "
                         "segmentation.default_model_path())")
    ap.add_argument("--json", default=None)
    ap.add_argument("--overlay-out", default=None,
                    help="把每帧复核图写到这个目录（原图 + 引擎漆线=蓝 + "
                         "候选命中=绿/未命中=红 + 题注），供人工反例复核")
    ap.add_argument("--overlay-limit", type=int, default=None,
                    help="最多写多少张复核图（帧集仍由 --limit 决定，"
                         "不影响测量范围）")
    ap.add_argument("--crop-limit", type=int, default=0,
                    help="每帧最多写几张『未匹配候选』的 2x 局部放大图，"
                         "供人工判真线/假线（0=不写）")
    args = ap.parse_args(argv)
    meta = None
    if args.meta:
        mp = Path(args.meta)
        if not mp.is_file():
            print(f"[ident] no meta at {mp}; the camera model is unavailable, "
                  f"so per-candidate identity is UNKNOWN for every frame "
                  f"(T09: 缺相机 -> UNKNOWN, not a measured zero)")
        else:
            meta = json.loads(mp.read_text(encoding="utf-8"))
    res = probe(Path(args.run), meta, view=args.view, limit=args.limit,
                model_path=args.model, null_shift_m=args.null_shift_m,
                overlay_dir=(Path(args.overlay_out) if args.overlay_out
                             else None),
                overlay_limit=args.overlay_limit,
                crop_limit=args.crop_limit or None)
    if "reason" in res:
        print(f"[ident] {res['reason']}")
        return 2
    s = res["summary"]
    print(f"[ident] {res['run']} view={res['view']} frames={s['frames']} "
          f"(engine line in {s['frames_with_engine_line']}) camera_model="
          f"{s['camera_model_used']}")
    if s.get("frames_skipped"):
        # 被跳过的帧必须看得见：帧名 + 第一条错误，绝不静默少测
        first = (s.get("errors") or [{}])[0]
        print(f"[ident] WARNING {s['frames_skipped']} frame(s) skipped "
              f"({s['n_errors']} error(s)), e.g. {first.get('frame')}: "
              f"{first.get('error')}: {str(first.get('message'))[:120]}")
    if s.get("match_rate_null") is not None:
        print(f"[ident] NULL control (lines shifted {s['null_shift_m']} m): "
              f"match {s['candidates_matched_null']}/{s['candidates_total']} "
              f"= {s['match_rate_null']}")
    print(f"[ident] precision p50={s['precision_p50']} recall p50="
          f"{s['recall_p50']} iou p50={s['iou_p50']} | side recall L/R="
          f"{s['recall_left_p50']}/{s['recall_right_p50']} | both sides: "
          f"engine {s['frames_both_engine_sides']} / perception "
          f"{s['frames_both_perception_sides']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"[ident] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
