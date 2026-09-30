"""把受控场景批次里**通过资格与几何校验**的站点导出成带凭证的训练数据（T16 §4.3）。

输入：`m5_controlled_scenes.py` 的一次运行产物（`scene_report.json` +
`scene_*.json` + `frames/*.npz`）。**不需要游戏**——复核与导出都在离线做。

四道门槛，缺一不可（方案 §4.3"逐通道赋资格"）：

1. **annotation 线类覆盖** ≥ ``--min-coverage``（默认 0.8）：训练标签就是
   annotation，所以只认 annotation 的覆盖——外观再可见也不能当标签来源，
   否则等于教模型"这里有线的地方没有线"（扩量实测：部分路段 annotation 线类
   像素只有个位数）。**逐实例**判定，且要求**每个**线实例都达标：只过一个
   实例时另一条的标签是断续的，同样在教模型"线可以断"。
2. **几何一致性**：校准后逐帧 ``after_px`` 均值 ≤ ``--max-after-px``（默认
   2.0）且**中位** ≤ ``--max-after-median-px``（默认 1.0）。中位门是必须的：
   均值会被个别坏帧掩盖（多档批次里同一场景 0.9 px 与 50 px 并存）。
3. **横向档位** |``lateral_m``| ≤ ``--max-line-lat-m``（默认 2.0）：多档批次
   的 2.4 m 档落在 annotation 标注带之外（实测残差中位 2.125 px），只收
   1.2/1.8 档。
4. **漆面外观一致**：认证实例的真值点投影处，RGB 邻域必须像**漆**（亮的无彩
   脊，或黄漆；阈值按开发集人工标签的线像素实测标定），合格点占比 ≥
   ``--min-paint-agreement``（默认 0.9）。这条是 2026-09-29 找到的**标签中毒
   机制**：被遮挡车/覆盖贴花挡住的线，引擎 annotation 仍把它标成线（annotation
   通道不做深度测试），于是标签在 RGB 是车/泥土的地方写着"线"。实测已导出的包
   里有 3/7 的漆面一致率只有 0.03–0.74（好包 1.00）；把它们混进训练后身份率从
   0.4316 崩到 0.1728。门与新判据在实测站点上的判别力：好站 1.00，被车挡住的
   线 0.00–0.32（`occluded_line_a1s2` 0.0、`slope_curve_a1s3` 右 0.32）。

通过 -> 写一份训练可用的数据目录（与采集包同构）：

```
<out>/m5auto_a<锚点>_<站点>/
    meta.json                      # map_name / source_id=场景族 / map_name_source
    annotation.json                # write_truth_credentials：engine_verified + v1 凭证
    front_main/frame_0000N.npz     # colour + label（+ annotation_raw）
```

未通过 -> 只记进 ``export_report.json`` 的 isolated 清单（带原因），**不写数据**：
"失败时隔离并自动换场景，不盲目扩量"（方案 §4.3）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_auto_truth_export.py `
        --batch logs\\experiments\\t16_scenes_scale_20260927 `
        --out logs\\experiments\\t16_autotruth_scale_20260927
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.auto_truth import (  # noqa: E402
    TRUTH_CONTRACT, VERIFIER_VERSION, _rotate_camera_basis,
    frame_content_shas, line_evidence_coverage, line_evidence_mask,
    palette_sha, verify_batch, write_truth_credentials,
)
from beamng_autopilot_tech.annotations import annotation_palette  # noqa: E402

EXPORTER_VERSION = "auto_truth_export_v1"


def _load_scenes_module():
    spec = importlib.util.spec_from_file_location(
        "m5_controlled_scenes", ROOT / "scripts" / "m5_controlled_scenes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_controlled_scenes"] = mod
    spec.loader.exec_module(mod)
    return mod


def palette_from_frames(frames: list) -> dict:
    """离线重建批次调色板：静态兼容盘（路面/漆线类色）+ 帧内实际出现的类色。

    为什么必须带上：`verify_batch` 没有调色板会直接判 `PALETTE_CHANGED`，
    于是整个批次复核不过、写不出 `engine_verified`（实测踩到）。采集侧用
    `bng.get_annotations()` 拿活调色板；离线导出没有游戏，就按同一口径重建
    （登记观测色，避免 UNKNOWN_CLASS 把"未登记"和"没登记"混在一起）。
    """
    base = annotation_palette(None)
    classes = {str(k): list(v) for k, v in (base.get("classes") or {}).items()}
    colors = set()
    for fr in frames:
        a = np.asarray(fr["annotation"])[:, :, :3].reshape(-1, 3)
        if not a.size:
            continue
        for row in np.unique(a, axis=0):
            colors.add(tuple(int(v) for v in row))
    for c in sorted(colors):
        classes.setdefault(f"observed_{c[0]:02x}{c[1]:02x}{c[2]:02x}", list(c))
    pal = {"version": base.get("version"), "source": base.get("source"),
           "classes": classes, "road": base.get("road"), "line": base.get("line")}
    pal["sha"] = palette_sha(pal)
    return pal


def args_default_margin():
    """导出用的遮挡余量（米）。1.5 = 3 × 默认 0.5：该通道精度不够时用。"""
    class _V:
        value = 1.5
    return _V()


def _sha16_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _labels_digest(label_shas: list) -> str:
    return _sha16_bytes("|".join(sorted(str(s) for s in label_shas)).encode())


def scene_anchor(name: str) -> int:
    """站点名 -> 锚点号（场景族 = 锚点，划分 train/dev/final 按它分）。

    **必须在分支之前算**：负例分支也要用它。实测踩到——旧代码把它写在负例
    分支后面，于是首个负例 `UnboundLocalError: 'anchor'`，后面的负例又沿用
    上一站的锚点，包名/meta 的 `source_id` 全是错的（`known_no_line_a5s1`
    写成了 `m5auto_a2_...`）。
    """
    if "_a" not in name:
        return 0
    try:
        return int(name.rsplit("_a", 1)[1].split("s")[0])
    except (IndexError, ValueError):
        return 0


def calibrated_frames(frames: list, rec: dict) -> list:
    """套上逐帧静态投影标定（标定是场景的渲染/相机常数，随批次落盘）。

    不套的话投影残差会把复核判成不通过、写不出 engine_verified（实测踩到：
    9 个站点全部 verified=False）。`evaluate_scene` 与 `export_scene` 共用，
    免得两条路径的判定口径漂移。
    """
    pf = ((rec.get("camera_alignment") or {}).get("per_frame") or [])
    out = []
    for i, fr in enumerate(frames):
        cam = fr["camera"]
        if i < len(pf) and pf[i].get("status") == "measured":
            cam = _rotate_camera_basis(cam, float(pf[i].get("yaw_deg") or 0.0),
                                       float(pf[i].get("pitch_deg") or 0.0))
        out.append(cam)
    return out


#: 漆面外观判据（**与材质声明无关**）。阈值按开发集人工标签实测标定
#: （2026-09-29，`review_pack_20260926/reviewed_full` 的 12 个目录、26 万线像素）：
#: 人工标注的线像素 max 通道 p10≈154 / p50≈205，色度（max-min）中位 ≈1——即
#: "亮的、近无彩的脊"。原来的 185 绝对门会把 30% 的人工线像素判成非漆，
#: 而实测 dim 站点（material_mix 的线只有 162 灰）本来就在开发集分布内。
#: 黄漆另给一支（生成场景里有黄线材质）。
#: **蓝漆不认证**：蓝漆与遮挡车（场景里是蓝车）在颜色上分不开，宁可保守隔离
#: （蓝线只出现在 symmetric 约定的 material_mix，从未进过训练包）。
PAINT_MIN_BRIGHT = 140
PAINT_MAX_CHROMA = 60
PAINT_YELLOW_RG_MIN = 120
PAINT_YELLOW_B_MAX = 140
PAINT_YELLOW_RB_MIN = 60
PAINT_CRITERION = "achromatic_bright_or_yellow"


def paint_like(rgb, u, v, *, radius_px: int = 2) -> bool | None:
    """投影点邻域是否像**漆**（亮的无彩脊，或黄漆）；``None`` = 画面外。

    画面外必须与"不是漆"分开：`line_appearance` 对越界返回 line_like=False，
    拿它当"不是漆"会把画面外的点算成不合格（UNKNOWN≠FAIL 的同一个道理）。
    """
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3 or a.size == 0:
        return None
    if not (np.isfinite(u) and np.isfinite(v)):
        return None          # 相机背后的点投影出 NaN：先判有限再取整（实测踩到）
    h, w = a.shape[:2]
    x, y = int(round(float(u))), int(round(float(v)))
    if not (0 <= x < w and 0 <= y < h):
        return None
    win = a[max(0, y - radius_px):y + radius_px + 1,
            max(0, x - radius_px):x + radius_px + 1, :3].astype(np.int16)
    mx = win.max(axis=2)
    mn = win.min(axis=2)
    r, g, b = win[:, :, 0], win[:, :, 1], win[:, :, 2]
    achromatic_bright = ((mx >= PAINT_MIN_BRIGHT)
                         & ((mx - mn) <= PAINT_MAX_CHROMA))
    yellow = ((r >= PAINT_YELLOW_RG_MIN) & (g >= PAINT_YELLOW_RG_MIN)
              & (b <= PAINT_YELLOW_B_MAX) & ((r - b) >= PAINT_YELLOW_RB_MIN)
              & (np.abs(r - g) <= 45))
    return bool((achromatic_bright | yellow).any())


def paint_agreement(lines: list, frames: list, cal_frames: list, *,
                    radius_px: int = 2) -> dict:
    """逐线实例：真值点投影处的 RGB 像不像漆（判据见 ``PAINT_CRITERION``）。

    只查 ``class == 2`` 的点（路面点 ``class == 1`` 落在沥青上，查它们等于
    拿"路面不是漆"去否掉整个场景——实测合成批次因此从 1.00 掉到 0.586）。

    返回 ``{role: {checked, like, ratio}}``；``checked == 0`` 的实例
    ``ratio = None``（没有可查的点 -> UNKNOWN，不当通过）。
    """
    from beamng_autopilot.experiments.auto_truth import project_point
    out: dict = {}
    for inst in lines or []:
        pts = [p for p in (inst.get("truth_points") or [])
               if p.get("class") in (None, 2)]     # 只查线类点（class 1 是路面）
        checked = like = 0
        for fr, cam in zip(frames, cal_frames):
            rgb = np.asarray(fr["rgb"])
            for p in pts:
                try:
                    u, v = project_point(p.get("world"), cam)
                except (TypeError, ValueError):
                    continue
                got = paint_like(rgb, u, v, radius_px=radius_px)
                if got is None:
                    continue
                checked += 1
                like += int(got)
        out[str(inst.get("role") or "?")] = {
            "checked": checked, "like": like,
            "ratio": (round(like / checked, 4) if checked else None)}
    return out


#: 结构足迹的投影半径（像素）：结构是 2–3 m 的实物，投影中心 ±该半径即其足迹。
#: 取 60 是保守值（宁可把"结构附近"算进足迹，也不放过远处的真漆线——
#: 后者由 outside 绝对量上限兜住）。
STRUCT_FOOTPRINT_PX = 60


def structure_appearance_split(frames: list, structures: list, *,
                               radius_px: int = STRUCT_FOOTPRINT_PX) -> dict:
    """把"像线的外观像素"分成落在**已声明结构投影足迹内/外**两份。

    结构负例的认证靠这两条一起成立：annotation 线类像素 = 0（引擎自己说这里
    不是漆线）+ 像线的外观能归因到已声明的非漆结构（石墙/护栏）上。只看前者
    会把"护栏=线"教给模型；只看后者则等于拿外观当标签来源（方案 §4.2 禁止）。
    """
    from beamng_autopilot.experiments.auto_truth import project_point
    inside = outside = 0
    per_frame = []
    for fr in frames:
        mask, _mode, _why = line_evidence_mask(
            {"rgb": fr["rgb"], "label": fr["label"]}, mode="appearance")
        if mask is None:
            continue
        h, w = np.asarray(mask).shape
        foot = np.zeros((h, w), dtype=bool)
        for st in structures or []:
            try:
                u, v = project_point(st.get("pos"), fr["camera"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            x, y = int(round(u)), int(round(v))
            if not (-radius_px <= x < w + radius_px
                    and -radius_px <= y < h + radius_px):
                continue
            x0, x1 = max(0, x - radius_px), min(w, x + radius_px + 1)
            y0, y1 = max(0, y - radius_px), min(h, y + radius_px + 1)
            foot[y0:y1, x0:x1] = True
        n_in = int(np.sum(mask & foot))
        n_out = int(np.sum(mask & ~foot))
        inside += n_in
        outside += n_out
        per_frame.append({"path": str(fr.get("path") or ""),
                          "inside": n_in, "outside": n_out})
    return {"criterion": "appearance_line_like_px_attributed_to_declared_"
                         "structures",
            "radius_px": int(radius_px), "n_structures": len(structures or []),
            "inside": inside, "outside": outside, "per_frame": per_frame}


def scene_frame_paths(batch: Path, name: str) -> list:
    return sorted((batch / "frames").glob(f"{name}_*.npz"))


def load_scene_frames(batch: Path, name: str) -> list:
    """读回某站点的落盘帧（rgb / annotation / label / camera）。"""
    out: list = []
    for fp in scene_frame_paths(batch, name):
        fr = np.load(fp, allow_pickle=False)
        out.append({"path": fp,
                    "rgb": np.asarray(fr["rgb"]),
                    "annotation": np.asarray(fr["annotation"]),
                    "label": np.asarray(fr["label"]),
                    # 深度是**帧身份检查**的第三路（缺它直接 RESOLUTION_MISMATCH：
                    # 判据要求三路齐全且同分辨率）；落盘的是生成时转好的米制深度
                    "depth": np.asarray(fr["depth_raw"], dtype=np.float32),
                    "camera": json.loads(str(fr["camera_json"]))})
    return out


def evaluate_scene(batch: Path, name: str, rec: dict, *, min_coverage: float,
                   max_after_px: float, max_after_median_px: float | None = None,
                   max_line_lat_m: float | None = None,
                   min_paint_agreement: float | None = None) -> dict:
    """资格 + 几何 + 档位 + 漆面四道门；返回判定与逐项证据。

    只读产物，不写任何东西——先判定再决定要不要导出。新增的三道门默认关闭
    （``None``），调用方（``main`` 的 CLI 默认值）显式开启，老测试不受影响。
    """
    out: dict = {"scene": name, "eligible": False, "reasons": [],
                 "coverage": None, "after_px": None}
    lines = ((rec.get("generated") or {}).get("lines") or [])
    truth = [p for inst in lines for p in (inst.get("truth_points") or [])]
    out["n_line_points"] = len(truth)
    frames = load_scene_frames(batch, name)
    if not frames:
        out["reasons"].append("no dumped frames for this scene")
        return out
    if not truth:
        # 无线场景按**合格负例**评估（方案 §4.2：无线负例来自"确定未生成漆线
        # 且无内嵌漆线纹理"的场景）。证据 = 生成器声明 line_generated=False
        # + 帧内 annotation 线类像素为 0 + 外观也找不到线状像素。
        gen = rec.get("generated") or {}
        if gen.get("line_generated") is not False:
            out["reasons"].append("no line truth points and the generator did "
                                  "not declare this scene line-free")
            return out
        ann_px = 0
        app_like = 0
        for fr in frames:
            ann_px += int(np.sum(np.asarray(fr["label"]) == 2))
            _mask, _mode, _why = line_evidence_mask(
                {"rgb": fr["rgb"], "label": fr["label"]}, mode="appearance")
            app_like += int(_mask.sum()) if _mask is not None else 0
        out.update(line_free=True, line_px_in_frames=ann_px,
                   appearance_like_px=app_like)
        # **外观判据不作负例门**（2026-09-30 实测）：`line_evidence_mask(
        # mode="appearance")` 在**已验证的无线负例包**上就有 13k–68k px/8 帧
        # （亮铺装、亮碎石都命中"白/暖白"），拿它当门会把整个负例通道杀光。
        # 负例的权威证据是**引擎 annotation 的线类像素为 0**（引擎自己说这里
        # 没有漆线）；外观数只上报、不判定。旧代码在这里调 `appearance_line_check`
        # （需要真值点，无线场景恒返回 0）——那是个**恒不生效**的门，已去掉，
        # 免得读代码的人以为外观被查过。
        structs = gen.get("structures") or []
        if structs:
            # **结构负例**（T16 §16.1）：场景故意放了像线的非漆结构（石墙/护栏），
            # 用来教"亮的细长结构 ≠ 线"。判据仍是 annotation 线类像素为 0；
            # 额外上报"线状外观落在结构投影足迹内/外"的分解（证据，不判定）。
            split = structure_appearance_split(frames, structs)
            out.update(structure_appearance=split,
                       structures=[{"id": s.get("id"), "kind": s.get("kind"),
                                    "shape": s.get("shape")} for s in structs])
        if ann_px != 0:
            out["reasons"].append(
                f"declared line-free but {ann_px} line-class pixels present: "
                "cannot be a confirmed negative")
        out["eligible"] = not out["reasons"]
        out["frames"] = frames
        return out
    # **横向档位门**：超出认证带的实例不进真值（多档批次的 2.4 m 档落在
    # annotation 标注带外，实测残差中位 2.125 px，是崩溃那轮的坏标签来源）。
    kept, excluded = [], []
    for inst in lines:
        lat = inst.get("lateral_m")
        if (max_line_lat_m is not None and lat is not None
                and abs(float(lat)) > float(max_line_lat_m)):
            excluded.append({"role": inst.get("role"),
                             "lateral_m": float(lat)})
        else:
            kept.append(inst)
    out["excluded_instances"] = excluded
    if not kept:
        out["reasons"].append(
            f"every line instance is outside the certified lateral band "
            f"|lat| <= {max_line_lat_m} m (excluded: {excluded})")
        out["frames"] = frames
        return out
    # **逐线实例**覆盖：按实例算覆盖（多线场景里某条线没被 annotation 覆盖，
    # 不该靠另一条把它带过门）；**每个**实例都必须达标，否则标签是断续的。
    inst_cov: dict = {}
    for inst in kept:
        role = str(inst.get("role") or "?")
        pts = inst.get("truth_points") or []
        covs = []
        for fr in frames:
            c = line_evidence_coverage(
                {"camera": fr["camera"], "label": fr["label"],
                 "rgb": fr["rgb"], "truth_points": pts},
                radius_px=6, evidence="annotation")
            if c.get("coverage") is not None:
                covs.append(float(c["coverage"]))
        inst_cov[role] = round(float(np.mean(covs)), 4) if covs else None
    out["instance_coverage"] = inst_cov
    ok_roles = [r for r, c in inst_cov.items()
                if c is not None and c >= float(min_coverage)]
    failed = [r for r, c in inst_cov.items()
              if c is None or c < float(min_coverage)]
    out["certified_roles"] = sorted(ok_roles)
    out["failed_roles"] = sorted(failed)
    out["coverage"] = (round(max((c for c in inst_cov.values()
                                  if c is not None), default=0.0), 4)
                       if inst_cov else None)
    if failed:
        out["reasons"].append(
            f"line instance(s) {sorted(failed)} below annotation coverage "
            f"{min_coverage} (per-instance: {inst_cov}): the engine label does "
            "not carry these lines here, so the site cannot be a line-truth "
            "source")
    aft = [p.get("after_px") for p in
           ((rec.get("camera_alignment") or {}).get("per_frame") or [])]
    aft = [float(x) for x in aft if x is not None]
    out["after_px"] = round(float(np.mean(aft)), 3) if aft else None
    out["after_px_median"] = round(float(np.median(aft)), 3) if aft else None
    if not aft:
        out["reasons"].append("no projection calibration record (after_px)")
    elif float(np.mean(aft)) > float(max_after_px):
        out["reasons"].append(
            f"calibrated projection residual {np.mean(aft):.2f} px > "
            f"{max_after_px}: geometry not verified")
    elif (max_after_median_px is not None
          and float(np.median(aft)) > float(max_after_median_px)):
        out["reasons"].append(
            f"calibrated projection residual median {np.median(aft):.2f} px > "
            f"{max_after_median_px}: geometry not verified")
    # **漆面外观门**（最后一道，也最贵）：认证实例的每个真值点投影处必须是
    # 该实例颜色的漆。挡掉的是"标签在 RGB 是车/泥土的地方写线"的中毒包。
    if min_paint_agreement is not None:
        agree = paint_agreement(kept, frames, calibrated_frames(frames, rec))
        out["paint_agreement"] = agree
        checked = sum(int(v["checked"]) for v in agree.values())
        like = sum(int(v["like"]) for v in agree.values())
        out["paint_agreement_pooled"] = (round(like / checked, 4)
                                         if checked else None)
        if not checked:
            out["reasons"].append(
                "no line truth point is projectable into the frames: paint "
                "appearance cannot be verified (UNKNOWN is not a pass)")
        elif float(like / checked) < float(min_paint_agreement):
            out["reasons"].append(
                f"paint appearance agreement {like / checked:.3f} < "
                f"{min_paint_agreement} over {checked} projected truth points "
                f"(per-instance: {agree}): the label marks line where the RGB "
                "does not show paint (annotation pass ignores occluders)")
    out["eligible"] = not out["reasons"]
    out["frames"] = frames
    return out


def certify_points(truth: list, frames: list, cal_frames: list, *,
                   radius_px: int = 2) -> dict:
    """逐点认证：保留"校准后投影 2 px 内有 annotation 线像素"的点。

    方案 §4.3 的口径是"不确定的像素 ignore，有效面积与拒绝率分开统计"——
    真值点同理：放弃认证不了的点（画面外/线不在该处），把**丢弃比例**写进凭证，
    而不是拿它们去换一个 ok=True。标签（annotation）一个像素都不改。
    """
    from beamng_autopilot.experiments.auto_truth import _dilate, project_point
    kept_by_frame: list = []
    drops = {"out_of_frame": 0, "no_line_within_radius": 0}
    for fr, cam in zip(frames, cal_frames):
        lab = np.asarray(fr["label"])
        line = (lab == 2)
        if line.any():
            near = _dilate(line, int(radius_px))
        else:
            near = None
        h, w = lab.shape
        kept = []
        for p in truth:
            u, v = project_point(p.get("world"), cam)
            if not (np.isfinite(u) and np.isfinite(v)):
                drops["out_of_frame"] += 1
                continue
            x, y = int(round(u)), int(round(v))
            if not (0 <= x < w and 0 <= y < h):
                drops["out_of_frame"] += 1
                continue
            if near is not None and bool(near[y, x]):
                kept.append(p)
            else:
                drops["no_line_within_radius"] += 1
        kept_by_frame.append(kept)
    return {"kept_by_frame": kept_by_frame, "drops": drops,
            "n_total": len(truth) * max(1, len(frames))}


def export_scene(rec: dict, truth: list, frames: list, *, out_root: Path,
                 scene: str, anchor: int, map_name: str,
                 generator: dict, camera_name: str = "front_main") -> dict:
    """写一份训练可用、带凭证的数据目录；返回导出摘要。"""
    d = out_root / f"m5auto_a{anchor}_{scene}"
    view = d / camera_name
    view.mkdir(parents=True, exist_ok=True)
    frame_records = []
    label_shas = []
    src_shas = []
    for i, fr in enumerate(frames):
        np.savez_compressed(view / f"frame_{i:05d}.npz", colour=fr["rgb"],
                            label=fr["label"], annotation_raw=fr["annotation"])
        src_sha, lab_sha = frame_content_shas(fr["rgb"], fr["label"])
        label_shas.append(lab_sha)
        src_shas.append(src_sha)
        frame_records.append({"i": i, "view": camera_name, "exposure": i,
                              "t_wall": 1000.0 + i, "path":
                              f"{camera_name}/frame_{i:05d}.npz"})
    (d / "meta.json").write_text(json.dumps({
        "map_name": map_name,
        # 场景族 = 锚点（同一路段的所有站点同族）：划分 train/dev/final 时按它分，
        # 不按站点分（方案 §4.4）
        "source_id": f"m5auto_a{anchor}",
        "map_name_source": "m5_controlled_scenes.py (generated scenario)",
        "generated_by": f"{EXPORTER_VERSION}",
        "frames": frame_records}, indent=1, ensure_ascii=False), encoding="utf-8")

    # 出口复核前**套上逐帧静态投影标定**（标定是场景的渲染/相机常数，已在
    # 生成批次里落盘）：不套的话投影残差会把复核判成不通过，写不出
    # engine_verified——实测踩到（9 个站点全部 verified=False）。
    cal_frames = calibrated_frames(frames, rec)
    # 出口复核：用**annotation 证据**重跑 verify_batch（标签就是 annotation），
    # ok=True 才可能写 engine_verified（write_truth_credentials 里再做一次与）
    batch = {"frames": [{"frame_id": f"{scene}_{i:05d}",
                         "timestamp": 1000.0 + i,
                         "channel_ids": {"rgb": f"{scene}_{i:05d}",
                                         "annotation": f"{scene}_{i:05d}",
                                         "depth": f"{scene}_{i:05d}"},
                         "camera": cam, "rgb": fr["rgb"],
                         "annotation": fr["annotation"],
                         "depth": fr["depth"],
                         "label": fr["label"],
                         "label_sha": sh,
                         "source_image_sha": sa,
                         "truth_points": truth}
                        for i, (fr, sh, sa, cam) in enumerate(
                            zip(frames, label_shas, src_shas, cal_frames))],
             "scene": {"map": map_name, "segment": f"m5auto_a{anchor}_{scene}",
                       "run_id": f"m5_controlled_{scene}", "scene_seed": 0,
                       "game_version": "beamng_tech", "renderer": "beamng_tech",
                       "line_generated": True, "line_texture_embedded": False,
                       "roles_defined": True, "road_defined": True,
                       "label_source": "engine_annotation"},
             "generator": {"name": "m5_controlled_scenes.py",
                           "version": str(generator.get("version") or ""),
                           "sha": str(generator.get("script_sha16") or "")},
             "asset": {"map": map_name},
             "palette": palette_from_frames(frames),
             "run": {"id": f"m5_controlled_{scene}", "scene_seed": 0}}
    # 逐点认证后再做出口复核：被丢弃的点不进真相集，丢弃量进凭证
    cert = certify_points(truth, frames, cal_frames)
    kept_union = {id(p) for kf in cert["kept_by_frame"] for p in kf}
    truth_certified = [p for p in truth if id(p) in kept_union]
    for i, fr in enumerate(batch["frames"]):
        fr["truth_points"] = [dict(p) for p in cert["kept_by_frame"][i]]
    n_kept = sum(len(kf) for kf in cert["kept_by_frame"])
    # 遮挡余量按该深度通道的实测精度给：uint8 量化 + 标定残差 ~0.6 m，默认
    # 0.5 m 低于噪声底 -> 成片假 OCCLUSION_INSERT。3× 默认并记录进凭证。
    occl_margin = float(getattr(args_default_margin(), "value", 1.5))
    # 遮挡通道：**上报不判码**（occlusion_mode="report"）。原因写进凭证：
    # uint8 量化 + 标定残差 ~0.6 m 的深度在 17 m 处分不出 1.5 m 的差，逐点判码
    # 会把整批生成场景判成 OCCLUSION_INSERT（实测 10/11 站）；而遮挡能力本身在
    # 合成反例套件里已验证（inject("occlude") 必被拒）。按"逐通道赋资格"，
    # 这一通道对本批次记不可判，不替线通道背书也不拦它。
    rep = verify_batch(batch, line_evidence="annotation",
                       occlusion_margin_m=occl_margin,
                       occlusion_mode="report")
    provenance = {
        "generator": {"name": "m5_controlled_scenes.py",
                      "version": str(generator.get("version") or ""),
                      "sha": str(generator.get("script_sha16") or "")},
        "asset": {"map": map_name, "segment": f"m5auto_a{anchor}_{scene}"},
        "run": {"id": f"m5_controlled_{scene}", "scene_seed": 0,
                "game_version": "beamng_tech", "renderer": "beamng_tech"},
        "camera": {"name": camera_name,
                   "calibration_sha": str(frames[0]["camera"].get("fov_y_deg")),
                   "frame_ids": [f"m5_controlled_{scene}:{i}"
                                 for i in range(len(frames))]},
        "labels": {"source_image_sha": _sha16_bytes(
                       "|".join(sorted(src_shas)).encode()),
                   "label_sha": _labels_digest(label_shas),
                   "channel_valid_area": "full frame (generated scene: no "
                                         "unknown pixels)",
                   "unknown_reason": "",
                   "occlusion_margin_m": occl_margin,
                   "occlusion_mode": "report",
                   "certified_points": n_kept,
                   "declared_points": cert["n_total"],
                   "dropped_points": cert["drops"]},
        "report": {"test_report_sha": _sha16_bytes(json.dumps(
                       rep.get("rejections"), sort_keys=True).encode()),
                   "verifier_version": VERIFIER_VERSION,
                   "verified": bool(rep.get("ok"))},
    }
    cred = write_truth_credentials(d, batch_report=rep, provenance=provenance)
    return {"dir": str(view), "n_frames": len(frames),
            "certified_points": n_kept, "declared_points": cert["n_total"],
            "dropped_points": cert["drops"],
            "label_source": cred.get("label_source"),
            "verified": bool((cred.get("truth_provenance") or {})
                             .get("report", {}).get("verified")),
            "rejections": sorted({r.get("code") for r in
                                  (rep.get("rejections") or [])}),
            "truth_contract": TRUTH_CONTRACT}


def export_line_free_package(frames: list, *, out_root: Path, scene: str,
                             anchor: int, map_name: str, generator: dict,
                             evidence: dict,
                             camera_name: str = "front_main") -> dict:
    """写一个**已确认无线**的负例包（生成器声明 + 帧内零线像素 + 外观无）。

    负例包同样带凭证（`engine_verified`），凭证里写明 `line_generated=False`
    与判定依据——训练侧就能合法把这些帧当"确实没有线"的负例（§4.2/§6.2），
    而不是"没标注"。
    """
    d = out_root / f"m5auto_a{anchor}_{scene}"
    view = d / camera_name
    view.mkdir(parents=True, exist_ok=True)
    frame_records = []
    label_shas = []
    src_shas = []
    for i, fr in enumerate(frames):
        np.savez_compressed(view / f"frame_{i:05d}.npz", colour=fr["rgb"],
                            label=fr["label"], annotation_raw=fr["annotation"])
        src_sha, lab_sha = frame_content_shas(fr["rgb"], fr["label"])
        label_shas.append(lab_sha)
        src_shas.append(src_sha)
        frame_records.append({"i": i, "view": camera_name, "exposure": i,
                              "t_wall": 1000.0 + i,
                              "path": f"{camera_name}/frame_{i:05d}.npz"})
    (d / "meta.json").write_text(json.dumps({
        "map_name": map_name, "source_id": f"m5auto_a{anchor}",
        "map_name_source": "m5_controlled_scenes.py (generated scenario)",
        "generated_by": EXPORTER_VERSION, "truth_kind": "line_free",
        "frames": frame_records}, indent=1, ensure_ascii=False), encoding="utf-8")
    batch = {"frames": [{"frame_id": f"{scene}_{i:05d}", "timestamp": 1000.0 + i,
                         "channel_ids": {"rgb": f"{scene}_{i:05d}",
                                         "annotation": f"{scene}_{i:05d}",
                                         "depth": f"{scene}_{i:05d}"},
                         "camera": fr["camera"], "rgb": fr["rgb"],
                         "annotation": fr["annotation"], "depth": fr["depth"],
                         "label": fr["label"], "label_sha": sh,
                         "source_image_sha": sa, "truth_points": []}
                        for i, (fr, sh, sa) in enumerate(
                            zip(frames, label_shas, src_shas))],
             "scene": {"map": map_name, "segment": f"m5auto_a{anchor}_{scene}",
                       "run_id": f"m5_controlled_{scene}", "scene_seed": 0,
                       "game_version": "beamng_tech", "renderer": "beamng_tech",
                       "line_generated": False, "line_texture_embedded": False,
                       "roles_defined": False, "road_defined": True,
                       "label_source": "engine_annotation"},
             "palette": palette_from_frames(frames),
             "generator": {"name": "m5_controlled_scenes.py",
                           "version": str(generator.get("version") or ""),
                           "sha": str(generator.get("script_sha16") or "")},
             "asset": {"map": map_name},
             "run": {"id": f"m5_controlled_{scene}", "scene_seed": 0}}
    rep = verify_batch(batch, line_evidence="annotation",
                       occlusion_mode="report")
    provenance = {
        "generator": {"name": "m5_controlled_scenes.py",
                      "version": str(generator.get("version") or ""),
                      "sha": str(generator.get("script_sha16") or "")},
        "asset": {"map": map_name, "segment": f"m5auto_a{anchor}_{scene}"},
        "run": {"id": f"m5_controlled_{scene}", "scene_seed": 0,
                "game_version": "beamng_tech", "renderer": "beamng_tech"},
        "camera": {"name": camera_name,
                   "frame_ids": [f"m5_controlled_{scene}:{i}"
                                 for i in range(len(frames))]},
        "labels": {"source_image_sha": _sha16_bytes(
                       "|".join(sorted(src_shas)).encode()),
                   "label_sha": _labels_digest(label_shas),
                   "channel_valid_area": "full frame",
                   "unknown_reason": "",
                   "truth_kind": "line_free",
                   "line_free_evidence": evidence},
        "report": {"test_report_sha": _sha16_bytes(json.dumps(
                       rep.get("rejections"), sort_keys=True).encode()),
                   "verifier_version": VERIFIER_VERSION,
                   "verified": bool(rep.get("ok"))}}
    cred = write_truth_credentials(d, batch_report=rep, provenance=provenance)
    return {"dir": str(view), "n_frames": len(frames),
            "label_source": cred.get("label_source"),
            "verified": bool((cred.get("truth_provenance") or {})
                             .get("report", {}).get("verified")),
            "rejections": sorted({r.get("code") for r in
                                  (rep.get("rejections") or [])})}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True,
                    help="m5_controlled_scenes.py 的一次运行目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-coverage", type=float, default=0.8,
                    help="逐线实例的 annotation 线类覆盖门（每个实例都要过）")
    ap.add_argument("--max-after-px", type=float, default=2.0)
    ap.add_argument("--max-after-median-px", type=float, default=1.0,
                    help="逐帧投影残差**中位**门；<0 关闭")
    ap.add_argument("--max-line-lat-m", type=float, default=2.0,
                    help="只认证 |lateral_m| <= 该值的线实例（多档批次收 1.2/1.8 档）；"
                         "<0 关闭")
    ap.add_argument("--min-paint-agreement", type=float, default=0.9,
                    help="真值点投影处 RGB 是该实例颜色的漆的占比门；<0 关闭")
    ap.add_argument("--map", default="italy")
    args = ap.parse_args()

    batch = Path(args.batch)
    report = json.loads((batch / "scene_report.json").read_text(encoding="utf-8"))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    scenes = report.get("scenes") or {}
    types = report.get("scene_types") or []
    names = list(scenes)
    gates = {
        "min_coverage": float(args.min_coverage),
        "max_after_px": float(args.max_after_px),
        "max_after_median_px": (None if float(args.max_after_median_px) < 0
                                else float(args.max_after_median_px)),
        "max_line_lat_m": (None if float(args.max_line_lat_m) < 0
                           else float(args.max_line_lat_m)),
        "min_paint_agreement": (None if float(args.min_paint_agreement) < 0
                                else float(args.min_paint_agreement)),
    }
    exported, isolated = [], []
    for k, name in enumerate(names):
        rec = scenes[name]
        # 锚点**在分支之前**算：负例包也要用它（旧代码写在负例分支后面，
        # 首个负例直接 UnboundLocalError，后续负例沿用上一站锚点）
        anchor = scene_anchor(name)
        ev = evaluate_scene(batch, name, rec, **gates)
        if not ev["eligible"]:
            isolated.append({kk: vv for kk, vv in ev.items()
                             if kk != "frames"})
            continue
        lines = ((rec.get("generated") or {}).get("lines") or [])
        _ok_roles = set(ev.get("certified_roles") or [])
        _excluded = {str(x.get("role") or "?")
                     for x in (ev.get("excluded_instances") or [])}
        truth = [p for inst in lines
                 if (str(inst.get("role") or "?") in _ok_roles
                     and str(inst.get("role") or "?") not in _excluded)
                 for p in (inst.get("truth_points") or [])]
        if ev.get("line_free") and not truth:
            _structs = ((rec.get("generated") or {}).get("structures") or [])
            try:
                res = export_line_free_package(
                    ev["frames"], out_root=out_root, scene=name, anchor=anchor,
                    map_name=str(args.map),
                    generator=report.get("generator") or {},
                    evidence={"line_px": ev.get("line_px_in_frames"),
                              "appearance_like_px": ev.get("appearance_like_px"),
                              # 结构负例：把"像线的外观"的来源写进凭证
                              "structures": [{"id": s.get("id"),
                                              "kind": s.get("kind"),
                                              "shape": s.get("shape")}
                                             for s in _structs],
                              "structure_appearance": ev.get(
                                  "structure_appearance")})
                exported.append({"scene": name,
                                 "type": (types[k] if k < len(types) else ""),
                                 "line_free": True,
                                 "structures": len(_structs),
                                 **res})
                print(f"[export] {name}: {res['n_frames']} 帧（负例）-> "
                      f"{res['label_source'] or '(无来源声明)'} "
                      f"verified={res['verified']}", flush=True)
            except Exception as exc:                          # noqa: BLE001
                isolated.append({"scene": name,
                                 "reasons": [f"export failed: "
                                             f"{type(exc).__name__}: {exc}"]})
            continue
        try:
            res = export_scene(rec, truth, ev["frames"], out_root=out_root,
                               scene=name, anchor=anchor,
                               map_name=str(args.map),
                               generator=report.get("generator") or {})
            exported.append({"scene": name,
                             "type": (types[k] if k < len(types) else ""),
                             "coverage": ev["coverage"],
                             "instance_coverage": ev.get("instance_coverage"),
                             "certified_roles": ev.get("certified_roles"),
                             "excluded_instances": ev.get("excluded_instances"),
                             "after_px": ev["after_px"],
                             "after_px_median": ev.get("after_px_median"),
                             "paint_agreement_pooled": ev.get(
                                 "paint_agreement_pooled"), **res})
            print(f"[export] {name}: {res['n_frames']} 帧 -> "
                  f"{res['label_source'] or '(无来源声明)'} "
                  f"verified={res['verified']}", flush=True)
        except Exception as exc:                              # noqa: BLE001
            isolated.append({"scene": name,
                             "reasons": [f"export failed: {type(exc).__name__}: {exc}"]})
    blob = {"exporter": EXPORTER_VERSION, "batch": str(batch),
            "map": str(args.map), **gates,
            "scenes_total": len(names), "exported": exported,
            "isolated": isolated,
            "frames_exported": int(sum(e["n_frames"] for e in exported)),
            "families": sorted({f"m5auto_a{e['dir'].split('m5auto_a')[-1].split('_')[0]}"
                                for e in exported})}
    (out_root / "export_report.json").write_text(
        json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[export] 导出 {len(exported)} / 隔离 {len(isolated)} 站 | "
          f"帧 {blob['frames_exported']} | 场景族 {len(blob['families'])} -> "
          f"{out_root / 'export_report.json'}")
    return 0 if exported else 1


if __name__ == "__main__":
    raise SystemExit(main())
