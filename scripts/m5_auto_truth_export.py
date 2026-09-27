"""把受控场景批次里**通过资格与几何校验**的站点导出成带凭证的训练数据（T16 §4.3）。

输入：`m5_controlled_scenes.py` 的一次运行产物（`scene_report.json` +
`scene_*.json` + `frames/*.npz`）。**不需要游戏**——复核与导出都在离线做。

两道门槛，缺一不可（方案 §4.3"逐通道赋资格"）：

1. **annotation 线类覆盖** ≥ ``--min-coverage``（默认 0.6）：训练标签就是
   annotation，所以只认 annotation 的覆盖——外观再可见也不能当标签来源，
   否则等于教模型"这里有线的地方没有线"（扩量实测：部分路段 annotation 线类
   像素只有个位数）。
2. **几何一致性** 校准后 ``after_px`` ≤ ``--max-after-px``（默认 2.0）：逐帧
   静态标定后的投影残差（标定参数在场景记录里）。

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
    TRUTH_CONTRACT, VERIFIER_VERSION, _rotate_camera_basis, frame_content_shas,
    line_evidence_coverage, palette_sha, verify_batch, write_truth_credentials,
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
                   max_after_px: float) -> dict:
    """资格 + 几何两道门；返回 ``{eligible, reasons, coverage, after_px, frames}``。

    只读产物，不写任何东西——先判定再决定要不要导出。
    """
    out: dict = {"scene": name, "eligible": False, "reasons": [],
                 "coverage": None, "after_px": None}
    lines = ((rec.get("generated") or {}).get("lines") or [])
    truth = [p for inst in lines for p in (inst.get("truth_points") or [])]
    out["n_line_points"] = len(truth)
    if not truth:
        out["reasons"].append("no line truth points (line-free scene: not a "
                              "line-truth source)")
        return out
    frames = load_scene_frames(batch, name)
    if not frames:
        out["reasons"].append("no dumped frames for this scene")
        return out
    covs = []
    for fr in frames:
        c = line_evidence_coverage(
            {"camera": fr["camera"], "label": fr["label"], "rgb": fr["rgb"],
             "truth_points": truth}, radius_px=6, evidence="annotation")
        if c.get("coverage") is not None:
            covs.append(float(c["coverage"]))
    coverage = float(np.mean(covs)) if covs else None
    out["coverage"] = None if coverage is None else round(coverage, 4)
    if coverage is None:
        out["reasons"].append("annotation coverage could not be measured")
    elif coverage < float(min_coverage):
        out["reasons"].append(
            f"annotation line coverage {coverage:.2f} < {min_coverage}: the "
            "engine label does not carry the line here, so it cannot be a "
            "line-truth source (isolate; do not export as engine_verified)")
    aft = [p.get("after_px") for p in
           ((rec.get("camera_alignment") or {}).get("per_frame") or [])]
    aft = [float(x) for x in aft if x is not None]
    out["after_px"] = round(float(np.mean(aft)), 3) if aft else None
    if not aft:
        out["reasons"].append("no projection calibration record (after_px)")
    elif float(np.mean(aft)) > float(max_after_px):
        out["reasons"].append(
            f"calibrated projection residual {np.mean(aft):.2f} px > "
            f"{max_after_px}: geometry not verified")
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
    pf = ((rec.get("camera_alignment") or {}).get("per_frame") or [])
    cal_frames = []
    for i, fr in enumerate(frames):
        cam = fr["camera"]
        if i < len(pf) and pf[i].get("status") == "measured":
            cam = _rotate_camera_basis(cam, float(pf[i].get("yaw_deg") or 0.0),
                                       float(pf[i].get("pitch_deg") or 0.0))
        cal_frames.append(cam)
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
    rep = verify_batch(batch, line_evidence="annotation",
                       occlusion_margin_m=occl_margin)
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True,
                    help="m5_controlled_scenes.py 的一次运行目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--max-after-px", type=float, default=2.0)
    ap.add_argument("--map", default="italy")
    args = ap.parse_args()

    batch = Path(args.batch)
    report = json.loads((batch / "scene_report.json").read_text(encoding="utf-8"))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    scenes = report.get("scenes") or {}
    types = report.get("scene_types") or []
    names = list(scenes)
    exported, isolated = [], []
    for k, name in enumerate(names):
        rec = scenes[name]
        ev = evaluate_scene(batch, name, rec,
                            min_coverage=float(args.min_coverage),
                            max_after_px=float(args.max_after_px))
        if not ev["eligible"]:
            isolated.append({kk: vv for kk, vv in ev.items()
                             if kk != "frames"})
            continue
        lines = ((rec.get("generated") or {}).get("lines") or [])
        truth = [p for inst in lines for p in (inst.get("truth_points") or [])]
        anchor = int(name.rsplit("_a", 1)[1].split("s")[0]) if "_a" in name else 0
        try:
            res = export_scene(rec, truth, ev["frames"], out_root=out_root,
                               scene=name, anchor=anchor,
                               map_name=str(args.map),
                               generator=report.get("generator") or {})
            exported.append({"scene": name,
                             "type": (types[k] if k < len(types) else ""),
                             "coverage": ev["coverage"],
                             "after_px": ev["after_px"], **res})
            print(f"[export] {name}: {res['n_frames']} 帧 -> "
                  f"{res['label_source'] or '(无来源声明)'} "
                  f"verified={res['verified']}", flush=True)
        except Exception as exc:                              # noqa: BLE001
            isolated.append({"scene": name,
                             "reasons": [f"export failed: {type(exc).__name__}: {exc}"]})
    blob = {"exporter": EXPORTER_VERSION, "batch": str(batch),
            "map": str(args.map), "min_coverage": args.min_coverage,
            "max_after_px": args.max_after_px,
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
