"""Decompose the line channel's false positives by ENGINE material (T08/T11).

The identity work measured the line mask's precision collapsing on an urban
junction (0.130 overall, 0.000 on the left half).  This tool answers WHERE
those false positives come from, using the engine's own palette as the
reference: per engine material, how much of it the model calls "line", and -
for the connected components - whether they are line-SHAPED or blobs.

The distinction decides the fix: blobs are already removed by the candidate
shape gate; line-shaped false positives pass it and need a different cue.

The palette is fetched live (``bng.get_annotations()``) only when the run
does not carry a saved ``palette_classes.json`` next to its meta - the same
session that made the capture is what makes the colours meaningful.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_fp_material_decomp.py \\
        --run logs/m5_seg/ident_material_20260923 --view front_main \\
        --model logs/m5_seg/seg_model_v13b/best.pt \\
        --json logs/goal_20260921/fp_material.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from beamng_autopilot.vision.hydra import FrameContext, HydraNet  # noqa: E402
from beamng_autopilot.vision.heads.semantic import SemanticHead  # noqa: E402
from beamng_autopilot.vision.segmentation import Segmenter  # noqa: E402
from m5_marking_identity_probe import camera_from_meta  # noqa: E402

CLS_LINE = 2
MIN_COMPONENT_PX = 25
LINE_SHAPE_ASPECT = 2.5


def load_palette(run_dir: Path, *, fetch_live: bool = True) -> dict:
    """Class name -> RGB, from the run's sidecar or the live engine."""
    side = Path(run_dir) / "palette_classes.json"
    if side.is_file():
        return json.loads(side.read_text(encoding="utf-8"))
    if not fetch_live:
        return {}
    from beamng_autopilot import config
    from beamng_autopilot.connector import BeamNGConnector
    from beamng_autopilot_tech.annotations import annotation_palette
    conn = BeamNGConnector("italy", "etk800", port=config.runtime_port("tech"),
                           home=config.runtime_home("tech"))
    try:
        conn.open(launch=False)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        get_ann = getattr(conn.bng, "get_annotations", None)
        pal = dict(annotation_palette(
            get_ann() if callable(get_ann) else None)["classes"])
    finally:
        conn.close()
    side.write_text(json.dumps(pal, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return pal


def material_lut(palette: dict) -> dict:
    """``{(r, g, b): name}`` - the reverse lookup the pixel test needs."""
    out = {}
    for name, rgb in (palette or {}).items():
        try:
            out[(int(rgb[0]), int(rgb[1]), int(rgb[2]))] = str(name)
        except (TypeError, ValueError, IndexError):
            continue
    return out


def decompose(run_dir: Path, palette: dict, *, view: str = "front_main",
              model_path: str | None = None, limit: int | None = None) -> dict:
    """Per-material false positives and their component shapes."""
    meta_path = Path(run_dir) / "meta.json"
    meta = (json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file() else {})
    cam = camera_from_meta(meta, view)
    frames = sorted((Path(run_dir) / view).glob("frame_*.npz"))
    if limit:
        frames = frames[:int(limit)]
    if not frames:
        return {"reason": f"no frames under {Path(run_dir) / view}"}
    first = np.load(frames[0])
    if "annotation_raw" not in first.files:
        return {"reason": "frames carry no annotation_raw: re-collect with "
                          "--save-annotation to get material-level evidence"}
    lut = material_lut(palette)
    if not lut:
        return {"reason": "no palette available (sidecar missing and the "
                          "live fetch returned nothing)"}
    net = HydraNet()
    net.add(SemanticHead(segmenter=Segmenter(model_path=model_path))
            if model_path else SemanticHead())
    pixels: dict = {}
    comps: dict = {}
    n_frames = 0
    for f in frames:
        z = np.load(f)
        colour = np.asarray(z["colour"])
        ann = np.asarray(z["annotation_raw"])[:, :, :3].astype(int)
        out = net.run(FrameContext(frame_rgb=colour, cam=cam, pos=np.zeros(3),
                                   heading=0.0, ground_z=0.0,
                                   role=view)).get("semantic")
        if out is None or "line" not in out.masks:
            continue
        n_frames += 1
        line = np.asarray(out.masks["line"], dtype=bool)
        fp = line & ~(np.asarray(z["label"]) == CLS_LINE)
        h, w = line.shape
        for side, sl in (("left", slice(0, w // 2)),
                         ("right", slice(w // 2, None))):
            am = ann[:, sl]
            sel = fp[:, sl]
            n_lab, lab, boxes, _ = cv2.connectedComponentsWithStats(
                sel.astype(np.uint8), connectivity=8)
            for i in range(1, n_lab):
                area = int(boxes[i, cv2.CC_STAT_AREA])
                if area < MIN_COMPONENT_PX:
                    continue
                bh = int(boxes[i, cv2.CC_STAT_HEIGHT])
                bw = int(boxes[i, cv2.CC_STAT_WIDTH])
                aspect = max(bh, bw) / max(1, min(bh, bw))
                # the material AT the component: the modal class over its
                # pixels, not the centre pixel (a centre can land on an
                # edge and mislabel the whole component)
                ys, xs = np.nonzero(lab == i)
                cols = am[ys, xs]
                keys = [tuple(int(v) for v in row) for row in cols[::7]]
                counts: dict = {}
                for k in keys:
                    name = lut.get(k, "?")
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get) if counts else "?"
                rec = comps.setdefault((side, name), {"n": 0, "area": [],
                                                      "line_shape": 0,
                                                      "aspect": []})
                rec["n"] += 1
                rec["area"].append(area)
                rec["aspect"].append(aspect)
                if aspect >= LINE_SHAPE_ASPECT:
                    rec["line_shape"] += 1
            for key_arr in np.unique(am.reshape(-1, 3), axis=0):
                key = tuple(int(v) for v in key_arr)
                label_name = lut.get(key)
                if label_name is None:
                    continue
                m = ((am[:, :, 0] == key[0]) & (am[:, :, 1] == key[1])
                     & (am[:, :, 2] == key[2]))
                n_mat = int(m.sum())
                if n_mat < 200:
                    continue
                r = pixels.setdefault((side, label_name),
                                      {"fp_px": 0, "material_px": 0,
                                       "line_px": 0})
                r["fp_px"] += int((m & fp[:, sl]).sum())
                r["material_px"] += n_mat
                r["line_px"] += int((m & line[:, sl]).sum())
    out_rows = {}
    for side in ("left", "right"):
        tot_fp = sum(v["fp_px"] for (s, _), v in pixels.items()
                     if s == side) or 1
        out_rows[side] = [
            {"material": name, "fp_px": v["fp_px"],
             "share_of_fp": round(v["fp_px"] / tot_fp, 4),
             "material_px": v["material_px"],
             "fp_rate_of_material": round(v["line_px"]
                                          / max(1, v["material_px"]), 4)}
            for (s, name), v in sorted(pixels.items(),
                                       key=lambda kv: -kv[1]["fp_px"])
            if s == side and v["fp_px"] > 0]
    out_comps = {}
    for side in ("left", "right"):
        entries = [(name, v) for (s, name), v in comps.items() if s == side]
        tot = sum(v["n"] for _, v in entries) or 1
        out_comps[side] = [
            {"material": name, "components": v["n"],
             "share": round(v["n"] / tot, 4),
             "area_p50_px": int(np.percentile(v["area"], 50)),
             "aspect_p50": round(float(np.percentile(v["aspect"], 50)), 2),
             "line_shaped_frac": round(v["line_shape"] / v["n"], 4)}
            for name, v in sorted(entries, key=lambda kv: -kv[1]["n"])]
    return {"run": str(run_dir), "view": view,
            "model": str(model_path) if model_path else "default_model_path",
            "frames": n_frames, "palette_classes": len(lut),
            "pixels": out_rows, "components": out_comps,
            "limits": [
                "Engine labels are the reference: worn/faded paint and "
                "materials the palette does not render are counted against "
                "the model.",
                "DRIVING_INSTRUCTIONS is road paint but NOT in the engine's "
                "line class, so the model calling it 'line' registers as a "
                "false positive; that is a reference-definition gap.",
                "One scene, one checkpoint, one time of day.",
            ]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--view", default="front_main")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-live-palette", action="store_true",
                    help="refuse to open the simulator for the palette")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    pal = load_palette(run_dir, fetch_live=not args.no_live_palette)
    res = decompose(run_dir, pal, view=args.view, model_path=args.model,
                    limit=args.limit)
    if "reason" in res:
        print(f"[fp] {res['reason']}")
        return 2
    print(f"[fp] {res['run']} view={res['view']} frames={res['frames']} "
          f"classes={res['palette_classes']}")
    for side in ("left", "right"):
        print(f"[fp] --- {side}")
        for r in res["pixels"][side][:5]:
            print("      %-16s FP %7d (%5.1f%% of FP) | material %8d | "
                  "called-line %5.1f%%" % (
                      r["material"], r["fp_px"], 100 * r["share_of_fp"],
                      r["material_px"], 100 * r["fp_rate_of_material"]))
        for r in res["components"][side][:4]:
            print("      comps %-14s n=%3d area p50 %5d aspect p50 %4.2f "
                  "line-shaped %4.1f%%" % (
                      r["material"], r["components"], r["area_p50_px"],
                      r["aspect_p50"], 100 * r["line_shaped_frac"]))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"[fp] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
