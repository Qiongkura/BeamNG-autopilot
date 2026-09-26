"""Manual frame annotator: paint lane-line labels for training.

Paint the LINE markings on captured frames; every save writes a
``colour``/``label`` npz in the exact ``m5_train_seg.py`` contract
(0 = background, 1 = road, 2 = line), so hand annotations fold directly
into the segmentation training set.  Road class stays available for a
later pass; the typical line-only session just uses class 1.

Every export also carries the **identity** of the source frame
(``map_name``/``source_id``/``pos``/``heading``) plus a ``meta.json``
sidecar, because a label with unrecorded provenance cannot be grouped,
placed or split - measured: an audit rejected all 48 human-revised frames
with "no map identity in the recording" while the annotator had the source
file in hand.  Identity is *carried* from the recording, never invented: a
missing field stays empty and is listed in ``identity_missing`` so the
audit rejects the frame instead of silently trusting it.

Tools, all of them clickable in the toolbar at the top of the window:
    pen         left-drag to paint with the active brush
    bucket      click a region to flood-fill it with the active brush
                (draw a closed outline, then click inside it)
    straight    drag A->B, or click A then B: a straight line, every painted
                pixel within the brush radius of the ideal segment
    curve       drag = freehand, resampled + de-jittered + Catmull-Rom, so
                the stroke comes out smooth; or click control points and
                press ENTER - the curve passes through every clicked point
    undo        pops the last control point while a curve is being drawn,
                otherwise the last stroke / fill / clear
    clear / zoom / prev / save+next   the other one-click actions

Brushes (toolbar buttons too): line / road / erase plus the three
"cannot judge" brushes (occluded / blurred / undecidable) which write
255(IGNORE) and record the reason in the ``unknown_kind`` array.

Controls (keyboard kept from the old flow; the buttons show their keys):
    1 / 2 / 3   brush = line / road / background(erase)
    4 / 5 / 6   brush = occluded / blurred / undecidable
    p / f / b   pen / bucket / toggle between them
    l / v       straight line / smooth curve
    u           undo (draft-aware)      c clear      z zoom 2x / 1x
    a / Left    previous frame (back)
    s           save + next frame       q quit
    ENTER       finish the curve being drawn
    ESC         abandon the draft (draft never touches the label)
    trackbar    brush size 1-40

Interaction lives in ``beamng_autopilot.labeling.annotate_session`` and the
toolbar/geometry in ``beamng_autopilot.labeling.annotate_tools`` (both are
unit-tested without a GUI); this script keeps the CLI, frame loading, the
identity recording and the export contract.

    .venv\\Scripts\\python.exe scripts\\m5_annotate_manual.py \\
        --frames-dir logs/m5_seg/manual_capture
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from beamng_autopilot import config
from beamng_autopilot.labeling import curve_schema as cs
from beamng_autopilot.labeling.annotate_session import AnnotateSession
# 左右侧统计只有一份实现：状态行与导出时的覆盖核查共用它（两个实现迟早会
# 出现"HUD 说 R=0、导出说右侧漏标"这种自相矛盾）。
from beamng_autopilot.labeling.annotate_tools import side_line_counts

# The pixel classes live in the schema (T10), so the annotator and every
# metric agree on what 0/1/2/255 mean: 255 is IGNORE, never background.
CLS_LINE, CLS_ROAD, CLS_BG = cs.CLS_LINE, cs.CLS_ROAD, cs.CLS_BACKGROUND
CLASS_NAME = {CLS_LINE: "line", CLS_ROAD: "road", CLS_BG: "erase"}
WIN = "annotate"

#: 除 line/road/erase 之外的第四类标记：**"这里不能判断"**。它们写进 label 的
#: 255(ignore)，同时把原因记在独立的 ``unknown_kind`` 数组里（1/2/3）——方案第 1 项
#: 要求"同时标记遮挡、模糊和无法判断的区域"，而 label 只有 0/1/2/255 四个值，
#: 原因必须另存，否则审计只能看到"这里被忽略了"，看不到为什么。
UNKNOWN_KINDS = {1: "occluded", 2: "blurred", 3: "undecidable"}
UNKNOWN_KEY_TO_KIND = {"4": 1, "5": 2, "6": 3}
#: 画笔的显示名：label 三色 + 三个"不能判断"的原因
BRUSH_NAME = {**CLASS_NAME, **UNKNOWN_KINDS}


def unknown_kind_for_key(ch: str) -> int | None:
    """键盘 4/5/6 → 遮挡/模糊/无法判断（其它键返回 None）。"""
    return UNKNOWN_KEY_TO_KIND.get(str(ch))


def side_coverage_note(human_label, engine_label) -> dict:
    """对比"人工 label"与"引擎 label"的逐侧 line 覆盖，给出可核查的提示。

    方案第 1 项的验收要求"左右侧各有可核查的正反例"。单侧为 0 而引擎在那一侧
    有线，通常意味着漏标（也可能是引擎把路缘当线）——两种都要写出来让人看，
    不能静默通过。
    """
    h = side_line_counts(human_label)
    e = side_line_counts(engine_label)
    warn = []
    for side in ("left", "right"):
        if h[side] == 0 and e[side] > 0:
            warn.append(f"{side}: 人工未标但引擎有 {e[side]} px"
                        "（漏标？或引擎把路缘/墙根当线）")
        elif h[side] > 0 and e[side] == 0:
            warn.append(f"{side}: 人工标了 {h[side]} px 但引擎为 0"
                        "（人工新增/引擎漏标——按方案这属于独立真值，保留）")
    return {"human": {"left": h["left"], "right": h["right"]},
            "engine": {"left": e["left"], "right": e["right"]},
            "warnings": warn}


def unknown_counts(kind_array) -> dict:
    """unknown_kind 数组 → {occluded/blurred/undecidable: 像素数}。"""
    arr = np.asarray(kind_array) if kind_array is not None else None
    out = {name: 0 for name in UNKNOWN_KINDS.values()}
    if arr is None or arr.size == 0:
        return out
    for k, name in UNKNOWN_KINDS.items():
        out[name] = int((arr == int(k)).sum())
    return out


#: Fields an exported frame must carry so a later audit can group it
#: (map/source), place it (pos/heading) and keep splits isolated.
IDENTITY_FIELDS = ("map_name", "source_id", "pos", "heading")


# ---------------------------------------------------------------------------
# Identity recording
#
# Priority, highest first: the frame npz itself, the frame's entry in the
# recording's ``meta.json``, then the run-level meta.  A field no layer
# proves is reported in ``identity_missing``; nothing here defaults to a map
# name, because a defaulted map name is indistinguishable from a measured one
# downstream (measured defect: a collector wrote the literal ``italy`` while
# ``east_coast_usa`` was loaded).
# ---------------------------------------------------------------------------


def _read_dir_meta(frames_dir: Path) -> tuple[dict, str]:
    """``(meta, source)`` from ``<dir>/meta.json`` else ``<parent>/meta.json``.

    Same lookup order as ``experiments.manifest._read_meta``, so what an
    annotated directory carries is exactly what the audit will read.
    """
    for cand, tag in ((Path(frames_dir) / "meta.json", "self"),
                      (Path(frames_dir).parent / "meta.json", "parent")):
        if cand.exists():
            try:
                return (json.loads(cand.read_text(encoding="utf-8")),
                        f"{tag}:{cand}")
            except Exception as exc:                      # noqa: BLE001
                return {}, f"unreadable:{cand}:{exc}"
    return {}, "unavailable"


def _norm_identity(layer) -> dict:
    """Normalise one identity layer to str / [float] / float; drop blanks."""
    out: dict = {}
    for k, v in dict(layer or {}).items():
        if k not in IDENTITY_FIELDS or v is None:
            continue
        if k == "pos":
            try:
                arr = np.asarray(v, dtype=float).reshape(-1)
            except Exception:                             # noqa: BLE001
                continue
            if arr.size:
                out[k] = [float(x) for x in arr[:3]]
        elif k == "heading":
            try:
                out[k] = float(v)
            except Exception:                             # noqa: BLE001
                continue
        else:
            s = str(v).strip()
            if s:
                out[k] = s
    return out


def _frame_in_meta(meta, name: str) -> dict:
    """The ``meta['frames']`` entry for one npz file (empty when absent)."""
    by: dict = {}
    for f in (meta or {}).get("frames") or []:
        p = str(f.get("path") or "")
        if p:
            by[p] = f
            by[Path(p).name] = f
    return dict(by.get(name) or {})


def load_engine_labels(frames_dir: Path) -> list:
    """逐帧读**引擎** label（与 load_frame_dir 同一排序，逐帧对齐）。

    人工标注时用来做左右侧覆盖核查：人工在某一侧没标、而引擎那一侧有线，
    就提示"漏标？或引擎把路缘当线"；人工标了而引擎没有，则按独立真值保留。
    """
    out: list = []
    for f in sorted(glob.glob(os.path.join(str(frames_dir), "*.npz"))):
        try:
            with np.load(f) as z:
                out.append(np.asarray(z["label"], dtype=np.uint8)
                           if "label" in z.files else None)
        except Exception:                                 # noqa: BLE001
            out.append(None)
    return out


def frame_identity(layers, *, source_path: str = "",
                   context: dict | None = None) -> dict:
    """Identity for one frame: first layer that carries a field wins.

    ``layers`` is ``[(tag, fields), ...]`` in priority order.  The result
    always has the four identity keys (``None`` when unproven),
    ``identity_source`` (what proved them), ``identity_provenance`` (per
    field) and ``identity_missing`` (what nothing proved).
    """
    vals: dict = {}
    prov: dict = {}
    for tag, layer in layers:
        for k, v in _norm_identity(layer).items():
            if k not in vals:
                vals[k] = v
                prov[k] = tag
    ident = {k: vals.get(k) for k in IDENTITY_FIELDS}
    ident.update({k: v for k, v in (context or {}).items()})
    ident["identity_provenance"] = prov
    ident["identity_missing"] = [k for k in IDENTITY_FIELDS if k not in vals]
    ident["identity_source"] = (",".join(sorted(set(prov.values())))
                                if prov else "unavailable")
    ident["source_path"] = str(source_path)
    return ident


def _npz_identity(z) -> tuple[dict, dict]:
    """Identity read from an open frame npz: ``(fields, context)``."""
    fields = {k: z[k] for k in IDENTITY_FIELDS if k in z.files}
    ctx: dict = {}
    if "identity_json" in z.files:
        try:
            blob = json.loads(str(z["identity_json"]))
        except Exception:                                 # noqa: BLE001
            blob = {}
        if isinstance(blob, dict):
            for k in IDENTITY_FIELDS:
                fields.setdefault(k, blob.get(k))
            for k in ("view", "exposure", "map_name_source"):
                if blob.get(k) is not None:
                    ctx.setdefault(k, blob[k])
    for k in ("view", "exposure"):
        if k in z.files:
            v = z[k]
            ctx[k] = (str(v) if k == "view" else int(v))
    return fields, ctx


def load_frame_dir(frames_dir: Path, *, out_dir=None):
    """Read one frame directory -> ``(frames, idents, resume_labels, paths)``.

    ``frames`` is ``[(rgb, source_index), ...]`` as the annotator uses it;
    ``idents`` is index-aligned identity records; ``resume_labels`` and
    ``paths`` keep the in-place resume behaviour (a revisited frame must
    overwrite its own output, not create a duplicate sample).
    """
    frames_dir = Path(frames_dir)
    meta, meta_source = _read_dir_meta(frames_dir)
    frames: list = []
    idents: list = []
    engine_labels: list = []
    resume_labels: dict[int, np.ndarray] = {}
    resume_paths: dict[int, tuple[Path, Path]] = {}
    in_place = out_dir is not None and \
        Path(out_dir).resolve() == frames_dir.resolve()
    for f in sorted(glob.glob(os.path.join(str(frames_dir), "*.npz"))):
        name = Path(f).name
        with np.load(f) as z:
            rgb = np.asarray(z["colour"], dtype=np.uint8)
            label = (np.asarray(z["label"], dtype=np.uint8).copy()
                     if "label" in z.files else None)
            npz_fields, npz_ctx = _npz_identity(z)
        try:
            idx = int(name.split("_")[-1].split(".")[0])
        except ValueError:
            idx = len(frames)
        rec = _frame_in_meta(meta, name)
        layers = [
            (f"npz:{Path(f).name}", npz_fields),
            ("meta:frame", rec),
            ("meta:run", {"map_name": (meta or {}).get("map_name"),
                          "source_id": (meta or {}).get("source_id")}),
        ]
        ident = frame_identity(layers, source_path=str(f), context=dict(
            npz_ctx, view=rec.get("view") or npz_ctx.get("view"),
            exposure=rec.get("exposure", npz_ctx.get("exposure")),
            meta_source=meta_source,
            map_name_source=(meta or {}).get("map_name_source") or ""))
        frames.append((rgb, idx))
        idents.append(ident)
        if label is not None:
            resume_labels[idx] = label
            if in_place:
                resume_paths[idx] = (
                    Path(f), Path(f).with_name(
                        name.replace("frame_", "preview_", 1)
                            .replace(".npz", ".png")))
    return frames, idents, resume_labels, resume_paths


def identity_npz_extras(ident: dict) -> dict:
    """npz extras for one frame - plain str/float arrays only (no objects).

    Object dtype would break strict checkpoint/payload loaders, so every
    value is a plain numpy array of str/float.
    """
    extras = {
        "map_name": np.array(str(ident.get("map_name") or "")),
        "source_id": np.array(str(ident.get("source_id") or "")),
        "identity_json": np.array(json.dumps(ident, ensure_ascii=False,
                                             sort_keys=True)),
    }
    if ident.get("pos") is not None:
        extras["pos"] = np.asarray(ident["pos"], dtype=float)
    if ident.get("heading") is not None:
        extras["heading"] = np.array(float(ident["heading"]))
    if ident.get("view") is not None:
        extras["view"] = np.array(str(ident["view"]))
    if ident.get("exposure") is not None:
        extras["exposure"] = np.array(int(ident["exposure"]))
    return extras


def export_frame(path: Path, rgb: np.ndarray, label: np.ndarray,
                 ident: dict, unknown_kind=None) -> None:
    """Save one labelled frame *with* its identity.

    ``unknown_kind``（可选）逐像素记录"为什么这里是 255"：1=遮挡 2=模糊
    3=无法判断。只有非零时才写，保持既有消费者不变。
    """
    extras = {}
    if unknown_kind is not None and int(np.asarray(unknown_kind).sum()) > 0:
        extras["unknown_kind"] = np.asarray(unknown_kind, dtype=np.uint8)
    np.savez_compressed(str(path), colour=rgb, label=label,
                        **identity_npz_extras(ident), **extras)


def _sidecar_seed(out_dir: Path) -> tuple[dict, str]:
    """Seed for ``<out_dir>/meta.json``: its own meta, else its parent's.

    An in-place annotation must not drop the exposure counters and frame
    records of the frames it did not touch, so the parent's records for this
    view are carried over when the directory has no meta of its own.
    """
    out_dir = Path(out_dir)
    if (out_dir / "meta.json").exists():
        m, s = _read_dir_meta(out_dir)
        return m, s
    if (out_dir.parent / "meta.json").exists():
        m, _ = _read_dir_meta(out_dir.parent)
        sub = [f for f in (m.get("frames") or [])
               if str(f.get("view") or "").strip() in ("", out_dir.name)]
        return dict(m, frames=sub), f"inherited:parent:{out_dir.parent}"
    return {}, "unavailable"


def _default_reviewer() -> str:
    """复核人缺省值：当前系统用户名（拿不到就写空，不编名字）。"""
    try:
        import getpass
        return getpass.getuser()
    except Exception:                                      # noqa: BLE001
        return ""


def write_sidecar(out_dir: Path, records: list, *, identity: dict,
                  annotation_reviewer: str = "",
                  seed: tuple | None = None) -> Path:
    """Write/merge ``<out_dir>/meta.json``: run identity + per-frame identity.

    Merges instead of overwriting (keyed by output file name): an in-place
    run must add to the recording's frame list, not replace it.  A conflicting
    non-empty run-level value is kept and reported, never silently changed.
    """
    out_dir = Path(out_dir)
    seed_meta, seed_tag = seed if seed is not None else _sidecar_seed(out_dir)
    side = dict(seed_meta or {})
    notes: list = []
    for k in ("map_name", "source_id"):
        new, old = str(identity.get(k) or ""), str(side.get(k) or "")
        if new and old and old != new:
            notes.append(f"identity conflict on {k}: kept {old!r}, "
                         f"annotation carried {new!r}")
        elif new and not old:
            side[k] = new
    if identity.get("map_name_source"):
        side.setdefault("map_name_source", identity["map_name_source"])
    side.setdefault("map_name", "")
    side.setdefault("source_id", "")
    side["label_source"] = "human_revision"
    side["annotation"] = {
        "tool": "m5_annotate_manual.py",
        "reviewer": str(annotation_reviewer or ""),
        "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "identity_source": identity.get("identity_source", "unavailable"),
        "identity_missing": list(identity.get("identity_missing") or []),
        "meta_source": identity.get("meta_source", "unavailable"),
        "frames_seed": seed_tag,
        "n_frames_saved": len(records),
    }
    if identity.get("identity_missing"):
        side["identity_note"] = (
            "map/source identity was NOT available in the recording "
            f"({', '.join(identity['identity_missing'])}); frames without it "
            "are rejected by the dataset audit - do not train on them as if "
            "their group were known")
    if notes:
        side["identity_conflicts"] = notes
    by_name = {}
    for f in side.get("frames") or []:
        p = str(f.get("path") or Path(str(f.get("rel") or "")).name)
        if p:
            by_name[Path(p).name] = f
    for r in records:
        entry = {k: v for k, v in r.items() if v not in (None, "", [])}
        by_name[Path(str(r["path"])).name] = entry
    side["frames"] = [by_name[k] for k in sorted(by_name)]
    fp = out_dir / "meta.json"
    fp.write_text(json.dumps(side, indent=1, ensure_ascii=False),
                  encoding="utf-8")
    return fp


def _live_map_name(conn) -> tuple[str, str]:
    """``(map_name, source)`` from the RUNNING session - never the argument.

    Same rule the ring collector had to learn: the constructor argument is
    what the connector was *told* to load, which on ``--attach`` says nothing
    about the session.
    """
    try:
        sc = getattr(getattr(conn, "bng", None), "scenario", None)
        cur = sc.get_current() if sc is not None else None
        level = str(getattr(cur, "level", "") or "")
        if level:
            return level, "session.get_current().level"
    except Exception:                                     # noqa: BLE001
        pass
    return "", "unavailable"


def _frames_from_episode(path: str):
    d = np.load(path, allow_pickle=True)
    return [(np.asarray(rgb, dtype=np.uint8), i)
            for i, rgb in enumerate(d["rgb"])]


def _frames_from_live(conn, n: int):
    """``(frames, idents)`` grabbed from the running game, identity included."""
    from beamng_autopilot.runtime import build_camera_ring_provider
    ring, _ = build_camera_ring_provider(conn, "tech", 400, 300,
                                         roles=("front_main",))
    level, level_src = _live_map_name(conn)
    source_id = f"grab_{time.strftime('%Y%m%d_%H%M%S')}"
    out, idents = [], []
    for i in range(n):
        snap = ring.grab_ring()
        role = "front_main" if "front_main" in snap else next(iter(snap))
        out.append((snap[role][0], i))
        pos = hdg = None
        try:
            st = conn.get_state()
            pos = [round(float(v), 3) for v in st.pos]
            hdg = round(float(getattr(st, "heading", 0.0) or 0.0), 5)
        except Exception as exc:                          # noqa: BLE001
            print(f"[annotate] frame {i}: no vehicle state ({exc}) - pos/"
                  "heading stay unknown")
        idents.append(frame_identity(
            [("live:session", {"map_name": level, "source_id": source_id,
                               "pos": pos, "heading": hdg})],
            context={"map_name_source": level_src, "exposure": i,
                     "view": role}))
        conn.step(10)
    return out, idents


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="manual lane annotation")
    ap.add_argument("--episode", type=str, default=None)
    ap.add_argument("--frames-dir", type=str, default=None)
    ap.add_argument("--grab", type=int, default=0,
                    help="capture N fresh frames from the live game first")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--reviewer", type=str, default=None,
                    help="复核人标识（缺省取 BEAMNG_REVIEWER，其次当前用户）："
                         "验收要能查'这帧是谁复核的'")
    ap.add_argument("--prefill-model", type=str, default=None,
                    help="用分割模型预填 road/line，人工只需修正误检/漏检")
    ap.add_argument("--review-incomplete", action="store_true",
                    help="只打开已有标签中 road 占比低于 --min-road-frac 的帧")
    ap.add_argument("--min-road-frac", type=float, default=0.10,
                    help="--review-incomplete 的 road 标签比例阈值")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    frames: list = []
    idents: list = []
    resume_labels: dict[int, np.ndarray] = {}
    resume_paths_by_source: dict[int, tuple[Path, Path]] = {}
    run_identity: dict = {}
    meta_source = "unavailable"
    if args.grab:
        from beamng_autopilot.connector import BeamNGConnector
        rt = getattr(args, "runtime", "tech")
        conn = BeamNGConnector("italy", "etk800",
                               port=config.runtime_port(rt),
                               home=config.runtime_home(rt))
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
        frames, idents = _frames_from_live(conn, int(args.grab))
        run_identity = idents[0] if idents else {}
    elif args.frames_dir:
        frames, idents, resume_labels, resume_paths_by_source = \
            load_frame_dir(args.frames_dir, out_dir=args.out)
        engine_labels = load_engine_labels(Path(args.frames_dir))
        meta_source = idents[0].get("meta_source", "unavailable") if idents \
            else "unavailable"
        run_identity = idents[0] if idents else {}
    elif args.episode:
        ep = (sorted(glob.glob(str(config.LOGS_DIR / "m5_e2e"
                               / "shadow_fsd_*.npz")),
                    key=os.path.getmtime)[-1]
              if args.episode == "latest" else args.episode)
        frames = _frames_from_episode(ep)
        # An episode recording carries no map identity: recorded as missing,
        # never guessed from the episode path.
        idents = [frame_identity([], source_path=ep) for _ in frames]
        run_identity = idents[0] if idents else {}
    if not frames:
        print("no frames to annotate (use --episode / --frames-dir / --grab)")
        return 1
    if args.review_incomplete:
        before = len(frames)
        keep = [i for i, (rgb, idx) in enumerate(frames)
                if idx not in resume_labels
                or float(np.mean(resume_labels[idx] == CLS_ROAD))
                < float(args.min_road_frac)]
        frames = [frames[i] for i in keep]
        idents = [idents[i] for i in keep]
        if engine_labels:
            engine_labels = [engine_labels[i] for i in keep]
        print(f"[annotate] review-incomplete: {len(frames)}/{before} frames "
              f"(road < {args.min_road_frac:.2f})")
        if not frames:
            print("[annotate] no incomplete frames found")
            return 0

    reviewer = (args.reviewer or os.environ.get("BEAMNG_REVIEWER")
                or _default_reviewer())
    print(f"[annotate] 复核人 -> {reviewer}（写进 meta.json 的 annotation.reviewer）")
    prefill = None
    if args.prefill_model:
        try:
            from beamng_autopilot.vision.segmentation import Segmenter
            prefill = Segmenter(model_path=args.prefill_model)
            print(f"[annotate] prefill model -> {args.prefill_model}")
        except Exception as exc:
            print(f"[annotate] prefill disabled: {exc}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.out) if args.out
               else config.LOGS_DIR / "m5_seg" / f"manual_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_seed = _sidecar_seed(out_dir)
    print(f"[annotate] {len(frames)} frames | output -> {out_dir}")
    _ident_txt = (f"{run_identity.get('map_name') or 'UNKNOWN'}/"
                  f"{run_identity.get('source_id') or 'UNKNOWN'}")
    if run_identity.get("identity_missing"):
        print(f"[annotate] identity {_ident_txt} via "
              f"{run_identity.get('identity_source')} - UNPROVEN: "
              f"{', '.join(run_identity['identity_missing'])} (the audit will "
              "reject frames that cannot name their map; fix the recording, "
              "not this flag)")
    else:
        print(f"[annotate] identity {_ident_txt} via "
              f"{run_identity.get('identity_source')} "
              f"(meta: {run_identity.get('meta_source', meta_source)})")

    # 工具 / 类别 / 撤回栈现在由 AnnotateSession 持有；脚本这边只留导出需要的
    # 帧号计数与逐帧记录。
    save_i = [0]
    saved_records: list = []

    def _initial_label(frame, source_idx=None):
        """这一帧的起始 ``(label, unknown_kind)``：续标优先，其次 prefill。

        与 label 同形的 unknown_kind 数组跟 label 一起进会话缓存/撤销；续标
        时从零开始（来源 npz 里的旧原因不重放，避免上一轮的标记粘住）。
        """
        blank = np.zeros(frame.shape[:2], dtype=np.uint8)
        if source_idx in resume_labels:
            cached = resume_labels[source_idx]
            if cached.shape == frame.shape[:2]:
                return cached.copy(), blank.copy()
        if prefill is None:
            return blank.copy(), blank.copy()
        try:
            road, line = prefill.predict(frame)
            lab = np.zeros(frame.shape[:2], dtype=np.uint8)
            lab[np.asarray(road, dtype=bool)] = CLS_ROAD
            lab[np.asarray(line, dtype=bool)] = CLS_LINE
            return lab, blank.copy()
        except Exception as exc:
            print(f"[annotate] prefill frame failed: {exc}")
            return blank.copy(), blank.copy()

    # 会话（工具选择 / 笔画几何 / 撤回 / 画布）在库里，脚本只留 CLI、身份与
    # 导出：下面两个回调告诉它"这一帧从什么 label 开始""保存一帧要写什么"。
    # 逐帧缓存与复标覆盖由会话负责（回上一帧不能丢未保存的修改，复标帧要覆盖
    # 它自己的输出而不是多出一份训练样本）。
    saved_paths: dict[int, tuple[Path, Path]] = {
        fi0: resume_paths_by_source[src_idx]
        for fi0, (_rgb0, src_idx) in enumerate(frames)
        if src_idx in resume_paths_by_source}

    def _on_save(fi, rgb, src_idx, label, unk, ident):
        """保存一帧：导出 npz + 预览图 + 逐帧记录（复标帧覆盖自己的输出）。"""
        if fi not in saved_paths:
            save_i[0] += 1
            saved_paths[fi] = (
                out_dir / f"frame_{save_i[0]:05d}.npz",
                out_dir / f"preview_{save_i[0]:05d}.png")
        fp, prev = saved_paths[fi]
        export_frame(fp, rgb, label, ident, unknown_kind=unk)
        engine = (engine_labels[fi]
                  if fi < len(engine_labels) and engine_labels[fi] is not None
                  else None)
        cov = side_coverage_note(label, engine) if engine is not None else None
        if cov and cov["warnings"]:
            print(f"[warn] {fp.name} 左右侧覆盖：{'; '.join(cov['warnings'])}")
        saved_records.append({
            "path": fp.name,
            "source_path": ident.get("source_path") or "",
            "map_name": ident.get("map_name") or "",
            "source_id": ident.get("source_id") or "",
            "view": ident.get("view"),
            "exposure": ident.get("exposure"),
            "pos": ident.get("pos"),
            "heading": ident.get("heading"),
            "identity_source": ident.get("identity_source", "unavailable"),
            "identity_missing": list(ident.get("identity_missing") or []),
            "unknown_px": unknown_counts(unk),
            "side_coverage": cov,
            # 验收要查"对**哪些类别**做了复核"：直接记这一帧各类别的像素数
            "classes_painted": {
                "line": int((np.asarray(label) == CLS_LINE).sum()),
                "road": int((np.asarray(label) == CLS_ROAD).sum()),
                "background": int((np.asarray(label) == 0).sum()),
                "unknown": int((np.asarray(label) == 255).sum()),
            },
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        ov = rgb.copy()
        ov[label == CLS_ROAD] = (ov[label == CLS_ROAD] * 0.6
                                 + np.array([255, 120, 0]) * 0.4
                                 ).astype(np.uint8)
        ov[label == CLS_LINE] = (0, 255, 0)
        cv2.imwrite(str(prev), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        print(f"[saved] {fp.name} (src#{src_idx}, ident="
              f"{ident.get('map_name') or 'UNKNOWN'}/"
              f"{ident.get('source_id') or 'UNKNOWN'})")

    def _brush_cb(v):
        pass

    session = AnnotateSession(
        frames, idents, label_for=_initial_label, on_save=_on_save,
        zoom=2, brush=6, cls=CLS_LINE, tool="pen", window_title=WIN)
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(
        WIN, lambda event, x, y, flags, _param: session.on_mouse(event, x, y,
                                                                flags))
    cv2.createTrackbar("brush", WIN, 6, 40, _brush_cb)
    while True:
        session.set_brush(max(1, cv2.getTrackbarPos("brush", WIN)))
        # waitKey 的原始值直接交给会话：方向键是大码，先 & 0xFF 就再也认不出来
        action = session.on_key(cv2.waitKey(20))
        cv2.imshow(WIN, session.canvas())
        if action == "finish":
            print("[annotate] all frames done")
            break
        if action == "quit":
            break
    cv2.destroyAllWindows()
    if saved_records:
        side_fp = write_sidecar(out_dir, saved_records, identity=run_identity,
                                seed=sidecar_seed,
                                annotation_reviewer=reviewer)
        print(f"[annotate] sidecar -> {side_fp.name} "
              f"(map={run_identity.get('map_name') or 'UNKNOWN'} "
              f"source={run_identity.get('source_id') or 'UNKNOWN'}, "
              f"{len(saved_records)} frames)")
    print(f"[annotate] annotations in {out_dir} - training-ready by "
          "m5_train_seg.py --runs <this dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
