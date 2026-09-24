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

Tools:
    pen (left-drag)         paint with the active class
    bucket (right-click/f)  flood-fill the connected same-label region
                            with the active class - draw a closed
                            outline, then click inside it
Controls:
    1 / 2 / 3   class = line / road / background(erase)
    b           toggle pen / bucket
    f           switch directly to bucket
    p           switch directly to pen
    u           undo (last stroke / fill / clear)
    c           clear the whole label
    trackbar    brush size 1-40
    z           zoom 2x / 1x
    a / Left    previous frame (back)
    s           save + next frame
    q           quit

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

# The pixel classes live in the schema (T10), so the annotator and every
# metric agree on what 0/1/2/255 mean: 255 is IGNORE, never background.
CLS_LINE, CLS_ROAD, CLS_BG = cs.CLS_LINE, cs.CLS_ROAD, cs.CLS_BACKGROUND
CLASS_NAME = {CLS_LINE: "line", CLS_ROAD: "road", CLS_BG: "erase"}
WIN = "annotate"

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
                 ident: dict) -> None:
    """Save one labelled frame *with* its identity."""
    np.savez_compressed(str(path), colour=rgb, label=label,
                        **identity_npz_extras(ident))


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


def write_sidecar(out_dir: Path, records: list, *, identity: dict,
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
        print(f"[annotate] review-incomplete: {len(frames)}/{before} frames "
              f"(road < {args.min_road_frac:.2f})")
        if not frames:
            print("[annotate] no incomplete frames found")
            return 0

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

    cls = CLS_LINE
    tool = "pen"                     # pen | bucket
    zoom = 2
    fi = 0
    save_i = [0]
    undo_stack: list = []
    saved_records: list = []

    def _initial_label(frame, source_idx=None):
        if source_idx in resume_labels:
            cached = resume_labels[source_idx]
            if cached.shape == frame.shape[:2]:
                return cached.copy()
        if prefill is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        try:
            road, line = prefill.predict(frame)
            lab = np.zeros(frame.shape[:2], dtype=np.uint8)
            lab[np.asarray(road, dtype=bool)] = CLS_ROAD
            lab[np.asarray(line, dtype=bool)] = CLS_LINE
            return lab
        except Exception as exc:
            print(f"[annotate] prefill frame failed: {exc}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    rgb, fidx = frames[0]
    label = _initial_label(rgb, fidx)
    # Keep the current label in memory by SOURCE frame.  Going back must
    # restore the work (including unsaved fixes), and saving a revisited
    # frame must overwrite its existing output instead of creating a
    # duplicate training sample.
    label_cache: dict[int, np.ndarray] = {0: label.copy()}
    saved_paths: dict[int, tuple[Path, Path]] = {
        fi0: resume_paths_by_source[src_idx]
        for fi0, (_rgb0, src_idx) in enumerate(frames)
        if src_idx in resume_paths_by_source}
    painting = False
    last_pt = None

    def _cache_current() -> None:
        label_cache[fi] = label.copy()

    def _load_frame(target: int) -> None:
        nonlocal fi, rgb, fidx, label
        _cache_current()
        fi = int(target)
        rgb, fidx = frames[fi]
        cached = label_cache.get(fi)
        label = (cached.copy() if cached is not None
                 else _initial_label(rgb, fidx))
        undo_stack.clear()

    def _save_current() -> None:
        _cache_current()
        if fi not in saved_paths:
            save_i[0] += 1
            saved_paths[fi] = (
                out_dir / f"frame_{save_i[0]:05d}.npz",
                out_dir / f"preview_{save_i[0]:05d}.png")
        fp, prev = saved_paths[fi]
        ident = idents[fi] if fi < len(idents) else {}
        export_frame(fp, rgb, label, ident)
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
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        ov = rgb.copy()
        ov[label == CLS_ROAD] = (ov[label == CLS_ROAD] * 0.6
                                 + np.array([255, 120, 0]) * 0.4
                                 ).astype(np.uint8)
        ov[label == CLS_LINE] = (0, 255, 0)
        cv2.imwrite(str(prev), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        print(f"[saved] {fp.name} (src#{fidx}, ident="
              f"{ident.get('map_name') or 'UNKNOWN'}/"
              f"{ident.get('source_id') or 'UNKNOWN'})")

    def _push_undo():
        undo_stack.append(label.copy())
        if len(undo_stack) > 25:
            undo_stack.pop(0)

    def _render():
        ov = rgb.copy()
        m_road = label == CLS_ROAD
        m_line = label == CLS_LINE
        ov[m_road] = (ov[m_road] * 0.6
                      + np.array([255, 120, 0]) * 0.4).astype(np.uint8)
        ov[m_line] = (0, 255, 0)
        big = cv2.resize(ov, (ov.shape[1] * zoom, ov.shape[0] * zoom),
                         interpolation=cv2.INTER_NEAREST)
        tool_txt = f"tool={tool} class={CLASS_NAME[cls]} " \
                   f"undo={len(undo_stack)}"
        idn = idents[fi] if fi < len(idents) else {}
        ident_txt = (f"{idn.get('map_name') or 'UNKNOWN'}/"
                     f"{idn.get('source_id') or 'UNKNOWN'}")
        for txt, row in (
                (f"[{fi + 1}/{len(frames)}] src#{fidx} {ident_txt} "
                 f"{tool_txt}", 20),
                ("1/2/3 class  b=tool  f=fill  p=pen  a/Left=back  "
                 "u=undo  c=clear  z=zoom  s=save+next  q=quit", 40)):
            cv2.putText(big, txt, (8, row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(big, txt, (8, row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
        cv2.imshow(WIN, big)

    def _paint(x, y):
        r, c = int(y / zoom), int(x / zoom)
        h, w = label.shape
        cv2.circle(label, (min(max(c, 0), w - 1), min(max(r, 0), h - 1)),
                   brush, cls, -1)

    def _bucket(x, y):
        r, c = int(y / zoom), int(x / zoom)
        h, w = label.shape
        if not (0 <= r < h and 0 <= c < w):
            return
        old = int(label[r, c])
        if old == cls:
            return
        m = (label == old).astype(np.uint8)
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(m, ff, (c, r), 0, loDiff=0, upDiff=0, flags=4)
        region = (m == 0) & (label == old)
        label[region] = cls

    def _mouse(event, x, y, flags, param):
        nonlocal painting, last_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            _push_undo()
            if tool == "bucket":
                _bucket(x, y)
            else:
                painting = True
                last_pt = (x, y)
                _paint(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and painting:
            if last_pt is not None:
                cv2.line(label,
                         (int(last_pt[0] / zoom), int(last_pt[1] / zoom)),
                         (int(x / zoom), int(y / zoom)), cls,
                         thickness=max(1, brush * 2))
            last_pt = (x, y)
            _paint(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
            last_pt = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            _push_undo()
            _bucket(x, y)

    def _brush_cb(v):
        pass

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, _mouse)
    cv2.createTrackbar("brush", WIN, 6, 40, _brush_cb)
    _render()
    while True:
        brush = max(1, cv2.getTrackbarPos("brush", WIN))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            cls = CLS_LINE
        elif key == ord("2"):
            cls = CLS_ROAD
        elif key == ord("3"):
            cls = CLS_BG
        elif key == ord("b"):
            tool = "bucket" if tool == "pen" else "pen"
        elif key == ord("f"):
            tool = "bucket"
        elif key == ord("p"):
            tool = "pen"
        elif key == ord("u") and undo_stack:
            label[:] = undo_stack.pop()
        elif key == ord("c"):
            _push_undo()
            label[:] = 0
        elif key == ord("z"):
            zoom = 1 if zoom == 2 else 2
        elif key in (ord("a"), 81):       # 81 = left arrow
            if fi > 0:
                _load_frame(fi - 1)
        elif key == ord("s"):
            _save_current()
            if fi >= len(frames) - 1:
                print("[annotate] all frames done")
                break
            _load_frame(fi + 1)
        _render()
    cv2.destroyAllWindows()
    if saved_records:
        side_fp = write_sidecar(out_dir, saved_records, identity=run_identity,
                                seed=sidecar_seed)
        print(f"[annotate] sidecar -> {side_fp.name} "
              f"(map={run_identity.get('map_name') or 'UNKNOWN'} "
              f"source={run_identity.get('source_id') or 'UNKNOWN'}, "
              f"{len(saved_records)} frames)")
    print(f"[annotate] annotations in {out_dir} - training-ready by "
          "m5_train_seg.py --runs <this dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
