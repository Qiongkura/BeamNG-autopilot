"""不可变数据版本：内容哈希、逐通道质量、**真正的整组隔离**与审计。

方案 §2 的两条硬要求，本模块用"能失败的检查"实现：

1. **不可变 ``dataset_id``**：对（帧内容哈希、标签哈希、逐通道质量、身份
   元数据、划分清单）整体做 sha256。写盘后拒绝覆盖——版本一旦被训练用过，
   改一个字节就该是**新版本**，而不是让旧结果失去可追溯性。
2. **整组隔离**：现有 ``--split by-map-scene`` 会在**组内**取时间尾部做
   验证（方案点名："不能把'共享组 6'写成组隔离"）。这里按**整个采集组**
   分配 train / dev / final，任一组跨集合即报错；同一路段的相邻帧、同次
   曝光的多视角、字节复制都不允许跨集合。

``final``（最终集）只有一次确认的机会：``freeze_final`` 之后
``assert_final_unused`` 会拒绝把它用于搜索。
"""

from __future__ import annotations

import glob
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from beamng_autopilot.experiments.labels import (
    audit_label,
    audit_summary,
)

SCHEMA = 1


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def content_sha16(colour: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(colour, dtype=np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


@dataclass
class FrameRecord:
    path: str                     # repo-relative
    run: str
    view: str
    group: str
    map_name: str
    source_id: str
    t_wall: float | None
    exposure: int | None
    content_sha16: str
    label_sha16: str
    road_px: int
    line_px: int
    quality: dict
    split: str = "none"
    reject_reason: str = ""

    @property
    def trainable(self) -> bool:
        return bool(self.quality.get("road", {}).get("valid")
                    or self.quality.get("paint", {}).get("valid"))


@dataclass
class DatasetManifest:
    dataset_id: str
    created: str
    records: list[FrameRecord] = field(default_factory=list)
    groups: dict = field(default_factory=dict)     # group -> split
    notes: list[str] = field(default_factory=list)
    final_frozen: bool = False
    schema: int = SCHEMA

    # ---- construction --------------------------------------------------
    @classmethod
    def build(cls, runs, *, root: Path, paint_sources: dict | None = None,
              dev_groups: list[str] | None = None,
              final_groups: list[str] | None = None,
              min_road_px: int = 200) -> "DatasetManifest":
        """扫描 run 目录并**整组**划分；有泄漏/重复的帧被隔离并说明原因。

        ``paint_sources`` 按 run 名给漆线真值来源（默认
        ``engine_annotation``，即"不可信"）。``dev_groups``/``final_groups``
        指定整组归属；其余组进 train。
        """
        paint_sources = paint_sources or {}
        dev_groups = list(dev_groups or [])
        final_groups = list(final_groups or [])
        records: list[FrameRecord] = []
        notes: list[str] = []
        for rd in [Path(r) for r in runs]:
            files = sorted(glob.glob(str(rd / "frame_*.npz")))
            if not files:
                notes.append(f"{rd}: 0 frames - an empty collection is not a "
                             f"success, it is rejected")
                continue
            meta, meta_level = _read_meta(rd)
            if meta_level == "parent":
                meta = dict(meta, frames=[
                    f for f in (meta.get("frames") or [])
                    if str(f.get("view") or "").strip() == rd.name])
            view = rd.name
            map_name = str((meta or {}).get("map_name") or "")
            source_id = str((meta or {}).get("source_id") or "")
            if map_name and not str((meta or {}).get("map_name_source") or ""):
                # Measured defect: a legacy collector wrote the literal
                # "italy" while east_coast_usa / gridmap_v2 were loaded, so a
                # map name without a recorded provenance cannot be trusted in
                # a decision.  This is a note, not a rejection: the record is
                # still grouped and counted.
                notes.append(
                    f"{rd}: map identity {map_name!r} has no provenance "
                    f"(meta lacks map_name_source) - a legacy collector may "
                    f"have written its argument as the map name; confirm with "
                    f"independent evidence before a decision")
            src = paint_sources.get(rd.name) or paint_sources.get(str(rd)) \
                or "engine_annotation"
            for f in files:
                z = np.load(f)
                colour = np.asarray(z["colour"], np.uint8) \
                    if "colour" in z.files else None
                label = np.asarray(z["label"], np.uint8) \
                    if "label" in z.files else None
                if colour is None or label is None:
                    notes.append(f"{f}: missing colour/label - RGB/annotation "
                                 f"alignment is a gate, not a warning")
                    continue
                audit = audit_label(label, paint_source=src,
                                    road_min_px=min_road_px)
                rec = FrameRecord(
                    path=str(Path(f).resolve()), run=rd.name, view=view,
                    group=_group_of(map_name, source_id, rd),
                    map_name=map_name, source_id=source_id,
                    t_wall=None, exposure=None,
                    content_sha16=content_sha16(colour),
                    label_sha16=audit.label_sha256_16,
                    road_px=audit.road.pixels, line_px=audit.paint.pixels,
                    quality=audit.as_dict())
                records.append(rec)
            # identities come from the collection's own frame list
            for rec, frame in _zip_frames(records, files, meta):
                if frame:
                    rec.t_wall = (None if frame.get("t_wall") is None
                                  else float(frame["t_wall"]))
                    rec.exposure = (None if frame.get("exposure") is None
                                    else int(frame["exposure"]))
        # map identity must be real: a collection that cannot name its map is
        # not admissible (the T13 round found a collector hardcoding "italy")
        for rec in records:
            if not rec.map_name:
                rec.reject_reason = ("no map identity in the recording: the "
                                     "group would be a directory name only")
        _reject_content_duplicates(records)
        groups = _assign_splits(records, dev_groups=dev_groups,
                               final_groups=final_groups, notes=notes)
        did = _dataset_id(records, groups)
        mf = cls(dataset_id=did,
                 created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 records=records, groups=groups, notes=notes)
        return mf

    # ---- io ------------------------------------------------------------
    def save(self, path: Path | str, *, force: bool = False) -> Path:
        """写盘；已存在且 ``force=False`` 时**拒绝覆盖**（不可变版本）。"""
        p = Path(path)
        if p.exists() and not force:
            old = json.loads(p.read_text(encoding="utf-8"))
            if old.get("dataset_id") != self.dataset_id:
                raise FileExistsError(
                    f"{p} holds dataset {old.get('dataset_id')}; refusing to "
                    f"overwrite it with {self.dataset_id} - a version change "
                    f"is a NEW file, not an edit")
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {"schema": self.schema, "dataset_id": self.dataset_id,
                "created": self.created, "groups": self.groups,
                "notes": self.notes, "final_frozen": self.final_frozen,
                "records": [asdict(r) for r in self.records]}
        p.write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> "DatasetManifest":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dataset_id=blob["dataset_id"], created=blob.get("created", ""),
                   records=[FrameRecord(**r) for r in blob.get("records", [])],
                   groups=blob.get("groups", {}), notes=blob.get("notes", []),
                   final_frozen=bool(blob.get("final_frozen")),
                   schema=int(blob.get("schema", SCHEMA)))

    # ---- queries -------------------------------------------------------
    def by_split(self, split: str) -> list[FrameRecord]:
        return [r for r in self.records
                if r.split == split and not r.reject_reason]

    def rejected(self) -> list[FrameRecord]:
        return [r for r in self.records if r.reject_reason]

    def audit(self) -> dict:
        """跨集合审计：组重叠、帧重叠、字节复制、曝光跨视角、身份缺失。"""
        splits = {}
        for r in self.records:
            splits.setdefault(r.split, []).append(r)
        group_sides: dict[str, set] = {}
        for r in self.records:
            if r.reject_reason:
                continue
            group_sides.setdefault(r.group, set()).add(r.split)
        shared = {g: sorted(s) for g, s in group_sides.items() if len(s) > 1}
        content_sides: dict[str, set] = {}
        for r in self.records:
            if r.reject_reason:
                continue
            content_sides.setdefault(r.content_sha16, set()).add(r.split)
        dup = {c: sorted(s) for c, s in content_sides.items() if len(s) > 1}
        exp_sides: dict[str, set] = {}
        for r in self.records:
            if r.reject_reason or r.exposure is None:
                continue
            exp_sides.setdefault(f"{r.group}#e{r.exposure}", set()).add(
                r.split)
        exp_shared = {k: sorted(s) for k, s in exp_sides.items()
                      if len(s) > 1}
        return {
            "dataset_id": self.dataset_id,
            "n_records": len(self.records),
            "n_rejected": len(self.rejected()),
            "reject_reasons": _count(r.reject_reason for r in self.rejected()),
            "n_by_split": {k: len(v) for k, v in sorted(splits.items())},
            "group_overlap": {"n": len(shared), "detail": list(shared)[:8],
                              "checked": True},
            "content_overlap": {"n": len(dup), "detail": list(dup)[:8],
                                "checked": True},
            "exposure_overlap": {"n": len(exp_shared),
                                 "detail": list(exp_shared)[:8],
                                 "checked": any(
                                     r.exposure is not None
                                     for r in self.records)},
            "faces_known_to_contain": self.faces_known_to_contain(),
        }

    def faces_known_to_contain(self) -> list[str]:
        """方案要求：T13 已查看过的帧只能当开发诊断集，不得进最终集。"""
        hits = []
        for r in self.records:
            if r.split == "final" and _is_consumed(r):
                hits.append(r.path)
        return hits

    def coverage(self) -> dict:
        """逐通道覆盖（方案：每类有效帧/像素与 UNKNOWN 分开报）。"""
        out = {}
        for split in ("train", "dev", "final", "none"):
            rows = [r for r in self.records
                    if r.split == split and not r.reject_reason]
            if not rows:
                continue
            audits = [_audit_from_dict(r.quality) for r in rows]
            out[split] = audit_summary(audits)
        return out

    # ---- final set discipline -----------------------------------------
    def freeze_final(self) -> dict:
        if not self.by_split("final"):
            raise ValueError("no group is assigned to 'final': freezing an "
                             "empty final set would report success")
        self.final_frozen = True
        return {"final_frozen": True, "n_final": len(self.by_split("final")),
                "dataset_id": self.dataset_id}

    def assert_final_unused(self, *, purpose: str) -> None:
        """搜索阶段调用：最终集冻结后只能在 dry_run 之外的确认轮使用一次。"""
        if self.final_frozen and purpose != "final_confirmation":
            raise PermissionError(
                f"the final set of {self.dataset_id} is frozen; it may only "
                f"be used once for 'final_confirmation', not for {purpose!r}")


# ---------------------------------------------------------------------------
def _read_meta(rd: Path):
    if (rd / "meta.json").exists():
        return json.loads((rd / "meta.json").read_text(encoding="utf-8")), "self"
    if (rd.parent / "meta.json").exists():
        return json.loads((rd.parent / "meta.json").read_text(
            encoding="utf-8")), "parent"
    return None, None


def _zip_frames(records, files, meta):
    """把 meta 的 frames 按文件名顺序对回刚加的记录（元数据可能缺项）。"""
    if not meta:
        return [(None, None)] * len(files)
    by_path = {str(f.get("path") or ""): f for f in (meta.get("frames") or [])}
    out = []
    for f in files:
        rec = records[-len(files) + files.index(f)]
        frame = by_path.get(f"{Path(f).parent.name}/{Path(f).name}") or \
            by_path.get(Path(f).name)
        out.append((rec, frame))
    return out


def _group_of(map_name: str, source_id: str, rd: Path) -> str:
    if map_name and source_id:
        return f"{map_name}/{source_id}"
    return f"dir/{rd.name}"


def _reject_content_duplicates(records: list[FrameRecord]) -> None:
    """字节复制：同内容只保留第一条，其余标隔离（方案点名的泄漏类型）。"""
    seen: dict[str, FrameRecord] = {}
    for r in records:
        if r.reject_reason:
            continue
        first = seen.get(r.content_sha16)
        if first is None:
            seen[r.content_sha16] = r
            continue
        r.reject_reason = (f"byte-identical to {Path(first.path).name} in "
                           f"{first.run}: a copied sample inflates whichever "
                           f"side it lands on")


def _assign_splits(records: list[FrameRecord], *, dev_groups, final_groups,
                   notes) -> dict:
    known = {r.group for r in records}
    for g in list(dev_groups) + list(final_groups):
        if g not in known:
            notes.append(f"requested group {g!r} is not in the data (typo?)")
    groups: dict[str, str] = {}
    for g in sorted(known):
        if g in final_groups:
            groups[g] = "final"
        elif g in dev_groups:
            groups[g] = "dev"
        else:
            groups[g] = "train"
    for r in records:
        r.split = groups.get(r.group, "none")
        if r.reject_reason:
            r.split = "none"
    # a group may not be split across sets: assignment is per group, so the
    # only way this fails is a bad map/source identity - check it anyway
    sides: dict[str, set] = {}
    for r in records:
        if not r.reject_reason:
            sides.setdefault(r.group, set()).add(r.split)
    bad = {g: sorted(s) for g, s in sides.items() if len(s) > 1}
    if bad:
        raise ValueError(f"group straddles splits: {bad}")
    return groups


def _dataset_id(records: list[FrameRecord], groups: dict) -> str:
    parts = []
    for r in sorted(records, key=lambda x: (x.run, x.path)):
        parts.append("|".join([
            r.run, r.view, Path(r.path).name, r.content_sha16, r.label_sha16,
            str(r.road_px), str(r.line_px),
            str(r.quality.get("paint", {}).get("valid")), r.split,
            r.group, r.reject_reason[:40]]))
    parts.append(json.dumps(groups, sort_keys=True))
    return _sha(*parts)


def _is_consumed(rec: FrameRecord) -> bool:
    """T13 轮已反复查看/用于选择的集合（方案点名只能当开发诊断）。"""
    consumed = ("t13_testA", "t13_testB", "holdout_wide2", "holdout_lines2",
                "holdout_street2", "holdout_town", "holdout_lines",
                "diverse_", "ident_probe", "ident_material")
    return any(k in rec.path.replace("\\", "/") for k in consumed)


def _audit_from_dict(q: dict):
    from beamng_autopilot.experiments.labels import ClassQuality, LabelAudit
    return LabelAudit(
        road=ClassQuality(**q["road"]), paint=ClassQuality(**q["paint"]),
        pavement=ClassQuality(**q["pavement"]),
        unknown_reason=q.get("unknown_reason", ""),
        line_masked_px=q.get("line_masked_px", 0),
        frame_unknown_frac=q.get("frame_unknown_frac", 0.0),
        label_sha256_16=q.get("label_sha256_16", ""),
        notes=q.get("notes", []))


def _count(items) -> dict:
    out: dict[str, int] = {}
    for i in items:
        if not i:
            continue
        key = i[:70]
        out[key] = out.get(key, 0) + 1
    return out


def content_digest_index(dirs, *, sample: int | None = None) -> dict:
    """按**内容**（不是路径）建索引，返回重复组。

    为什么不能只看路径：采集器写的是"采集内相对路径"
    （``front_main/frame_00000.npz``），两组的同名帧路径完全一样；而真正的
    复制样本往往换了名字或换了采集。内容哈希是唯一能同时抓住这两种情况的
    证据。``sample`` 限制每组抽样帧数（大集合上更快），抽样时如实报告。
    """
    index: dict[str, list] = {}
    per_dir: dict[str, int] = {}
    n_frames = 0
    for d in [Path(x) for x in dirs]:
        files = sorted(d.glob("frame_*.npz"))
        if sample:
            step = max(1, len(files) // int(sample))
            files = files[::step][:int(sample)]
            per_dir[str(d)] = len(files)
        else:
            per_dir[str(d)] = len(files)
        for f in files:
            try:
                z = np.load(f)
                a = np.ascontiguousarray(np.asarray(z["colour"], np.uint8))
            except Exception:                # noqa: BLE001
                continue
            n_frames += 1
            dig = hashlib.sha256(a.tobytes()).hexdigest()[:20]
            try:
                rel = str(Path(f).resolve().relative_to(ROOT_DIR))
            except ValueError:
                rel = str(f)
            index.setdefault(dig, []).append(rel)
    dups = {k: v for k, v in index.items() if len(v) > 1}
    return {"n_frames": n_frames, "n_unique_images": len(index),
            "n_duplicate_groups": len(dups),
            "duplicate_groups": dict(sorted(dups.items())),
            "per_dir_frames": per_dir, "sampled": bool(sample)}


#: 仓库根：把相对路径写进报告时用，避免把机器绝对路径带进证据
ROOT_DIR = Path(__file__).resolve().parents[2]
