"""不可变数据版本：内容哈希、逐通道质量、**真正的整组隔离**与审计。

方案 §2 的两条硬要求，本模块用"能失败的检查"实现：

1. **不可变 ``dataset_id``**：对（帧内容哈希、标签哈希、逐通道质量、身份
   元数据、划分清单）整体做 sha256。写盘后拒绝覆盖——版本一旦被训练用过，
   改一个字节就该是**新版本**，而不是让旧结果失去可追溯性。
2. **整组隔离**：现有 ``--split by-map-scene`` 会在**组内**取时间尾部做
   验证（方案点名："不能把'共享组 6'写成组隔离"）。这里按**整个采集组**
   分配 train / dev / final，任一组跨集合即报错；同一路段的相邻帧、同次
   曝光的多视角、字节复制都不允许跨集合。
3. **显式的重复/冲突归属**（方案 v2 §3.1）：按图像内容哈希分组，组内再比
   标签哈希与几何身份（图像尺寸、相机内外参摘要、位姿存在性）：完全一致
   ⇒ 只留**一个**评价样本、全部来源路径记为别名；标签不同 ⇒ 记标签冲突
   并隔离（只有"恰好一侧有 verified 凭证"时才保留该侧并记继承）；几何身份
   不同 ⇒ 记身份冲突并隔离。生存者按 ``(run, path)`` 排序选取——**绝不**
   依赖目录传入顺序。

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

from beamng_autopilot.experiments.credentials import read_dir_credentials
from beamng_autopilot.experiments.labels import (
    audit_label,
    audit_summary,
)
from beamng_autopilot.experiments.protocol import effective_source

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
    # 机器可读的拒绝代码（``reject_reason`` 是给人看的说明）：
    # "no_map_identity" / "alias_duplicate" / "label_conflict" /
    # "label_conflict_superseded" / "identity_conflict"
    reject_code: str = ""
    # 逐帧位姿（来自采集 meta）：空间隔离要用它判断"同地点重采"。
    # 缺失就是 None —— 空间判定不猜（方案 W2：不能只靠字节重复）。
    pos: tuple | None = None
    heading: float | None = None
    # ---- 几何身份（方案 v2 §3.1）----------------------------------------
    # 同 RGB 但尺寸/相机/位姿不一致的帧：不能合并成一个样本，也不能当两个
    # 独立样本。尺寸取 RGB 实际像素；相机摘要元数据缺失时为空串（不猜）。
    image_hw: str = ""            # "高x宽"
    camera_sha16: str = ""        # 相机内外参摘要（meta.cameras[view] 等）
    pose_state: str = "unknown"   # pos+heading / pos / heading / unknown
    geometry_key: str = ""        # sha16(image_hw, camera_sha16, pose_state)
    # (a) 完全一致的分组里，同一样本的全部来源路径（含本记录自己的路径）；
    # 无别名时为空列表。生存者是谁见 ``aliases[content_sha16]["kept"]``。
    aliases: list[str] = field(default_factory=list)

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
    # 内容冲突（同图像、标签/几何不一致）逐对记录，可定位、可报告；
    # 结构见 ``_resolve_content_groups`` 的 docstring。顺序与目录传入顺序无关。
    conflicts: list = field(default_factory=list)
    # content_sha16 -> {"content_sha16", "n", "kept", "paths", "groups",
    #                   "splits", "n_accepted"}：同图像同标签同几何的分组，
    # 只留一个评价样本，全部来源路径保留为别名。
    aliases: dict = field(default_factory=dict)

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
        _missing_cred: list = []
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
            # 资格以**目录自己的凭证**为准（方案 §6.1 / A1）：sidecar
            # （annotation.json / meta.json 的 label_source）优先于调用方的声明，
            # 调用方**不能**用字符串把来源抬高质量。原来这里 `or
            # "engine_annotation"` 从不读凭证——人工复核过的帧会被判成
            # unreliable / valid=False（实测踩到）。
            _declared = (paint_sources.get(rd.name)
                         or paint_sources.get(str(rd)) or "")
            _cred = read_dir_credentials(rd)
            _cred_src = None if _cred is None else str(
                _cred.get("label_source") or "")
            # 没有凭证也没有声明时，保持**旧默认**（引擎标注 = unreliable）：
            # 这些帧确实带引擎 label，只是不能当门槛真值；写成 absent 会把
            # "有引擎标注但不可信"误报成"来源不明"。
            src, _src_notes = effective_source(
                _declared or "engine_annotation", _cred_src)
            if _cred is None:
                _missing_cred.append(str(rd))
            for _n in _src_notes:
                notes.append(f"{rd}: {_n}")
            if _cred is not None and _cred.get("readable") is False:
                notes.append(f"{rd}: credential file is not parseable "
                             f"({_cred.get('path')}) - treated as absent")
            elif _cred is not None and not _cred_src:
                notes.append(
                    f"{rd}: credential file {_cred.get('path')} declares no "
                    f"label_source ({_cred.get('why')}) - the declared value "
                    f"or the engine default is used, which is not a verified "
                    f"source")
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
                # 路型列（标注器另存）：有它 pavement 通道才可判（方案 §6.3
                # "两种道路类型分别标记"）。老帧没有这一列 -> 保持"不可分"。
                _rt = (np.asarray(z["road_type"], np.uint8)
                       if "road_type" in z.files else None)
                audit = audit_label(label, paint_source=src,
                                    road_min_px=min_road_px, road_type=_rt)
                rec = FrameRecord(
                    path=str(Path(f).resolve()), run=rd.name, view=view,
                    group=_group_of(map_name, source_id, rd),
                    map_name=map_name, source_id=source_id,
                    t_wall=None, exposure=None,
                    content_sha16=content_sha16(colour),
                    label_sha16=audit.label_sha256_16,
                    road_px=audit.road.pixels, line_px=audit.paint.pixels,
                    quality=audit.as_dict(),
                    image_hw=_image_hw(colour))
                records.append(rec)
            # identities come from the collection's own frame list
            for rec, frame in _zip_frames(records, files, meta):
                if frame:
                    rec.t_wall = (None if frame.get("t_wall") is None
                                  else float(frame["t_wall"]))
                    rec.exposure = (None if frame.get("exposure") is None
                                    else int(frame["exposure"]))
                    _p = frame.get("pos")
                    rec.pos = (None if not _p else tuple(
                        float(v) for v in _p[:2]))
                    _h = frame.get("heading")
                    rec.heading = (None if _h is None else float(_h))
                    rec.camera_sha16 = _camera_fingerprint(meta, view, frame)
                elif rec is not None:
                    # 有 meta 但没有逐帧记录：相机摘要只取 meta 级（不猜）
                    rec.camera_sha16 = _camera_fingerprint(meta, view, None)
        # 几何身份在 pos/heading 到位之后才算（方案 v2 §3.1：位姿是否缺失
        # 本身就是身份的一部分）
        for rec in records:
            rec.pose_state = _pose_state(rec.pos, rec.heading)
            rec.geometry_key = _geometry_key(rec)
        # map identity must be real: a collection that cannot name its map is
        # not admissible (the T13 round found a collector hardcoding "italy")
        for rec in records:
            if not rec.map_name:
                rec.reject_reason = ("no map identity in the recording: the "
                                     "group would be a directory name only")
                rec.reject_code = "no_map_identity"
        conflicts, aliases = _resolve_content_groups(records)
        groups = _assign_splits(records, dev_groups=dev_groups,
                               final_groups=final_groups, notes=notes)
        _attach_alias_split_info(aliases, records)
        if conflicts:
            notes.append(
                f"{len(conflicts)} content conflict pair(s) recorded: identical "
                f"image bytes with a different label or geometry - the "
                f"affected copies are isolated, never merged (see "
                f"manifest.conflicts for both paths and reasons)")
        if aliases:
            notes.append(
                f"{len(aliases)} alias group(s): the same image+label+geometry "
                f"is counted once, every source path is kept in "
                f"manifest.aliases")
        if _missing_cred:
            notes.append(
                f"{len(_missing_cred)} dir(s) have no credential sidecar "
                f"(annotation.json/meta.json label_source): their paint source "
                f"falls back to the declared value or engine_annotation - "
                f"not a verified source. dirs: {_missing_cred[:4]}")
        did = _dataset_id(records, groups, conflicts=conflicts,
                          aliases=aliases)
        mf = cls(dataset_id=did,
                 created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 records=records, groups=groups, notes=notes,
                 conflicts=conflicts, aliases=aliases)
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
                "conflicts": self.conflicts, "aliases": self.aliases,
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
                   schema=int(blob.get("schema", SCHEMA)),
                   conflicts=list(blob.get("conflicts") or []),
                   aliases=dict(blob.get("aliases") or {}))

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
            # 同图像、标签/几何不一致的冲突：可定位、可计数（方案 v2 §3.1/T02）
            "conflicts": {
                "n": len(self.conflicts),
                "n_isolated_copies": sum(
                    1 for r in self.records
                    if r.reject_code in ("label_conflict",
                                         "identity_conflict",
                                         "label_conflict_superseded")),
                "checked": True,
                "detail": list(self.conflicts)[:8],
            },
            # 同图像同标签同几何的分组：一个评价样本 + 全部来源别名
            "alias_groups": {
                "n": len(self.aliases),
                "n_extra_paths": sum(max(0, int(a.get("n", 0)) - 1)
                                     for a in self.aliases.values()),
                "checked": True,
                "detail": list(self.aliases.values())[:8],
            },
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


def dir_group(d: str | Path) -> str:
    """一个数据目录的场景/组键：``map_name/source_id``。

    与 manifest 的分组规则**同一个定义**（方案 W2）：分场景硬门、空间隔离审计
    和数据集清单必须指同一个东西，否则"这个场景过没过门"会随入口而变。
    身份读 ``meta.json``（先本目录、再父目录），**不从目录名猜地图**。

    没有身份时退到 ``dir/<完整路径>``：**绝不**用目录名兜底——两个采集的
    ``front_main`` 目录同名，用名字当键会把它们合并成"一个场景"，
    正是"坏场景被别处稀释"要防的事（这类目录本来就该被审计拒收）。
    """
    d = Path(d)
    meta, _where = _read_meta(d)
    if isinstance(meta, dict):
        return _group_of(str(meta.get("map_name") or ""),
                         str(meta.get("source_id") or ""), d)
    try:
        key = str(d.resolve())
    except OSError:                                    # pragma: no cover
        key = str(d)
    return f"dir/{key}"


def _image_hw(colour: np.ndarray) -> str:
    c = np.asarray(colour)
    if c.ndim >= 2:
        return f"{int(c.shape[0])}x{int(c.shape[1])}"
    return f"{int(c.size)}"


def _pose_state(pos, heading) -> str:
    """位姿存在性：缺失不猜，缺失本身就是几何身份的一部分（方案 v2 §3.1）。"""
    if pos is not None and heading is not None:
        return "pos+heading"
    if pos is not None:
        return "pos"
    if heading is not None:
        return "heading"
    return "unknown"


#: 逐帧 meta 里可能出现的相机参数键（boundary 采集器与人工标注器都用这一组）
_CAMERA_FRAME_KEYS = ("cam_offset", "cam_fwd", "cam_up", "cam_fov",
                      "cam_w", "cam_h", "width", "height")


def _canon(obj):
    """把相机参数变成可哈希的规范 JSON 值（numpy 标量 -> Python，浮点定精度）。"""
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in
                sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canon(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def _camera_fingerprint(meta, view: str, frame_meta=None) -> str:
    """相机内/外参摘要（sha16）。元数据没有任何相机信息时返回 ``""``——不猜。

    来源按"同一目录的 meta 对同一 view 的声明"取：``meta['cameras'][view]``、
    meta 级 width/height、以及逐帧 meta 的 cam_* 键。**不**从目录名或图像
    尺寸反推相机。
    """
    blob: dict = {}
    if isinstance(meta, dict):
        cams = meta.get("cameras")
        if isinstance(cams, dict) and isinstance(cams.get(view), dict):
            blob["cameras"] = {view: cams[view]}
        for k in ("width", "height"):
            if meta.get(k) is not None:
                blob[k] = meta[k]
    if isinstance(frame_meta, dict):
        for k in _CAMERA_FRAME_KEYS:
            if k in frame_meta:
                blob[f"frame.{k}"] = frame_meta[k]
    if not blob:
        return ""
    try:
        payload = json.dumps(_canon(blob), sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"))
    except (TypeError, ValueError):                    # pragma: no cover
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _geometry_key(rec: FrameRecord) -> str:
    """几何身份键：尺寸 + 相机摘要 + 位姿存在性一起哈希（同键才算同一个样本）。"""
    return _sha(f"hw={rec.image_hw}", f"cam={rec.camera_sha16}",
                f"pose={rec.pose_state}")[:16]


#: 漆线来源 rank 的强度阶梯（取值与 labels.PAINT_SOURCE_RANK 对应）：
#: 只有 ``verified`` 是"已确认"的修订/凭证继承，能在标签冲突中胜出；
#: agent/pseudo 都只是可测的研究来源（protocol.SOURCE_ELIGIBILITY）。
_RANK_TIER = {"verified": 3, "agent": 2, "pseudo": 1, "unreliable": 0}


def _paint_rank(rec: FrameRecord) -> str:
    return str((rec.quality.get("paint") or {}).get("rank") or "absent")


def _tier(rank: str) -> int:
    return _RANK_TIER.get(str(rank), -1)          # 未知/absent 一律最低


def _rel(p: str | Path) -> str:
    """报告用路径：能相对仓库根就相对，否则原样（不猜、不丢信息）。"""
    try:
        return str(Path(p).resolve().relative_to(ROOT_DIR))
    except (ValueError, OSError):
        return str(p)


def _alias_reason(sha: str, kept: FrameRecord) -> str:
    return (f"byte-identical content sha16={sha}: an evaluation sample of this "
            f"image is kept at {_rel(kept.path)}; this copy is an alias, not a "
            f"second sample (the survivor is chosen by sorted (run, path), "
            f"never by directory order)")


def _superseded_reason(sha: str, kept: FrameRecord) -> str:
    return (f"content conflict sha16={sha}: superseded by the verified "
            f"revision kept at {_rel(kept.path)}; this copy is isolated, not "
            f"silently merged")


def _conflict_reason(kind: str, sha: str, labels: list, geoms: list) -> str:
    """冲突的**机器可读**拒绝理由：两侧的标签/几何摘要都在字符串里。"""
    bits = []
    if "label" in kind:
        bits.append("label " + ", ".join(labels) + " differ")
    if "identity" in kind:
        bits.append("geometry " + "; ".join(geoms) + " differ")
    lead = f"content conflict sha16={sha}: " + " and ".join(bits)
    if "identity" in kind:
        return (lead + "; identical pixels with a different identity are "
                "neither one sample nor two independent samples -> every "
                "copy is isolated")
    return (lead + "; no confirmed revision lineage -> every copy is "
            "isolated (directory order must not pick a winner)")


def _geom_desc(rec: FrameRecord) -> str:
    return (f"{rec.image_hw or 'hw-unknown'}/"
            f"cam:{rec.camera_sha16 or 'unknown'}/{rec.pose_state}")


def _verified_winner(sigs: dict) -> tuple | None:
    """标签冲突里唯一可胜出的签名：**恰好一侧**的漆线 rank 是 verified。

    "已确认的修订继承"= 该侧目录自己的凭证把来源定为 ``human_revision`` /
    ``engine_verified``（``protocol.effective_source`` 只在凭证支持时才给
    verified；调用方声明抬不上去）。两侧都 verified ⇒ 两个权威版本互相矛盾，
    不选，全隔离。
    """
    tiers = {sig: max(_tier(_paint_rank(r)) for r in vs)
             for sig, vs in sigs.items()}
    winners = [sig for sig, t in tiers.items() if t == 3]
    if len(winners) != 1:
        return None
    return winners[0]


def _resolve_content_groups(records: list[FrameRecord]) -> tuple:
    """按**图像内容**分组，把重复/冲突变成显式归属（方案 v2 §3.1）。

    规则（写死，与 ``runs`` 的传入顺序无关）：

    (a) 同内容 + 同标签 + 同几何（尺寸 / 相机摘要 / 位姿存在性）⇒ 只计
        **一个**评价样本；生存者按 ``(run, path)`` 排序取最小（不是目录
        顺序），其余副本标 ``alias_duplicate`` 隔离，但全部来源路径记进
        ``aliases[content_sha16]["paths"]`` 和生存者的 ``aliases`` 字段。
    (b) 同内容、标签不同 ⇒ 标签冲突。只有**恰好一侧**的漆线来源 rank 是
        ``verified``（来自目录自己的凭证）且其他各侧严格低于它时，才保留
        该侧并记 ``inheritance``（"有明确修订继承关系时按已确认版本选择"）；
        否则每一份都隔离（``label_conflict``）——绝不按路径排序或"后读到"
        静默选一个。
    (c) 同内容、几何身份不同 ⇒ 身份冲突（``identity_conflict``），每一份
        都隔离：相同像素不是独立样本，几何不同也不能强行合并成一个样本。
        身份冲突**不**适用 (b) 的继承规则（几何不同就无法确认是同一视图）。

    返回 ``(conflicts, aliases)``：

    * ``conflicts``：逐对（每条两个签名各取一个代表）的 ``dict``，含
      ``group_key``(内容 sha16)、``kind``(label/identity/label+identity)、
      ``path_a/path_b``(绝对路径)、``label_sha16_a/b``、``geometry_key_a/b``
      与各自 ``image_hw/camera_sha16/pose_state``、两侧 ``source_rank``、
      ``why``、``resolution``("isolated"/"kept_verified")、``kept``、
      ``superseded``；
    * ``aliases``：``content_sha16 -> {content_sha16, n, kept, paths}``，
      ``paths`` 是**全部**来源路径（含生存者，按 (run, path) 排序）；只在
      同一样本确有多个来源（(a) 或 (b) 胜出侧自己有多份）时记录，
      单来源样本不出现在这里（``FrameRecord.aliases`` 同理保持空）。

    两个结构都只依赖帧集合本身（内容/标签/几何/rank/路径），不依赖传入
    顺序，因此 ``dataset_id`` 可复现、不会"换个目录顺序换一个标签赢家"。
    """
    by_content: dict[str, list[FrameRecord]] = {}
    for r in records:
        if not r.reject_reason:
            by_content.setdefault(r.content_sha16, []).append(r)
    conflicts: list[dict] = []
    aliases: dict[str, dict] = {}
    for sha in sorted(by_content):
        members = sorted(by_content[sha], key=lambda r: (r.run, r.path))
        if len(members) < 2:
            continue
        labels = sorted({r.label_sha16 for r in members})
        geoms = sorted({r.geometry_key for r in members})
        geoms_desc = sorted({_geom_desc(r) for r in members})
        if len(labels) == 1 and len(geoms) == 1:
            # (a) 完全一致：一个样本 + 全部别名
            kept = members[0]
            kept.aliases = [r.path for r in members]
            aliases[sha] = {"content_sha16": sha, "n": len(members),
                            "kept": kept.path,
                            "paths": [r.path for r in members]}
            for r in members[1:]:
                r.reject_reason = _alias_reason(sha, kept)
                r.reject_code = "alias_duplicate"
            continue
        kind = "+".join([k for k, on in (("label", len(labels) > 1),
                                         ("identity", len(geoms) > 1)) if on])
        sigs: dict[tuple, list[FrameRecord]] = {}
        for r in members:
            sigs.setdefault((r.label_sha16, r.geometry_key), []).append(r)
        reps = {sig: vs[0] for sig, vs in sigs.items()}   # vs 已按 (run,path) 排序
        winner = _verified_winner(sigs) if "identity" not in kind else None
        ordered = sorted(reps)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                sig_a, sig_b = ordered[i], ordered[j]
                a, b = reps[sig_a], reps[sig_b]
                kept_sig = None
                if winner is not None and winner in (sig_a, sig_b):
                    kept_sig = winner
                kept_rec = reps[kept_sig] if kept_sig is not None else None
                other = reps[sig_b if kept_sig == sig_a else sig_a] \
                    if kept_rec is not None else None
                if kept_rec is not None:
                    why = ("label differs; exactly one side has a confirmed "
                           "(verified) revision lineage, so that side is kept "
                           "and the other copy is isolated")
                    resolution, kept_path = "kept_verified", kept_rec.path
                    superseded = other.path
                else:
                    why = _conflict_reason(kind, sha, labels, geoms_desc)
                    resolution, kept_path, superseded = "isolated", None, None
                conflicts.append({
                    "group_key": sha, "kind": kind,
                    "path_a": a.path, "path_b": b.path,
                    "run_a": a.run, "run_b": b.run,
                    "label_sha16_a": a.label_sha16,
                    "label_sha16_b": b.label_sha16,
                    "geometry_key_a": a.geometry_key,
                    "geometry_key_b": b.geometry_key,
                    "image_hw_a": a.image_hw, "image_hw_b": b.image_hw,
                    "camera_sha16_a": a.camera_sha16,
                    "camera_sha16_b": b.camera_sha16,
                    "pose_state_a": a.pose_state, "pose_state_b": b.pose_state,
                    "source_rank_a": _paint_rank(a),
                    "source_rank_b": _paint_rank(b),
                    "why": why, "resolution": resolution,
                    "kept": kept_path, "superseded": superseded,
                })
        if winner is None:
            reason = _conflict_reason(kind, sha, labels, geoms_desc)
            code = ("identity_conflict" if "identity" in kind
                    else "label_conflict")
            for r in members:
                r.reject_reason = reason
                r.reject_code = code
            continue
        # (b) 唯一 verified 侧胜出：该侧只留一个样本，其余签名全隔离
        kept = reps[winner]
        win_members = sigs[winner]
        if len(win_members) > 1:            # 该侧自己也有多份来源 -> 记别名
            kept.aliases = [r.path for r in win_members]
            aliases[sha] = {"content_sha16": sha, "n": len(win_members),
                            "kept": kept.path,
                            "paths": [r.path for r in win_members]}
        for r in win_members:
            if r is not kept:
                r.reject_reason = _alias_reason(sha, kept)
                r.reject_code = "alias_duplicate"
        for sig, vs in sigs.items():
            if sig == winner:
                continue
            for r in vs:
                r.reject_reason = _superseded_reason(sha, kept)
                r.reject_code = "label_conflict_superseded"
    return conflicts, aliases


def _attach_alias_split_info(aliases: dict, records: list) -> None:
    """别名组补充分组/划分信息（要在 ``_assign_splits`` 之后调用）。"""
    by_path = {r.path: r for r in records}
    for info in aliases.values():
        members = [by_path[p] for p in info.get("paths", []) if p in by_path]
        info["groups"] = sorted({m.group for m in members})
        info["splits"] = sorted({m.split for m in members})
        info["n_accepted"] = sum(1 for m in members if not m.reject_reason)


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


def _dataset_id(records: list[FrameRecord], groups: dict, *,
                conflicts: list | None = None,
                aliases: dict | None = None) -> str:
    """数据版本哈希。贡献**只依赖帧集合本身**，与 runs 的传入顺序无关。

    除逐帧记录与划分外，冲突/别名的归属也进哈希（用仓库相对路径，避免机器
    路径差异）：目录顺序换了若真能改变接受集或谁胜出，``dataset_id`` 必然
    可见地不同——不会出现"同一个 dataset_id 两种标签赢家"。
    """
    parts = []
    for r in sorted(records, key=lambda x: (x.run, x.path)):
        parts.append("|".join([
            r.run, r.view, Path(r.path).name, r.content_sha16, r.label_sha16,
            str(r.road_px), str(r.line_px),
            str(r.quality.get("paint", {}).get("valid")), r.split,
            r.group, r.reject_reason[:40]]))
    parts.append(json.dumps(groups, sort_keys=True))
    for c in sorted(conflicts or (),
                    key=lambda c: (str(c.get("group_key", "")),
                                   _rel(str(c.get("path_a", ""))),
                                   _rel(str(c.get("path_b", ""))))):
        parts.append("conflict|" + "|".join(
            [str(c.get("kind", "")), str(c.get("group_key", "")),
             _rel(str(c.get("path_a", ""))), _rel(str(c.get("path_b", "")))]
            + [str(c.get(k, "")) for k in
               ("label_sha16_a", "label_sha16_b",
                "geometry_key_a", "geometry_key_b",
                "resolution", "kept")]))
    for sha, info in sorted((aliases or {}).items()):
        parts.append("alias|" + sha + "|" + "|".join(
            _rel(p) for p in (info.get("paths") or [])))
        parts.append("alias-kept|" + sha + "|"
                     + _rel(str(info.get("kept") or "")))
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
        notes=q.get("notes", []),
        # 路型计数要回到重建的审计对象（覆盖表按它统计"有路型图的帧"）
        road_type_px=dict(q.get("road_type_px") or {}))


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
