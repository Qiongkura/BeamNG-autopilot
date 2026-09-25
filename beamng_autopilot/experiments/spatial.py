"""空间隔离：同地点重采与同曝光多视角都不能绕过训练/开发隔离（方案 W2 §7.3）。

为什么单靠"组"（``map_name/source_id``）不够：两次采集可能走同一段路、只是
``source_id`` 不同（本项目实测：`diverse_town` 与 `holdout_town` 首帧相距 30 m、
`diverse_straightstreet` 与 `ident_probe_straight` 相距 **0 m**）——整组隔离的
字面实现会放过这种泄漏。另外同一次采集的 8 个视角共享 ``exposure``：它们必须
**同组**（方案原文："同曝光多视角必须同组"）。

判据（冻结在协议里，可复算）：

* ``same_group``：两边组键相同（已有的整组隔离）；
* ``same_exposure_cross_view``：同一 ``source_id`` + 同一 ``exposure`` 出现在两边
  （多视角被拆开 = 同一瞬间同时进训练与开发）；
* ``within_buffer``：两帧位置距离 < 空间缓冲（默认 50 m，依据前向相机在
  536×403 下的可见范围取保守值；相机或分辨率变了要重新标定并升协议版本）。

本模块只做**几何与分组判定**（纯函数），不读盘、不做 I/O 决策。
"""

from __future__ import annotations

import math

#: 空间缓冲（米）。冻结值：前向相机 536×403 在典型路段上可见约 40–60 m，
#: 取保守的 50 m；改相机/分辨率必须重新标定并递增协议版本。
SPATIAL_BUFFER_M = 50.0


def _xy(pos) -> tuple | None:
    if pos is None:
        return None
    try:
        return (float(pos[0]), float(pos[1]))
    except (TypeError, ValueError, IndexError):
        return None


def distance_m(a, b) -> float | None:
    """两个位置的水平距离；任一缺失返回 ``None``（不猜）。"""
    pa, pb = _xy(a), _xy(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def frame_key(rec) -> dict:
    """从 manifest 记录里取判定需要的字段（缺失就是缺失）。"""
    return {"path": str(getattr(rec, "path", "") or ""),
            "group": str(getattr(rec, "group", "") or ""),
            "map_name": str(getattr(rec, "map_name", "") or ""),
            "source_id": str(getattr(rec, "source_id", "") or ""),
            "exposure": getattr(rec, "exposure", None),
            "view": str(getattr(rec, "view", "") or ""),
            "pos": getattr(rec, "pos", None)}


def spatial_conflicts(train_records, dev_records, *,
                     buffer_m: float = SPATIAL_BUFFER_M,
                     max_report: int = 50) -> dict:
    """训练侧 × 开发侧的空间/曝光冲突清单（每条都带证据）。

    返回 ``{buffer_m, violations, n_pairs_checked, min_distance_m,
    n_missing_position}``；``violations`` 每项含 ``why`` 与距离，
    ``min_distance_m`` 是**有位置**的帧对里的最小距离（用于复核缓冲是否合适）。
    """
    tr = [frame_key(r) for r in train_records]
    dv = [frame_key(r) for r in dev_records]
    violations: list = []
    missing = 0
    best: float | None = None
    n_pairs = 0
    n_other_map = 0
    for a in tr:
        for b in dv:
            n_pairs += 1
            # 跨地图不比坐标：不同地图的世界坐标不可比。实测踩到——两张地图的
            # 默认出生点都落在原点附近（west_coast_usa 首帧 (-0.03,-0.01)、
            # italy gm_walk 首帧 (-0.03, 0.003)），按距离算就是"相距 0.016 m
            # 的同地点重采"，纯属假冲突（方案 §7.3 的本意是同一地图内的泄漏）。
            if a["map_name"] and b["map_name"] and                     a["map_name"] != b["map_name"]:
                n_other_map += 1
                continue
            if a["group"] and a["group"] == b["group"]:
                violations.append({"why": "same_group", "train": a["path"],
                                   "dev": b["path"], "group": a["group"],
                                   "distance_m": distance_m(a["pos"],
                                                            b["pos"])})
                continue
            if (a["source_id"] and a["source_id"] == b["source_id"]
                    and a["exposure"] is not None
                    and a["exposure"] == b["exposure"]):
                violations.append({
                    "why": "same_exposure_cross_view", "train": a["path"],
                    "dev": b["path"], "source_id": a["source_id"],
                    "exposure": a["exposure"],
                    "views": [a["view"], b["view"]]})
                continue
            d = distance_m(a["pos"], b["pos"])
            if d is None:
                missing += 1
                continue
            best = d if best is None or d < best else best
            if d < float(buffer_m):
                violations.append({"why": "within_buffer", "train": a["path"],
                                   "dev": b["path"],
                                   "distance_m": round(d, 2)})
            if len(violations) >= int(max_report):
                break
        if len(violations) >= int(max_report):
            break
    return {"buffer_m": float(buffer_m), "violations": violations,
            "n_pairs_checked": n_pairs,
            "n_pairs_skipped_other_map": n_other_map,
            "min_distance_m": None if best is None else round(best, 2),
            "n_missing_position": missing,
            "truncated": len(violations) >= int(max_report)}


def group_exposure_leak(records) -> list:
    """同一次采集里，同一 ``exposure`` 的多视角是否被拆到不同组（必须同组）。"""
    by_exp: dict = {}
    for r in records:
        k = frame_key(r)
        if k["exposure"] is None or not k["source_id"]:
            continue
        by_exp.setdefault((k["source_id"], k["exposure"]), set()).add(
            k["group"])
    return [{"source_id": sid, "exposure": e, "groups": sorted(g)}
            for (sid, e), g in sorted(by_exp.items()) if len(g) > 1]


def group_spread(records, *, expect_step_m: float | None = None) -> list:
    """每个组的**实际覆盖范围**：位姿跨度与"相邻帧间距"。

    为什么审计要报它（实测教训）：一次采集 30 帧 × 2 m 步长，期望沿路走 ~58 m，
    实测位姿只在 ~13 m 内变化（车辆几乎没动）——这样的 30 帧是**近重复集**，
    不是"新增了 30 个独立样本"。帧数审计（身份/计数）看不出这一点，所以把
    跨度单独报出来，让人和选样规则都能看见。

    ``expect_step_m`` 给出期望步长时，另报 ``median_step_m`` 与
    ``step_ratio``（实测中位间距 / 期望步长）；两者不可比时记 ``None``。
    """
    by_group: dict = {}
    for r in records:
        k = frame_key(r)
        # 缺位姿的帧也要**留下组名**：整组都缺位姿时不能在报告里消失
        # （"没有测量"必须看得见，不能变成"没有这一组"）。
        by_group.setdefault(k["group"], [])
        if k["pos"] is not None:
            by_group[k["group"]].append(k)
    out = []
    for g, ks in sorted(by_group.items()):
        pts = [_xy(k["pos"]) for k in ks]
        pts = [p for p in pts if p is not None]
        if len(pts) < 2:
            out.append({"group": g, "n_positioned": len(pts),
                        "extent_m": None, "median_step_m": None,
                        "note": ("no positioned frames: spread unknown"
                                 if not pts else
                                 "fewer than 2 positioned frames: spread "
                                 "unknown")})
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        steps = sorted(math.hypot(pts[i + 1][0] - pts[i][0],
                                  pts[i + 1][1] - pts[i][1])
                       for i in range(len(pts) - 1))
        med = steps[len(steps) // 2] if steps else None
        row = {"group": g, "n_positioned": len(pts),
               "extent_m": round(extent, 2),
               "median_step_m": None if med is None else round(med, 3),
               "min_step_m": round(steps[0], 3) if steps else None,
               "max_step_m": round(steps[-1], 3) if steps else None}
        if expect_step_m:
            row["expected_step_m"] = float(expect_step_m)
            row["step_ratio"] = (None if not med
                                 else round(med / float(expect_step_m), 3))
            # 覆盖比 = 实际跨度 / "一直朝前走"应有的距离。实测例子：30 帧 × 2 m
            # 步长、间距正常（step_ratio≈1），但跨度只有 13.9 m —— 路径在小范围
            # 折返：帧不重复，**场景覆盖小**。两件事必须分开报（间距正常不等于
            # 覆盖够，覆盖够也不等于帧不重复）。
            span = max(1e-9, (len(pts) - 1) * float(expect_step_m))
            row["coverage_ratio"] = round(extent / span, 3)
        out.append(row)
    return out
