"""标注工具的几何与工具栏布局：不碰 GUI 的纯函数，可离线测试。

为什么要单独成模块：`scripts/m5_annotate_manual.py` 原来把"选哪个工具"、
"笔画几何"和 cv2 窗口循环混在一个 ``main()`` 里，键盘是唯一入口，而且
"平滑曲线"这种几何没有任何可测的边界。这里把两样东西抽出来：

* 工具栏的**布局与命中测试**——给定画布宽度算出每个按钮的矩形，以及
  "点在哪一个按钮上"。纯算术：能断言"点按钮中心必命中、点缝里必不命中、
  按钮互不重叠、全部落在画布内"。
* 笔画的**几何**——直线（严格两点）、平滑曲线（Catmull-Rom，穿过控制点）、
  自由手绘路径的抽稀与平滑，以及把折线栅格化成掩码再落到
  ``(label, unknown_kind)`` 两个数组上。

工具栏项集中在 `ITEMS` 一张表里：按钮文字、快捷键、单选组、取值全部来自
它，脚本只负责画。这样"顶端能点到的操作"和"按键能触发的操作"不可能各说
一套（表里没有的操作两边都没有）。

按钮文字用 ASCII（``cv2.putText`` 只有 Hershey 字体，画不出中文），与
界面其它文字一致；中文对应关系：

    Road=路  Lane=标线  Erase=擦除
    Occl=遮挡  Blur=模糊  Undec=无法判断
    Pen=笔  Bucket=油漆桶  Straight=直线  Curve=曲线
    Undo=撤回  Clear=清空  Zoom=缩放  Prev=上一帧  Save+Next=保存并下一帧
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from beamng_autopilot.labeling import curve_schema as cs
# 曲线数学只有一份：planning 的路线重采样就是这个 Catmull-Rom，直接复用，
# 不在这里再写一遍（同一段公式有两个实现时，两边迟早会漂开）。
from beamng_autopilot.planning.local_route import _resample as _catmull_rom


# ---------------------------------------------------------------------------
# 工具栏：一张表定义全部操作
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """一个可点选的操作。

    ``group`` 决定单选语义：同一个组里选中一个就取消同组其它项；
    ``momentary`` 的动作项（撤回/清空/缩放/翻帧/保存）点一下执行一次，
    不保持选中状态。
    """

    id: str
    label: str                 # 按钮文字（ASCII）
    group: str                 # "class" | "unknown" | "tool" | "action"
    key: str | None = None     # 快捷键字符（显示在按钮上，也是按键入口）
    value: int | None = None   # class/unknown 项的像素取值
    momentary: bool = False
    hint: str = ""             # 中文名（写进文档与帮助行）


#: 顺序 = 工具栏上从左到右的顺序。类别组按原快捷键 1/2/3 排列，避免按钮上的
#: 数字看起来是乱的。
ITEMS: tuple[Item, ...] = (
    Item("cls_line", "Lane", "class", key="1", value=cs.CLS_LINE, hint="标线"),
    Item("cls_road", "Road", "class", key="2", value=cs.CLS_ROAD, hint="路"),
    Item("cls_erase", "Erase", "class", key="3", value=cs.CLS_BACKGROUND,
         hint="擦除"),
    Item("unk_occluded", "Occl", "unknown", key="4", value=1, hint="遮挡"),
    Item("unk_blurred", "Blur", "unknown", key="5", value=2, hint="模糊"),
    Item("unk_undecidable", "Undec", "unknown", key="6", value=3, hint="无法判断"),
    Item("tool_pen", "Pen", "tool", key="p", hint="笔"),
    Item("tool_bucket", "Bucket", "tool", key="f", hint="油漆桶"),
    Item("tool_straight", "Straight", "tool", key="l", hint="直线"),
    Item("tool_curve", "Curve", "tool", key="v", hint="平滑曲线"),
    Item("act_undo", "Undo", "action", key="u", momentary=True, hint="撤回"),
    Item("act_clear", "Clear", "action", key="c", momentary=True, hint="清空"),
    Item("act_zoom", "Zoom", "action", key="z", momentary=True, hint="缩放"),
    Item("act_prev", "Prev", "action", key="a", momentary=True, hint="上一帧"),
    Item("act_next", "Save+Next", "action", key="s", momentary=True,
         hint="保存并下一帧"),
)

ITEM_BY_ID: dict[str, Item] = {it.id: it for it in ITEMS}
ITEM_BY_KEY: dict[str, Item] = {it.key: it for it in ITEMS if it.key}
GROUP_ORDER = ("class", "unknown", "tool", "action")
TOOL_IDS = {"tool_pen": "pen", "tool_bucket": "bucket",
            "tool_straight": "straight", "tool_curve": "curve"}


@dataclass(frozen=True)
class Button:
    """工具栏上一个按钮的位置（画布坐标，左上原点）。"""

    item: Item
    x0: int
    y0: int
    x1: int
    y1: int

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= int(x) < self.x1 and self.y0 <= int(y) < self.y1

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2)


@dataclass(frozen=True)
class Layout:
    """一整条工具栏：按钮清单 + 它占的高度。"""

    buttons: tuple[Button, ...]
    total_h: int
    width: int

    def hit(self, x: int, y: int) -> Item | None:
        """点选命中：返回按钮对应的操作，点在缝里/条外返回 None。"""
        for b in self.buttons:
            if b.contains(x, y):
                return b.item
        return None

    def button(self, item_id: str) -> Button | None:
        for b in self.buttons:
            if b.item.id == item_id:
                return b
        return None


def text_width(text: str, *, scale: float = 0.45) -> int:
    """一段文字在画布上的像素宽度（按钮排布与状态行截断共用同一把尺）。"""
    (w, _h), _b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    return int(w)


def toolbar_layout(width: int, items: tuple[Item, ...] = ITEMS, *,
                   btn_h: int = 24, gap: int = 4, group_gap: int = 10,
                   pad: int = 6, scale: float = 0.45) -> Layout:
    """把操作排成若干行（装不下就换行），返回按钮矩形与总高度。

    宽度取按钮文字实测宽度（``cv2.getTextSize``），所以加一个按钮或改文字
    不会让布局和文字错位。
    """
    width = max(80, int(width))
    widths = []
    for it in items:
        text = f"{it.label}({it.key})" if it.key else it.label
        widths.append(min(max(34, text_width(text, scale=scale) + 18),
                          max(34, width - 2 * pad)))
    buttons: list[Button] = []
    x, y = pad, pad
    row_started = False
    for i, it in enumerate(items):
        w = widths[i]
        if row_started and x + w + pad > width:
            x, y, row_started = pad, y + btn_h + gap, False
        buttons.append(Button(it, x, y, x + w, y + btn_h))
        last_in_group = (i + 1 >= len(items) or items[i + 1].group != it.group)
        x += w + (group_gap if last_in_group else gap)
        row_started = True
    total_h = y + btn_h + pad
    return Layout(tuple(buttons), total_h, width)


# ---------------------------------------------------------------------------
# 笔画几何
# ---------------------------------------------------------------------------

DRAG_MIN_PX = 4.0            # 小于它算"点击"（放控制点），大于它算"拖拽"
PATH_MIN_STEP_PX = 2.0       # 自由手绘取样间距：更密的点只是抖动，没有信息


def dedupe(points, *, min_dist: float = 1.0) -> np.ndarray:
    """去掉重复/过近的点（零长度段会让 Catmull-Rom 的除法炸掉）。"""
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(arr) == 0:
        return arr
    keep = [arr[0]]
    for p in arr[1:]:
        if float(np.linalg.norm(p - keep[-1])) >= float(min_dist):
            keep.append(p)
    return np.asarray(keep, dtype=float)


def straight_points(a, b) -> np.ndarray:
    """直线工具：严格两点，不做任何重采样——画出来必须笔直。"""
    return np.asarray([a, b], dtype=float).reshape(-1, 2)


def catmull_rom(points, *, step_px: float = 3.0) -> np.ndarray:
    """穿过控制点的平滑曲线（复用 planning 的 Catmull-Rom 重采样）。

    两个点以内原样返回（两点之间只有一条直线，没有可平滑的自由度）。
    """
    pts = dedupe(points, min_dist=1.0)
    if len(pts) <= 2:
        return pts
    return np.asarray(_catmull_rom(pts, step=float(step_px)), dtype=float)


def simplify_path(points, *, eps_px: float = 2.5) -> np.ndarray:
    """自由手绘抽稀：容差 ``eps_px`` 的多段线近似，去掉手抖但保留形状。"""
    pts = dedupe(points, min_dist=1.0)
    if len(pts) < 3:
        return pts
    approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2),
                             float(eps_px), False)
    return approx.reshape(-1, 2).astype(float)


def resample_uniform(points, *, step_px: float = PATH_MIN_STEP_PX) -> np.ndarray:
    """按弧长等距重采样。

    鼠标事件间距是不均匀的（手快时相邻事件能差 30 px），后面的盒滤波按点数
    加权——不先等距化，滤波权重就落在"事件密度"上而不是"路径长度"上。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], seg > 1e-9])]
    if len(pts) < 2:
        return pts
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
        np.diff(pts, axis=0), axis=1))])
    total = float(arc[-1])
    if total < float(step_px) * 2:
        return pts
    n = max(2, int(np.ceil(total / float(step_px))) + 1)
    s = np.linspace(0.0, total, n)
    return np.stack([np.interp(s, arc, pts[:, 0]),
                     np.interp(s, arc, pts[:, 1])], axis=1)


def box_smooth(points, *, window: int = 5) -> np.ndarray:
    """滑动平均去手抖（两端按端点值延拓，避免把端点拉向内部）。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    window = int(window)
    if window < 3 or len(pts) < window:
        return pts
    pad = window // 2
    ext = np.vstack([np.repeat(pts[:1], pad, axis=0), pts,
                     np.repeat(pts[-1:], pad, axis=0)])
    k = np.ones(window) / window
    return np.stack([np.convolve(ext[:, 0], k, "valid"),
                     np.convolve(ext[:, 1], k, "valid")], axis=1)


def smooth_path(points, *, eps_px: float = 2.5, step_px: float = 3.0,
                window: int = 5) -> np.ndarray:
    """自由手绘 → 平滑曲线：等距化 → 去抖 → 抽稀 → Catmull-Rom。

    顺序不能换。2026-09-26 标定（40 点、±2 px 抖动的弧线）：只做"抽稀 +
    Catmull-Rom"时最大拐角 1.9-2.2 rad（等于没平滑，锯齿原样保留——抽稀的
    容差去不掉同量级的手抖）；加上窗口 5 的滑动平均后降到 0.12 rad，而轨迹
    相对理想弧线只偏 1.39 px。所以去抖必须**在**抽稀之前。
    """
    return catmull_rom(
        simplify_path(box_smooth(resample_uniform(points), window=window),
                      eps_px=eps_px),
        step_px=step_px)


def brush_thickness(brush: int) -> int:
    """画笔滑块值 → 像素线宽（与原实现一致：``thickness = 2 * brush``）。"""
    return max(1, 2 * int(brush))


def side_line_counts(label, centre_col: int | None = None) -> dict:
    """按图像左右半分列 line 像素数（左右两侧都标了没有，一眼可查）。

    状态行与导出时的左右侧覆盖核查共用同一个口径：同一份统计有两个实现时，
    "HUD 说有 R=0" 和 "导出说右侧漏标" 迟早会互相矛盾。
    """
    arr = np.asarray(label)
    h, w = arr.shape[:2]
    c = int(w // 2) if centre_col is None else int(centre_col)
    left = int((arr[:, :c] == cs.CLS_LINE).sum())
    right = int((arr[:, c:] == cs.CLS_LINE).sum())
    return {"left": left, "right": right, "centre_col": c}


def rasterize(shape, points, thickness: int) -> np.ndarray:
    """把点/折线栅格化成 0/1 掩码。

    单点 → 圆点；多点是折线 + 两端圆帽（圆帽让线条两端和画笔一致，不会
    出现方切口）。用 ``LINE_8`` 而不是抗锯齿：掩码最终是训练标签，
    边缘半像素只会让线宽不可复现。
    """
    mask = np.zeros(tuple(shape[:2]), dtype=np.uint8)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return mask
    t = max(1, int(thickness))
    r = max(1, t // 2)
    cx = lambda p: (int(round(float(p[0]))), int(round(float(p[1]))))  # noqa: E731
    if len(pts) == 1:
        cv2.circle(mask, cx(pts[0]), r, 1, -1, cv2.LINE_8)
        return mask
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [poly], False, 1, thickness=t, lineType=cv2.LINE_8)
    for p in (pts[0], pts[-1]):
        cv2.circle(mask, cx(p), r, 1, -1, cv2.LINE_8)
    return mask


def apply_mask(label: np.ndarray, unk: np.ndarray, mask: np.ndarray, *,
               cls: int, unknown_kind: int = 0) -> None:
    """把掩码落到 ``(label, unknown_kind)``：两边必须一起改。

    "不能判断"的画笔写 255(ignore) 并把原因写进 ``unk``；普通画笔写类别值
    并把 ``unk`` 清零——只写一边会让导出留下"255 但不知道为什么"的帧。
    """
    sel = mask > 0
    if not sel.any():
        return
    if unknown_kind:
        label[sel] = cs.CLS_IGNORE
        unk[sel] = int(unknown_kind)
    else:
        label[sel] = int(cls)
        unk[sel] = 0
