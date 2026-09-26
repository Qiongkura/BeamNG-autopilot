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

界面文字用中文：``cv2.putText`` 只有 Hershey 字体、画不出汉字，所以工具栏与
状态行改用 PIL 渲染（字体在 ``%WINDIR%\\Fonts`` 里找微软雅黑 / 黑体 / 等线）。
找不到中文字体时退回 ``Item.ascii`` 的英文名——**退回的是字，不是功能**：
按钮集合、快捷键、取值、几何全都不变。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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

    ``label`` 是界面显示名（中文，PIL 渲染）；``ascii`` 是找不到中文字体时的
    退回显示名——退回的是**字**，不是功能：按钮集合、快捷键、取值都不变。
    """

    id: str
    label: str                 # 界面显示名（中文）
    group: str                 # "class" | "roadtype" | "unknown" | "tool" | "action"
    ascii: str = ""            # 无中文字体时的退回名
    key: str | None = None     # 快捷键字符（显示在按钮上，也是按键入口）
    value: int | None = None   # class/unknown/roadtype 项的取值
    momentary: bool = False


#: 顺序 = 工具栏上从左到右的顺序。类别组按原快捷键 1/2/3 排列，避免按钮上的
#: 数字看起来是乱的；**路型组单独一组**（方案要求"两种道路类型分别标记，不能
#: 混成 road 一类"），用空闲的 7/8/9，不占用原来的 4/5/6。
ITEMS: tuple[Item, ...] = (
    Item("cls_line", "标线", "class", ascii="Lane", key="1", value=cs.CLS_LINE),
    Item("cls_road", "路面", "class", ascii="Road", key="2", value=cs.CLS_ROAD),
    Item("cls_erase", "擦除", "class", ascii="Erase", key="3",
         value=cs.CLS_BACKGROUND),
    Item("rt_asphalt", "沥青", "roadtype", ascii="Asphalt", key="7", value=1),
    Item("rt_gravel", "碎石", "roadtype", ascii="Gravel", key="8", value=2),
    Item("rt_shoulder", "路肩", "roadtype", ascii="Shoulder", key="9", value=3),
    Item("unk_occluded", "遮挡", "unknown", ascii="Occl", key="4", value=1),
    Item("unk_blurred", "模糊", "unknown", ascii="Blur", key="5", value=2),
    Item("unk_undecidable", "未知", "unknown", ascii="Undec", key="6", value=3),
    Item("tool_pen", "画笔", "tool", ascii="Pen", key="p"),
    Item("tool_bucket", "油漆桶", "tool", ascii="Bucket", key="f"),
    Item("tool_straight", "直线", "tool", ascii="Straight", key="l"),
    Item("tool_curve", "曲线", "tool", ascii="Curve", key="v"),
    Item("act_undo", "撤回", "action", ascii="Undo", key="u", momentary=True),
    Item("act_clear", "清空", "action", ascii="Clear", key="c", momentary=True),
    Item("act_zoom", "缩放", "action", ascii="Zoom", key="z", momentary=True),
    Item("act_prev", "上一帧", "action", ascii="Prev", key="a", momentary=True),
    Item("act_next", "保存并下一帧", "action", ascii="Save+Next", key="s",
         momentary=True),
)

ITEM_BY_ID: dict[str, Item] = {it.id: it for it in ITEMS}
ITEM_BY_KEY: dict[str, Item] = {it.key: it for it in ITEMS if it.key}
GROUP_ORDER = ("class", "roadtype", "unknown", "tool", "action")
TOOL_IDS = {"tool_pen": "pen", "tool_bucket": "bucket",
            "tool_straight": "straight", "tool_curve": "curve"}

#: 路型（``road_type`` 数组取值）：与 ``unknown_kind`` 同一套设计——像素类别只有
#: 0/1/2/255 不够表达"这是什么路面"，原因/类型另存一列，导出时一起带走。
ROAD_TYPE_NAMES = {1: "asphalt", 2: "gravel", 3: "shoulder"}
ROAD_TYPE_BY_ITEM = {"rt_asphalt": 1, "rt_gravel": 2, "rt_shoulder": 3}

#: 路型对应的**像素类别**，按 AGENTS.md 的驾驶约束定，不按"画了就算路"：
#:
#: * 沥青：铺装面 -> 路面(1)；
#: * 碎石/土路：路线本身就是土路时土才算路面（``strip_soil_from_road`` 的
#:   ``route_is_dirt`` 同一口径）-> 路面(1)，类型另记为 gravel；
#: * 路肩：**有铺装路面时土肩不得算作道路**（约束 1/2），所以写背景(0)而不是
#:   路面——否则一次手滑就能把土肩算进路面掩码。类型记 3，导出可查，
#:   界面上也会着色显示，不存在"画了看不见"。
ROAD_TYPE_CLS = {1: cs.CLS_ROAD, 2: cs.CLS_ROAD, 3: cs.CLS_BACKGROUND}
UNKNOWN_NAMES = {1: "occluded", 2: "blurred", 3: "undecidable"}


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
    """一整条工具栏：按钮清单 + 它占的高度（附渲染用的字号与文字变体）。"""

    buttons: tuple[Button, ...]
    total_h: int
    width: int
    variant: str = "cjk"
    label_size: int = 14

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


def text_width(text: str, *, size: int = 14) -> int:
    """估算一段文字的像素宽度：CJK 一字约等于字号，ASCII 约 0.55 字号。

    刻意不调用任何字体后端：布局必须是纯函数（测试要断言"点按钮中心必命中、
    按钮互不重叠、全都落在画布内"），估宽只要够准就行。
    """
    return sum(size if ord(ch) > 0x2E7F else max(1, int(size * 0.55))
               for ch in (text or " "))


#: 找中文字体的候选顺序（Windows 自带；``.ttc`` 是字体集合，PIL 能直接读）。
CJK_FONT_FILES = ("msyh.ttc", "msyhl.ttc", "simhei.ttf", "deng.ttf",
                  "simsun.ttc", "msjh.ttc")


def fonts_dir() -> Path:
    """系统字体目录（从 ``WINDIR`` 推，不硬编码机器路径）。"""
    win = os.environ.get("WINDIR") or r"C:\Windows"
    return Path(win) / "Fonts"


def cjk_font_path() -> Path | None:
    """第一个存在的中文字体文件；一个都没有返回 None（调用方退回 ASCII）。"""
    for name in CJK_FONT_FILES:
        cand = fonts_dir() / name
        if cand.exists():
            return cand
    return None


_FONT_CACHE: dict[int, object] = {}


def load_font(size: int):
    """按字号取 PIL 字体（缓存）；没有中文字体或没装 PIL 时返回 None。"""
    size = int(size)
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    path = cjk_font_path()
    if path is not None:
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(str(path), size)
        except Exception:                                  # noqa: BLE001
            font = None
    _FONT_CACHE[size] = font
    return font


def button_text(item: Item, *, variant: str = "cjk") -> str:
    """按钮上的文字：显示名 + 键位（无中文字体时用 ``ascii`` 名）。"""
    name = item.label if variant == "cjk" else (item.ascii or item.label)
    return f"{name}({item.key})" if item.key else name


def toolbar_layout(width: int, items: tuple[Item, ...] = ITEMS, *,
                   btn_h: int = 26, gap: int = 5, group_gap: int = 11,
                   pad: int = 7, label_size: int = 14,
                   variant: str = "cjk") -> Layout:
    """把操作排成若干行（装不下就换行），返回按钮矩形与总高度。

    宽度按文字实测宽度估（``text_width``），所以加一个按钮或改文字不会让布局
    和文字错位；``variant`` 决定量的是中文名还是退回的英文名。
    """
    width = max(80, int(width))
    widths = []
    for it in items:
        text = button_text(it, variant=variant)
        widths.append(min(max(30, text_width(text, size=label_size) + 16),
                          max(30, width - 2 * pad)))
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
    return Layout(tuple(buttons), total_h, width, variant, label_size)


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


def road_type_counts(road_type) -> dict:
    """``road_type`` 数组 → ``{asphalt/gravel/shoulder: 像素数}``（全 0 也返回）。"""
    arr = np.asarray(road_type) if road_type is not None else None
    out = {name: 0 for name in ROAD_TYPE_NAMES.values()}
    if arr is None or arr.size == 0:
        return out
    for value, name in ROAD_TYPE_NAMES.items():
        out[name] = int((arr == int(value)).sum())
    return out


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


@dataclass
class PaintState:
    """一帧的三个并列数组：像素类别 / 忽略原因 / 路面类型。

    为什么分三列：``label`` 的取值被训练与评测契约钉死（0/1/2/255），而"这里
    为什么标不了"和"这是什么路面"都不是那个契约能表达的东西——把它们硬塞进
    label 会悄悄改掉训练格式，另存一列则既保住了契约，又能被审计查到。
    """

    label: np.ndarray
    unknown: np.ndarray
    road_type: np.ndarray

    @classmethod
    def zeros(cls, shape) -> PaintState:
        first = np.zeros(tuple(shape[:2]), dtype=np.uint8)
        return cls(first.copy(), first.copy(), first.copy())

    def copy(self) -> PaintState:
        return PaintState(self.label.copy(), self.unknown.copy(),
                          self.road_type.copy())


def coerce_paint_state(label, unknown=None, road_type=None) -> PaintState:
    """把外部给的数组（或数组组）整理成 ``PaintState``，形状不对外报错。"""
    lab = np.asarray(label, dtype=np.uint8)
    if isinstance(label, PaintState):
        return label.copy()
    unk = (np.zeros(lab.shape, np.uint8) if unknown is None
           else np.asarray(unknown, dtype=np.uint8))
    rt = (np.zeros(lab.shape, np.uint8) if road_type is None
          else np.asarray(road_type, dtype=np.uint8))
    return PaintState(lab.copy(), unk.copy(), rt.copy())


def apply_mask(state: PaintState, mask: np.ndarray, *, cls: int,
               unknown_kind: int = 0, road_type: int = 0) -> None:
    """把掩码落到三个数组上，三边的口径必须同时成立。

    * "不能判断"画笔：label 写 255(ignore)、原因进 ``unknown``、路型清 0；
    * 路型画笔：label 写该路型的像素类别（路肩是背景，见 ``ROAD_TYPE_CLS``）、
      类型进 ``road_type``、忽略原因清 0；
    * 普通画笔：label 写类别，另外两列都清 0。

    只写其中一边会让导出留下"255 但不知道为什么""是路面但不知道什么材质"的帧。
    """
    sel = mask > 0
    if not sel.any():
        return
    if unknown_kind:
        state.label[sel] = cs.CLS_IGNORE
        state.unknown[sel] = int(unknown_kind)
        state.road_type[sel] = 0
    else:
        state.label[sel] = int(cls)
        state.unknown[sel] = 0
        state.road_type[sel] = int(road_type)
