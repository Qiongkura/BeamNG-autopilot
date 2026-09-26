"""标注会话：工具选择、笔画绘制、撤回、画布合成——不调用任何 GUI。

为什么把状态机从脚本搬进库：脚本里的闭包没法离线测试，而"点顶端某个按钮
应该切到哪个工具""拖完一条曲线到底落了哪些像素""缩放后窗口坐标 (x,y) 对应
哪个图像像素"全是最容易错的地方。搬进库以后测试可以直接喂合成的鼠标事件，
断言 ``(label, unknown_kind, road_type)``、撤回栈和命中结果。

分工：本模块只管**交互与显示**（选择、草稿、栅格化、画布）；CLI、来源身份、
文件导出与 sidecar 仍留在 ``scripts/m5_annotate_manual.py``，由 ``on_save``
回调把落盘接回去。

交互约定（鼠标为主，键盘保留原快捷键）：

* 顶端工具栏每一项都能点：类别 / 路型 / "不能判断" / 工具是单选，动作项点一次
  执行。**路型与普通路面分成两组**（方案要求两种道路类型分别标记，不能混成
  road 一类）。
* pen：按住拖动画笔；bucket：点一下填连通区（右键同）。
* straight（直线）：拖动 = 起止两点；或点两下（先起点、再终点）。
* curve（平滑曲线）：拖动 = 自由手绘，松手时等距化 + 去抖 + 抽稀 + Catmull-Rom；
  或点若干下放控制点，Enter 落笔。
* 草稿可以放弃：Esc，或再点一次当前工具按钮；Undo 在草稿上表示"退掉最后一个
  控制点"，没有草稿时退掉上一笔笔画。

显示用中文（PIL 渲染微软雅黑一类字体）；找不到中文字体时退回英文名——退回的
是字，不是功能。路肩像素在 label 里是背景（AGENTS.md 约束：有铺装时土肩不得
算作道路），所以它**单独着色**，否则"画了看不见"。

键盘保留原来的快捷键（按钮上印的就是生效的键）：1/2/3 类别、7/8/9 路型、
4/5/6 不能判断、p 画笔、f 油漆桶、b 画笔↔油漆桶互切、l 直线、v 曲线、u 撤回、
c 清空、z 缩放、a 与左方向键上一帧、s 保存并下一帧、q 退出；ENTER 落笔曲线、
ESC 放弃草稿。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from beamng_autopilot.labeling import annotate_tools as at
from beamng_autopilot.labeling import curve_schema as cs

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.45
#: 配色（RGB）：深底 + 扁平圆角块 + 单一强调色，尽量少用边框。
BAND_BG = (23, 24, 27)
BAND_LINE = (42, 45, 51)
CHIP_BG = (38, 40, 45)
CHIP_HOVER = (52, 55, 62)
CHIP_ACTIVE = (61, 111, 181)
CHIP_BORDER = (56, 59, 66)
TEXT_ON = (255, 255, 255)
TEXT_OFF = (200, 204, 212)
TEXT_STATUS = (143, 150, 163)
LABEL_SIZE = 14
STATUS_SIZE = 13
STATUS_LINE_H = 17
#: 类别叠加色（RGB，与既有渲染一致：路面半透明橙、标线绿、未知洋红）
ROAD_RGB = (255, 120, 0)
LINE_RGB = (0, 255, 0)
UNK_RGB = (255, 0, 255)
GRAVEL_RGB = (198, 152, 96)        # 碎石/土路：土色，叠在路面上分得出来
SHOULDER_RGB = (116, 124, 140)     # 路肩：青灰（label 是背景，必须单独着色）
BRUSH_RGB = {cs.CLS_ROAD: (255, 170, 0), cs.CLS_LINE: (0, 255, 0),
             cs.CLS_BACKGROUND: (230, 230, 230)}
DRAFT_RGB = (255, 220, 90)         # 草稿中心线：亮黄
UNKNOWN_BY_VALUE = {it.value: it.id for it in at.ITEMS if it.group == "unknown"}
ROAD_TYPE_BY_VALUE = {it.value: it.id for it in at.ITEMS
                      if it.group == "roadtype"}
TOOL_ID_BY_NAME = {v: k for k, v in at.TOOL_IDS.items()}
#: 左方向键的码：不同后端不一样（GTK 65361、Windows 0x250000），81 是旧代码
#: 里用的那个值，一并保留。
LEFT_ARROW_CODES = (81, 65361, 2424832)


@dataclass
class _Draft:
    """未提交的几何草稿：直线只有起点，曲线是若干控制点。"""

    tool: str
    points: list = field(default_factory=list)
    cursor: tuple | None = None


class AnnotateSession:
    """一帧一帧的标注状态机（鼠标事件 / 按键 / 画布）。"""

    def __init__(self, frames, idents, *, label_for, on_save=None,
                 zoom: int = 2, brush: int = 6, cls: int = cs.CLS_LINE,
                 tool: str = "pen", window_title: str = "annotate") -> None:
        if not frames:
            raise ValueError("AnnotateSession needs at least one frame")
        self.window_title = str(window_title)
        self._frames = list(frames)
        self._idents = list(idents or [{}] * len(self._frames))
        self._label_for = label_for
        self._on_save = on_save
        self.fi = 0
        self.zoom = int(zoom)
        self.brush = int(brush)
        self.cls = int(cls)
        self.tool = str(tool)
        self.unknown_kind = 0
        self.road_kind = 0

        self._rgb, self._src_idx = self._frames[0]
        self.paint = self._initial(0)
        self._cache: dict[int, at.PaintState] = {0: self.paint.copy()}
        self._undo: list[at.PaintState] = []

        self._painting = False
        self._last_pt: tuple | None = None
        self._gesture: dict | None = None
        self._draft: _Draft | None = None
        self._hover_id: str | None = None
        self._layout: at.Layout | None = None

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def src_idx(self) -> int:
        return int(self._src_idx)

    @property
    def rgb(self) -> np.ndarray:
        return self._rgb

    @property
    def ident(self) -> dict:
        return self._idents[self.fi] if self.fi < len(self._idents) else {}

    @property
    def label(self) -> np.ndarray:
        return self.paint.label

    @property
    def unk(self) -> np.ndarray:
        return self.paint.unknown

    @property
    def road_type(self) -> np.ndarray:
        """逐像素路型（0=未指定 1=沥青 2=碎石 3=路肩）。"""
        return self.paint.road_type

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    @property
    def draft_points(self) -> list:
        return list(self._draft.points) if self._draft else []

    @property
    def painting(self) -> bool:
        return self._painting

    @property
    def cjk(self) -> bool:
        """有中文字体就用中文，否则退回英文（布局与显示同时切，不许半中半英）。"""
        return at.load_font(LABEL_SIZE) is not None

    def layout(self) -> at.Layout:
        """工具栏布局：按当前画布宽度缓存（缩放变化时宽度变，重新算）。"""
        width = int(self._rgb.shape[1]) * self.zoom
        variant = "cjk" if self.cjk else "ascii"
        if (self._layout is None or self._layout.width != max(80, width)
                or self._layout.variant != variant):
            self._layout = at.toolbar_layout(width, variant=variant,
                                             label_size=LABEL_SIZE)
        return self._layout

    @property
    def band_h(self) -> int:
        return self.layout().total_h + 2 * STATUS_LINE_H + 6

    # ------------------------------------------------------------------
    # 状态查询（渲染与测试都用它，避免"高亮"和"实际生效"两套判断）
    # ------------------------------------------------------------------

    def active_item_id(self) -> str:
        """当前生效的画笔（渲染高亮与状态行共用）。"""
        if self.unknown_kind:
            return UNKNOWN_BY_VALUE.get(int(self.unknown_kind), "")
        if self.road_kind:
            return ROAD_TYPE_BY_VALUE.get(int(self.road_kind), "")
        for it in at.ITEMS:
            if it.group == "class" and int(it.value) == int(self.cls):
                return it.id
        return ""

    def is_active(self, item: at.Item) -> bool:
        """工具栏高亮：动作项从不常亮，其余按组单选。"""
        if item.momentary:
            return False
        if item.group in ("class", "unknown", "roadtype"):
            return item.id == self.active_item_id()
        if item.group == "tool":
            return at.TOOL_IDS.get(item.id) == self.tool
        return False

    def brush_name(self) -> str:
        item = at.ITEM_BY_ID.get(self.active_item_id())
        if item is None:
            return "?"
        return item.label if self.cjk else (item.ascii or item.label)

    def tool_name(self) -> str:
        item = at.ITEM_BY_ID.get(TOOL_ID_BY_NAME.get(self.tool, ""))
        if item is None:
            return self.tool
        return item.label if self.cjk else (item.ascii or item.label)

    # ------------------------------------------------------------------
    # 坐标
    # ------------------------------------------------------------------

    def image_xy(self, x: int, y: int) -> tuple[int, int] | None:
        """窗口坐标 → 图像像素坐标；落在工具栏/状态条或画布外返回 None。"""
        y = int(y) - self.band_h
        if y < 0:
            return None
        h, w = self.label.shape[:2]
        r, c = int(y // self.zoom), int(int(x) // self.zoom)
        if 0 <= r < h and 0 <= c < w:
            return int(c), int(r)
        return None

    # ------------------------------------------------------------------
    # 选择与动作
    # ------------------------------------------------------------------

    def activate(self, item_id: str) -> str | None:
        """执行一个工具栏项。返回给主循环的信号：``quit`` / ``finish``。"""
        item = at.ITEM_BY_ID.get(str(item_id))
        if item is None:
            return None
        if item.group == "class":
            self.cls, self.unknown_kind, self.road_kind = int(item.value), 0, 0
        elif item.group == "roadtype":
            # 路型是一支"带材质的画笔"：同时定像素类别（路肩是背景，见
            # ROAD_TYPE_CLS）与 road_type 列，并清掉忽略原因。
            self.road_kind = int(item.value)
            self.cls = int(at.ROAD_TYPE_CLS[int(item.value)])
            self.unknown_kind = 0
        elif item.group == "unknown":
            self.unknown_kind = int(item.value)
            self.road_kind = 0
        elif item.group == "tool":
            tool = at.TOOL_IDS[item.id]
            if tool == self.tool:
                self.cancel_draft()            # 再点一次当前工具 = 放弃草稿
            else:
                self.cancel_draft()
                self.tool = tool
            self._painting = False
            self._last_pt = None
        elif item.group == "action":
            if item.id == "act_undo":
                self.undo()
            elif item.id == "act_clear":
                self.cancel_draft()
                self._push_undo()
                self.paint = at.PaintState.zeros(self.label.shape)
            elif item.id == "act_zoom":
                self.toggle_zoom()
            elif item.id == "act_prev":
                if self.fi > 0:
                    self.load_frame(self.fi - 1)
            elif item.id == "act_next":
                return "finish" if not self.save_and_advance() else None
        return None

    def toggle_zoom(self) -> None:
        self.zoom = 1 if self.zoom == 2 else 2

    def set_brush(self, value) -> None:
        self.brush = max(1, min(40, int(value)))

    def undo(self) -> bool:
        """撤回：草稿上退掉最后一个控制点，否则退掉上一笔笔画。"""
        if self._draft is not None and self._draft.points:
            self._draft.points.pop()
            if not self._draft.points:
                self._draft = None
            return True
        if self._undo:
            self.paint = self._undo.pop()
            return True
        return False

    # ------------------------------------------------------------------
    # 帧间
    # ------------------------------------------------------------------

    def _initial(self, index: int) -> at.PaintState:
        got = self._label_for(self._frames[index][0], self._frames[index][1])
        if isinstance(got, at.PaintState):
            return got.copy()
        if isinstance(got, tuple):
            return at.coerce_paint_state(*got)
        return at.coerce_paint_state(got)

    def _cache_current(self) -> None:
        self._cache[self.fi] = self.paint.copy()

    def load_frame(self, target: int) -> None:
        """切帧：先把当前帧（含未保存修改）缓存起来，再恢复目标帧。"""
        target = int(target)
        if not 0 <= target < len(self._frames):
            return
        self._cache_current()
        self.cancel_draft()
        self._painting = False
        self._last_pt = None
        self.fi = target
        self._rgb, self._src_idx = self._frames[target]
        cached = self._cache.get(target)
        self.paint = cached.copy() if cached is not None else self._initial(target)
        self._undo.clear()

    def save_and_advance(self) -> bool:
        """保存当前帧并前进；已是最后一帧则只保存并返回 False。"""
        self._cache_current()
        if self._on_save is not None:
            self._on_save(self.fi, self._rgb, self._src_idx, self.label,
                          self.unk, self.road_type, self.ident)
        if self.fi >= len(self._frames) - 1:
            return False
        self.load_frame(self.fi + 1)
        return True

    # ------------------------------------------------------------------
    # 画
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        self._undo.append(self.paint.copy())
        if len(self._undo) > 25:
            self._undo.pop(0)

    def _stamp(self, points) -> None:
        """把一段几何落进三个数组（不压撤回栈：调用方决定时机）。"""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            return
        mask = at.rasterize(self.label.shape, pts, at.brush_thickness(self.brush))
        at.apply_mask(self.paint, mask, cls=self.cls,
                      unknown_kind=self.unknown_kind,
                      road_type=self.road_kind)

    def _commit(self, points) -> bool:
        """落笔一笔：先存撤销栈，再整段盖上去。"""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            return False
        self._push_undo()
        self._stamp(pts)
        return True

    def _pen_to(self, p: tuple[int, int]) -> None:
        self._stamp([self._last_pt, p] if self._last_pt is not None else [p])
        self._last_pt = p

    def _bucket(self, p: tuple[int, int]) -> None:
        c, r = p
        h, w = self.label.shape[:2]
        if not (0 <= r < h and 0 <= c < w):
            return
        old = int(self.label[r, c])
        want = cs.CLS_IGNORE if self.unknown_kind else int(self.cls)
        if old == want and not self.road_kind:
            return
        m = (self.label == old).astype(np.uint8)
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(m, ff, (c, r), 0, loDiff=0, upDiff=0, flags=4)
        region = (m == 0) & (self.label == old)
        self.label[region] = want
        self.unk[region] = int(self.unknown_kind) if self.unknown_kind else 0
        self.road_type[region] = int(self.road_kind) if self.road_kind else 0

    def cancel_draft(self) -> None:
        self._draft = None
        self._gesture = None

    def commit_draft(self) -> bool:
        """Enter：把曲线草稿落笔（直线草稿只有起点时无法落笔）。"""
        d = self._draft
        if d is None or d.tool != "curve" or len(d.points) < 2:
            return False
        ok = self._commit(at.catmull_rom(d.points))
        self._draft = None
        return ok

    # ------------------------------------------------------------------
    # 鼠标
    # ------------------------------------------------------------------

    def on_mouse(self, event: int, x: int, y: int, flags: int = 0) -> bool:
        """处理一个 cv2 鼠标事件；返回是否需要重画。"""
        hit = self.layout().hit(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            changed = False
            hover = hit.item.id if hit is not None else None
            if hover != self._hover_id:
                self._hover_id = hover
                changed = True
            p = self.image_xy(x, y)
            if p is None:
                return changed
            if self._painting:
                self._pen_to(p)
                return True
            if self._gesture is not None:
                start = self._gesture["start"]
                if (float(np.hypot(p[0] - start[0], p[1] - start[1]))
                        >= at.DRAG_MIN_PX):
                    self._gesture["moved"] = True
                self._gesture["cursor"] = p
                if self.tool == "curve" and self._gesture["moved"]:
                    path = self._gesture["path"]
                    if (not path or float(np.hypot(p[0] - path[-1][0],
                                                   p[1] - path[-1][1]))
                            >= at.PATH_MIN_STEP_PX):
                        path.append(p)
                return True
            if self._draft is not None and self._draft.cursor != p:
                self._draft.cursor = p
                return True
            return changed

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            if hit is not None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.activate(hit.id)
                    return True
                return False                      # 右键落在工具栏：忽略，不填色
            p = self.image_xy(x, y)
            if p is None:
                return False
            if event == cv2.EVENT_RBUTTONDOWN:
                self._push_undo()
                self._bucket(p)
                return True
            if self.tool == "bucket":
                self._push_undo()
                self._bucket(p)
                return True
            if self.tool == "pen":
                self._push_undo()
                self._painting = True
                self._last_pt = None
                self._pen_to(p)
                return True
            self._gesture = {"start": p, "cursor": p, "moved": False,
                             "path": [p] if self.tool == "curve" else []}
            return True

        if event == cv2.EVENT_LBUTTONUP:
            if self._painting:
                self._painting = False
                self._last_pt = None
                return True
            if self._gesture is not None:
                return self._finish_gesture()
        return False

    def _finish_gesture(self) -> bool:
        """松手：拖动过的按一次手势落笔；没动的算一次点击（放点/起手）。"""
        g = self._gesture
        self._gesture = None
        if g is None:
            return False
        start, cursor = g["start"], g["cursor"] or g["start"]
        far = float(np.hypot(cursor[0] - start[0], cursor[1] - start[1]))
        if self.tool == "straight":
            if g["moved"] and far >= at.DRAG_MIN_PX:
                return self._commit(at.straight_points(start, cursor))
            if self._draft is None:
                self._draft = _Draft("straight", [start], cursor=start)
                return True
            first = self._draft.points[0]
            self._draft = None
            if float(np.hypot(start[0] - first[0], start[1] - first[1])) >= 1.0:
                return self._commit(at.straight_points(first, start))
            return True
        if self.tool == "curve":
            if g["moved"] and len(g["path"]) >= 2:
                return self._commit(at.smooth_path(g["path"]))
            if self._draft is None:
                self._draft = _Draft("curve", [], cursor=start)
            self._draft.points.append(start)
            self._draft.cursor = start
            return True
        return False

    # ------------------------------------------------------------------
    # 键盘（工具栏表是唯一来源：按钮上写的键就是这里认的键）
    # ------------------------------------------------------------------

    def on_key(self, key: int) -> str | None:
        """按键入口。接受 ``waitKey`` 的原始值（含方向键的大码），也接受低字节。

        注意 81 既是旧代码里"左方向键"的码、又等于 ``ord("Q")``：方向键必须在
        可打印字符之前判断，否则按左方向键会被当成退出。
        """
        raw = int(key)
        if raw in (-1, 255):
            return None
        if raw in LEFT_ARROW_CODES:
            return self.activate("act_prev")
        if raw == 13:                          # Enter：曲线落笔
            self.commit_draft()
            return None
        if raw == 27:                          # Esc：放弃草稿
            self.cancel_draft()
            return None
        if raw == ord("q"):
            return "quit"
        ch = chr(raw) if 32 <= raw < 127 else ""
        if ch == "b":                          # 旧键：画笔 / 油漆桶 互切
            self.activate("tool_bucket" if self.tool != "bucket" else "tool_pen")
            return None
        item = at.ITEM_BY_KEY.get(ch)
        if item is None:
            return None
        return self.activate(item.id)

    # ------------------------------------------------------------------
    # 画布
    # ------------------------------------------------------------------

    def brush_rgb(self) -> tuple:
        if self.unknown_kind:
            return UNK_RGB
        if int(self.road_kind) == 2:
            return GRAVEL_RGB
        if int(self.road_kind) == 3:
            return SHOULDER_RGB
        if self.road_kind:
            return ROAD_RGB
        return BRUSH_RGB.get(int(self.cls), LINE_RGB)

    def status_line(self) -> str:
        idn = self.ident
        sides = at.side_line_counts(self.label)
        n_shoulder = int((self.road_type == 3).sum())
        n_gravel = int((self.road_type == 2).sum())
        if self.cjk:
            return (f"[{self.fi + 1}/{self.n_frames}] 源#{self.src_idx} "
                    f"{idn.get('map_name') or '未知地图'}/"
                    f"{idn.get('source_id') or '未知来源'}　"
                    f"工具={self.tool_name()}　画笔={self.brush_name()}　"
                    f"撤回={len(self._undo)}　左={sides['left']}　"
                    f"右={sides['right']}　碎石={n_gravel}　路肩={n_shoulder}　"
                    f"缩放={self.zoom}x")
        return (f"[{self.fi + 1}/{self.n_frames}] src#{self.src_idx} "
                f"{idn.get('map_name') or 'UNKNOWN'}/"
                f"{idn.get('source_id') or 'UNKNOWN'} tool={self.tool} "
                f"brush={self.brush_name()} undo={len(self._undo)} "
                f"L={sides['left']} R={sides['right']} gravel={n_gravel} "
                f"shoulder={n_shoulder} zoom={self.zoom}x")

    def hint_line(self) -> str:
        """当前工具怎么用（键位不在这里重复——按钮上已经印了）。"""
        cjk = self.cjk
        if self.tool == "straight":
            head = ("直线：拖动 A→B，或先点 A 再点 B" if cjk
                    else "straight: drag A->B, or click A then B")
            if self._draft:
                head += "　[已起手，点终点]" if cjk else "  [draft: click B]"
        elif self.tool == "curve":
            n = len(self._draft.points) if self._draft else 0
            head = (f"曲线：拖动=自由手绘（自动平滑），或点若干点后按回车落笔　"
                    f"[已放点 {n}]" if cjk else
                    "curve: drag = freehand (auto-smoothed), or click points "
                    f"then ENTER  [points={n}]")
        elif self.tool == "bucket":
            head = ("油漆桶：点一下填连通区域（右键同）" if cjk
                    else "bucket: click a region to fill it (right-click too)")
        else:
            head = ("画笔：按住左键拖动，涂当前画笔" if cjk
                    else "pen: hold the left button and drag to paint")
        tail = ("　｜　回车=落笔　ESC=放弃草稿　q=退出" if cjk
                else "   |   ENTER finish curve   ESC cancel   q quit")
        return head + tail

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def _fit_text(self, canvas: np.ndarray, text: str, x: int, y: int,
                  size: int, color: tuple, font=None) -> None:
        """画一行文字，超出画布宽度就先截短加省略号；无字体时退回 cv2 英文。"""
        room = int(canvas.shape[1]) - x - 6
        while text and at.text_width(text, size=size) > room:
            text = text[:-1]
            if text and at.text_width(text + "…", size=size) <= room:
                text += "…"
                break
        if not text:
            return
        if font is not None:
            from PIL import Image, ImageDraw
            img = Image.fromarray(canvas)
            ImageDraw.Draw(img).text((x, y), text, font=font, fill=color)
            canvas[:] = np.asarray(img)
        else:
            cv2.putText(canvas, text, (x, y + size - 3), FONT,
                        size / 30.0, color, 1, cv2.LINE_AA)

    def _draw_band(self, band: np.ndarray, layout: at.Layout) -> None:
        """工具栏 + 两行状态：PIL 圆角块与中文，无字体时退回 cv2 英文块。"""
        band[:] = BAND_BG
        font = at.load_font(layout.label_size) if layout.variant == "cjk" else None
        status_font = at.load_font(STATUS_SIZE) if font is not None else None
        if font is not None:
            from PIL import Image, ImageDraw
            img = Image.fromarray(band)
            draw = ImageDraw.Draw(img)
            for b in layout.buttons:
                if self.is_active(b.item):
                    fill, fg = CHIP_ACTIVE, TEXT_ON
                elif self._hover_id == b.item.id:
                    fill, fg = CHIP_HOVER, TEXT_ON
                else:
                    fill, fg = CHIP_BG, TEXT_OFF
                draw.rounded_rectangle([b.x0, b.y0, b.x1 - 1, b.y1 - 1],
                                       radius=6, fill=fill,
                                       outline=CHIP_BORDER, width=1)
                text = at.button_text(b.item, variant=layout.variant)
                tw = at.text_width(text, size=layout.label_size)
                draw.text((b.x0 + max(4, (b.w - tw) // 2),
                           b.y0 + (b.h - layout.label_size) // 2 + 1),
                          text, font=font, fill=fg)
            self._group_separators(draw, layout)
            band[:] = np.asarray(img)
        else:
            self._draw_band_cv2(band, layout)
        self._draw_status(band, layout, status_font)

    def _group_separators(self, draw, layout: at.Layout) -> None:
        """分组：只在同排换组处画一条浅竖线，不加多余边框（简约）。"""
        for i in range(len(layout.buttons) - 1):
            a, b = layout.buttons[i], layout.buttons[i + 1]
            if a.y0 != b.y0 or a.item.group == b.item.group:
                continue
            x = (a.x1 + b.x0) // 2
            draw.line([(x, a.y0 + 5), (x, a.y1 - 5)], fill=BAND_LINE, width=1)

    def _draw_band_cv2(self, band: np.ndarray, layout: at.Layout) -> None:
        """没有中文字体时的英文退回（功能不变，只是字换掉）。"""
        for b in layout.buttons:
            if self.is_active(b.item):
                fill, fg = CHIP_ACTIVE, TEXT_ON
            elif self._hover_id == b.item.id:
                fill, fg = CHIP_HOVER, TEXT_ON
            else:
                fill, fg = CHIP_BG, TEXT_OFF
            cv2.rectangle(band, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1), fill, -1)
            cv2.rectangle(band, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1),
                          CHIP_BORDER, 1)
            text = at.button_text(b.item, variant=layout.variant)
            (tw, th), _b = cv2.getTextSize(text, FONT, TEXT_SCALE, 1)
            cv2.putText(band, text,
                        (b.x0 + max(2, (b.w - tw) // 2), b.y0 + (b.h + th) // 2),
                        FONT, TEXT_SCALE, fg, 1, cv2.LINE_AA)
        for i in range(len(layout.buttons) - 1):
            a, b = layout.buttons[i], layout.buttons[i + 1]
            if a.y0 != b.y0 or a.item.group == b.item.group:
                continue
            x = (a.x1 + b.x0) // 2
            cv2.line(band, (x, a.y0 + 5), (x, a.y1 - 5), BAND_LINE, 1)

    def _draw_status(self, band: np.ndarray, layout: at.Layout, font) -> None:
        y = layout.total_h + 4
        for text in (self.status_line(), self.hint_line()):
            self._fit_text(band, text, 8, y, STATUS_SIZE, TEXT_STATUS, font)
            y += STATUS_LINE_H
        cv2.line(band, (0, layout.total_h - 2),
                 (band.shape[1] - 1, layout.total_h - 2), BAND_LINE, 1)

    def _draw_preview(self, canvas: np.ndarray) -> None:
        """把草稿/手势画在**显示层**上（不进 label，所以放弃草稿不脏数据）。"""
        z = self.zoom
        color = self.brush_rgb()
        width = max(1, at.brush_thickness(self.brush) * z)
        segs: list[np.ndarray] = []
        if self._gesture is not None and self._gesture["moved"]:
            if self.tool == "curve":
                segs.append(np.asarray(self._gesture["path"], dtype=float))
            else:
                segs.append(np.asarray([self._gesture["start"],
                                        self._gesture["cursor"]], dtype=float))
        elif self._draft is not None:
            pts = list(self._draft.points)
            if self._draft.cursor is not None:
                pts = pts + [self._draft.cursor]
            if len(pts) >= 2:
                segs.append(at.catmull_rom(pts) if self._draft.tool == "curve"
                            else at.straight_points(pts[0], pts[-1]))
            for p in self._draft.points:       # 控制点
                cv2.circle(canvas, (int(p[0] * z), int(p[1] * z)), 3,
                           DRAFT_RGB, -1, cv2.LINE_8)
        for seg in segs:
            if len(seg) == 0:
                continue
            poly = np.round(seg * z).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [poly], False, color, thickness=width,
                          lineType=cv2.LINE_8)
            cv2.polylines(canvas, [poly], False, DRAFT_RGB, thickness=1,
                          lineType=cv2.LINE_8)

    def _overlay(self) -> np.ndarray:
        """标签叠加：路面/标线/未知 + 路型（碎石叠土色，路肩单独着色）。"""
        ov = self._rgb.copy()
        m_road = self.label == cs.CLS_ROAD
        ov[m_road] = (ov[m_road] * 0.6 + np.array(ROAD_RGB) * 0.4).astype(np.uint8)
        ov[self.label == cs.CLS_LINE] = LINE_RGB
        rt = self.road_type
        gravel = (rt == 2) & m_road
        if gravel.any():
            ov[gravel] = (ov[gravel] * 0.55 + np.array(GRAVEL_RGB) * 0.45
                          ).astype(np.uint8)
        shoulder = rt == 3
        if shoulder.any():
            ov[shoulder] = (ov[shoulder] * 0.45 + np.array(SHOULDER_RGB) * 0.55
                            ).astype(np.uint8)
        if self.unk.any():
            ov[self.unk > 0] = UNK_RGB
        return ov

    def canvas(self) -> np.ndarray:
        """合成一帧显示画布：顶部工具栏与状态条 + 图像（含标签叠加与草稿）。"""
        z = self.zoom
        ov = self._overlay()
        big = cv2.resize(ov, (ov.shape[1] * z, ov.shape[0] * z),
                         interpolation=cv2.INTER_NEAREST)
        self._draw_preview(big)
        layout = self.layout()
        band = np.zeros((self.band_h, big.shape[1], 3), np.uint8)
        self._draw_band(band, layout)
        return np.vstack([band, big])
