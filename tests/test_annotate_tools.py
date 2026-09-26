"""标注界面：顶端工具栏、直线与平滑曲线工具。

背景（用户要求）：原来的标注器只能按键换工具（1/2/3 选类、b/f/p 切笔和
油漆桶、u 撤回），而画标线需要的是**鼠标**——手一直按在键上就没法连续画。
这组测试钉住三件事：

1. 工具栏上的**每一项都能点**：逐个点按钮中心，断言状态真的切过去了
   （类别、未知原因、工具、撤回、缩放、翻帧、保存），不是画着好看；
2. 点击的**坐标映射**正确：工具栏那一条不能被当成图像画到 label 上，
   缩放 1x/2x 下点同一个像素必须落在同一个像素；
3. 两种新几何**名副其实**：直线每个像素都在理想线段 ±(半径+1) 内、水平线
   每列行集合完全一致；曲线穿过控制点且拐角远小于手绘原轨迹（自由手绘
   先抽稀再 Catmull-Rom，不是把抖动原样插值进去）。

测试用合成鼠标事件驱动，不需要窗口：``AnnotateSession`` 把交互从 GUI 循环
里分出来了，所以"点哪个按钮、拖到哪、落哪些像素"全都能离线断言。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from beamng_autopilot.labeling import annotate_session as ann
from beamng_autopilot.labeling import annotate_tools as at
from beamng_autopilot.labeling import curve_schema as cs

W, H = 96, 64


def _mk(n_frames: int = 1, *, w: int = W, h: int = H, **kw) -> ann.AnnotateSession:
    frames = [(np.zeros((h, w, 3), np.uint8), i) for i in range(n_frames)]
    idents = [{"map_name": "italy", "source_id": "s"} for _ in range(n_frames)]
    kw.setdefault("label_for", lambda rgb, idx: np.zeros(rgb.shape[:2], np.uint8))
    return ann.AnnotateSession(frames, idents, **kw)


def _btn(sess: ann.AnnotateSession, item_id: str) -> tuple[int, int]:
    b = sess.layout().button(item_id)
    assert b is not None, f"工具栏缺少 {item_id}"
    return b.center


def _click_item(sess: ann.AnnotateSession, item_id: str) -> None:
    x, y = _btn(sess, item_id)
    sess.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y)
    sess.on_mouse(cv2.EVENT_LBUTTONUP, x, y)


def _win(sess: ann.AnnotateSession, c: float, r: float) -> tuple[int, int]:
    """图像像素 → 窗口坐标（含工具栏偏移与缩放）。"""
    z = sess.zoom
    return int(c * z + z // 2), int(sess.band_h + r * z + z // 2)


def _press(sess, p, **kw):
    x, y = _win(sess, *p)
    sess.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, **kw)


def _move_to(sess, p, **kw):
    x, y = _win(sess, *p)
    sess.on_mouse(cv2.EVENT_MOUSEMOVE, x, y, **kw)


def _release(sess, p=None, **kw):
    if p is None:
        sess.on_mouse(cv2.EVENT_LBUTTONUP, 0, 0, **kw)
    else:
        x, y = _win(sess, *p)
        sess.on_mouse(cv2.EVENT_LBUTTONUP, x, y, **kw)


def _click_at(sess, p, button: int = cv2.EVENT_LBUTTONDOWN) -> None:
    _press(sess, p)
    _release(sess, p) if button == cv2.EVENT_LBUTTONDOWN else None


def _drag(sess, a, b, steps: int = 10) -> None:
    _press(sess, a)
    for i in range(1, steps + 1):
        _move_to(sess, (a[0] + (b[0] - a[0]) * i / steps,
                        a[1] + (b[1] - a[1]) * i / steps))
    _release(sess, b)


def _drag_path(sess, pts) -> None:
    _press(sess, pts[0])
    for p in pts[1:]:
        _move_to(sess, p)
    _release(sess, pts[-1])


def _painted(mask_like) -> np.ndarray:                     # (N, 2) 的 (x, y)
    ys, xs = np.nonzero(np.asarray(mask_like) != 0)
    return np.stack([xs, ys], axis=1).astype(float)


def _max_dist_to_segment(pix: np.ndarray, a, b) -> float:
    """每个点到理想线段的最短距离取最大（"笔直"的定量判据）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-9:
        return float(np.linalg.norm(pix - a, axis=1).max())
    t = np.clip(((pix - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(pix - proj, axis=1).max())


def _min_dist_to_polyline(p: np.ndarray, poly: np.ndarray) -> float:
    best = float("inf")
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 1e-9:
            d = float(np.linalg.norm(p - a))
        else:
            t = min(1.0, max(0.0, float((p - a) @ ab) / denom))
            d = float(np.linalg.norm(p - (a + t * ab)))
        best = min(best, d)
    return best


def _max_dev_to_polyline(points, poly) -> float:
    poly = np.asarray(poly, float).reshape(-1, 2)
    return max(_min_dist_to_polyline(np.asarray(p, float), poly)
               for p in points)


def _arc(n: int, *, jitter: float = 0.0, seed: int = 7):
    """一段弧线采样：x 均匀推进，y 走正弦；``jitter`` 加手抖。"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = i / (n - 1)
        x = 10 + 70 * t
        out.append((x, 32 + 14 * np.sin(t * np.pi)
                    + (rng.normal(0, jitter) if jitter else 0.0)))
    return out


def _turn_rad(pts) -> float:
    """折线的最大拐角（相邻段方向差，弧度）。"""
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    d = np.diff(p, axis=0)
    keep = np.linalg.norm(d, axis=1) > 1e-9
    d = d[keep]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    diff = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    return float(np.abs(diff).max())


# ---------------------------------------------------------------------------
# 工具栏：每一项都能点，且真的生效
# ---------------------------------------------------------------------------


def test_every_item_is_on_the_toolbar_once_and_clickable():
    """表里每一项都在工具栏上出现一次，点它的中心就命中它自己。"""
    sess = _mk()
    layout = sess.layout()
    ids = [b.item.id for b in layout.buttons]
    assert ids == [it.id for it in at.ITEMS]
    assert len(set(ids)) == len(ids)
    for b in layout.buttons:
        assert layout.hit(*b.center).id == b.item.id


def test_buttons_do_not_overlap_and_stay_inside_the_canvas():
    for width in (320, 640, 800, 1280):
        layout = at.toolbar_layout(width)
        for b in layout.buttons:
            assert b.x0 >= 0 and b.x1 <= layout.width, (width, b)
        by_row: dict[int, list] = {}
        for b in layout.buttons:
            by_row.setdefault(b.y0, []).append(b)
        for row in by_row.values():
            row.sort(key=lambda b: b.x0)
            for a, b in zip(row, row[1:]):
                assert a.x1 <= b.x0, (width, a, b)
        assert layout.total_h > 0


def test_narrow_canvas_wraps_instead_of_dropping_buttons():
    """画布再窄也不能把操作挤出界面：换行，而不是丢掉按钮。"""
    layout = at.toolbar_layout(160)
    assert len(layout.buttons) == len(at.ITEMS)
    rows = {b.y0 for b in layout.buttons}
    assert len(rows) > 1
    assert layout.total_h > at.toolbar_layout(1280).total_h


def test_gap_and_outside_clicks_hit_nothing():
    layout = at.toolbar_layout(800)
    first = layout.buttons[0]
    mid = (first.x1 + layout.buttons[1].x0) // 2
    assert layout.hit(mid, first.y0 + 2) is None
    assert layout.hit(first.center[0], layout.total_h + 40) is None


@pytest.mark.parametrize("item_id,check", [
    ("cls_road", lambda s: (s.cls, s.unknown_kind) == (cs.CLS_ROAD, 0)),
    ("cls_line", lambda s: (s.cls, s.unknown_kind) == (cs.CLS_LINE, 0)),
    ("cls_erase", lambda s: (s.cls, s.unknown_kind) == (cs.CLS_BACKGROUND, 0)),
    ("unk_occluded", lambda s: s.unknown_kind == 1),
    ("unk_blurred", lambda s: s.unknown_kind == 2),
    ("unk_undecidable", lambda s: s.unknown_kind == 3),
    ("tool_bucket", lambda s: s.tool == "bucket"),
    ("tool_pen", lambda s: s.tool == "pen"),
    ("tool_straight", lambda s: s.tool == "straight"),
    ("tool_curve", lambda s: s.tool == "curve"),
    ("rt_asphalt", lambda s: (s.road_kind, s.cls) == (1, cs.CLS_ROAD)),
    ("rt_gravel", lambda s: (s.road_kind, s.cls) == (2, cs.CLS_ROAD)),
    ("rt_shoulder", lambda s: (s.road_kind, s.cls) == (3, cs.CLS_BACKGROUND)),
    ("act_zoom", lambda s: s.zoom == 1),
])
def test_clicking_a_button_changes_the_state(item_id, check):
    sess = _mk()
    _click_item(sess, item_id)
    assert check(sess), item_id


def test_clicking_a_class_button_leaves_unknown_brush():
    """选了类别就离开"不能判断"画笔——否则会继续往 label 里写 255。"""
    sess = _mk()
    _click_item(sess, "unk_blurred")
    assert sess.unknown_kind == 2
    _click_item(sess, "cls_line")
    assert (sess.cls, sess.unknown_kind) == (cs.CLS_LINE, 0)


def test_active_highlight_follows_the_clicked_button():
    sess = _mk()
    _click_item(sess, "tool_curve")
    _click_item(sess, "unk_occluded")
    active = {b.item.id for b in sess.layout().buttons if sess.is_active(b.item)}
    assert active == {"tool_curve", "unk_occluded"}


def test_momentary_actions_never_look_selected():
    """撤回/保存这类动作点完不该常亮（否则看不出当前画笔是哪个）。"""
    sess = _mk()
    for item in at.ITEMS:
        if item.momentary:
            assert not sess.is_active(item), item.id


def test_undo_button_restores_the_last_stroke():
    sess = _mk(cls=cs.CLS_LINE, brush=3)
    _click_item(sess, "tool_pen")
    _drag(sess, (10, 20), (40, 20))
    after = sess.label.copy()
    assert after.any()
    _click_item(sess, "act_undo")
    assert not sess.label.any()


def test_clear_button_wipes_the_label_and_is_undoable():
    sess = _mk(cls=cs.CLS_LINE, brush=3)
    _drag(sess, (10, 20), (40, 20))
    _click_item(sess, "act_clear")
    assert not sess.label.any()
    _click_item(sess, "act_undo")
    assert sess.label.any()


def test_save_and_prev_buttons_drive_the_frame():
    saved: list = []
    sess = _mk(3, cls=cs.CLS_LINE, brush=3,
               on_save=lambda fi, rgb, src, lab, unk, rt, ident:
               saved.append(fi))
    _click_item(sess, "act_next")
    assert saved == [0] and sess.fi == 1
    _click_item(sess, "act_prev")
    assert sess.fi == 0


def test_save_on_the_last_frame_stops_instead_of_wrapping():
    saved: list = []
    sess = _mk(1, on_save=lambda *a: saved.append(a[0]))
    assert sess.activate("act_next") == "finish"
    assert saved == [0] and sess.fi == 0


# ---------------------------------------------------------------------------
# 坐标映射：工具栏不能被画进 label，缩放不能错位
# ---------------------------------------------------------------------------


def test_toolbar_clicks_never_paint_into_the_label():
    sess = _mk(cls=cs.CLS_LINE, brush=6)
    for item in at.ITEMS:
        _click_item(sess, item.id)
        assert not sess.label.any(), item.id


def test_clicking_below_the_band_maps_to_row_zero():
    sess = _mk(cls=cs.CLS_LINE, brush=2)
    z = sess.zoom
    assert sess.image_xy(10, sess.band_h) == (10 // z, 0)
    assert sess.image_xy(10, sess.band_h - 1) is None
    assert sess.image_xy(10, sess.band_h + H * z) is None          # 图像下边界外


@pytest.mark.parametrize("zoom", [1, 2])
def test_zoom_keeps_the_same_pixel_under_the_cursor(zoom):
    """同一个像素在 1x/2x 下点它，落点必须一致（差 1 px 就是错位）。"""
    hit = {}
    for z in (zoom,):
        sess = _mk(cls=cs.CLS_LINE, brush=2, zoom=z)
        if sess.zoom != z:
            sess.toggle_zoom()
        _click_item(sess, "tool_pen")
        _press(sess, (30, 20))
        _release(sess, (30, 20))
        ys, xs = np.nonzero(sess.label)
        hit[z] = (int(xs.mean()), int(ys.mean()))
    assert abs(hit[zoom][0] - 30) <= 1 and abs(hit[zoom][1] - 20) <= 1


def test_zoom_toggle_relayouts_the_toolbar_to_the_new_width():
    sess = _mk(w=200)
    wide = sess.layout().width
    sess.toggle_zoom()
    assert sess.layout().width == wide // 2
    assert sess.band_h > 0


# ---------------------------------------------------------------------------
# 直线：必须笔直
# ---------------------------------------------------------------------------


def test_straight_drag_is_exactly_straight():
    sess = _mk(cls=cs.CLS_LINE, brush=4, tool="straight")
    a, b = (12, 30), (70, 30)
    _drag(sess, a, b)
    pix = _painted(sess.label == cs.CLS_LINE)
    assert len(pix) > 0
    assert _max_dist_to_segment(pix, a, b) <= at.brush_thickness(sess.brush) / 2 + 1
    cols = {}
    for x, y in pix:
        cols.setdefault(int(x), set()).add(int(y))
    inner = [cols[c] for c in range(a[0] + 3, b[0] - 2) if c in cols]
    assert len(inner) > 40
    assert all(rows == inner[0] for rows in inner), "水平直线的每列行集合必须一致"


def test_straight_click_then_click_equals_a_drag():
    """点两下画线：结果必须和拖出来的一模一样（同一条线段）。"""
    a, b = (14, 12), (60, 44)
    drag = _mk(cls=cs.CLS_LINE, brush=3, tool="straight")
    _drag(drag, a, b)
    clicks = _mk(cls=cs.CLS_LINE, brush=3, tool="straight")
    _click_at(clicks, a)
    _click_at(clicks, b)
    assert np.array_equal(drag.label, clicks.label)


def test_diagonal_line_stays_within_the_brush_band():
    sess = _mk(cls=cs.CLS_LINE, brush=5, tool="straight")
    a, b = (10, 10), (60, 50)
    _drag(sess, a, b)
    pix = _painted(sess.label == cs.CLS_LINE)
    assert _max_dist_to_segment(pix, a, b) <= at.brush_thickness(sess.brush) / 2 + 1


def test_a_click_without_a_second_point_paints_nothing():
    """直线工具只点一下不能落笔（没有终点），但可以放弃。"""
    sess = _mk(cls=cs.CLS_LINE, brush=3, tool="straight")
    _click_at(sess, (20, 20))
    assert not sess.label.any()
    assert sess.draft_points == [(20, 20)]
    sess.on_key(27)                                     # Esc
    assert sess.draft_points == []
    assert not sess.label.any()


def test_switching_tool_abandons_the_draft_without_touching_the_label():
    sess = _mk(cls=cs.CLS_LINE, brush=3, tool="straight")
    _click_at(sess, (20, 20))
    _click_item(sess, "tool_pen")
    assert sess.draft_points == [] and not sess.label.any()


# ---------------------------------------------------------------------------
# 曲线：穿过控制点、且比手绘原轨迹平滑得多
# ---------------------------------------------------------------------------


def test_curve_through_clicked_points_pass_through_them():
    sess = _mk(cls=cs.CLS_LINE, brush=3, tool="curve")
    pts = [(15, 40), (35, 15), (60, 40)]
    for p in pts:
        _click_at(sess, p)
    assert sess.on_key(13) is None                      # Enter
    pix = _painted(sess.label == cs.CLS_LINE)
    assert len(pix) > 0
    for p in pts:
        d = np.linalg.norm(pix - np.asarray(p, float), axis=1).min()
        assert d <= at.brush_thickness(sess.brush) / 2 + 2, (p, d)


def test_curve_preview_does_not_touch_the_label_until_enter():
    sess = _mk(cls=cs.CLS_LINE, brush=3, tool="curve")
    for p in [(15, 40), (35, 15), (60, 40)]:
        _click_at(sess, p)
    _move_to(sess, (70, 30))
    assert not sess.label.any()
    assert sess.on_key(13) is None
    assert sess.label.any()


def test_curve_needs_two_points_before_enter_commits():
    sess = _mk(cls=cs.CLS_LINE, brush=3, tool="curve")
    _click_at(sess, (30, 20))
    sess.on_key(13)
    assert not sess.label.any()
    assert sess.draft_points == [(30, 20)]


def test_freehand_curve_is_smoothed_not_interpolated_verbatim():
    """自由手绘：等距化 + 去抖 + 抽稀 + Catmull-Rom 之后拐角要降一个量级。

    判据不是"看起来平滑"：40 点、±2 px 抖动的手绘轨迹相邻段方向差本来就大
    （实测 2.32 rad），只做"抽稀 + Catmull-Rom"时仍有 1.9-2.2 rad（抽稀容差
    去不掉同量级的手抖，等于没平滑）。2026-09-26 标定：加窗口 5 的滑动平均后
    降到 0.12 rad。这条测试同时钉死相对（< 1/3）与绝对（< 0.30 rad）两条界，
    去掉去抖那一步必然失败。
    """
    raw = _arc(40, jitter=2.0)
    smooth = at.smooth_path(raw)
    assert len(smooth) >= 3
    assert _turn_rad(smooth) < 0.30
    assert _turn_rad(smooth) < _turn_rad(raw) / 3.0


def test_freehand_smoothing_keeps_the_shape_it_was_given():
    """平滑不许把形状改坏：相对理想弧线的偏离要有界（实测 2.1 px）。"""
    clean = _arc(40)
    smooth = at.smooth_path(clean)
    assert _max_dev_to_polyline(smooth, clean) <= 3.0


def test_freehand_smoothing_does_not_depend_on_event_spacing():
    """手快时事件能差 20 px：等距化之后，稀疏输入的平滑结果必须同样平滑。

    不对间距做等距化时，滑动平均的权重落在"事件密度"上——手快的地方被
    过度平均、手慢的地方几乎不平滑。
    """
    dense = _arc(40)
    sparse = _arc(40)[::10] + [_arc(40)[-1]]
    dense_turn = _turn_rad(at.smooth_path(dense))
    sparse_turn = _turn_rad(at.smooth_path(sparse))
    assert dense_turn < 0.30 and sparse_turn < 0.30
    assert abs(dense_turn - sparse_turn) < 0.15


def test_freehand_drag_commits_a_curved_stroke():
    """拖一条弧形，落笔后要有像素落在弧顶附近，而不是退化成直线。"""
    sess = _mk(cls=cs.CLS_LINE, brush=3, tool="curve")
    arc = [(15 + 1.6 * i, 45 - 25 * np.sin((1.6 * i) / 60.0 * np.pi))
           for i in range(36)]
    _drag_path(sess, arc)
    pix = _painted(sess.label == cs.CLS_LINE)
    assert len(pix) > 0
    apex = np.asarray(arc[len(arc) // 2], float)
    assert np.linalg.norm(pix - apex, axis=1).min() <= 6.0


def test_curve_control_points_interpolate_not_approximate():
    """Catmull-Rom 穿过控制点（不是"靠近就好"）。"""
    pts = [(0, 0), (20, 30), (50, 10), (80, 40)]
    curve = at.catmull_rom(pts)
    for p in pts:
        d = np.linalg.norm(curve - np.asarray(p, float), axis=1).min()
        assert d <= 2.0, (p, d)


def test_undo_pops_curve_control_points_before_strokes():
    sess = _mk(cls=cs.CLS_LINE, brush=3, tool="curve")
    _click_at(sess, (15, 40))
    _click_at(sess, (35, 15))
    assert len(sess.draft_points) == 2
    sess.on_key(ord("u"))
    assert len(sess.draft_points) == 1
    sess.on_key(ord("u"))
    assert sess.draft_points == []
    sess.on_key(ord("u"))                                # 空草稿上撤回：无副作用
    assert not sess.label.any()


# ---------------------------------------------------------------------------
# 未知类别 + 键盘兼容（旧工作流不能坏）
# ---------------------------------------------------------------------------


def test_unknown_brush_writes_ignore_pixels_with_a_reason():
    sess = _mk(brush=3)
    _click_item(sess, "unk_occluded")
    _drag(sess, (10, 20), (40, 20))
    assert (sess.label == cs.CLS_IGNORE).any()
    assert (sess.unk == 1).any()
    assert not (sess.label[10:40, 20:22] == cs.CLS_LINE).any()


def test_painting_after_unknown_clears_the_reason():
    """普通画笔要清掉 unk，否则导出会出现"255 但不知道原因"的帧。"""
    sess = _mk(cls=cs.CLS_LINE, brush=4)
    _click_item(sess, "unk_blurred")
    _drag(sess, (10, 20), (40, 20))
    assert (sess.unk > 0).any()
    _click_item(sess, "cls_line")
    _drag(sess, (10, 20), (40, 20))
    assert not (sess.unk > 0).any()
    assert not (sess.label == cs.CLS_IGNORE).any()


@pytest.mark.parametrize("key,check", [
    (ord("1"), lambda s: (s.cls, s.unknown_kind) == (cs.CLS_LINE, 0)),
    (ord("2"), lambda s: (s.cls, s.unknown_kind) == (cs.CLS_ROAD, 0)),
    (ord("3"), lambda s: (s.cls, s.unknown_kind) == (cs.CLS_BACKGROUND, 0)),
    (ord("4"), lambda s: s.unknown_kind == 1),
    (ord("5"), lambda s: s.unknown_kind == 2),
    (ord("6"), lambda s: s.unknown_kind == 3),
    (ord("7"), lambda s: (s.road_kind, s.cls) == (1, cs.CLS_ROAD)),
    (ord("8"), lambda s: (s.road_kind, s.cls) == (2, cs.CLS_ROAD)),
    (ord("9"), lambda s: (s.road_kind, s.cls) == (3, cs.CLS_BACKGROUND)),
    (ord("l"), lambda s: s.tool == "straight"),
    (ord("v"), lambda s: s.tool == "curve"),
    (ord("p"), lambda s: s.tool == "pen"),
    (ord("f"), lambda s: s.tool == "bucket"),
    (ord("b"), lambda s: s.tool == "bucket"),
    (ord("z"), lambda s: s.zoom == 1),
])
def test_legacy_and_new_keys_still_work(key, check):
    sess = _mk()
    sess.on_key(key)
    if key == ord("b"):                                  # 再按一次切回 pen
        assert sess.tool == "bucket"
        sess.on_key(key)
        check = lambda s: s.tool == "pen"                # noqa: E731
    assert check(sess)


def test_key_letters_match_the_buttons_on_screen():
    """按钮上印的键就是真正生效的键（一张表，不许两边漂开）。"""
    for item in at.ITEMS:
        if not item.key:
            continue
        sess = _mk()
        sess.on_key(ord(item.key))
        if item.group in ("class", "unknown"):
            assert sess.active_item_id() == item.id, item.id
        elif item.group == "tool":
            assert sess.tool == at.TOOL_IDS[item.id], item.id


def test_q_key_quits_and_left_arrow_steps_back():
    sess = _mk(3)
    sess.on_key(ord("a"))
    assert sess.fi == 0
    sess.load_frame(2)
    sess.on_key(81)                                      # 左方向键
    assert sess.fi == 1
    assert sess.on_key(ord("q")) == "quit"


# ---------------------------------------------------------------------------
# 帧缓存与保存回调（旧的"回上一帧不丢修改"行为）
# ---------------------------------------------------------------------------


def test_unsaved_edits_survive_going_back_and_forth():
    sess = _mk(2, cls=cs.CLS_LINE, brush=3)
    _drag(sess, (10, 20), (40, 20))
    first = sess.label.copy()
    _click_item(sess, "act_next")
    assert not sess.label.any()
    _click_item(sess, "act_prev")
    assert np.array_equal(sess.label, first)


def test_undo_stack_resets_per_frame():
    sess = _mk(2, cls=cs.CLS_LINE, brush=3)
    _drag(sess, (10, 20), (40, 20))
    assert sess.undo_depth == 1
    sess.load_frame(1)
    assert sess.undo_depth == 0
    sess.load_frame(0)
    assert sess.undo_depth == 0


def test_save_callback_receives_the_frame_and_its_identity():
    seen: list = []
    sess = _mk(2, cls=cs.CLS_LINE, brush=3,
               on_save=lambda fi, rgb, src, lab, unk, rt, ident:
               seen.append((fi, src, lab.any(), rt.max(), ident.get("source_id"))))
    _drag(sess, (10, 20), (40, 20))
    sess.on_key(ord("s"))
    assert seen == [(0, 0, True, 0, "s")]
    assert sess.fi == 1


def test_canvas_is_band_plus_image_and_never_empty():
    sess = _mk(cls=cs.CLS_LINE, brush=3, tool="curve")
    _click_at(sess, (20, 20))
    _move_to(sess, (40, 30))
    canvas = sess.canvas()
    assert canvas.shape[0] == sess.band_h + H * sess.zoom
    assert canvas.shape[1] == W * sess.zoom
    assert canvas.dtype == np.uint8
    # 工具栏那一条必须有内容（按钮画上去了），不是纯背景色
    band = canvas[:sess.band_h]
    assert int(band.max()) > 0
    assert len(np.unique(band.reshape(-1, 3), axis=0)) > 3


def test_label_for_may_return_a_pair_for_prefilled_frames():
    """预填/续标：label_for 可以同时给出 unknown_kind 数组。"""
    def _label_for(rgb, idx):
        lab = np.zeros(rgb.shape[:2], np.uint8)
        unk = np.zeros(rgb.shape[:2], np.uint8)
        lab[5:10, 5:10] = cs.CLS_ROAD
        unk[5:10, 5:10] = 2
        return lab, unk

    sess = _mk(label_for=_label_for)
    assert (sess.label == cs.CLS_ROAD).any()
    assert (sess.unk == 2).any()


# ---------------------------------------------------------------------------
# 路型（沥青/碎石/路肩）：与普通路面分开，像素类别按驾驶约束定
# ---------------------------------------------------------------------------


def test_roadtype_is_its_own_group_right_after_the_classes():
    """三个路型自成一组：方案要求两种道路类型分别标记，不能混成 road 一类。"""
    ids = [it.id for it in at.ITEMS]
    assert at.GROUP_ORDER.index("roadtype") == at.GROUP_ORDER.index("class") + 1
    assert ids.index("rt_asphalt") > ids.index("cls_erase")
    assert [it.id for it in at.ITEMS if it.group == "roadtype"] == [
        "rt_asphalt", "rt_gravel", "rt_shoulder"]
    assert sorted(at.ROAD_TYPE_NAMES.values()) == ["asphalt", "gravel", "shoulder"]


@pytest.mark.parametrize("item_id,pixel_cls,rt", [
    ("rt_asphalt", cs.CLS_ROAD, 1),
    ("rt_gravel", cs.CLS_ROAD, 2),
    ("rt_shoulder", cs.CLS_BACKGROUND, 3),
])
def test_road_type_paints_its_pixel_class_and_records_the_material(
        item_id, pixel_cls, rt):
    """沥青与纯土路算路面；**路肩不算路面**（约束：有铺装时土肩不得算道路）。"""
    sess = _mk(brush=4)
    _click_item(sess, item_id)
    assert (sess.road_kind, sess.cls) == (rt, pixel_cls)
    _drag(sess, (10, 20), (40, 20))
    band = np.s_[18:23, 12:38]
    assert (sess.label[band] == pixel_cls).all()
    assert (sess.paint.road_type[band] == rt).all()
    assert not (sess.paint.unknown[band] > 0).any()


def test_two_road_types_coexist_in_one_frame():
    """同帧里沥青与碎石必须能分别标出，而不是都变成同一个"路面"。"""
    sess = _mk(brush=3)
    _click_item(sess, "rt_asphalt")
    _drag(sess, (10, 12), (60, 12))
    _click_item(sess, "rt_gravel")
    _drag(sess, (10, 40), (60, 40))
    rt = sess.paint.road_type
    assert (rt[10:15, 12:58] == 1).all(), "上半幅应为沥青"
    assert (rt[38:43, 12:58] == 2).all(), "下半幅应为碎石"
    assert (sess.label[10:15, 12:58] == cs.CLS_ROAD).all()
    assert (sess.label[38:43, 12:58] == cs.CLS_ROAD).all()
    counts = at.road_type_counts(rt)
    assert counts["asphalt"] > 0 and counts["gravel"] > 0
    assert counts["shoulder"] == 0


def test_selecting_a_class_or_an_unknown_brush_leaves_the_road_type():
    sess = _mk()
    _click_item(sess, "rt_gravel")
    assert sess.road_kind == 2
    _click_item(sess, "cls_line")
    assert (sess.road_kind, sess.cls) == (0, cs.CLS_LINE)
    _click_item(sess, "rt_shoulder")
    assert (sess.road_kind, sess.cls) == (3, cs.CLS_BACKGROUND)
    _click_item(sess, "unk_blurred")
    assert (sess.road_kind, sess.unknown_kind) == (0, 2)


def test_unknown_brush_clears_the_road_type_column():
    """255 的像素不该还留着路型，否则导出会出现"忽略但知道材质"的怪字段。"""
    sess = _mk(brush=4)
    _click_item(sess, "rt_gravel")
    _drag(sess, (10, 20), (40, 20))
    assert (sess.paint.road_type > 0).any()
    _click_item(sess, "unk_blurred")
    _drag(sess, (10, 20), (40, 20))
    assert not (sess.paint.road_type > 0).any()
    assert (sess.label == cs.CLS_IGNORE).any()


def test_a_shoulder_stroke_is_visible_on_the_canvas():
    """路肩写的是背景，必须单独着色——否则"画了看不见"，复核人会以为没生效。"""
    sess = _mk(brush=5)
    before = sess.canvas().copy()
    _click_item(sess, "rt_shoulder")
    _drag(sess, (10, 30), (60, 30))
    after = sess.canvas()
    assert (sess.paint.road_type == 3).any()
    assert not (sess.label == cs.CLS_ROAD).any()
    band = sess.band_h
    changed = (before[band:] != after[band:]).any(axis=2)
    assert int(changed.sum()) > 100, "路肩画完画面上没有变化"


def test_every_button_has_a_name_in_both_languages():
    """中文与英文名都不能缺：缺中文名会显示空白，缺英文名就没有退回方案。"""
    for it in at.ITEMS:
        assert it.label.strip(), it.id
        assert it.ascii.strip(), it.id
    assert at.button_text(at.ITEM_BY_ID["cls_road"]) == "路面(2)"
    assert at.button_text(at.ITEM_BY_ID["cls_road"], variant="ascii") == "Road(2)"
