"""M5 决策层实时可视化：直接读 autopilot 发布的遥测（live.json）。

与 lane_state_view（独立跑一遍感知前端）不同，本窗口显示的是
autopilot 进程内部**实际运行**的决策状态：感知融合来源、planner
模式、速度决策链、交通规则、安全保护与控制输出。每帧从
logs/telemetry/live.json 读取最新快照（原子覆写，安全轮询）。

用法（autopilot 运行中，另开终端）:
    .venv\Scripts\python.exe scripts\m5_decision_view.py

按键: q/ESC 退出, s 保存当前画面到 logs/telemetry/decision_*.png
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.telemetry import read_live

W, H = 1400, 860
BG = (16, 18, 24)
PANEL = (26, 30, 38)
TEXT = (225, 228, 235)
DIM = (140, 146, 158)
GREEN = (70, 210, 120)
YELLOW = (80, 190, 255)
RED = (70, 80, 230)
CYAN = (230, 200, 80)
MAGENTA = (200, 120, 255)
GRID = (44, 50, 60)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BEV_W, BEV_H = 680, 620
BEV_SCALE = 16.0  # px per metre


def _put(img, text, x, y, color=TEXT, scale=0.5, thick=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color, label, val_txt):
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, 1)
    f = max(0.0, min(1.0, float(frac)))
    if f > 0.01:
        cv2.rectangle(img, (x + 1, y + 1),
                      (x + int((w - 2) * f), y + h - 1), color, -1)
    _put(img, label, x + 6, y + h // 2 + 4, DIM, 0.42)
    _put(img, val_txt, x + w - 6, y + h // 2 + 4,
         color if f > 0.01 else DIM, 0.42, 1)


def _bev(img, data):
    """鸟瞰：路线 / 障碍物 / 标线 / ego。"""
    ex = data.get("pos") or [0.0, 0.0, 0.0]
    hdg = float(data.get("heading") or 0.0)
    extra = data.get("extra") or {}
    x0, y0 = 20, 90

    def to_screen(wx, wy):
        dx, dy = wx - ex[0], wy - ex[1]
        rx = dx * math.cos(hdg) + dy * math.sin(hdg)
        ry = -dx * math.sin(hdg) + dy * math.cos(hdg)
        return int(x0 + BEV_W / 2 + rx * BEV_SCALE), \
            int(y0 + BEV_H / 2 - ry * BEV_SCALE)

    cv2.rectangle(img, (x0, y0), (x0 + BEV_W, y0 + BEV_H), PANEL, -1)
    cv2.rectangle(img, (x0, y0), (x0 + BEV_W, y0 + BEV_H), GRID, 1)
    _put(img, "BEV  (route / obstacles / markings)", x0 + 10, y0 + 22, DIM)

    # 距离网格
    for r in (10, 20, 30):
        cx = x0 + BEV_W // 2
        cy = y0 + BEV_H // 2
        rr = int(r * BEV_SCALE)
        cv2.circle(img, (cx, cy), rr, GRID, 1)

    # 路线
    rte = extra.get("rte") or []
    if len(rte) >= 2:
        pts = [to_screen(p[0], p[1]) for p in rte]
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, CYAN, 2)
        # 起点/终点标记
        cv2.circle(img, pts[0], 5, GREEN, -1)
        cv2.circle(img, pts[-1], 5, RED, -1)
    # 标线（世界 polyline）
    for mk in (extra.get("markings") or [])[:12]:
        poly = mk.get("poly") or []
        if len(poly) < 2:
            continue
        col = (0, 230, 230) if mk.get("color") == "yellow" else (200, 200, 210)
        pts = [to_screen(p[0], p[1]) for p in poly]
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, col, 1)
    # 障碍物
    for b in (extra.get("boxes") or [])[:14]:
        bx, by, hw, hh, label = b[0], b[1], b[2], b[3], str(b[4])
        axis = b[5] if len(b) > 5 else None
        half_len = float(b[6]) if len(b) > 6 and b[6] else 0.0
        half_thick = float(b[7]) if len(b) > 7 and b[7] else 0.0
        if axis is not None and half_len > 0:
            ax, ay = float(axis[0]), float(axis[1])
            vx, vy = -ay, ax
            pts = []
            for s1, s2 in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                pts.append(to_screen(
                    bx + ax * half_len * s1 + vx * half_thick * s2,
                    by + ay * half_len * s1 + vy * half_thick * s2))
            cv2.polylines(img, [np.asarray(pts, np.int32)], True,
                          MAGENTA, 1)
        else:
            p1 = to_screen(bx - hw, by - hh)
            p2 = to_screen(bx + hw, by + hh)
            cv2.rectangle(img, p1, p2, MAGENTA, 1)
        cx, cy = to_screen(bx, by)
        _put(img, label, cx + 4, cy - 4, MAGENTA, 0.4)
    # ego
    cv2.arrowedLine(img, (x0 + BEV_W // 2, y0 + BEV_H // 2 + 16),
                    (x0 + BEV_W // 2, y0 + BEV_H // 2 - 16), GREEN, 3,
                    cv2.LINE_AA, tipLength=0.4)
    cv2.circle(img, (x0 + BEV_W // 2, y0 + BEV_H // 2), 6, GREEN, -1)


def _panel(img, data):
    """右侧决策链面板。"""
    extra = data.get("extra") or {}
    x, y0 = 720, 90
    w = W - x - 20

    def row(yy, label, value, color=TEXT, scale=0.5):
        _put(img, label, x + 12, yy, DIM, scale)
        _put(img, value, x + 250, yy, color, scale, 1)

    # 模式徽章
    mode = str(extra.get("mode") or "?")
    auto = int(extra.get("auto") or 0)
    if auto:
        badge = f"AUTOPILOT ON  [{mode}]"
        bcol = GREEN if mode not in ("blocked",) else RED
    else:
        badge = "AUTOPILOT OFF"
        bcol = DIM
    cv2.rectangle(img, (x, y0), (x + w, y0 + 44), PANEL, -1)
    _put(img, badge, x + 14, y0 + 28, bcol, 0.7, 2)
    _put(img, f"v={float(data.get('speed') or 0):5.1f} m/s  "
              f"target={float(extra.get('target') or 0):5.1f}  "
              f"goal={float(extra.get('goal_d') or 0):5.1f} m",
         x + 330, y0 + 28, TEXT, 0.55)

    y = y0 + 64
    # 1) 感知
    cv2.rectangle(img, (x, y), (x + w, y + 118), PANEL, -1)
    _put(img, "PERCEPTION", x + 12, y + 20, CYAN, 0.55, 1)
    lane_src = str(extra.get("lane_src") or "-")
    conf = float(extra.get("lane_conf") or 0)
    row(y + 42, "sensors", str(extra.get("sen") or "?"))
    row(y + 66, "lane src", lane_src,
        GREEN if "lidar" in lane_src else TEXT)
    row(y + 90, "lane conf", f"{conf:.2f}  paired={int(extra.get('lane_paired') or 0)}"
                              f"  jump={int(extra.get('lane_jump') or 0)}")
    row(y + 112, "obstacles", f"obs={extra.get('obs')}  "
                              f"nearest={float(extra.get('obs_d') or 999):.0f}m  "
                              f"vis={extra.get('vis')}/{extra.get('vconf')}  "
                              f"lidar_hits={extra.get('lidar_hits')}")
    y += 128

    # 2) 规划
    cv2.rectangle(img, (x, y), (x + w, y + 118), PANEL, -1)
    _put(img, "PLANNER", x + 12, y + 20, CYAN, 0.55, 1)
    row(y + 42, "plan mode", f"{mode}  lane_mode={extra.get('plan_mode')}  "
                             f"offset={float(extra.get('plan_offset') or 0):+.2f}m")
    blk = str(extra.get("blk") or "")
    row(y + 66, "blocker", blk, RED if blk else DIM)
    row(y + 90, "route", f"{extra.get('route')} pts  "
                         f"sharp={int(extra.get('sharp') or 0)}")
    row(y + 112, "hdg dev", f"{float(extra.get('hdg_dev') or 0):.1f} deg  "
                            f"guard={int(extra.get('hdg_g') or 0)}")
    y += 128

    # 3) 速度决策链
    cv2.rectangle(img, (x, y), (x + w, y + 150), PANEL, -1)
    _put(img, "SPEED CHAIN", x + 12, y + 20, CYAN, 0.55, 1)
    cruise = float(extra.get("cruise") or 0)
    corner = float(extra.get("corner") or 0)
    obslim = extra.get("obslim")
    obslim_txt = "None" if obslim is None else f"{float(obslim):.1f}"
    desired = float(extra.get("desired") or 0)
    target = float(extra.get("target") or 0)
    rule_reason = str(extra.get("rule_reason") or "-")
    chain = (f"cruise {cruise:.1f} -> corner {corner:.1f} -> "
             f"obslim {obslim_txt} -> desired {desired:.1f} -> "
             f"target {target:.1f}")
    row(y + 42, "chain", chain)
    row(y + 66, "rule", f"{rule_reason}  "
                        f"limit={extra.get('speed_limit')} km/h",
        YELLOW if rule_reason not in ("-", "None") else DIM)
    sig = str(extra.get("signal_name") or "")
    sig_a = str(extra.get("signal_action") or 0)
    sig_d = extra.get("signal_dist")
    row(y + 90, "signal",
        f"{sig or '-'} action={sig_a} "
        f"dist={'-' if sig_d is None else f'{float(sig_d):.0f}m'}",
        YELLOW if sig_a not in ("0", "None") else DIM)
    row(y + 114, "comfort", f"creep={int(extra.get('creep') or 0)}  "
                            f"slip={int(extra.get('slip') or 0)}  "
                            f"rev={float(extra.get('rev') or 0):.2f}m  "
                            f"rev_g={int(extra.get('rev_g') or 0)}")
    _put(img, "target speed ramp:", x + 12, y + 140, DIM, 0.42)
    _bar(img, x + 170, y + 130, w - 190, 16,
         target / max(cruise, 1e-6), GREEN, "", f"{target:.1f}")
    y += 160

    # 4) 控制输出
    cv2.rectangle(img, (x, y), (x + w, y + 110), PANEL, -1)
    _put(img, "OUTPUT", x + 12, y + 20, CYAN, 0.55, 1)
    thr = float(data.get("throttle") or 0)
    brk = float(data.get("brake") or 0)
    st = float(data.get("steer") or 0)
    _bar(img, x + 12, y + 30, w - 24, 18, thr, GREEN, "throttle",
         f"{thr:.2f}")
    _bar(img, x + 12, y + 54, w - 24, 18, brk, RED, "brake", f"{brk:.2f}")
    _bar(img, x + 12, y + 78, w - 24, 18, abs(st), YELLOW, "steer",
         f"{st:+.2f}")
    _put(img, f"g_lat={float(data.get('g_lat') or 0):+.2f}  "
              f"g_lon={float(data.get('g_lon') or 0):+.2f}",
         x + 12, y + 102, DIM, 0.42)


def _history_plot(img, hist: collections.deque):
    """底部速度时序图：actual / desired / target。"""
    x0, y0, w, h = 20, 740, W - 40, 100
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), GRID, 1)
    _put(img, "speed history (m/s)  — green=actual  cyan=desired  "
              "yellow=target", x0 + 10, y0 + 18, DIM)
    if len(hist) < 2:
        return
    arr = np.asarray(list(hist), dtype=float)
    vmax = max(1.0, float(np.nanmax(arr)) * 1.2)
    n = len(arr)
    for k, (col, idx) in enumerate(((GREEN, 0), (CYAN, 1), (YELLOW, 2))):
        pts = []
        for i in range(n):
            px = x0 + 10 + int((w - 20) * i / max(1, n - 1))
            py = y0 + h - 14 - int((h - 30) * arr[i, idx] / vmax)
            pts.append((px, py))
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, col,
                      1 if k else 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="决策层实时可视化")
    ap.add_argument("--file", default=None, help="live.json 路径")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="刷新率 Hz（默认 10）")
    ap.add_argument("--max-samples", type=int, default=900,
                    help="时序图最大样本数（默认 900）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    hist: collections.deque = collections.deque(maxlen=args.max_samples)
    cv2.namedWindow("Decision Layer", cv2.WINDOW_NORMAL)
    print("[decview] 等待 autopilot 遥测 (logs/telemetry/live.json) ...")

    while True:
        t0 = time.time()
        img = np.full((H, W, 3), BG, np.uint8)
        _put(img, "M5 DECISION LAYER  (live.json)",
             20, 36, CYAN, 0.7, 2)
        _put(img, "q/ESC quit   s save", W - 240, 36, DIM, 0.45)

        data = read_live(args.file)
        if data is None or not (data.get("extra") or {}):
            _put(img, "no telemetry yet - start m5_autopilot.py first",
                 60, 300, DIM, 0.7)
        else:
            hist.append((float(data.get("speed") or 0.0),
                         float((data.get("extra") or {}).get("desired") or 0.0),
                         float((data.get("extra") or {}).get("target") or 0.0)))
            _bev(img, data)
            _panel(img, data)
        _history_plot(img, hist)

        cv2.imshow("Decision Layer", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            p = config.LOGS_DIR / "telemetry" / (
                f"decision_{time.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(str(p), img)
            print(f"[decview] saved -> {p}")

        rem = 1.0 / args.rate - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)
        try:
            if cv2.getWindowProperty("Decision Layer",
                                     cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()