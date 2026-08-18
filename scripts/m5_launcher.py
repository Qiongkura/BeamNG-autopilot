"""M5 自动驾驶控制台 - 软件式启动界面 + 环境信息显示 (EID).

一个桌面 GUI：一键启动游戏 / 一键启动自动驾驶助手 / 一键开关自动驾驶，
并实时显示环境信息（速度、目标速度、油门/刹车/转向、G 力、障碍物、
最近障碍距离、传感器状态、路线与车辆俯视图）。

用法::

    python scripts/m5_launcher.py

界面按钮会通过 ``logs/autopilot_ctl.json`` 命令桥（beamng_autopilot.bridge）
向正在运行的 m5_autopilot 助手发送命令，等价于在游戏里按 F9/F10/F11。
"""

from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

import psutil
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.bridge import ControlBridge
from beamng_autopilot.telemetry import read_live
from beamngpy.beamng.filesystem import determine_binary

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# ---------------------------------------------------------------- theme
BG = "#0d1117"
PANEL = "#161b22"
PANEL2 = "#1c2128"
BORDER = "#30363d"
TEXT = "#e6edf3"
DIM = "#8b949e"
ACCENT = "#2f81f7"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
CYAN = "#39c5cf"
FONT = "Microsoft YaHei UI"
MONO = "Consolas"

MODE_CN = {
    "IDLE": "待机",
    "ENDED": "会话结束",
    "follow": "巡航",
    "detour": "避障中",
    "deform": "微调绕行",
    "blocked": "阻塞停车",
    "cruise": "巡航",
}
MODE_COLOR = {
    "IDLE": DIM,
    "ENDED": CYAN,
    "follow": GREEN,
    "detour": AMBER,
    "deform": CYAN,
    "blocked": RED,
    "cruise": GREEN,
}
BEAMNG_PROCS = set(config.BEAMNG_PROCESS_NAMES)


def _beamng_running() -> bool:
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] in BEAMNG_PROCS:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


class BirdView(tk.Canvas):
    """车辆为中心的俯视图：路线、障碍物框、目标点。"""

    RADIUS_M = 45.0

    def render(self, data: dict | None) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 60:
            return
        if data is None or not data.get("pos"):
            self.create_text(w // 2, h // 2, text="等待遥测数据...",
                             fill=DIM, font=(FONT, 11))
            return
        pos = data["pos"]
        heading = float(data.get("heading") or 0.0)
        extra = data.get("extra") or {}
        cx, cy = w // 2, h // 2
        scale = min(w, h) / (2.0 * self.RADIUS_M)
        cos_h, sin_h = math.cos(heading), math.sin(heading)

        def to_canvas(x: float, y: float) -> tuple[float, float]:
            dx, dy = x - pos[0], y - pos[1]
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            return cx - lat * scale, cy - fwd * scale

        for r in (10.0, 20.0, 30.0, self.RADIUS_M):
            rr = r * scale
            self.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                             outline=BORDER, width=1)
        self.create_line(cx - self.RADIUS_M * scale, cy,
                         cx + self.RADIUS_M * scale, cy,
                         fill=BORDER, width=1)
        self.create_line(cx, cy - self.RADIUS_M * scale,
                         cx, cy + self.RADIUS_M * scale,
                         fill=BORDER, width=1)

        # 道路标线：附近道路中心线（车道示意），导航路线盖在其上
        for poly in extra.get("roads") or []:
            if len(poly) < 2:
                continue
            pts = []
            for rx, ry in poly:
                px, py = to_canvas(float(rx), float(ry))
                pts.append((max(0, min(w, px)), max(0, min(h, py))))
            self.create_line(pts, fill="#232a34", width=5, smooth=True)
            self.create_line(pts, fill="#4b5563", width=1, smooth=True)

        # 检测到的道路标线：白色实线 / 黄色虚线
        for mk in extra.get("markings") or []:
            poly = mk.get("poly") or []
            if len(poly) < 2:
                continue
            pts = []
            for mx, my in poly:
                px, py = to_canvas(float(mx), float(my))
                pts.append((max(0, min(w, px)), max(0, min(h, py))))
            fill = "#ffd166" if mk.get("color") == "yellow" else "#e6e6e6"
            dash = (8, 6) if mk.get("kind") == "dashed" else ()
            self.create_line(pts, fill=fill, width=3, smooth=True,
                             dash=dash)

        # 导航路线
        rte = extra.get("rte") or []
        if len(rte) >= 2:
            pts = []
            for x, y in rte:
                px, py = to_canvas(x, y)
                pts.append((max(0, min(w, px)), max(0, min(h, py))))
            self.create_line(pts, fill=GREEN, width=3, smooth=True)
            gx, gy = to_canvas(rte[-1][0], rte[-1][1])
            self.create_oval(gx - 7, gy - 7, gx + 7, gy + 7,
                             fill=GREEN, outline="")

        # 障碍物框
        OB_COLORS = {
            "car": ("#12355c", "#2f81f7"),
            "vehicle": ("#12355c", "#2f81f7"),
            "truck": ("#4a2c17", "#db6d28"),
            "bus": ("#4a3d10", "#d29922"),
            "person": ("#5a1f1f", "#f85149"),
            "pedestrian": ("#5a1f1f", "#f85149"),
            "raycast": ("#3d2b56", "#a371f7"),
            "wall": ("#3a3440", "#8b949e"),
            "scenario": ("#334155", "#94a3b8"),
        }
        for box in extra.get("boxes") or []:
            if len(box) < 4:
                continue
            bx, by, hw, hh = box[0], box[1], box[2], box[3]
            label = str(box[4]) if len(box) > 4 else ""
            fill_c, out_c = OB_COLORS.get(label, ("#12355c", ACCENT))
            corners = [(bx + hw, by + hh), (bx - hw, by + hh),
                       (bx - hw, by - hh), (bx + hw, by - hh)]
            pts = [to_canvas(x, y) for x, y in corners]
            self.create_polygon(pts, fill=fill_c, outline=out_c, width=2)
            # 附近车辆状态：每个障碍物下方标出与本车的直线距离
            dist_m = math.hypot(bx - pos[0], by - pos[1])
            if dist_m <= self.RADIUS_M + 2:
                ty = min(h - 8, max(10, max(p[1] for p in pts) + 10))
                self.create_text(cx, ty, text=f"{dist_m:.0f}m",
                                 fill=out_c, font=(FONT, 8))

        # 本车箭头（车头朝上）
        tip = (cx, cy - 16)
        base_l = (cx - 10, cy + 12)
        base_r = (cx + 10, cy + 12)
        self.create_polygon(tip, base_l, base_r, fill=TEXT, outline="")
        self.create_text(cx, h - 8, text=f"范围 {self.RADIUS_M:.0f} m",
                         fill=DIM, font=(FONT, 8))


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.ctl = ControlBridge()
        self.game_proc: subprocess.Popen | None = None
        self.m5_proc: subprocess.Popen | None = None
        self.log_q: queue.Queue[str] = queue.Queue()
        self.auto_on = False
        self.live = None
        self.m5_log: TextIO | None = None
        self.m5_log_path: Path | None = None
        self.m5_run_dir: Path | None = None
        self._m5_run_meta: dict | None = None

        self.map_var = tk.StringVar(value=config.DEFAULT_MAP)
        self.veh_var = tk.StringVar(value=config.DEFAULT_VEHICLE)
        self.speed_var = tk.StringVar(value="60")
        self.attach_var = tk.BooleanVar(value=True)
        self.markers_var = tk.BooleanVar(value=True)
        self.nav_world_var = tk.BooleanVar(value=False)
        self._nav_world_sent: bool | None = None
        self.runtime_var = tk.StringVar(value=config.RUNTIME_MODE)
        self._chart_img = None
        self._chart_pil = None
        self._chart_summary = None
        self._last_session_mtime = None
        self._chart_visible = False

        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(120, self.poll)

    # ------------------------------------------------------------ UI
    def _build(self) -> None:
        root = self.root
        root.title("BeamNG 自动驾驶控制台 · M5")
        root.geometry("1220x820")
        root.minsize(1080, 720)
        root.configure(bg=BG)

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=(FONT, 10), padding=(10, 7),
                        background=PANEL2, foreground=TEXT,
                        borderwidth=0, focusthickness=0)
        style.map("TButton",
                  background=[("active", "#232a33"), ("pressed", "#2d3741")])
        style.configure("Accent.TButton", background=ACCENT,
                        foreground="#ffffff", font=(FONT, 10, "bold"))
        style.map("Accent.TButton",
                  background=[("active", "#3a8bfd"),
                              ("pressed", "#1f6feb"),
                              ("disabled", "#30363d")],
                  foreground=[("disabled", "#8b949e")])
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT,
                        font=(FONT, 10))
        style.map("TCheckbutton",
                  background=[("active", PANEL)],
                  foreground=[("active", TEXT)])
        style.configure("TEntry", fieldbackground="#0d1117",
                        foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER)

        # 顶栏
        header = tk.Frame(root, bg=PANEL, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="BEAMNG 自动驾驶控制台",
                 bg=PANEL, fg=TEXT, font=(FONT, 14, "bold")).pack(
            side="left", padx=18, pady=16)
        self.chip_game = tk.Label(header, bg=PANEL, fg=DIM,
                                  font=(FONT, 10))
        self.chip_m5 = tk.Label(header, bg=PANEL, fg=DIM,
                                font=(FONT, 10))
        self.chip_auto = tk.Label(header, bg=PANEL, fg=DIM,
                                  font=(FONT, 10))
        for chip in (self.chip_game, self.chip_m5, self.chip_auto):
            chip.pack(side="right", padx=6)
        self.chip_auto.pack(side="right", padx=(6, 18))

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True)

        # 左：控制面板
        left = tk.Frame(body, bg=PANEL, width=290)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_left(left)

        # 右：EID + 俯视图
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self._build_eid(right)

        # 底部：日志
        log_panel = tk.Frame(root, bg=PANEL, height=150)
        log_panel.pack(fill="x", side="bottom")
        log_panel.pack_propagate(False)
        self._build_log(log_panel)

    def _panel(self, parent, title: str):
        box = tk.Frame(parent, bg=PANEL2, highlightbackground=BORDER,
                       highlightthickness=1)
        tk.Label(box, text=title, bg=PANEL2, fg=DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        return box

    def _build_left(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="控 制", bg=PANEL, fg=DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=14, pady=(14, 4))

        self.btn_game = ttk.Button(parent, text="① 启动游戏", style="TButton",
                                   command=self.launch_game)
        self.btn_m5 = ttk.Button(parent, text="② 启动自动驾驶助手",
                                 style="TButton", command=self.start_m5)
        self.btn_toggle = ttk.Button(parent, text="③ 一键启动自动驾驶 (F9)",
                                     style="Accent.TButton",
                                     command=self.toggle_autopilot)
        self.btn_route = ttk.Button(parent, text="抓取导航路线 (F10)",
                                    style="TButton", command=self.grab_route)
        self.btn_clear = ttk.Button(parent, text="清空路线 (F11)",
                                    style="TButton", command=self.clear_route)
        self.btn_stop = ttk.Button(parent, text="停止并退出助手",
                                   style="TButton", command=self.stop_m5)
        for btn in (self.btn_game, self.btn_m5, self.btn_toggle,
                    self.btn_route, self.btn_clear, self.btn_stop):
            btn.pack(fill="x", padx=14, pady=4)

        tk.Label(parent, text="设 置", bg=PANEL, fg=DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w",
                                              padx=14, pady=(18, 4))
        for label, var in (("限速 (km/h)", self.speed_var),
                           ("地图 (新场景)", self.map_var),
                           ("车型 (新场景)", self.veh_var)):
            row = tk.Frame(parent, bg=PANEL)
            row.pack(fill="x", padx=14, pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=TEXT,
                     font=(FONT, 10), width=16, anchor="w").pack(side="left")
            entry = ttk.Entry(row, textvariable=var, width=14)
            if label.startswith("限速"):
                apply_btn = ttk.Button(row, text="应用", style="TButton",
                                       width=6, command=self.apply_speed)
                apply_btn.pack(side="right")
                entry.pack(side="right", padx=(0, 6))
                entry.bind("<Return>", lambda e: self.apply_speed())
            else:
                entry.pack(side="right")
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=14, pady=3)
        tk.Label(row, text="运行时", bg=PANEL, fg=TEXT,
                 font=(FONT, 10), width=16, anchor="w").pack(side="left")
        ttk.Combobox(row, textvariable=self.runtime_var,
                     values=("auto", "steam", "tech"),
                     state="readonly", width=12).pack(side="right")
        ttk.Checkbutton(parent, text="接管运行中的游戏 (--attach)",
                        variable=self.attach_var).pack(anchor="w",
                                                       padx=14, pady=8)
        ttk.Checkbutton(parent, text="显示红/黄标记点（起点黄球、终点红球）",
                        variable=self.markers_var).pack(anchor="w",
                                                        padx=14, pady=(0, 8))
        ttk.Checkbutton(parent, text="世界内显示导航线（地图上始终显示）",
                        variable=self.nav_world_var,
                        command=self.toggle_nav_world).pack(anchor="w",
                                                            padx=14,
                                                            pady=(0, 8))

        self.lbl_hint = tk.Label(
            parent, text="提示：\n1. 先点「启动游戏」进入地图\n"
                         "2. 启动助手后按 M 选目的地\n"
                         "3. 一键自动驾驶即可接管车辆",
            bg=PANEL, fg=DIM, font=(FONT, 9), justify="left",
            anchor="nw", wraplength=255)
        self.lbl_hint.pack(fill="both", expand=True, padx=14, pady=10)

    def _build_eid(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg=BG)
        top.pack(fill="x")

        # 速度
        speed_box = self._panel(top, "速 度")
        speed_box.pack(side="left", fill="y", padx=(0, 8))
        inner = tk.Frame(speed_box, bg=PANEL2)
        inner.pack(padx=16, pady=(0, 8))
        self.lbl_speed = tk.Label(inner, text="--", bg=PANEL2, fg=TEXT,
                                  font=("Segoe UI", 44, "bold"))
        self.lbl_speed.pack(side="left")
        unit = tk.Frame(inner, bg=PANEL2)
        unit.pack(side="left", padx=(8, 0))
        tk.Label(unit, text="km/h", bg=PANEL2, fg=DIM,
                 font=(FONT, 10)).pack(anchor="w")
        self.lbl_target = tk.Label(unit, text="目标 --", bg=PANEL2, fg=CYAN,
                                   font=(FONT, 10))
        self.lbl_target.pack(anchor="w", pady=(6, 0))
        self.lbl_cruise = tk.Label(unit, text="限速 --", bg=PANEL2, fg=DIM,
                                   font=(FONT, 10))
        self.lbl_cruise.pack(anchor="w", pady=(4, 0))

        # 模式徽章
        self.lbl_mode = tk.Label(speed_box, text="待机", bg=PANEL2,
                                 fg=DIM, font=(FONT, 11, "bold"),
                                 padx=14, pady=4)
        self.lbl_mode.pack(anchor="w", padx=12, pady=(0, 10))

        # 踏板 / 转向
        bars_box = self._panel(top, "油 门 / 刹 车 / 转 向")
        bars_box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.cv_throttle = tk.Canvas(bars_box, width=300, height=26,
                                     bg=PANEL2, highlightthickness=0)
        self.cv_throttle.pack(padx=10, pady=(4, 4))
        self.cv_brake = tk.Canvas(bars_box, width=300, height=26,
                                  bg=PANEL2, highlightthickness=0)
        self.cv_brake.pack(padx=10, pady=4)
        self.cv_steer = tk.Canvas(bars_box, width=300, height=26,
                                  bg=PANEL2, highlightthickness=0)
        self.cv_steer.pack(padx=10, pady=(4, 10))

        # G 力表
        g_box = self._panel(top, "G 力")
        g_box.pack(side="left", fill="y")
        self.cv_g = tk.Canvas(g_box, width=170, height=170, bg=PANEL2,
                              highlightthickness=0)
        self.cv_g.pack(padx=10, pady=(2, 10))

        # 统计网格
        stats = self._panel(parent, "环境信息 (EID)")
        stats.pack(fill="x", pady=(8, 8))
        grid = tk.Frame(stats, bg=PANEL2)
        grid.pack(fill="x", padx=10, pady=(0, 10))
        self.stat_labels: dict[str, tk.Label] = {}
        items = [("障碍物", "obs"), ("最近障碍", "obs_d"),
                 ("视觉目标", "vis"), ("传感器", "sen"),
                 ("路线点", "route"), ("距目标", "goal_d"),
                 ("标线", "lanes"),
                 ("已运行", "run_t"), ("航向", "heading"),
                 ("阻塞原因", "blk"), ("地图", "map"), ("车型", "veh")]
        for i, (cn, key) in enumerate(items):
            cell = tk.Frame(grid, bg=PANEL2)
            cell.grid(row=i // 4, column=i % 4, sticky="w",
                      padx=(4, 24), pady=3)
            tk.Label(cell, text=cn, bg=PANEL2, fg=DIM,
                     font=(FONT, 9)).pack(anchor="w")
            val = tk.Label(cell, text="--", bg=PANEL2, fg=TEXT,
                           font=(FONT, 12, "bold"))
            val.pack(anchor="w")
            self.stat_labels[key] = val

        # 最后会话遥测：会话结束自动弹出图表
        session_box = tk.Frame(parent, bg=PANEL2,
                               highlightbackground=BORDER,
                               highlightthickness=1)
        head = tk.Frame(session_box, bg=PANEL2)
        head.pack(fill="x", padx=10, pady=(6, 2))
        tk.Label(head, text="最后会话遥测", bg=PANEL2, fg=DIM,
                 font=(FONT, 9, "bold")).pack(side="left")
        self.btn_chart = ttk.Button(head, text="显示图表", style="TButton",
                                    command=self.toggle_chart)
        self.btn_chart.pack(side="right")
        session_box.pack(fill="x", pady=(0, 8))
        self.cv_chart = tk.Canvas(session_box, height=170, bg="#0b0f14",
                                  highlightbackground=BORDER,
                                  highlightthickness=1)
        self.cv_chart.bind("<Configure>", lambda e: self._redraw_chart())
        self.lbl_session = tk.Label(session_box, text="暂无会话数据",
                                    bg=PANEL2, fg=DIM, font=(FONT, 9))
        self.lbl_session.pack(anchor="w", padx=12, pady=(0, 8))

        # 俯视图
        view_box = self._panel(parent, "俯 视 图")
        view_box.pack(fill="both", expand=True)
        self.bird = BirdView(view_box, bg="#0b0f14",
                             highlightbackground=BORDER,
                             highlightthickness=1)
        self.bird.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    def _build_log(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="日 志", bg=PANEL, fg=DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=12, pady=(6, 0))
        wrap = tk.Frame(parent, bg=PANEL)
        wrap.pack(fill="both", expand=True, padx=12, pady=(2, 8))
        self.txt_log = tk.Text(wrap, bg="#0b0f14", fg=TEXT,
                               font=(MONO, 9), wrap="word",
                               insertbackground=TEXT, relief="flat",
                               state="disabled")
        sb = ttk.Scrollbar(wrap, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt_log.pack(side="left", fill="both", expand=True)

    # --------------------------------------------------------- actions
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", line + "\n")
        if float(self.txt_log.index("end-1c").split(".")[0]) > 400:
            self.txt_log.delete("1.0", "100.0")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def launch_game(self) -> None:
        runtime_mode = config.resolve_launch_runtime(self.runtime_var.get())
        try:
            binary = determine_binary(config.runtime_home(runtime_mode))
        except Exception as exc:
            self.log(f"未找到游戏：{exc}")
            return
        if _beamng_running():
            self.log("游戏已在运行，直接启动助手即可")
            return
        cmd = [str(binary)]
        if runtime_mode != "tech":
            cmd.append("-nosteam")
        cmd += ["-tcom", "-tport",
                str(config.runtime_port(runtime_mode)), "-console"]
        runtime_user = config.runtime_user(runtime_mode)
        if runtime_user:
            cmd += ["-userpath", str(runtime_user)]
        try:
            self.game_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            self.log(f"已启动游戏（runtime={runtime_mode}，带 tcom 通信端口），"
                     "请进入地图...")
        except Exception as exc:
            self.log(f"启动游戏失败：{exc}")

    def _m5_alive(self) -> bool:
        return self.m5_proc is not None and self.m5_proc.poll() is None

    def _m5_running_elsewhere(self):
        """Return the PID of another m5_autopilot.py process, or None."""
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                cmd = proc.info["cmdline"] or []
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if not any("m5_autopilot.py" in c for c in cmd):
                continue
            pid = proc.pid
            if self.m5_proc is not None and pid == self.m5_proc.pid:
                continue
            return pid
        return None

    def start_m5(self) -> None:
        if self._m5_alive():
            self.log("自动驾驶助手已在运行")
            return
        dup = self._m5_running_elsewhere()
        if dup is not None:
            self.log(f"检测到另一个助手进程 (PID {dup})，请先关闭多余的启动器窗口")
            return
        try:
            speed_ms = float(self.speed_var.get()) / 3.6
            speed_ms = max(1.0, min(60.0, speed_ms))
        except ValueError:
            speed_ms = 60.0 / 3.6
        args = [sys.executable,
                str(Path(__file__).resolve().parent / "m5_autopilot.py"),
                "--speed", f"{speed_ms:.2f}",
                "--map", self.map_var.get().strip() or config.DEFAULT_MAP,
                "--vehicle",
                self.veh_var.get().strip() or config.DEFAULT_VEHICLE,
                "--runtime",
                self.runtime_var.get().strip() or config.RUNTIME_MODE,
                "--port",
                str(config.runtime_port(self.runtime_var.get()))]
        if self.attach_var.get():
            args.append("--attach")
        if not self.markers_var.get():
            args.append("--no-markers")
        args += ["--nav-world",
                 "1" if self.nav_world_var.get() else "0"]
        try:
            self.m5_proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        except Exception as exc:
            self.log(f"启动助手失败：{exc}")
            self.m5_proc = None
            return
        self._open_m5_log(args, self.m5_proc.pid)
        self._m5_log_line(
            f"[launcher] started pid={self.m5_proc.pid} "
            f"args={' '.join(args)}")
        threading.Thread(target=self._reader, args=(self.m5_proc,),
                         daemon=True).start()
        self.log("自动驾驶助手已启动：游戏里按 M 选目的地，然后一键自动驾驶")

    def _reader(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                text = line.rstrip()
                self.log_q.put(text)
                self._m5_log_line(text)
        except Exception as exc:
            self._m5_log_line(f"[launcher] reader error: {exc}")
        finally:
            try:
                rc = proc.poll()
            except Exception:
                rc = None
            self._m5_log_line(f"[launcher] helper exited rc={rc}")
            self._close_m5_log(rc)

    def _open_m5_log(self, args: list[str], pid: int) -> None:
        """Give every launcher-spawned run a persistent manual-run record."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = config.LOGS_DIR / "manual_runs" / f"run_{stamp}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.m5_log_path = run_dir / "autopilot.log"
            self.m5_log = self.m5_log_path.open(
                "w", encoding="utf-8", errors="replace", buffering=1)
            self.m5_run_dir = run_dir
            self._m5_run_meta = {
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pid": pid,
                "args": [str(a) for a in args],
                "status": "running",
            }
            (run_dir / "run.json").write_text(
                json.dumps(self._m5_run_meta, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as exc:
            if self.m5_log is not None:
                try:
                    self.m5_log.close()
                except Exception:
                    pass
            self.m5_log = None
            self.m5_log_path = None
            self.m5_run_dir = None
            self._m5_run_meta = None
            self.log(f"创建手动运行日志失败：{exc}，本次不写日志文件")

    def _m5_log_line(self, text: str) -> None:
        if self.m5_log is None:
            return
        try:
            self.m5_log.write(f"{time.strftime('%H:%M:%S')} {text}\n")
            self.m5_log.flush()
        except Exception:
            pass

    def _close_m5_log(self, exit_code: int | None = None) -> None:
        f = self.m5_log
        self.m5_log = None
        run_dir = self.m5_run_dir
        meta = self._m5_run_meta
        self.m5_log_path = None
        self.m5_run_dir = None
        self._m5_run_meta = None
        if meta is not None and run_dir is not None:
            meta["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta["exit_code"] = exit_code
            meta["status"] = "exited"
            try:
                (run_dir / "run.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            except Exception:
                pass
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    def _send(self, cmd: str, label: str) -> None:
        if not self._m5_alive():
            self.log("助手未运行，先点「② 启动自动驾驶助手」")
            return
        if self.ctl.send(cmd):
            self.log(f"已发送：{label}")
            self._m5_log_line(f"[launcher] sent cmd={cmd} ({label})")
        else:
            self.log("命令发送失败")

    def toggle_autopilot(self) -> None:
        self._send("autopilot",
                   "停止自动驾驶" if self.auto_on else "启动自动驾驶")

    def apply_speed(self) -> None:
        """Apply the UI speed limit to the running assistant immediately."""
        try:
            kph = float(self.speed_var.get())
            speed_ms = max(1.0, min(60.0, kph / 3.6))
        except ValueError:
            self.log("限速无效，请输入数字 (km/h)")
            return
        self.speed_var.set(f"{speed_ms * 3.6:.0f}")
        if self._m5_alive():
            if self.ctl.send("set_speed", speed_ms):
                self.log(f"已应用限速：{speed_ms * 3.6:.0f} km/h")
                self._m5_log_line(
                    f"[launcher] sent cmd=set_speed value={speed_ms:.2f} "
                    f"({speed_ms * 3.6:.0f} km/h)")
            else:
                self.log("限速命令发送失败")
        else:
            self.log("助手未运行，限速已保存，启动时生效")

    def grab_route(self) -> None:
        self._send("navroute", "抓取导航路线")

    def clear_route(self) -> None:
        self._send("clear", "清空路线")

    def toggle_nav_world(self) -> None:
        """Apply the in-world nav line checkbox to the running assistant."""
        visible = self.nav_world_var.get()
        if not self._m5_alive():
            self.log("世界导航线设置已保存，启动助手时生效")
            return
        if self.ctl.send("nav_world", 1.0 if visible else 0.0):
            self._nav_world_sent = visible
            self.log("已发送：世界导航线"
                     f"{'显示' if visible else '隐藏'}")
            self._m5_log_line(
                f"[launcher] sent cmd=nav_world value="
                f"{'1.0' if visible else '0.0'}")
        else:
            self.log("命令发送失败")

    def stop_m5(self) -> None:
        if self._m5_alive():
            self.ctl.send("quit")
            self._m5_log_line("[launcher] stop requested")
            self.log("已发送退出命令，等待助手安全停车退出...")
        else:
            self.log("助手未在运行")

    # ---------------------------------------------------------- refresh
    def poll(self) -> None:
        try:
            self._refresh_auto()
            self._drain_log()
            self._update_chips()
            self._update_eid()
            self._check_last_session()
        except Exception as exc:
            self.log(f"界面刷新错误：{exc}")
        self.root.after(120, self.poll)

    def _refresh_auto(self) -> None:
        """Read the autopilot on/off flag before the chips are drawn."""
        data = read_live()
        if data is not None:
            self.live = data
            self.auto_on = bool((data.get("extra") or {}).get("auto"))
            nav_state = (data.get("extra") or {}).get("nav_world")
            if self._m5_alive() and isinstance(nav_state, int) and (
                    self._nav_world_sent is None
                    or bool(nav_state) == self._nav_world_sent):
                self._nav_world_sent = None
                self.nav_world_var.set(bool(nav_state))

    def _drain_log(self) -> None:
        try:
            while True:
                line = self.log_q.get_nowait()
                self.log(line)
        except queue.Empty:
            pass

    def _update_chips(self) -> None:
        if _beamng_running():
            self.chip_game.configure(text="● 游戏：运行中", fg=GREEN)
        else:
            self.chip_game.configure(text="○ 游戏：未运行", fg=RED)
        if self._m5_alive():
            self.chip_m5.configure(text="● 助手：运行中", fg=GREEN)
        else:
            self.chip_m5.configure(text="○ 助手：未运行", fg=RED)
        if self.auto_on:
            self.chip_auto.configure(text="● 自动驾驶：ON", fg=GREEN)
        else:
            self.chip_auto.configure(text="○ 自动驾驶：OFF", fg=DIM)
        enabled = self._m5_alive()
        state = "normal" if enabled else "disabled"
        for btn in (self.btn_toggle, self.btn_route, self.btn_clear,
                    self.btn_stop):
            btn.configure(state=state)
        self.btn_toggle.configure(
            text=("③ 停止自动驾驶 (F9)" if self.auto_on
                  else "③ 一键启动自动驾驶 (F9)"))

    def _update_eid(self) -> None:
        data = read_live()
        if data is None:
            return
        self.live = data
        extra = data.get("extra") or {}
        speed = float(data.get("speed") or 0.0) * 3.6
        target = float(extra.get("target") or 0.0) * 3.6
        cruise = float(extra.get("cruise") or 0.0) * 3.6
        throttle = float(data.get("throttle") or 0.0)
        brake = float(data.get("brake") or 0.0)
        steer = float(data.get("steer") or 0.0)
        g_lat = float(data.get("g_lat") or 0.0)
        g_lon = float(data.get("g_lon") or 0.0)

        self.auto_on = bool(extra.get("auto"))
        mode = extra.get("mode", "IDLE")
        mode_cn = MODE_CN.get(mode, str(mode))
        if extra.get("creep"):
            mode_cn += " · 爬行"
        if extra.get("slip"):
            mode_cn += " · 打滑"
        if extra.get("hdg_g"):
            mode_cn += f" · 偏航{int(extra.get('hdg_dev') or 0)}°"
        self.lbl_mode.configure(text=mode_cn, fg=MODE_COLOR.get(mode, DIM))
        self.lbl_speed.configure(text=f"{speed:3.0f}")
        self.lbl_target.configure(
            text=f"目标 {target:3.0f}" if target > 0 else "目标 --")
        self.lbl_cruise.configure(
            text=f"限速 {cruise:3.0f}" if cruise > 0 else "限速 --")

        self._draw_bar(self.cv_throttle, throttle, GREEN, "油门")
        self._draw_bar(self.cv_brake, brake, RED, "刹车")
        self._draw_steer(self.cv_steer, steer)
        self._draw_g(self.cv_g, g_lat, g_lon)

        obs = extra.get("obs", 0)
        obs_d = extra.get("obs_d")
        vis = extra.get("vis", 0)
        sen = extra.get("sen", "--")
        route = extra.get("route", 0)
        lanes = extra.get("lanes", 0)
        goal_d = extra.get("goal_d")
        blk = extra.get("blk") or ""
        run_t = float(data.get("t") or 0.0)
        heading = data.get("heading")
        self.stat_labels["obs"].configure(
            text=str(obs),
            fg=AMBER if obs else TEXT)
        self.stat_labels["obs_d"].configure(
            text="--" if obs_d is None else f"{float(obs_d):.0f} m")
        self.stat_labels["vis"].configure(text=str(vis))
        self.stat_labels["sen"].configure(
            text=sen, fg=GREEN if sen == "OK" else RED)
        self.stat_labels["route"].configure(text=str(route))
        self.stat_labels["lanes"].configure(text=str(lanes))
        self.stat_labels["goal_d"].configure(
            text="--" if goal_d is None else f"{float(goal_d):.0f} m")
        self.stat_labels["blk"].configure(
            text=blk or "--",
            fg=RED if blk else TEXT)
        self.stat_labels["run_t"].configure(
            text=f"{int(run_t)} s" if run_t > 0 else "--")
        self.stat_labels["heading"].configure(
            text="--" if heading is None else f"{math.degrees(heading):.0f}°")
        env = extra.get("env") or {}
        self.stat_labels["map"].configure(text=env.get("map") or "--")
        self.stat_labels["veh"].configure(text=env.get("vehicle") or "--")
        # 自动获取实际地图 / 车型：只在设置仍是默认值时回填，避免覆盖
        # 用户手动输入的自定义值。
        if env.get("map") and self.map_var.get().strip() in (
                "", config.DEFAULT_MAP):
            self.map_var.set(str(env["map"]))
        if env.get("vehicle") and self.veh_var.get().strip() in (
                "", config.DEFAULT_VEHICLE):
            self.veh_var.set(str(env["vehicle"]))
        self.bird.render(data)

    def toggle_chart(self) -> None:
        """显示 / 收起最后会话的遥测图表面板。"""
        self._chart_visible = not self._chart_visible
        if self._chart_visible:
            self.cv_chart.pack(fill="x", padx=10, pady=(2, 4))
            self.btn_chart.configure(text="收起图表")
            self.root.after(50, self._redraw_chart)
        else:
            self.cv_chart.pack_forget()
            self.btn_chart.configure(text="显示图表")
        self._update_session_label()

    def _check_last_session(self) -> None:
        """Watch last_session.json; a new file pops the chart automatically."""
        p = config.LOGS_DIR / "telemetry" / "last_session.json"
        try:
            mtime = p.stat().st_mtime
        except OSError:
            self._last_session_mtime = None
            return
        if mtime == self._last_session_mtime:
            return
        self._last_session_mtime = mtime
        try:
            summary = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            self.lbl_session.configure(text=f"会话数据读取失败：{exc}")
            return
        self._chart_summary = summary
        self._chart_pil = None
        png = summary.get("png")
        if png and Path(png).exists() and _HAS_PIL:
            try:
                self._chart_pil = Image.open(png).copy()
            except Exception as exc:
                self.lbl_session.configure(text=f"遥测图加载失败：{exc}")
        if not self._chart_visible:
            self.toggle_chart()  # 会话结束自动弹出
        self._update_session_label()

    def _redraw_chart(self) -> None:
        if not self._chart_visible or self._chart_summary is None:
            return
        cw, ch = self.cv_chart.winfo_width(), self.cv_chart.winfo_height()
        self.cv_chart.delete("all")
        if cw < 40 or ch < 40:
            return
        if self._chart_pil is None:
            self.cv_chart.create_text(cw // 2, ch // 2, text="（无图像）",
                                      fill=DIM, font=(FONT, 10))
            return
        iw, ih = self._chart_pil.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        photo = ImageTk.PhotoImage(
            self._chart_pil.resize((nw, nh), Image.LANCZOS))
        self._chart_img = photo  # 保持引用，防止被 GC
        self.cv_chart.create_image(cw // 2, ch // 2, image=photo)

    def _update_session_label(self) -> None:
        s = self._chart_summary
        if s is None:
            self.lbl_session.configure(text="暂无会话数据")
            return
        dur = float(s.get("duration") or 0.0)
        mx = float(s.get("max_speed") or 0.0) * 3.6
        av = float(s.get("avg_speed") or 0.0) * 3.6
        tr = float(s.get("throttle_ratio") or 0.0) * 100.0
        br = float(s.get("brake_ratio") or 0.0) * 100.0
        self.lbl_session.configure(
            text=(f"时长 {dur:.0f}s  |  最高 {mx:.0f} km/h  |  "
                  f"平均 {av:.0f} km/h  |  油门占比 {tr:.0f}%  |  "
                  f"刹车占比 {br:.0f}%"))

    def _draw_bar(self, cv: tk.Canvas, value: float, color: str,
                  label: str) -> None:
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 40:
            return
        v = max(0.0, min(1.0, value))
        cv.create_rectangle(2, 2, w - 2, h - 2, outline=BORDER)
        if v > 0.01:
            cv.create_rectangle(4, 4, 4 + (w - 8) * v, h - 4,
                                fill=color, outline="")
        cv.create_text(8, h // 2, text=f"{label} {value:.2f}",
                       anchor="w", fill=TEXT, font=(FONT, 9, "bold"))

    def _draw_steer(self, cv: tk.Canvas, value: float) -> None:
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 40:
            return
        cv.create_rectangle(2, 2, w - 2, h - 2, outline=BORDER)
        cx = w // 2
        cv.create_line(cx, 4, cx, h - 4, fill=BORDER)
        off = max(-1.0, min(1.0, value)) * (w // 2 - 10)
        px = cx + off
        cv.create_rectangle(px - 6, 4, px + 6, h - 4, fill=CYAN,
                            outline="")
        cv.create_text(8, h // 2, text=f"转向 {value:+.2f}",
                       anchor="w", fill=TEXT, font=(FONT, 9, "bold"))

    def _draw_g(self, cv: tk.Canvas, g_lat: float, g_lon: float) -> None:
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 40:
            return
        cx, cy = w // 2, h // 2
        scale = 30.0
        for g, col in ((0.5, "#30363d"), (1.0, BORDER), (2.0, "#21262d")):
            r = g * scale
            cv.create_oval(cx - r, cy - r, cx + r, cy + r,
                           outline=col, width=1)
        cv.create_line(cx - 2.2 * scale, cy, cx + 2.2 * scale, cy,
                       fill=BORDER)
        cv.create_line(cx, cy - 2.2 * scale, cx, cy + 2.2 * scale,
                       fill=BORDER)
        dx = max(-2.0, min(2.0, g_lat)) * scale
        dy = -max(-2.0, min(2.0, g_lon)) * scale
        px, py = cx + dx, cy + dy
        cv.create_oval(px - 8, py - 8, px + 8, py + 8,
                       fill=AMBER, outline="")
        cv.create_text(cx, h - 10,
                       text=f"横 {g_lat:+.2f} g   纵 {g_lon:+.2f} g",
                       fill=DIM, font=(FONT, 8))

    # ------------------------------------------------------------ close
    def on_close(self) -> None:
        if self._m5_alive():
            self.ctl.send("quit")
            deadline = time.time() + 3.0
            while self._m5_alive() and time.time() < deadline:
                time.sleep(0.1)
            if self._m5_alive():
                try:
                    self.m5_proc.terminate()
                except Exception:
                    pass
        rc = self.m5_proc.poll() if self.m5_proc is not None else None
        self._m5_log_line("[launcher] console closing")
        self._close_m5_log(rc)
        self.root.destroy()


def main() -> None:
    # 单实例锁：防止重复打开多个启动器窗口，避免它们各自拉起
    # 重复的 m5_autopilot.py 进程，互相抢同一个游戏连接导致卡死。
    import uuid
    _lock = config.LOGS_DIR / "m5_launcher.lock"
    try:
        _lock.parent.mkdir(parents=True, exist_ok=True)
        _fd = open(_lock, "w")
        try:
            import msvcrt
            if msvcrt.locking(_fd.fileno(), msvcrt.LK_NBLCK, 1):
                _lock.unlink()
        except OSError:
            import tkinter.messagebox as _mb
            import sys as _sys
            _root = tk.Tk()
            _root.withdraw()
            _mb.showwarning(
                "启动器已运行",
                "检测到另一个启动器实例已打开。\n"
                "请先关闭现有的启动器窗口，避免多个自动驾驶助手重复连接。")
            _root.destroy()
            _sys.exit(1)
    except Exception:
        pass

    app = LauncherApp(tk.Tk())
    try:
        app.root.mainloop()
    finally:
        try:
            _fd.close()
        except Exception:
            pass
        try:
            _lock.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
