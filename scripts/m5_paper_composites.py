"""正文组合图：把 26 张单图按主题拼成 7 张跨栏组合图（单图仍全部保留在补充材料）。

为什么：审稿意见要求正文压到 6–8 张图，且双栏下单图文字偏小。组合图跨栏（HTML 用
``div.wide``、LaTeX 用 ``figure*``），每格宽度≈单栏宽，因此格内文字大小不变。

输出：``logs/paper_composites/figc{1..7}_*.png``（英文图）与
``logs/paper_composites_zh/``（中文图）。数据仍来自各自的已记录产物。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EN_SRC = ROOT / "logs" / "paper_figures"
ZH_SRC = ROOT / "logs" / "paper_figures_zh"
EN_OUT = ROOT / "logs" / "paper_composites"
ZH_OUT = ROOT / "logs" / "paper_composites_zh"

#: 组合图定义：key -> (文件名, [面板单图], 每行格数)
COMPOSITES = {
    "c1": ("figc1_design_contract", ["fig31_pipeline", "fig30_counting_contract",
                                     "fig33_protocol_matrix", "fig34_final_set_flow"], 2),
    "c2": ("figc2_definition_acceptance", ["fig1_gate_matrix", "fig18_r3_per_seed",
                                           "fig11_recall_scopes", "fig10_identity_scopes"], 2),
    "c3": ("figc3_postproc_composition", ["fig2_boundary_map", "fig14_appearance_gate",
                                          "fig12_lateral_scan", "fig3_dose_response"], 2),
    "c4": ("figc4_one_shot_confirmation", ["fig4_final_confirm", "fig29_final_set_composition",
                                           "fig42_overlap_audit"], 2),
    "c5": ("figc5_closed_loop", ["fig8_closed_loop_tradeoff", "fig22_gate_heatmap",
                                 "fig28_metric_scaling"], 2),
    "c6": ("figc6_deadlock_mechanism", ["fig32_arbitration_ladder", "fig5_deadlock_anatomy",
                                        "fig23_reason_hist", "fig7_lane_gate_evidence"], 2),
    "c7": ("figc7_rigour_audits", ["fig39_scene_density", "fig40_negative_audit",
                                   "fig36_timing_retest", "fig38_lane_geometry"], 2),
}

PANEL_W = 1560          # 每格统一宽度（像素）
GAP = 26                # 格间距
LABEL_H = 46            # 每格顶部留给 (a)(b) 标号的高度
BG = "white"


def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_one(name: str, panels: list[str], cols: int, src: Path, out: Path) -> Path:
    imgs = []
    for p in panels:
        f = src / f"{p}.png"
        if not f.is_file():
            raise SystemExit(f"缺面板图: {f}")
        im = Image.open(f).convert("RGB")
        h = max(1, round(im.height * PANEL_W / im.width))
        imgs.append(im.resize((PANEL_W, h), Image.LANCZOS))
    rows = (len(imgs) + cols - 1) // cols
    row_h = [max(imgs[r * cols + c].height for c in range(cols) if r * cols + c < len(imgs))
             + LABEL_H for r in range(rows)]
    W = cols * PANEL_W + (cols + 1) * GAP
    H = sum(row_h) + (rows + 1) * GAP
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    font = _font(34)
    y = GAP
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i >= len(imgs):
                break
            x = GAP + c * (PANEL_W + GAP)
            im = imgs[i]
            canvas.paste(im, (x, y + LABEL_H))
            draw.text((x + 6, y + 4), f"({chr(97 + i)})", fill="black", font=font)
        y += row_h[r] + GAP
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{name}.png"
    canvas.save(dst, dpi=(300, 300))
    return dst


def main() -> int:
    made = []
    for key, (name, panels, cols) in COMPOSITES.items():
        made.append(build_one(name, panels, cols, EN_SRC, EN_OUT))
        made.append(build_one(name, panels, cols, ZH_SRC, ZH_OUT))
    for p in made:
        print(f"[composite] {p} ({p.stat().st_size // 1024} KB)")
    print(f"[composite] 共 {len(made)} 张（7 组 × 中英）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
