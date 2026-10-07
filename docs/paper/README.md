# T16 论文（英文版 + 中文版 + 补充材料 + 投稿源包）

**交付物**

| 文件 | 说明 |
|---|---|
| `T16_paper.pdf` | 英文版成品（13 页，双栏 IEEE 式；**正文 7 张跨栏组合图**） |
| `T16_paper_zh.pdf` | 中文版成品（15 页，图内文字中文） |
| `T16_supplement.pdf` | 补充材料（7 页，42 张图：S1–S16 原补充图 + S17–S42 正文组合图的单图版本） |
| `T16_supplement_zh.pdf` | 补充材料中文版（7 页） |
| `submission/` | **投稿源包**：四份 `.tex` + `figures/` + `figures_zh/` + `references.bib` + README（**未编译**：本机无 LaTeX 引擎） |
| `CLAIMS_20261007.md` | 冻结主张清单：主张 → 数值 → 来源文件 → 协议/种子/范围 → 状态 |
| `REVIEW_20261007.md` | 外部审查报告（本轮修正的依据） |
| `main.tex` / `main_zh.tex` / `supplement*.tex` | 与 `submission/` 同源（图目录为 `figures/`） |
| `paper.md` / `paper_zh.md` | 正文主源（26 张单图 → 7 张组合图；图号与引文编号由脚本生成） |
| `supplement.md` / `supplement_zh.md` | 补充材料主源（42 图） |
| `references.bib` | 15 条题录（9 篇中文按中文著录；朱威等已核对为《模式识别与人工智能》2021, 34(5): 434–445） |
| `build_paper.py` | 构建脚本（Markdown → LaTeX + HTML；组合图跨栏；`--lang zh` 出中文版） |

**图的结构**

正文 7 张组合图（跨栏，单格≈单栏宽，格内文字不缩小）：

| 组合图 | 格 | 主题 |
|---|---|---|
| 图 1 | (a)(b)(c)(d) | 研究设计与计数契约（流水线／契约／协议矩阵／确认流程，示意） |
| 图 2 | (a)(b)(c)(d) | 车为什么不动：仲裁阶梯／死锁解剖／仲裁原因 1383·512／车道接受证据 |
| 图 3 | (a)(b)(c)(d) | 严谨性与审计：场景筛选／负例认证／计时复测／标线几何 |
| 图 4 | (a)(b)(c)(d) | 验收结果的定义敏感性：门矩阵／逐种子／召回双范围／身份排序反转 |
| 图 5 | (a)(b)(c)(d) | 后处理与组成：边界图／外观门／横向扫描／剂量 |
| 图 6 | (a)(b)(c) | 一次性确认：按组指标与负例／封存组成／重叠审计 |
| 图 7 | (a)(b)(c) | 闭环：权衡／硬门清单／指标量纲 |

补充材料 42 张：S1–S16 为原补充图；**S17–S42 是正文七张组合图的单图版本**（原始尺寸，便于逐格查看）。

**重新构建（四步，无需安装 LaTeX）**

```pwsh
# 0) 中文图 + 组合图
.venv\Scripts\python.exe scripts\m5_figures_i18n.py          # 英文图脚本 -> 中文图
.venv\Scripts\python.exe scripts\m5_paper_composites.py      # 26 单图 -> 7 组合图（中英各一套）

# 1) 四个 Markdown 源 -> LaTeX + HTML
.venv\Scripts\python.exe docs\paper\build_paper.py --figs-src logs\paper_composites
.venv\Scripts\python.exe docs\paper\build_paper.py --md docs\paper\paper_zh.md `
    --tex docs\paper\main_zh.tex --html logs\paper_build\paper_zh.html --lang zh `
    --figs logs\paper_build\figures_zh --img-dir figures_zh --figs-src logs\paper_composites_zh
.venv\Scripts\python.exe docs\paper\build_paper.py --md docs\paper\supplement.md `
    --tex docs\paper\supplement.tex --html logs\paper_build\supplement.html --fig-prefix S `
    --figs-src logs\paper_figures
.venv\Scripts\python.exe docs\paper\build_paper.py --md docs\paper\supplement_zh.md `
    --tex docs\paper\supplement_zh.tex --html logs\paper_build\supplement_zh.html --lang zh `
    --fig-prefix S --figs logs\paper_build\figures_zh --img-dir figures_zh --figs-src logs\paper_figures_zh

# 2) HTML -> PDF（Edge headless，逐条执行，避免进程互锁）
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
& $edge --headless=new --disable-gpu --no-pdf-header-footer `
    --print-to-pdf="I:\projects\beamng-autopilot\docs\paper\T16_paper.pdf" `
    "file:///I:/projects/beamng-autopilot/logs/paper_build/paper.html"
# 中文正文与两份补充材料同理，只换 HTML 与输出名

# 3) 投稿源包 + 静态检查（本机无 LaTeX 引擎，只做静态检查）
.venv\Scripts\python.exe scripts\m5_build_submission.py
.venv\Scripts\python.exe scripts\m5_paper_texcheck.py docs\paper\submission\main.tex `
    docs\paper\submission\main_zh.tex docs\paper\submission\supplement.tex `
    docs\paper\submission\supplement_zh.tex
```

**排版自检**

- 图内文字拥挤：`scripts\m5_figures_qa.py`（42 图 × 2 语言，当前 0 条重叠）；
- LaTeX 静态检查：`scripts\m5_paper_texcheck.py`（环境配平／图存在／引文闭环／模板残留，当前 0 个问题）；
- 图注样式：`<figcaption>` 开标签曾在生成器里漏掉（题注按正文大小渲染），2026-10-07 修复。

## 已按你的回答填入

| 项 | 现状 |
|---|---|
| A1 作者信息 | 袁哲宇（Zheyu Yuan），华南农业大学，广州，中国 |
| A2 基金 | 未获外部资助 |
| A3 作者贡献 | 唯一作者，CRediT 全角色 |
| A4 利益冲突 | 无 |
| A5 数据可得性 | **已改为与事实一致**：代码与稿件源公开；记录产物在本地 `logs/`（未跟踪），公开归档尚不可用 |
| A6 致谢 | BeamNG.tech、父母、朋友、DeepSeek 与 OpenCode |
| B7–B9 口径 | 三级凭证阶梯；人工标注由作者本人对自己录制的仿真数据完成；负结果全量披露 |
| B12 参考文献 | 15 条全保留；9 篇中文按中文著录 |
| B13 图内文字 | 英文版英文图；中文版中文图 |
| B14/B15 | 中英分版；补充材料已出（中英各一份） |
| 摘要/关键词 | 英文摘要 **250 词**（目标 150–250）、关键词 **6** 个 |

## 仍需你定/核

1. **ORCID**（免费注册 orcid.org；期刊常要）。
2. **目标期刊与模板**：现为 IEEEtran；投 Elsevier 系需换模板并补 Graphical Abstract。
3. **投稿源包的实际编译**：本机没有 LaTeX 引擎，静态检查通过 ≠ 编译通过；建议在装好 TeX 的机器上编一次 `submission/main.tex`。
4. **公开归档**：若要把 `logs/` 里的判定 JSON／scorecard／manifest 发到 Zenodo，我可以生成最小归档清单。
