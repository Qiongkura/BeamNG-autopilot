# T16 论文（英文版 + 中文版 + 补充材料 + 投稿源包）

**交付物**

| 文件 | 说明 |
|---|---|
| `T16_paper.pdf` | 英文版成品（**19 页**，双栏 IEEE 式；正文 **7 张满栏单图**，图内文字按原尺寸可读） |
| `T16_paper_zh.pdf` | 中文版成品（**17 页**，图内文字中文） |
| `T16_supplement.pdf` | 补充材料（**21 页**，42 张满栏图：S1–S16 补充图 + S17–S42 历史排版的独立诊断／示意图） |
| `T16_supplement_zh.pdf` | 补充材料中文版（**22 页**） |
| `submission/` | **投稿源包**：四份 `.tex` + `figures/` + `figures_zh/` + `references.bib` + README（**未编译**：本机无 LaTeX 引擎） |
| `CLAIMS_20261007.md` | 冻结主张清单：主张 → 数值 → 来源文件 → 协议/种子/范围 → 状态 |
| `REVIEW_20261007.md` | 外部审查报告（本轮修正的依据） |
| `main.tex` / `main_zh.tex` / `supplement*.tex` | 与 `submission/` 同源（图目录为 `figures/`） |
| `paper.md` / `paper_zh.md` | 正文主源（26 张单图 → 7 张组合图；图号与引文编号由脚本生成） |
| `supplement.md` / `supplement_zh.md` | 补充材料主源（42 图） |
| `references.bib` | 20 条题录，核验范围与缺失字段见 `REFS_VERIFICATION_20261007.md` |
| `build_paper.py` | 构建脚本（Markdown → LaTeX + HTML；组合图跨栏；`--lang zh` 出中文版） |

**图的结构**

正文 **7 张满栏单图**（每主题一张承重图；满栏≈7 in，6.5 pt 图内文字显示约 6.5 pt，读者不必放大）：

| 图 | 内容 |
|---|---|
| 1 | 计数契约（示意）——冻结计数定义、覆盖率是候选参考可判率 |
| 2 | 两套协议定义的门指标与冻结阈值（开发池／有限类别池） |
| 3 | 外观门消融——唯一用标签召回换精度的开关（对应表 V） |
| 4 | 冻结协议 + 配对种子的剂量响应（对应表 VI） |
| 5 | 一次性确认：按组指标与负例一侧诊断 |
| 6 | 闭环权衡：压中心线 vs 传感器车道来源率 |
| 7 | 死锁解剖：规划穿越落在 4 m 阈值内、96% 帧静止 |

补充材料 **42 张满栏单图**：S1–S16 为补充图；**S17–S42 保留历史排版的 26 张独立图**，其中 7 张用于现正文
（含正文未放的 19 张：流水线、协议矩阵、确认流程、逐种子、召回/身份双范围、边界图、横向扫描、
封存组成、重叠审计、硬门清单、指标量纲、仲裁阶梯、铺装门证据、场景密度、计时复测、标线几何等）。

正文里对被移到补充材料的内容一律写「补充材料」（不写具体图号：补充材料编号由脚本生成）。

**重新构建（四步，无需安装 LaTeX）**

```pwsh
# 0) 中文图（英文图脚本 -> 中文图；组合图脚本已不用，正文改为满栏单图）
.venv\Scripts\python.exe scripts\m5_figures_i18n.py

# 1) 四个 Markdown 源 -> LaTeX + HTML
.venv\Scripts\python.exe docs\paper\build_paper.py --figs-src logs\paper_composites
.venv\Scripts\python.exe docs\paper\build_paper.py --md docs\paper\paper_zh.md `
    --tex docs\paper\main_zh.tex --html logs\paper_build\paper_zh.html --lang zh `
    --figs logs\paper_build\figures_zh --img-dir figures_zh --figs-src logs\paper_composites_zh
.venv\Scripts\python.exe docs\paper\build_paper.py --md docs\paper\supplement.md `
    --tex docs\paper\supplement.tex --html logs\paper_build\supplement.html --fig-prefix S `
    --figs-src logs\paper_figures --wide-all
.venv\Scripts\python.exe docs\paper\build_paper.py --md docs\paper\supplement_zh.md `
    --tex docs\paper\supplement_zh.tex --html logs\paper_build\supplement_zh.html --lang zh `
    --fig-prefix S --figs logs\paper_build\figures_zh --img-dir figures_zh --figs-src logs\paper_figures_zh

# 2) 两份正文与两份补充材料 -> 新 PDF（独立 Edge 配置、超时与新文件检查）
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/m5_paper_export.ps1 -IncludeSupplement
# 默认不加此参数只重建两份正文；每次导出后均需核对页数与版面

# 3) 证据归档（限定结果与对应源码 + SHA-256 清单；缺项记入清单并返回失败）
.venv\Scripts\python.exe scripts\m5_paper_evidence_archive.py

# 4) 投稿源包 + 静态检查（本机无 LaTeX 引擎，只做静态检查）
.venv\Scripts\python.exe scripts\m5_build_submission.py
.venv\Scripts\python.exe scripts\m5_paper_texcheck.py docs\paper\submission\main.tex `
    docs\paper\submission\main_zh.tex docs\paper\submission\supplement.tex `
    docs\paper\submission\supplement_zh.tex
```

**排版自检**

- 图内文字拥挤：`scripts\m5_figures_qa.py`；2026-10-09 本轮重绘 fig17／fig21／fig22／fig32 的中英图检查无重叠，不追认全部历史图已重新执行该检查。
- LaTeX 静态检查：`scripts\m5_paper_texcheck.py`（环境配平／图存在／引文闭环／模板残留，当前 0 个问题）；
- 图标识必须唯一：重复会由构建器拒绝；补充材料使用唯一的 S1–S42，防止旧编号误指不同证据。
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
| B12 参考文献 | 20 条；题录核验记录见 `REFS_VERIFICATION_20261007.md` |
| B13 图内文字 | 英文版英文图；中文版中文图 |
| B14/B15 | 中英分版；补充材料已出（中英各一份） |
| 摘要/关键词 | 英文摘要 **250 词**（目标 150–250）、关键词 **6** 个 |

## 仍需你定/核

1. **ORCID**（免费注册 orcid.org；期刊常要）。
2. **目标期刊与模板**：现为 IEEEtran；投 Elsevier 系需换模板并补 Graphical Abstract。
3. **投稿源包的实际编译**：本机没有 LaTeX 引擎，静态检查通过 ≠ 编译通过；建议在装好 TeX 的机器上编一次 `submission/main.tex`。
4. **公开归档**：本地限定证据包与摘要清单由 `scripts/m5_paper_evidence_archive.py` 生成；公开发布及 DOI 尚未完成。

## 2026-10-09 证据修正

旧 E5 的请求位姿不等于放置后真实驾驶位姿，曝光与收尾也未统一过门。中英正文撤回固定实际位姿的单点可走集合解释，保留历史观察与 UNKNOWN；参考点压线和车体包络压线分别报告。后续机械测量与几何回放见 `../T16_RUNTIME_LEARNING_REPAIR_20261009.md`。本轮基础设施开发不代表学习收益、strict 驾驶通过或公开可复现已完成。

fig21 固定六次 R3 短训练，fig22 固定 32 对历史运行输入，清单为 `figure_inputs_20261009.json`。前者不证明收敛或模型晋级，后者不追认驾驶资格。正文表格修复英文长文件名换行；图内中文翻译补齐整图图例，补充材料撤回“可用率失败所以安全通过”等过度结论。最新页数对应 2026-10-09 重新导出的四份 PDF；本地修改尚未提交／发布。

本次限定证据包为 `logs/paper_release/T16_evidence_20261009_195827.zip`：930 个输入文件，缺项与摘要违规均 0，逐成员字节核验通过；不包含模型与原始训练帧，不代表公开完整复现。最终测试、源码身份、PDF 摘要及八项测量审计见 `../T16_DEVELOPMENT_RESULT_20261009.md`。
