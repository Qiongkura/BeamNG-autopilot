# T16 论文（英文，SCI 期刊投稿格式）

**交付物**

| 文件 | 说明 |
|---|---|
| `T16_paper.pdf` | 成品 PDF（**当前是旧版**：仅 18 图、含已更正前的表述；你说"可以了"我再重建） |
| `main.tex` | 投稿用 LaTeX 源（IEEEtran journal；自带 `thebibliography`，单次 `pdflatex` 可编译） |
| `paper.md` | 唯一主稿（**8 400 词、42 图、3 表、15 条参考文献**） |
| `references.bib` | 参考文献（你提供的 11 篇 + 8 篇经典工作） |
| `build_paper.py` | 构建脚本：`paper.md` → `main.tex` + 双栏 HTML（图号/引文编号自动生成） |
| `logs/paper_build/T16_paper_preview.pdf` | 构建预览（15 页，gitignored；用于排版验收，非交付版） |

**重新构建**（两行，无需安装 LaTeX）

```pwsh
.venv\Scripts\python.exe docs\paper\build_paper.py
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu `
    --no-pdf-header-footer --print-to-pdf="I:\projects\beamng-autopilot\logs\paper_build\T16_paper.pdf" `
    "file:///I:/projects/beamng-autopilot/logs/paper_build/paper.html"
```

## 论文结构（SCI 投稿规范）

Title → Authors → **Highlights** → Abstract → Index Terms → I. Introduction → II. Related Work →
III. System and Method → IV. Experimental Setup → V. Results → VI. Discussion →
VII. Limitations and Threats to Validity → VIII. Conclusion → Reproducibility and Data
Availability → **Nomenclature** → **Open problems and unfinished work** → **Declarations** →
**Appendix A（逐节对应的已记录产物清单）** → References

**结果覆盖面（42 图全部来自已记录数据）**：门限矩阵与逐 seed、边界图、双口径（标签/漆、
身份 label/surface）、剂量-效应（五指标）、横向与并行扫描、外观门、阈值敏感性、SWA、β=0.8、
负例全家、实例覆盖、训练曲线、一次性确认（含最终集组成与重叠审计）、闭环权衡、硬门热图、
仲裁原因、死锁解剖、逐帧时间线、各臂箱线、车道偏差、安全裕度、计时复测、GPU 记账、
场景密度筛选、负例认证漏斗、空间隔离审计、重叠审计，以及 5 张示意图。

## 需要你填/定的（投稿前）

### A. 必须由你提供的

1. **作者信息**：姓名、单位、通讯作者邮箱、ORCID（现在是 `Anonymous Author(s) — Affiliation withheld for review`）。
2. **基金**：资助项目与编号（`Declarations → Funding` 现为占位）。
3. **作者贡献**：按 CRediT 角色分配（占位已列角色清单）。
4. **致谢**：如需（当前没有该节，可加）。
5. **利益冲突确认**：我写了 "no competing interests"，请你确认。
6. **数据可得性**：仓库地址；是否把已记录产物（scorecard/manifest/判定 JSON）作为补充数据集发布，以及是否申请 Zenodo DOI。

### B. 需要你确认口径的

7. **真值来源的公开表述**：论文现在写明三级凭证阶梯（human revision / **agent review** / engine-certified），并写明"评价集与 75 帧训练基座是人工修订、消融增量是引擎认证、本轮不新增人工标注"。请确认这三句与你的理解一致（尤其 `agent_revision` 的对外措辞）。
8. **人工标注的伦理声明**：论文写的是"人工标注由作者本人对自己录制的仿真数据完成"（`Declarations → Ethics`）。请确认。
9. **负面结果的披露程度**：驾驶验收 0/4、五个因子全部否决、死锁机制、铺装门被 BEV 障碍层驱动——都写进了论文。若你希望降低披露粒度（例如不点出具体阈值 4 m、不写 0/4 这类内部门），告诉我改哪几处。
10. **目标期刊与模板**：现按 IEEE T-ITS（双栏、IEEEtran）。若投 Elsevier 系（如 *Transportation Research Part C*、*Expert Systems with Applications*）需要改模板，并要求 **Graphical Abstract**（我可以从 42 张里选一张改成图文摘要）。
11. **正文/补充材料切分**：42 图对多数期刊偏多（常见上限 10–15 张正文图）。建议：正文保留 14 张（门限矩阵、边界图、双口径、剂量、外观门、最终确认、闭环权衡、死锁、原因直方图、硬门热图、计时、示意图×3），其余 28 张进补充材料——**需要你拍板**是否这样切。
12. **参考文献**：`references.bib` 里第 8 条（朱威等）**期刊名待核对**；页码/DOI 凡抽取文本里读不到的一律留空。是否保留 9 篇中文文献（部分 SCI 期刊对非英文文献有限制）。
13. **图内文字的语种**：所有图现在是英文（便于投稿）；若你要中文版用于国内期刊，我可以再出一套。

## 写作纪律（与项目一致）

负结果与正结果同等详细；未测口径一律 UNKNOWN；不放宽任何门限；图号与引文编号由脚本自动生成；
数字全部来自 `logs/` 里已记录的判定文件、scorecard/manifest 与逐帧遥测（出处见
`docs/T16_PAPER_FIGURES_20261006.md` 与论文 Appendix A）。
