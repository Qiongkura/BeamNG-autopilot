# T16 论文（英文，IEEE T-ITS 投稿格式）

**交付物**

| 文件 | 说明 |
|---|---|
| `T16_paper.pdf` | 渲染成品（9 页 A4 双栏、18 图、3 表、15 条参考文献；矢量文本、字体全嵌入） |
| `main.tex` | **投稿用 LaTeX 源**（IEEEtran, journal 选项；自带 `thebibliography`，单次 `pdflatex` 即可编译，无需 bibtex） |
| `paper.md` | 唯一主稿（内容只写在这里；两个产物都由它生成） |
| `references.bib` | 参考文献（11 条来自你提供的文献 + 8 条经典工作） |
| `build_paper.py` | 构建脚本：`paper.md` → `main.tex` + 双栏 HTML（并自动编号图/引文） |
| `logs/paper_build/paper.html` | 渲染用 HTML（build 产物，logs/ 不进 git） |
| `logs/paper_build/pages/*.png` | 逐页 PNG（验收用） |

**重新构建**

```pwsh
.venv\Scripts\python.exe docs\paper\build_paper.py     # 生成 main.tex + logs\paper_build\paper.html
# 渲染 PDF（无需安装 LaTeX：用系统自带 Edge 的 Chromium 内核）
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu `
    --no-pdf-header-footer --print-to-pdf="I:\projects\beamng-autopilot\logs\paper_build\T16_paper.pdf" `
    "file:///I:/projects/beamng-autopilot/logs/paper_build/paper.html"
# 若本机装了 LaTeX（Tectonic/TeX Live），投稿版直接编译 main.tex 即可
```

## 论文内容（全部数字来自已记录的运行产物）

- **I 引言**：人工标注成本 + "线"定义歧义两个问题；五条贡献
- **II 相关工作**：车道线方法（LaneNet/SCNN/CLRNet + 你提供的 9 篇中文文献）、数据集与真值生成（TuSimple/CULane/Borkar/McCall）、自动标注、评测纪律
- **III 方法**：系统架构；引擎真值 + 四道认证门；**计数契约 v5**（P/C/C_out/R/M/L/A）；**协议定义**（横向范围 / 同侧合并 / 外观门）与哈希绑定；训练组成设计空间；一次性最终集的封存-哈希门-单次消费；驾驶栈与安全仲裁阶梯
- **IV 实验设置**：两套不相交参考集、6 seed、冻结门限、驾驶 A/B 读法（4/4 硬门、安全取最大值、UNKNOWN 不释放）
- **V 结果**：① 两套定义各差一门（v7 身份率 0.433/0.414；v8 五门全过，漆召回 0.812/0.885，标签召回 0.559）；② 分歧是**定义性**的（约 30% 标注线像素是无局部对比度的暗漆）；③ 边界图（膨胀买召回付精度，R3 付 3 倍）；④ 外观门（精度 0.650→0.884 / 0.387→0.650，R3 IoU 0.347→0.532）；⑤ 剂量-效应（6× 峰值 0.763、8.5× 回落 0.701）；⑥ **一次性最终确认不复现**（标签召回 0.460 / 精度 0.763，如实作为该测试的头条）；⑦ 闭环（v8 中心压线 13→0，代价是可用率 0.851→0.292）；⑧ **驾驶验收 0/4 未过门** + 死锁机制（越界点中位 2.50 m、96% 静止、当前车体 512/512 在道内）；⑨ 五个单因子全部否决（含"帧计数被行进量污染"的规则缺陷）
- **VI 讨论 / VII 限制 / VIII 结论 / 可复现性与数据可得性**

**写作纪律（与项目一致）**：负结果与正结果同等详细；未测口径一律写 UNKNOWN；不放宽任何门限；所有图号/引文编号由脚本自动生成（手写编号曾错位一位）。

## 验收记录（QA）

`pdf_qa.py`（PDF 技能自带）：**通过 8 项**（元数据、页面尺寸一致、无空白页、字体全嵌入、无内容溢出、填充率、边距对称）。
保留的警告及理由：

1. *Line-start punctuation '—'*（3 处，第 7 页）：实为 `\xa0—`（**不换行空格**绑定在词后），破折号不会出现在行首；这是检查器未剥离 `\xa0` 的误报。
2. *Cover page not full-bleed*：论文没有封面页，此项不适用（已用 `--skip-cover` 运行）。

渲染/排版缺陷的修复记录（都在本轮内发现并修掉）：摘要被渲染两遍、Index Terms 重复、正文 `{@fig:}` 占位符未替换、`$…$` 数学记号原样显示且产生控制字符（`\1_\2` 被转义层吃掉）、6 列表格在单栏里被挤成 4–5 行（改为跨双栏）、References 标题与占位行重复。

## 预投稿待办（需要你决定/补齐）

1. **作者与单位**：现在写的是 "Anonymous Author(s) — Affiliation withheld for review"。
2. **目标期刊与模板**：按 IEEE T-ITS 的 journal 格式排版；若改投他刊（T-IV / RA-L / ICRA），需换模板与页数限制。
3. **参考文献元数据补全**：`references.bib` 里第 8 条（朱威等）**期刊名待核对**（提供的抽取文本里没有刊名）；其余条目按抽取文本填写，**页码/DOI 未从原文读出的一律留空**，投稿前按出版社记录补齐。
4. **补充材料**：正文用了 18 张图，另有 16 张（`docs/T16_PAPER_FIGURES_20261006.md` 列出的 fig9/10/13/15/17/19/20/21/24/25/26/27/29/31/33/34）适合放补充材料；如需要我可以生成一份补充材料 PDF。
5. **声明**：需要补作者贡献、资助、数据可得性声明（当前有一节"Reproducibility and Data Availability"）。
