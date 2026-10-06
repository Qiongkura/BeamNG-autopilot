# T16 论文（英文版 + 中文版 + 补充材料）

**交付物**

| 文件 | 说明 |
|---|---|
| `T16_paper.pdf` | 英文版成品（11 页，双栏 IEEE 式；已按你的回答填入作者/声明/致谢，并修正了"无人工标注"的旧表述） |
| `T16_paper_zh.pdf` | **中文版成品**（13 页，与英文版同结构、同图号顺序、同数字） |
| `T16_supplement.pdf` | **补充材料**（4 页，16 张补充图 Fig. S1–S16 + 读数约定与已声明缺口） |
| `main.tex` / `main_zh.tex` / `supplement.tex` | 投稿用 LaTeX 源（IEEEtran journal；自带 `thebibliography`，单次 `pdflatex` 可编译；中文版用 `ctex`） |
| `paper.md` | 英文主稿（唯一主源：26 张正文图、3 张表、15 条参考文献） |
| `paper_zh.md` | 中文主稿（与英文逐段对应，图号/引文编号由脚本自动生成） |
| `supplement.md` | 补充材料主源（16 图） |
| `references.bib` | 参考文献（你提供的 11 篇 + 8 篇经典工作；9 篇中文文献按中文题录著录，标 `[In Chinese]`） |
| `build_paper.py` | 构建脚本：Markdown → LaTeX + 双栏 HTML（图号、表题、引文编号自动生成；`--lang zh` 出中文版） |
| `logs/paper_build/` | 构建产物与预览（gitignored） |

**重新构建（三步，无需安装 LaTeX）**

```pwsh
# 1) 三个 Markdown 源 -> LaTeX + HTML
.venv\Scripts\python.exe docs\paper\build_paper.py
.venv\Scripts\python.exe docs\paper\build_paper.py --md docs\paper\paper_zh.md `
    --tex docs\paper\main_zh.tex --html logs\paper_build\paper_zh.html --lang zh
.venv\Scripts\python.exe docs\paper\build_paper.py --md docs\paper\supplement.md `
    --tex docs\paper\supplement.tex --html logs\paper_build\supplement.html --fig-prefix S

# 2) HTML -> PDF（Edge headless）
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
& $edge --headless=new --disable-gpu --no-pdf-header-footer `
    --print-to-pdf="I:\projects\beamng-autopilot\docs\paper\T16_paper.pdf" `
    "file:///I:/projects/beamng-autopilot/logs/paper_build/paper.html"
& $edge --headless=new --disable-gpu --no-pdf-header-footer `
    --print-to-pdf="I:\projects\beamng-autopilot\docs\paper\T16_paper_zh.pdf" `
    "file:///I:/projects/beamng-autopilot/logs/paper_build/paper_zh.html"
& $edge --headless=new --disable-gpu --no-pdf-header-footer `
    --print-to-pdf="I:\projects\beamng-autopilot\docs\paper\T16_supplement.pdf" `
    "file:///I:/projects/beamng-autopilot/logs/paper_build/supplement.html"
```

## 论文结构（SCI 投稿规范）

Title → Authors → **Highlights** → Abstract → Index Terms → I. Introduction → II. Related Work →
III. System and Method → IV. Experimental Setup → V. Results → VI. Discussion →
VII. Limitations and Threats to Validity → VIII. Conclusion → Reproducibility and Data
Availability → **Nomenclature** → **Open problems and unfinished work** → **Declarations** →
**Acknowledgements** → **Appendix A（逐节对应的已记录产物清单）** → References

中文版逐节对应（摘要／关键词／亮点／一至八节／可复现性与数据可得性／术语表／未解决的问题与未完成的工作／声明／致谢／附录 A／参考文献）。

**图与表**：正文 26 张图（5 张示意图 + 21 张数据图），3 张表（表 I 双定义验收、表 II 五个驾驶因子、表 III 控制节拍）；
补充材料 16 张图（Fig. S1–S16）。全部图由三个脚本从 `logs/` 里已记录的判定文件、scorecard/manifest 与逐帧遥测重建。

## 已按你的回答填入

| 项 | 现状 |
|---|---|
| A1 作者信息 | 袁哲宇（Zheyu Yuan），华南农业大学，广州，中国；邮箱 `2447402326@qq.com`（在 `Declarations → Data and code availability` 之外，如需在首页脚注显示请告知） |
| A2 基金 | "This research received no external funding." |
| A3 作者贡献 | 唯一作者，CRediT 全角色 |
| A4 利益冲突 | "The author declares no competing interests."（如与事实不符请改） |
| A5 数据可得性 | 仓库 `https://github.com/Qiongkura/BeamNG-autopilot`（分支 `fix/round3-hardening-20260921`）+ 三个重建脚本 |
| A6 致谢 | BeamNG.tech 团队、父母、朋友、DeepSeek 与 OpenCode |
| B7 真值口径 | 三级凭证阶梯（human revision / agent review / engine-certified）+ "本轮不新增人工标注" |
| B8 伦理 | 人工标注由作者本人对自己录制的仿真数据完成 |
| B9 负结果披露 | 全量披露（0/4、五个因子否决、死锁机制、4 m 阈值、铺装门被障碍层驱动） |
| B10 目标期刊 | 现按 IEEE T-ITS 双栏格式 |
| B11 正文/补充切分 | 正文 26 张 + 补充 16 张 |
| B12 参考文献 | 15 条全保留；9 篇中文文献按中文题录著录 |
| B13 图内文字 | 全英文（中英两版共用同一套图） |
| B14 语言版本 | 英文版与中文版分开（本目录两份 PDF） |
| B15 补充材料 PDF | 已出（`T16_supplement.pdf`） |

## 仍需你定/核

1. **ORCID**：你说不知道是什么——它是作者的 16 位唯一标识（免费注册 `orcid.org`）；若投稿期刊要求，注册后填进 `paper.md` / `paper_zh.md` 的 front matter 即可。
2. **目标期刊与模板**：现为 IEEEtran。若投 Elsevier 系（如 *TR Part C*、*ESWA*），需要换模板并补 **Graphical Abstract**（可从 42 张图里挑一张改）。
3. **第 8 条中文文献（朱威等）的期刊名**：抽取文本里读不到，现为 `Journal name to be confirmed`，请你核对后填 `references.bib`。
4. **邮箱是否上首页**：现只写在投稿系统层面，论文正文未出现邮箱；要显示的话告诉我放在哪。
5. **中文版用途**：中文版按同一套数据撰写，可直接投国内期刊（图注为英文，如需图内中文标注需要另出一套图）。

## 写作纪律（与项目一致）

负结果与正结果同等详细；未测口径一律 UNKNOWN；不放宽任何门限；图号、表题与引文编号由脚本自动生成；
数字全部来自 `logs/` 里已记录的判定文件、scorecard/manifest 与逐帧遥测（出处见
`docs/T16_PAPER_FIGURES_20261006.md` 与论文 Appendix A）。
