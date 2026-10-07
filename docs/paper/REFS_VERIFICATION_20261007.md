# 参考文献题录核验（全量 20 条）

日期：2026-10-07　依据：审查 `docs/paper/REVIEW_20261007.md` P2 #11（「全量文献核验」此前只做了 `zhu2024parallel` 一条）
方法：三类证据，逐条比对 `docs/paper/references.bib`；**核实不到的字段保持缺失，不猜**（与 bib 头部既有约定一致）。

| 证据类别 | 来源 | 覆盖 |
|---|---|---|
| A：作者提供的论文抽取文本 | `logs/lit_extract/*.txt`（GBK/UTF-8 解码） | 11 篇（9 篇中文 + Borkar + McCall） |
| B：Crossref API | `api.crossref.org`（按题名检索 / 按 DOI 直取） | 10 条经典文献 + 2 篇 ITS |
| C：arXiv API | `export.arxiv.org/api/query?id_list=1803.05407` | SWA（预印本注明 UAI 2018） |

## 逐条结果

| key | 核验内容 | 证据 | 处理 |
|---|---|---|---|
| `borkar2012groundtruth` | T-ITS 13(1), 2012-03, 页 365–374 | A：抽取文本页眉逐页 365…374；B：Crossref 同页、DOI `10.1109/TITS.2011.2173196` | 补页码 + DOI |
| `mccall2006video` | T-ITS 7(1), 2006, 页 20–37 | A：抽取文本首页「7(1)」；B：Crossref 页 20–37、DOI `10.1109/TITS.2006.869595` | 补页码 + DOI |
| `chen2026clrnetlight` | 软件工程 29(9), 2026-09 | A：抽取文本首页卷期年 | 页码不可读 → 如实标注 |
| `huang2026bspline` | 长安大学学报(自然科学版) 46(3):118–129, 2026-05 | A：期刊印刷引用行 + 文章编号 `1671-8879(2026)03-0118-12` | 补页码 + DOI |
| `jia2026gnn` | 工程设计学报 33(3), 2026-06 | A：抽取文本首页卷期年 | 页码不可读 → 如实标注 |
| `li2026xca` | 电子测量技术 49(7), 2026-04 | A：抽取文本首页 + DOI | 补 DOI；页码不可读 → 如实标注 |
| `zhu2024parallel` | 模式识别与人工智能 34(5):434–445, 2021-05 | A：期刊名/第 34 卷/2021 年 5 月；**页码来自作者单位主页**（非 PDF 证据） | 页码保留并标注来源 |
| `hu2026multiscale` | 汽车工程学报，网络首发 2026-09-03 | A：`引用格式 … [J/OL]．汽车工程学报` | 标注 online first（无卷期页） |
| `shuai2026deeplab` | 太原科技大学学报 47(3):208–214, 2026-06 | A：文章编号 `1673-2057(2026)03-0208-07`（7 页 → 208–214） | 补页码 + DOI |
| `lu2026symmetry` | 智能系统学报，网络首发 2026-08-28 | A：`引用格式 …[J]．智能系统学报，DOI：10.11992/tis.202602011` | 补 DOI；标注 online first |
| `chen2025threed` | 现代电子技术 48(24), 2025-12-15 | A：抽取文本首页卷期年 | 页码不可读 → 如实标注 |
| `pan2018scnn` | AAAI 2018 | B：Crossref DOI `10.1609/aaai.v32i1.12301`（Crossref 无页码） | 补 DOI；页码缺失（不猜） |
| `neven2018lanenet` | IEEE IV 2018, 页 286–291 | B：Crossref DOI `10.1109/IVS.2018.8500547` | 补页码 + DOI |
| `zheng2022clrnet` | CVPR 2022, 页 888–897 | B：Crossref DOI `10.1109/CVPR52688.2022.00097` | 补页码 + DOI |
| `tusimple2017` | 数据集（2017） | — | 保持原样（数据集无卷期页） |
| `salehi2017tversky` | MLMI 2017（MICCAI workshop）, 页 379–387 | B：Crossref DOI `10.1007/978-3-319-67389-9_44` | 补页码 + DOI（原稿曾疑为 240–248，以 Crossref 为准） |
| `izmailov2018swa` | UAI 2018 | C：arXiv 1803.05407 的 comment 字段「Appears at … UAI, 2018」 | 补 note（arXiv 号）；UAI 无 DOI |
| `chen2018deeplab` | ECCV 2018（LNCS）, 页 833–851 | B：Crossref DOI `10.1007/978-3-030-01234-2_49` | 补页码 + DOI |
| `ronneberger2015unet` | MICCAI 2015（LNCS）, 页 234–241 | B：Crossref DOI `10.1007/978-3-319-24574-4_28` | 补页码 + DOI |
| `cordts2016cityscapes` | CVPR 2016, 页 3213–3223 | B：Crossref DOI `10.1109/CVPR.2016.350` | 补页码 + DOI |

## 仍未闭合的三项（如实声明）

1. **4 篇中文文献的页码**（`chen2026clrnetlight`、`jia2026gnn`、`li2026xca`、`chen2025threed`）：
   抽取文本里没有印刷引用行或文章编号，Crossref 未收录这些中文期刊。bib 中**不写页码**并在
   `note` 里标注原因；若需要补全，请提供带页码的首页截图或 CNKI 引用行。
2. **`zhu2024parallel` 的页码**来自作者单位主页而非 PDF 证据——已在 note 里写明来源。
3. **`pan2018scnn` 的页码**：AAAI 论文集在 Crossref 里没有页码字段，故不写（不猜）。

## 呈现方式

* 参考文献表渲染 `pages` 与已核验的 `doi`（`build_paper.py` 的 `ieee_entry`）；
* `note` 在**可见列表**里只渲染第一段（如「[In Chinese]」），完整核验记录留在本文件与 bib 里，
  避免把审计文字带进投稿的参考文献表；
* 四份 PDF 重出后，正文数字审计（115 项）与 LaTeX 静态检查（0 问题）均复跑通过。
