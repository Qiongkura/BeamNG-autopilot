# 投稿源包（T16）

本目录由 `scripts/m5_build_submission.py` 生成，内容与 `docs/paper/` 下的成品 PDF 同源。

| 文件 | 说明 |
|---|---|
| `main.tex` | 英文正文（IEEEtran journal 类；自带 `thebibliography`，**单次 pdflatex 即可**，无需 bibtex） |
| `main_zh.tex` | 中文正文（同一类 + `ctex`；中文字体用 `fontset=windows`） |
| `supplement.tex` / `supplement_zh.tex` | 补充材料（图号 S1–S16） |
| `figures/` | 英文图（26 张，300 dpi） |
| `figures_zh/` | 中文图（同数据、图内文字中文） |
| `references.bib` | 题录源（.tex 已内嵌参考文献，仅在改用 bibtex 时需要） |
| `CLAIMS_20261007.md` | 主张→数值→来源→配置的冻结清单 |

## 编译

```bash
pdflatex main.tex          # 英文正文；两次可稳定交叉引用（本包未用 \ref 计数）
xelatex main_zh.tex        # 中文正文（ctex 建议 xelatex）
pdflatex supplement.tex
xelatex supplement_zh.tex
```

图路径为相对路径（`figures/xxx.png`），因此请在**本目录内**编译，或把整个目录拷到别处。

## 已验证 / 未验证

- **已验证（静态检查）**：环境配平、`\includegraphics` 指向的文件存在、`\cite` 与 `\bibitem` 闭环、
  无未替换的模板残留（`{@fig:`、`[@`）、花括号配平、摘要环境唯一。检查器：`scripts/m5_paper_texcheck.py`。
- **未验证**：本机**没有安装 LaTeX 引擎**（无 pdflatex/xelatex/tectonic），因此**未编译**，
  不声称编译通过。交付 PDF（`docs/paper/T16_*.pdf`）由同一 Markdown 源经 HTML→Edge 打印生成，
  不是 LaTeX 产物；两者内容同源但排版路径不同。
