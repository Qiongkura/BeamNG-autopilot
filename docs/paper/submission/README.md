# 投稿源包（T16）

本目录由 `scripts/m5_build_submission.py` 生成，内容与 `docs/paper/` 下的成品 PDF 同源。

| 文件 | 说明 |
|---|---|
| `main.tex` | 英文正文（IEEEtran journal 类；自带 `thebibliography`，**单次 pdflatex 即可**，无需 bibtex） |
| `main_zh.tex` | 中文正文（同一类 + `ctex`；中文字体用 `fontset=windows`） |
| `supplement.tex` / `supplement_zh.tex` | 补充材料（含正文单图版本） |
| `figures/` | 英文图（正文与补充材料引用图） |
| `figures_zh/` | 中文图（同数据、图内文字中文） |
| `references.bib` | 题录源（.tex 已内嵌参考文献，仅在改用 bibtex 时需要） |
| `CLAIMS_20261007.md` | 主张→数值→来源→配置的冻结清单 |

fig21 / fig22 的明确输入与 SHA-256 另存于 `figure_inputs_20261009.json`；输入产物在本地证据归档中，未因源包存在而公开。

## 编译

```pwsh
pdflatex main.tex          # 英文正文；两次可稳定交叉引用（本包未用 \ref 计数）
xelatex main_zh.tex        # 中文正文（ctex 建议 xelatex）
pdflatex supplement.tex
xelatex supplement_zh.tex
```

图路径为相对路径（`figures/xxx.png`），因此请在**本目录内**编译，或把整个目录拷到别处。

## 已验证 / 未验证

- **静态检查范围**：环境配平、`\includegraphics` 指向的文件存在、`\cite` 与 `\bibitem` 闭环、
  无未替换的模板残留（`{@fig:`、`[@`）、花括号配平、摘要环境唯一。检查器：`scripts/m5_paper_texcheck.py`。
- **未验证**：本构建命令不执行 LaTeX 编译，不声称编译通过；下列检查自动运行，结果以退出码为准。
  交付 PDF（`docs/paper/T16_*.pdf`）由 Markdown 源经 HTML→Edge 打印生成，修改源后应另行更新 PDF；
  不是 LaTeX 产物；两者内容同源但排版路径不同。
