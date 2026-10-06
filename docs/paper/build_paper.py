"""Build the paper from paper.md: IEEEtran LaTeX (submission) + two-column HTML (render).

    .venv\\Scripts\\python.exe docs\\paper\\build_paper.py

Outputs
-------
* ``docs/paper/main.tex``        IEEEtran manuscript, self-contained (thebibliography,
                                 so a single ``pdflatex`` run suffices; no bibtex step).
* ``logs/paper_build/paper.html`` two-column A4 HTML with the figures copied next to it
                                 (logs/ is gitignored: build artefacts stay out of git).
* ``logs/paper_build/figures/``  the figures actually cited by the paper.

Design notes
------------
* Figures are numbered **by order of appearance** and in-text references are
  written as ``{@fig:key}``; the builder substitutes the number, so reordering
  figures can never desynchronise the text (hand-written numbers were off by one
  when this was first drafted).
* Citations ``[@key]`` are resolved from ``references.bib`` and numbered by first
  appearance (IEEE style).  A missing key is a hard error, not a silent drop.
* Only the Markdown subset used by the paper is supported; anything else raises.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGS_SRC = ROOT / "logs" / "paper_figures"
BUILD = ROOT / "logs" / "paper_build"
MD = HERE / "paper.md"
BIB = HERE / "references.bib"

# ---------------------------------------------------------------- bib


def parse_bib(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        out[key] = {"kind": kind, **fields}
    return out


def ieee_entry(e: dict) -> str:
    """One IEEE-style reference string (no page numbers when the source lacked them)."""
    authors = e.get("author") or e.get("editor") or ""
    authors = authors.replace(" and ", ", ")
    title = e.get("title", "")
    if e["kind"] == "article":
        venue = e.get("journal", "")
        bits = [b for b in (e.get("volume"), f"no. {e['number']}" if e.get("number") else "",
                            e.get("pages"), e.get("year")) if b]
        tail = ", ".join(bits)
        note = f" [{e['note']}]" if e.get("note") else ""
        return f"{authors}, \u201c{title},\u201d {venue}, {tail}.{note}".replace("  ", " ")
    if e["kind"] in ("inproceedings", "conference"):
        venue = e.get("booktitle", "")
        tail = e.get("year", "")
        note = f" [{e['note']}]" if e.get("note") else ""
        return f"{authors}, \u201c{title},\u201d in {venue}, {tail}.{note}".replace("  ", " ")
    note = f" {e.get('note','')}" if e.get("note") else ""
    return f"{authors}, \u201c{title},\u201d {e.get('year','')}.{note}".replace("  ", " ")


# ---------------------------------------------------------------- markdown


def split_front(txt: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", txt, re.S)
    if not m:
        raise SystemExit("paper.md must start with a --- front-matter block")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, m.group(2)


def collect(md: str):
    """Number the figures by appearance and resolve citations by first use."""
    figs = re.findall(r"!\[(.*?)\]\((.*?)\)\{#(fig:[^}]+)\}", md)
    fig_no = {key: i + 1 for i, (_c, _f, key) in enumerate(figs)}
    order = [k for k in re.findall(r"\[@([A-Za-z0-9_]+)\]", md)]
    cites = []
    for k in order:
        if k not in cites:
            cites.append(k)
    return figs, fig_no, cites


def md_inline_to_tex(s: str, fig_no: dict) -> str:
    s = re.sub(r"\{@(fig:[^}]+)\}", lambda m: str(fig_no[m.group(1)]), s)
    s = re.sub(r"\[@([A-Za-z0-9_]+)\]", r"\\cite{\1}", s)
    s = s.replace("\\", "\\textbackslash{}") if False else s
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)
    for a, b in (("→", r"$\rightarrow$"), ("≥", r"$\ge$"), ("≤", r"$\le$"),
                 ("~", r"\textasciitilde{}"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    s = re.sub(r"(?<!\\)\$([^$]+)\$", lambda m: "$" + m.group(1) + "$", s)
    for a, b in (("×", r"$	imes$"), ("⊆", r"$\subseteq$"), ("–", "--"),
                 ("—", "---"), ("“", "``"), ("”", "''")):
        s = s.replace(a, b)
    return s




def _math_to_html(s: str) -> str:
    """$...$ -> 可读 HTML（无 MathJax 时的近似）。源码刻意不含字面反斜杠：
    早先的写法被转义层吃掉，生成了控制字符（渲染出方块 + 丢字）。"""
    BS = chr(92)
    DOLLAR = chr(36)
    reps = ((BS + "subseteq", "⊆"), (BS + "ge", "≥"), (BS + "le", "≤"),
            (BS + "times", "×"), (BS + "rightarrow", "→"),
            (BS + "mathrm", ""), (BS + ",", " "), (BS, ""))
    def conv(m):
        t = m.group(1)
        for a, b in reps:
            t = t.replace(a, b)
        t = t.replace("_{", "_").replace("{", "").replace("}", "")
        t = re.sub("_([A-Za-z]+)", lambda mm: "<sub>" + mm.group(1) + "</sub>", t)
        return "<i>" + t + "</i>"
    pat = BS + DOLLAR + "([^" + DOLLAR + "]+)" + BS + DOLLAR
    return re.sub(pat, conv, s)

def md_inline_to_html(s: str, fig_no: dict) -> str:
    s = re.sub(r"\{@(fig:[^}]+)\}", lambda m: str(fig_no[m.group(1)]), s)
    s = re.sub(r"\[@([A-Za-z0-9_]+)\]", lambda m: f"[{CITE_NO[m.group(1)]}]", s)
    s = _math_to_html(s)
    return s


CITE_NO: dict = {}


def emit_tex(md: str, meta: dict, bib: dict, fig_no: dict, cites: list) -> str:
    body = []
    for block in re.split(r"\n\s*\n", md):
        b = block.strip()
        if not b:
            continue
        fig = re.match(r"^!\[(.*?)\]\((.*?)\)\{#(fig:[^}]+)\}$", b, re.S)
        if fig:
            cap, img, key = fig.groups()
            body.append(
                "\\begin{{figure}}[!t]\n\\centering\n"
                f"\\includegraphics[width=\\columnwidth]{{figures/{img}}}\n"
                f"\\caption{{{md_inline_to_tex(cap, fig_no)}}}\n"
                f"\\label{{{key}}}\n\\end{{figure}}")
            continue
        if b.startswith("|"):                       # markdown table
            rows = [r for r in b.splitlines() if r.strip().startswith("|")]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
            ncol = len(cells[0])
            head = " & ".join(md_inline_to_tex(c, fig_no) for c in cells[0])
            body_rows = [" & ".join(md_inline_to_tex(c, fig_no) for c in r) + r" \\"
                         for r in cells[1:]]
            env = "table*" if ncol >= 6 else "table"
            width = r"\textwidth" if ncol >= 6 else r"\columnwidth"
            tex = [f"\\begin{{{env}}}[!t]", "\\centering",
                   f"\\begin{{tabular}}{{@{{}}{'l' * ncol}@{{}}}}", "\\toprule",
                   head + r" \\", "\\midrule", *body_rows, "\\bottomrule",
                   "\\end{tabular}", "\\end{" + env + "}"]
            body.append("\n".join(tex))
            continue
        if b.startswith("# "):
            title = b[2:].strip()
            if title.lower().startswith("abstract"):
                continue
            if title.lower().startswith("references"):
                break
            sec = re.match(r"^([IVX]+)\.\s+(.*)$", title)
            if sec:
                body.append(f"\\section{{{md_inline_to_tex(sec.group(2), fig_no)}}}")
            else:
                body.append(f"\\section*{{{md_inline_to_tex(title, fig_no)}}}")
            continue
        if b.startswith("## "):
            sub = re.match(r"^([A-Z])\.\s+(.*)$", b[3:].strip())
            body.append(f"\\subsection{{{md_inline_to_tex(sub.group(2) if sub else b[3:], fig_no)}}}")
            continue
        if b.startswith("# References"):
            break
        if b.startswith("**Index Terms**"):
            body.append("\\begin{IEEEkeywords}\n"
                        + md_inline_to_tex(b.replace("**Index Terms**—", ""), fig_no)
                        + "\n\\end{IEEEkeywords}")
            continue
        body.append(md_inline_to_tex(b, fig_no))

    abstract = re.search(r"# Abstract\n(.*?)\n# ", md, re.S)
    _abs = (abstract.group(1) if abstract else "").split("**Index Terms**")[0]
    abs_tex = md_inline_to_tex(" ".join(_abs.split()), fig_no)
    refs = "\n".join(f"\\bibitem{{{k}}} {ieee_entry(bib[k])}" for k in cites if k in bib)
    missing = [k for k in cites if k not in bib]
    if missing:
        raise SystemExit(f"citation keys missing from references.bib: {missing}")
    return f"""\\documentclass[journal]{{IEEEtran}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{booktabs}}
\\usepackage{{url}}
\\usepackage[utf8]{{inputenc}}

\\title{{{meta.get('title','')}}}
\\author{{{meta.get('authors','')}}}

\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{abs_tex}
\\end{{abstract}}

{chr(10).join(body)}

\\begin{{thebibliography}}{{99}}
{refs}
\\end{{thebibliography}}
\\end{{document}}
"""


CSS = """
@page { size: A4; margin: 18mm 15mm; }
body { font: 9.6pt/1.42 'Times New Roman', Times, serif; color: #111; margin: 0;
       column-count: 2; column-gap: 7mm; text-align: justify; hyphens: auto; }
h1.title { column-span: all; font-size: 19pt; text-align: center; margin: 0 0 4pt; }
p.authors { column-span: all; text-align: center; font-size: 11pt; margin: 0 0 10pt; }
div.abstract { column-span: all; margin: 0 0 10pt; }
div.abstract h2 { font-size: 11pt; text-align: center; margin: 0 0 3pt; letter-spacing: .5px; }
div.abstract p { margin: 0 0 6pt; }
p.indexterms { column-span: all; margin: 0 0 12pt; }
h2 { font-size: 11pt; margin: 12pt 0 4pt; page-break-after: avoid; }
h3 { font-size: 10pt; font-style: italic; margin: 9pt 0 3pt; page-break-after: avoid; }
figure { margin: 8pt 0 10pt; break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-size: 8.2pt; text-align: justify; margin-top: 3pt; }
div.wide { column-span: all; }
div.wide table { font-size: 8.6pt; }
table { width: 100%; border-collapse: collapse; font-size: 8.2pt; margin: 6pt 0 10pt; }
th, td { border-top: .5pt solid #444; border-bottom: .5pt solid #444; padding: 2pt 3pt;
         text-align: left; vertical-align: top; }
th { border-bottom: .8pt solid #222; }
ol.refs { font-size: 8.4pt; padding-left: 14pt; }
ol.refs li { margin-bottom: 2pt; }
"""


def emit_html(md: str, meta: dict, bib: dict, fig_no: dict, cites: list) -> str:
    import markdown as mdlib
    lines = []
    in_abstract = False
    for block in re.split(r"\n\s*\n", md):
        b = block.strip()
        if not b:
            continue
        # 摘要与 Index Terms 只出现在页首的全宽块：正文流跳过，避免重复渲染
        if b.startswith("# Abstract"):
            in_abstract = True
            continue
        if in_abstract:
            if b.startswith("# "):
                in_abstract = False
            else:
                continue
        if b.startswith("# References"):
            break
        if b.startswith("**Index Terms**"):
            continue
        fig = re.match(r"^!\[(.*?)\]\((.*?)\)\{#(fig:[^}]+)\}$", b, re.S)
        if fig:
            cap, img, key = fig.groups()
            lines.append(f'<figure id="{key}"><img src="figures/{img}" alt="">'
                         f'<figcaption><b>Fig. {fig_no[key]}.</b> '
                         f'{md_inline_to_html(cap, fig_no)}</figcaption></figure>')
            continue
        b = md_inline_to_html(b, fig_no).replace(" — ", "&nbsp;&mdash; ")
        lines.append(b)
    body = mdlib.markdown("\n\n".join(lines), extensions=["tables"])
    abstract = re.search(r"# Abstract\n(.*?)\n# ", md, re.S)
    _absh = (abstract.group(1) if abstract else "").split("**Index Terms**")[0]
    abs_html = mdlib.markdown(" ".join(_absh.split()))
    idx = re.search(r"\*\*Index Terms\*\*—(.*)", md)
    body = body.replace("<h1>Abstract</h1>", "")
    body = re.sub(r"<h1>(.*?)</h1>", lambda m: f"<h2>{m.group(1)}</h2>", body)
    refs = "".join(f"<li>{ieee_entry(bib[k])}</li>" for k in cites if k in bib)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{meta.get('title','')}</title><style>{CSS}</style></head><body>
<h1 class="title">{meta.get('title','')}</h1>
<p class="authors">{meta.get('authors','')} &mdash; {meta.get('affiliation','')}</p>
<div class="abstract"><h2>Abstract</h2>{abs_html}</div>
<p class="indexterms"><b>Index Terms</b>&mdash;{md_inline_to_html(idx.group(1), fig_no) if idx else ''}</p>
{body}
<h2>References</h2><ol class="refs">{refs}</ol>
</body></html>"""
    # 破折号不得出现在行首（含表头/题注/表格单元格）
    html = html.replace(" — ", "&nbsp;&mdash; ").replace("— ", "&nbsp;&mdash; ")
    # 6 列以上的表跨双栏（HTML 版的 table*）：否则单元格被挤成 4-5 行
    def _wide(m):
        tbl = m.group(0)
        head = tbl.split("</tr>")[0]
        return f'<div class="wide">{tbl}</div>' if head.count("<th") >= 6 else tbl
    html = re.sub(r"<table>.*?</table>", _wide, html, flags=re.S)
    return html


def main() -> int:
    global CITE_NO
    bib = parse_bib(BIB)
    meta, md = split_front(MD.read_text(encoding="utf-8"))
    figs, fig_no, cites = collect(md)
    CITE_NO = {k: i + 1 for i, k in enumerate(cites)}
    (HERE / "main.tex").write_text(emit_tex(md, meta, bib, fig_no, cites), encoding="utf-8")
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "figures").mkdir(exist_ok=True)
    for _c, img, _k in figs:
        src = FIGS_SRC / img
        if not src.is_file():
            raise SystemExit(f"missing figure: {src}")
        shutil.copy2(src, BUILD / "figures" / img)
    (BUILD / "paper.html").write_text(emit_html(md, meta, bib, fig_no, cites), encoding="utf-8")
    print(f"[paper] main.tex -> {HERE / 'main.tex'}")
    print(f"[paper] html     -> {BUILD / 'paper.html'}  ({len(figs)} figures, {len(cites)} refs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
