from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PAPER = Path(__file__).resolve().parents[1] / "docs" / "paper"
spec = importlib.util.spec_from_file_location("paper_builder", PAPER / "build_paper.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_reordered_figures_resolve_to_same_evidence_in_html_and_tex():
    md = ("![Candidate](candidate.png){#fig:candidate}\n\n"
          "![Baseline](baseline.png){#fig:baseline}\n\n"
          "Baseline {@fig:baseline}, candidate {@fig:candidate}.")
    _, numbers, _ = builder.collect(md)
    sentence = "Baseline {@fig:baseline}, candidate {@fig:candidate}."
    assert builder.md_inline_to_html(sentence, numbers) == "Baseline 2, candidate 1."
    assert builder.md_inline_to_tex(sentence, numbers) == "Baseline 2, candidate 1."


def test_distinct_evidence_cannot_share_a_figure_identifier():
    md = "![First](a.png){#fig:s24}\n\n![Different evidence](b.png){#fig:s24}"
    with pytest.raises(ValueError, match="duplicate figure identifier: fig:s24"):
        builder.collect(md)


@pytest.mark.parametrize("name", ["supplement.md", "supplement_zh.md"])
def test_actual_supplement_has_resolvable_unambiguous_references(name):
    _, md = builder.split_front((PAPER / name).read_text(encoding="utf-8"))
    figs, numbers, _ = builder.collect(md)
    assert len(figs) == 42
    builder.md_inline_to_html(md, numbers)
    builder.md_inline_to_tex(md, numbers)
