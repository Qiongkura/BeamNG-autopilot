"""复核工作表：土路与无标线的帧必须在 notes 里带上 road_type。

背景（方案 §6.3）：``dirt_shoulder``（铺装/土肩与纯土路）的要求原文是"两种道路
类型分别标记，不能混成 road 一类"。复核结论的唯一载体是 review_worksheet，
所以那一类（以及"真无线铺装路"）的帧要在 notes 里写
``road_type=（gravel / asphalt / shoulder）``——这段逻辑如果埋在 ``main()`` 里，
漏掉一个类别不会有任何东西报警，而漏掉的正是方案点名的两类之一。这里把它
抽成函数钉死。
"""

from __future__ import annotations

from scripts.m5_review_pack import (ROAD_TYPE_NOTE, ROAD_TYPE_VALUES,
                                    worksheet_notes, worksheet_rows,
                                    worksheet_table_md)


def _starter() -> list:
    return [
        {"path": "a/frame_00000.npz", "view": "front_main", "group": "g1",
         "category": "dirt_shoulder", "why": "土路：line_px=0"},
        {"path": "b/frame_00001.npz", "view": "front_main", "group": "g2",
         "category": "no_line_pavement", "why": "无标线：line_px=0"},
        {"path": "c/frame_00002.npz", "view": "front_main", "group": "g3",
         "category": "clear_paint", "why": "清晰漆线：line_px=4009"},
    ]


def test_dirt_and_no_line_frames_are_asked_for_a_road_type():
    rows = worksheet_rows(_starter())
    assert rows[0]["notes"] == ROAD_TYPE_NOTE
    assert rows[1]["notes"] == ROAD_TYPE_NOTE
    assert rows[2]["notes"] == "", "其它类别不该被塞路型问题"


def test_every_category_has_a_note_decision():
    """穷举六个类别：只有点名的两类带路型要求（少写一类这里会红）。"""
    from scripts.m5_review_pack import CATEGORIES
    got = {c for c, _l, _n in CATEGORIES if worksheet_notes(c)}
    assert got == {"dirt_shoulder", "no_line_pavement"}


def test_the_note_spells_out_the_three_allowed_values():
    for value in ("gravel", "asphalt", "shoulder"):
        assert value in ROAD_TYPE_NOTE
    assert sorted(ROAD_TYPE_VALUES) == ["asphalt", "gravel", "shoulder"]


def test_markdown_table_has_a_notes_column_carrying_the_note():
    """md 表里必须有 notes 列——没有它，复核人在 review_worksheet.md 上看不到
    这条要求（只有 json 里才有，而 review_worksheet.md 才是递给人看的那份）。"""
    rows = worksheet_rows(_starter())
    md = worksheet_table_md(rows)
    text = "\n".join(md)
    assert "| notes |" in text
    assert text.count(ROAD_TYPE_NOTE) == 2
    assert len([line for line in md if line.startswith("| ")]) == 5   # 表头2+3帧


def test_reviewer_fields_start_empty():
    """工作表是让人填的：预填的只有要求，不能有"已复核"的假记录。"""
    for r in worksheet_rows(_starter()):
        assert r["reviewer"] == "" and r["reviewed_at"] == ""
        assert r["verdict"] == "" and r["regions"] == ""


def test_road_type_vocabulary_matches_the_toolbar():
    """界面里的路型取值与 worksheet 让复核人填的三个词必须是同一套。

    两处各写一份字符串迟早会漂开：复核人按 worksheet 填 "gravel"，而标注器
    写的是别的词——那份真值就没人对得上了。
    """
    from beamng_autopilot.labeling import annotate_tools as at
    assert sorted(ROAD_TYPE_VALUES) == sorted(at.ROAD_TYPE_NAMES.values())
