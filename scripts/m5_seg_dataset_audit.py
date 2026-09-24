"""数据集准入审计：来源身份、逐通道覆盖、内容哈希与泄漏检查。

T14 阶段 A 的准入门入口（方案 §1/§2）。它只读数据、写报告，不改数据：

* 逐通道覆盖：road / paint / pavement 各自的**有效帧与像素**，以及 UNKNOWN
  原因计数——"有 Tech annotation 但无可靠标线真值"单独计数，不并进"干净"；
* 内容哈希：同一张图在不同采集里出现（字节复制）单独列组，因为复制样本会
  让"样本数"虚高；
* 泄漏：整个采集组不得跨训练/开发/最终集（方案点名现有 by-map-scene 会在
  组内取时间尾部，不能叫组隔离）；
* 已被其它集合消费过的（T13 用过的）**不得充当最终集**。

用法::

    pwsh> .venv\\Scripts\\python.exe scripts\\m5_seg_dataset_audit.py `
            --runs logs\\m5_seg\\diverse_town_20260924\\front_main ... `
            --dev-group italy/ring_20260923_135342 `
            --final-group italy/ring_20260924_101500 `
            --out logs\\experiments\\dataset_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.manifest import (  # noqa: E402
    DatasetManifest, content_digest_index,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="数据集准入审计（只读）")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="帧目录（<collection>/<view> 或单视角目录）")
    ap.add_argument("--dev-group", action="append", default=[],
                    help="整组划入开发集；可重复")
    ap.add_argument("--final-group", action="append", default=[],
                    help="整组划入最终集（只用一次）；可重复")
    ap.add_argument("--paint-source", action="append", default=[],
                    metavar="RUN=SOURCE",
                    help="按目录名指定漆线真值来源（human_revision / "
                         "engine_verified / pseudo）；缺省 engine_annotation")
    ap.add_argument("--digest-scan", action="store_true",
                    help="额外做全目录内容哈希扫描（更慢，但能发现跨集合复制）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    paint_sources = {}
    for spec in args.paint_source:
        if "=" in spec:
            k, v = spec.split("=", 1)
            paint_sources[k.strip()] = v.strip()
    mf = DatasetManifest.build([Path(r) for r in args.runs], root=ROOT,
                               paint_sources=paint_sources,
                               dev_groups=args.dev_group,
                               final_groups=args.final_group)
    report = {"dataset_id": mf.dataset_id, "groups": mf.groups,
              "audit": mf.audit(), "coverage": mf.coverage(),
              "notes": mf.notes,
              "consumed_sets_in_final": mf.faces_known_to_contain()}
    if args.digest_scan:
        report["content_digests"] = content_digest_index(
            [Path(r) for r in args.runs])

    print(f"[dataset-audit] dataset_id {mf.dataset_id[:16]}  "
          f"records {len(mf.records)}  rejected {len(mf.rejected())}")
    print(f"[dataset-audit] groups: " +
          ", ".join(f"{g}={s}" for g, s in sorted(mf.groups.items())))
    cov = report["coverage"]
    for split, c in sorted(cov.items()):
        print(f"[dataset-audit] {split:6s} frames={c['n_frames']:4d} "
              f"road_ok={c['road_valid_frames']:4d} "
              f"paint_ok={c['paint_valid_frames']:4d} "
              f"trainable={c['trainable_frames']:4d} "
              f"masked_line_px={c['line_masked_px']}")
        for reason, n in list(c.get("unknown_reasons", {}).items())[:2]:
            print(f"            UNKNOWN x{n}: {reason}")
    a = report["audit"]
    print(f"[dataset-audit] group_overlap={a['group_overlap']['n']} "
          f"content_overlap={a['content_overlap']['n']} "
          f"exposure_overlap={a['exposure_overlap']['n']} "
          f"(checked={a['exposure_overlap']['checked']})")
    if report["consumed_sets_in_final"]:
        print(f"[dataset-audit] REFUSED: {len(report['consumed_sets_in_final'])} "
              f"frames of already-consumed sets were assigned to the final set")
    for n in report["notes"]:
        print(f"[dataset-audit] note: {n}")
    if args.digest_scan:
        cd = report["content_digests"]
        print(f"[dataset-audit] content digests: {cd['n_frames']} frames, "
              f"{cd['n_unique_images']} unique, "
              f"{cd['n_duplicate_groups']} duplicate groups")
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        print(f"[dataset-audit] -> {p}")
    return 1 if report["consumed_sets_in_final"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
