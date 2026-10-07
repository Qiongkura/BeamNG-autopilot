"""生成论文证据归档（供第三方复算）+ 机器可读清单。

为什么：论文的「数据可得性」现在写的是「记录产物在本地 logs/，公开归档尚不可用」。
本脚本把**论文每个数字所依赖的最小产物集**打成一个 zip（在 logs/ 下，gitignored——
用户规则要求 logs 不入库），并生成 MANIFEST.json（路径 + sha256 + 大小），
使用者可直接上传 Zenodo 或按清单逐文件核对。

清单内容（与 docs/paper/CLAIMS_20261007.md 一一对应）：
* 判定 JSON：v7/v8、双范围、剂量、E1/E2/E3 重评、封存确认
* 运行记录：fsd_benchmark 的 scorecard/manifest（V-E/V-F 的依据）
* 采集/审计：final_pool_audit、collect_isolation、timing_retest、e1_negative_pool
* 论文源：paper.md / paper_zh.md / supplement*.md / references.bib / CLAIMS / REVIEW

用法::

    .venv\\Scripts\\python.exe scripts\\m5_paper_evidence_archive.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "logs" / "experiments"
BENCH = ROOT / "logs" / "fsd_benchmark"
PAPER = ROOT / "docs" / "paper"
OUT = ROOT / "logs" / "paper_release"

#: 判定与结果 JSON（论文每个数字的来源）
EXP_FILES = [
    "r2_verdict_v7_20261005.json", "r2_verdict_v8_ADOPTED_20261005.json",
    "t16_dual_scope_10seed.json",
    "dev_full_arms_20261005.json", "dev_full_dose_arms_s43.json", "dev_full_dose_arms_s44.json",
    "dev_full_dose_arms_s45.json", "dev_full_dose_arms_s46.json", "dev_full_dose_arms_s47.json",
    "e1_c0_v7_20261007.json", "e1_c1_lat_20261007.json", "e1_c2_merge_20261007.json",
    "e1_c3_appgate_20261007.json", "e1_c4_v8_20261007.json",
    "e2_dose0x_20261007.json", "e2_dose4x_20261007.json",
    "e3_second_arm_pairfar_20261007.json",
    "r3_acceptance_20261004.json", "dev_pixel_base_20261004.json",
    "dev_pixel_appgate_20261004.json", "dev_scan_lateral_20261004.json",
    "dev_scan_parallel_20261004.json", "r3b_pixel_lat3_20261004.json",
    "r3b_pixel_appgate_20261004.json", "r3_instance_ann_cov.json",
    "final_pool_audit_20261005.json", "collect_isolation_20260927.json",
    "timing_retest_20260927.json", "e1_negative_pool_20260927.json",
    "gpu_ledger_machine.json", "t16_r3_density_dev.json",
    "final_set_v5_20261005/seal/final_set_seal.json",
    "final_set_v5_20261005/seal/confirmation_base6x-seed42.json",
    "final_set_v5_20261005/seal/final_set_ledger.jsonl",
]
#: 驾驶运行（V-E/V-F）：全部 town scorecard/manifest
BENCH_GLOB = ("scorecard_*.json", "manifest_*.json", "town_*.json")
PAPER_FILES = ["paper.md", "paper_zh.md", "supplement.md", "supplement_zh.md",
               "references.bib", "CLAIMS_20261007.md", "REVIEW_20261007.md",
               "REFS_VERIFICATION_20261007.md"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    zp = OUT / f"T16_evidence_{stamp}.zip"
    manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "论文每个数字所依赖的最小产物集；路径为仓库内相对路径",
                "entries": []}
    missing = []
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        def add(p: Path, arc: str) -> None:
            if not p.is_file():
                missing.append(str(p.relative_to(ROOT)))
                return
            z.write(p, arc)
            manifest["entries"].append({"arc": arc, "bytes": p.stat().st_size,
                                        "sha256": sha256(p)})
        for name in EXP_FILES:
            add(E / name, f"experiments/{name}")
        for pat in BENCH_GLOB:
            for p in sorted(BENCH.glob(pat)):
                add(p, f"fsd_benchmark/{p.name}")
        for name in PAPER_FILES:
            add(PAPER / name, f"paper/{name}")
        add(ROOT / "docs" / "T16_PROTOCOL_FACTOR_DECOMP_PREREG_20261007.md",
            "docs/T16_PROTOCOL_FACTOR_DECOMP_PREREG_20261007.md")
        add(ROOT / "docs" / "T16_PROTOCOL_FACTOR_DECOMP_RESULT_20261007.md",
            "docs/T16_PROTOCOL_FACTOR_DECOMP_RESULT_20261007.md")
        add(ROOT / "docs" / "T16_E4_SECOND_MAP_RESULT_20261007.md",
            "docs/T16_E4_SECOND_MAP_RESULT_20261007.md")
        z.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / f"MANIFEST_{stamp}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[archive] {zp} ({zp.stat().st_size/1e6:.2f} MB, {len(manifest['entries'])} 个文件)")
    if missing:
        print(f"[archive] 缺失 {len(missing)} 项（清单里已跳过）: {missing[:6]}")
    print(f"[archive] 清单 -> {OUT / f'MANIFEST_{stamp}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
