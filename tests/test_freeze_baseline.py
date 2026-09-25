"""冻结基线工具（`scripts/m5_freeze_baseline.py`）的确定性。

W0 的快照要能回答"这一轮到底读了什么、更新了什么、谁持有控制权"：
读不到的记 UNKNOWN（不写 0）、未提交差异标记归属、W0 只读不更新。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.protocol import PROTOCOL_VERSION  # noqa: E402


def test_the_freeze_snapshot_records_the_protocol_and_unknowns(tmp_path):
    """冻结快照：读不到的东西记 UNKNOWN，不写 0；协议与哈希进快照。"""
    spec = importlib.util.spec_from_file_location(
        "m5_freeze_baseline", ROOT / "scripts" / "m5_freeze_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_freeze_baseline"] = mod
    spec.loader.exec_module(mod)

    snap = mod.build_snapshot(
        run_id="w0_test",
        models=["production=/no/such/model.pt"],
        run_dirs=["/no/such/dir=agent_revision"],
        thresholds=Path("/no/such/thresholds.json"))
    assert snap["protocol_hash"] and len(snap["protocol_hash"]) == 16
    assert snap["protocol"]["version"] == PROTOCOL_VERSION
    assert snap["models"][0]["status"] == "UNKNOWN"
    assert "error" in snap["models"][0]
    assert snap["runs"][0]["status"] == "UNKNOWN"
    assert snap["thresholds_file"]["status"] == "UNKNOWN"
    assert snap["control_ownership"]["game_session_started"] is False
    assert snap["control_ownership"]["vehicle_control"] == "none"
    assert snap["will_read"], "必须列出将要读取的路径"
    assert snap["will_update"] == [], "W0 只读，不更新任何目录"
    assert "dirty_note" in snap["git"], "未提交差异要标记归属"
    assert snap["env"]["camera_resolution_px"] == [536, 403]
