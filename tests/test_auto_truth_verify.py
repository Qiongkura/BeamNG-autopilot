"""自动真值校验的行为（T16 P1；冻结契约 ``docs/T16_ORDER0_FREEZE_20260927.md`` §3.4）。

钉住四件事：

1. 独立投影实现与仓库标定（``vision/projection.py``）在同一组相机基下给出同一
   像素：校验器的价值就在于它不是被校验对象的复刻；
2. 五类反例（翻转/尺寸/错帧/遮挡插入/篡改）各自触发**对应**拒绝码，正例全过；
3. 不确定不当通过：没有真值点 = unknown；无线场景由生成器声明 + 无漆线证据
   判 not_applicable，而不是"模型没预测线"；pseudo 档不可能是 measured；
4. 凭证写入：``ok=False`` 时绝不写 ``engine_verified``；``truth_verified`` 对
   旧 sidecar（无字段）与伪造 sidecar（有字符串但没有报告）都是 False。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments import auto_truth as at  # noqa: E402
from beamng_autopilot.experiments.credentials import (  # noqa: E402
    read_dir_credentials, truth_verified,
)
from beamng_autopilot.vision.projection import CameraModel  # noqa: E402


def _batch(**kw):
    kw.setdefault("n_frames", 2)
    return at.make_synthetic_batch(**kw)


def _codes(report) -> set:
    return {r["code"] for r in report["rejections"]}


def _touch(frame) -> None:
    """按内容重算 sha（修改图后调用，避免把"内容变了"误报成 hash 篡改）。"""
    frame["source_image_sha"], frame["label_sha"] = at.frame_content_shas(
        frame["rgb"], frame["label"])


# ── 1. 投影 ─────────────────────────────────────────────────────────────────
def test_projection_of_known_truth_points_matches_the_label():
    batch = _batch(with_line=True)
    for frame in batch["frames"]:
        rep = at.verify_projection(frame)
        assert rep["ok"] is True, rep
        assert rep["checked"] >= 20 and rep["mismatches"] == 0
        assert rep["residual_px"] is not None
        assert rep["residual_px"]["max"] <= at.PROJECTION_RADIUS_PX


def test_independent_projection_agrees_with_the_repo_calibration():
    """同一相机基下，本模块的独立针孔实现与 vision/projection.py 像素一致。"""
    width, height = 536, 403
    model = CameraModel(offset=np.array([0.0, 1.216, 1.386]),
                        fwd_local=np.array([0.0, 0.99994, -0.0112]),
                        up_local=np.array([0.0, 0.0112, 0.99994]),
                        fov_deg=65.0, width=width, height=height)
    pos = np.array([10.0, 20.0, 5.0])
    heading = 0.4
    c, right, fwd, up = model.camera_pose(pos, heading)
    camera = {"width": width, "height": height, "fov_y_deg": 65.0,
              "pose": {"pos": [float(v) for v in c],
                       "basis": {"right": [float(v) for v in right],
                                 "fwd": [float(v) for v in fwd],
                                 "up": [float(v) for v in up]}}}
    points = [pos + 8.0 * fwd + 0.7 * right + 0.3 * up,
              pos + 25.0 * fwd - 2.5 * right - 0.2 * up,
              pos + 5.0 * fwd]
    u_ref, v_ref, valid = model.project(np.asarray(points), pos, heading)
    for i, point in enumerate(points):
        assert bool(valid[i])
        u, v = at.project_point(point, camera)
        assert abs(u - float(u_ref[i])) < 1e-6
        assert abs(v - float(v_ref[i])) < 1e-6


def test_points_behind_the_camera_give_nan_not_a_guess():
    batch = _batch(n_frames=1)
    frame = batch["frames"][0]
    cam = frame["camera"]
    pos = np.asarray(cam["pose"]["pos"], dtype=float)
    fwd = at.camera_basis(cam)[2]
    u, v = at.project_point(pos - 5.0 * fwd, cam)
    assert np.isnan(u) and np.isnan(v)


def test_a_truth_point_without_its_class_is_a_projection_mismatch():
    batch = _batch(n_frames=1)
    frame = batch["frames"][0]
    cam = frame["camera"]
    w = int(cam["width"])
    h = int(cam["height"])
    road_pt = None
    for p in frame["truth_points"]:
        if int(p.get("class", 0)) != 1:
            continue
        u, v = at.project_point(p["world"], cam)
        if 4 <= u < w - 4 and 4 <= v < h - 4:
            road_pt = (p, u, v)
            break
    assert road_pt is not None, "场景里应有可见的路面真值点"
    _p, u, v = road_pt
    x, y = int(round(u)), int(round(v))
    lab = frame["label"].copy()
    lab[y - 4:y + 5, x - 4:x + 5] = 0            # 该处不再有 class 1（声明点还在）
    frame["label"] = lab
    frame["label_sha"] = at.label_sha16(lab)
    rep = at.verify_projection(frame)
    assert rep["mismatches"] >= 1 and rep["ok"] is False
    assert at.PROJECTION_MISMATCH in _codes(at.verify_batch(batch))


# ── 2. 帧身份 / 反例 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind,code", [
    ("flip", at.CAMERA_FLIP),
    ("resize", at.RESOLUTION_MISMATCH),
    ("time_shift", at.TIME_MISMATCH),
    ("occlude", at.OCCLUSION_INSERT),
    ("tamper", at.LABEL_TAMPERED),
])
def test_each_counterexample_is_rejected_by_its_own_code(kind, code):
    base = _batch(with_line=True)
    report = at.verify_batch(at.inject(base, kind))
    assert report["ok"] is False
    assert code in _codes(report), (kind, _codes(report))


def test_the_clean_base_batch_passes_and_is_not_mutated_by_inject():
    base = _batch(with_line=True)
    before = base["frames"][0]["label"].copy()
    assert at.verify_batch(base)["ok"] is True
    at.inject(base, "flip")
    assert np.array_equal(base["frames"][0]["label"], before)
    r = at.verify_batch(base)
    assert r["ok"] is True and not r["rejections"]


def test_the_flip_injection_also_loses_projection_but_flip_is_reported():
    """翻转同时会让真值点对不上（同一现象的两个证据）；两个码都保留，不藏证据。"""
    codes = _codes(at.verify_batch(at.inject(_batch(with_line=True), "flip")))
    assert at.CAMERA_FLIP in codes
    assert at.PROJECTION_MISMATCH in codes


def test_black_frame_is_rejected():
    batch = _batch(n_frames=1)
    frame = batch["frames"][0]
    assert at.detect_black_frame(frame["rgb"]) is False
    frame["rgb"][:] = 0
    _touch(frame)
    assert at.detect_black_frame(frame["rgb"]) is True
    report = at.verify_batch(batch)
    assert at.BLACK_FRAME in _codes(report)
    assert at.detect_black_frame(None) is True
    assert at.detect_black_frame(np.zeros((0, 0, 3), np.uint8)) is True


def test_a_resized_channel_is_a_resolution_mismatch():
    batch = _batch(n_frames=1)
    frame = batch["frames"][0]
    assert at.frame_identity_ok(frame) == []
    frame["depth"] = frame["depth"][::2, ::2]
    assert at.RESOLUTION_MISMATCH in at.frame_identity_ok(frame)
    # 相机声明与实际图像不符（同一码）
    frame2 = dict(_batch(n_frames=1)["frames"][0])
    frame2["camera"] = dict(frame2["camera"])
    frame2["camera"]["width"] = frame2["camera"]["width"] + 1
    assert at.RESOLUTION_MISMATCH in at.frame_identity_ok(frame2)


def test_an_async_stale_channel_is_rejected():
    batch = _batch(n_frames=1)
    frame = batch["frames"][0]
    frame["channel_ids"] = {"rgb": frame["frame_id"], "annotation": frame["frame_id"],
                            "depth": "older_depth_frame"}
    assert at.ASYNC_STALE_FRAME in at.frame_identity_ok(frame)
    report = at.verify_batch(batch)
    assert at.ASYNC_STALE_FRAME in _codes(report)
    assert at.TIME_MISMATCH not in _codes(report)


def test_a_missing_timestamp_or_a_backwards_one_is_a_time_mismatch():
    batch = _batch(n_frames=2)
    batch["frames"][1]["timestamp"] = float(batch["frames"][0]["timestamp"]) - 1.0
    report = at.verify_batch(batch)
    assert at.TIME_MISMATCH in _codes(report)
    frame = dict(batch["frames"][0])
    frame["timestamp"] = None
    assert at.TIME_MISMATCH in at.frame_identity_ok(frame)
    fresh = _batch(n_frames=1)["frames"][0]
    ts = float(fresh["timestamp"])
    assert at.TIME_MISMATCH not in at.frame_identity_ok(fresh, prev_timestamp=0.0)
    assert at.TIME_MISMATCH in at.frame_identity_ok(fresh,
                                                    prev_timestamp=ts + 1.0)
    # 同曝光多视角共享时间戳是合法的：只拒绝倒退，不要求严格递增
    assert at.TIME_MISMATCH not in at.frame_identity_ok(fresh, prev_timestamp=ts)


# ── 3. 调色板 ───────────────────────────────────────────────────────────────
def test_a_changed_palette_is_rejected_and_unregistered_colours_are_flagged():
    batch = _batch(n_frames=1)
    batch["recorded_palette"] = {
        "version": batch["palette"]["version"],
        "classes": {"road": [1, 2, 3], "line": [4, 5, 6]}}
    report = at.verify_batch(batch)
    assert at.PALETTE_CHANGED in _codes(report)

    batch2 = _batch(n_frames=1)
    batch2["frames"][0]["annotation"][5, 5] = (7, 7, 7)   # 未登记类色
    report2 = at.verify_batch(batch2)
    assert at.UNKNOWN_CLASS in _codes(report2)


def test_palette_helpers_are_content_addressed():
    p = _batch(n_frames=1)["palette"]
    same = json.loads(json.dumps(p))
    assert at.palette_sha(p) == at.palette_sha(same)
    assert at.check_palette(p, same) == []
    forged = dict(same)
    forged["sha"] = "deadbeefdeadbeef"
    assert at.PALETTE_CHANGED in at.check_palette(p, forged)
    assert at.check_palette({}, None) == [at.PALETTE_CHANGED]


# ── 4. 标签 ─────────────────────────────────────────────────────────────────
def test_label_tamper_and_invalid_values_are_both_refused():
    batch = _batch(n_frames=1)
    frame = batch["frames"][0]
    assert at.check_label(frame) == []
    tampered = dict(frame)
    lab = frame["label"].copy()
    lab[0, 0] = 2
    tampered["label"] = lab                      # 内容变了，hash 没跟着变
    assert at.LABEL_TAMPERED in at.check_label(tampered)
    invalid = dict(frame)
    lab2 = frame["label"].copy()
    lab2[0, 0] = 7                               # 取值域之外
    invalid["label"] = lab2
    invalid["label_sha"] = at.label_sha16(lab2)
    assert at.LABEL_TAMPERED in at.check_label(invalid)
    no_hash = dict(frame)
    no_hash["label_sha"] = ""
    assert at.LABEL_TAMPERED in at.check_label(no_hash)


# ── 5. 遮挡 ─────────────────────────────────────────────────────────────────
def test_the_occlusion_scene_marks_the_hidden_line_as_ignore():
    """正确地"看不见就不判"的场景必须过：hidden 区是 255，不是 0 也不是 2。"""
    batch = _batch(with_line=True, with_occluder=True)
    report = at.verify_batch(batch)
    assert report["ok"] is True, report["rejections"]
    assert report["stats"]["occlusion"]["occluded_px"] > 0
    audit = at.occlusion_audit(batch["frames"][0])
    assert audit["status"] == "measured" and audit["ignored_px"] > 0
    assert audit["line_marked_px"] == 0 and audit["background_marked_px"] == 0


def test_occlusion_kept_as_line_is_an_insert():
    batch = at.inject(_batch(with_line=True), "occlude", keep_line=True)
    audit = at.occlusion_audit(batch["frames"][0])
    assert audit["line_marked_px"] > 0
    assert at.OCCLUSION_INSERT in audit["codes"]
    assert at.OCCLUSION_INSERT in _codes(at.verify_batch(batch))


def test_occlusion_relabelled_as_background_is_also_an_insert():
    """把被挡的线当背景 = 制造假负例，同样拒绝（"遮挡不当漏检"不靠 label=0）。"""
    batch = at.inject(_batch(with_line=True), "occlude", keep_line=False)
    audit = at.occlusion_audit(batch["frames"][0])
    assert audit["background_marked_px"] > 0
    assert audit["line_marked_px"] == 0
    assert at.OCCLUSION_INSERT in audit["codes"]
    assert at.OCCLUSION_INSERT in _codes(at.verify_batch(batch))


def test_no_line_truth_means_occlusion_is_not_applicable_not_pass():
    audit = at.occlusion_audit(_batch(with_line=False, n_frames=1)["frames"][0])
    assert audit["status"] == "not_applicable" and audit["codes"] == []


# ── 6. 翻转 ─────────────────────────────────────────────────────────────────
def test_flip_detection_uses_truth_points_and_the_mirror_fallback():
    batch = _batch(n_frames=1)
    assert at.detect_camera_flip(batch["frames"][0]) is False
    flipped = at.inject(batch, "flip")["frames"][0]
    assert at.detect_camera_flip(flipped) is True
    # 无真值点时：只镜像 annotation（一路镜像的采集错误）也要能发现
    frame = batch["frames"][0]
    frame["truth_points"] = None
    frame["annotation"] = np.ascontiguousarray(frame["annotation"][:, ::-1])
    frame["label"] = np.ascontiguousarray(frame["label"][:, ::-1])
    frame["label_sha"] = at.label_sha16(frame["label"])
    audit = at.flip_audit(frame)
    assert audit["source"] == "mirror_consistency"
    assert at.detect_camera_flip(frame) is True


# ── 7. 逐通道资格 ───────────────────────────────────────────────────────────
def test_a_line_free_scene_is_not_applicable_and_that_does_not_back_other_channels():
    report = at.verify_batch(_batch(with_line=False))
    ch = report["channels"]
    assert ch["LINE"]["status"] == "not_applicable"
    assert ch["ROLE"]["status"] == "not_applicable"
    assert ch["ROAD"]["status"] == "measured", "无线不替路面通道背书，路面也有自己的证据"
    assert "line_generated=False" in ch["LINE"]["why"]
    assert "line_texture_embedded=False" in ch["LINE"]["why"]
    assert ch["LINE"]["n"] == 2


def test_a_declared_line_free_scene_that_shows_lines_is_unknown_not_na():
    batch = _batch(with_line=False)
    batch["frames"][0]["label"][:2, :2] = 2       # 声明无线却有漆线像素
    report = at.verify_batch(batch)
    assert report["channels"]["LINE"]["status"] == "unknown"
    assert "contradicts" in report["channels"]["LINE"]["why"]


def test_pseudo_labels_are_never_measured():
    batch = _batch(with_line=True, label_source="pseudo")
    ch = at.verify_batch(batch)["channels"]
    assert ch["LINE"]["status"] == "unverified_labels"
    assert ch["ROLE"]["status"] == "unverified_labels"
    assert "pseudo" in ch["LINE"]["why"]


def test_material_mix_gets_shoulder_and_roles_independently():
    ch = at.verify_batch(_batch(with_line=True, second_line=True,
                                material_mix=True))["channels"]
    assert ch["SHOULDER"]["status"] == "measured"
    assert ch["ROLE"]["status"] == "measured"
    assert ch["LINE"]["status"] == "measured"


def test_valid_area_and_rejection_rate_are_reported_separately():
    """ignore 像素既不算假阳也不算漏检，但它有多少必须看得见（冻结文档 §3.4）。"""
    report = at.verify_batch(_batch(with_line=True, with_occluder=True))
    area = report["stats"]["valid_area"]
    assert area["total_px"] == 192 * 144 * 2
    assert area["valid_px"] + area["ignore_px"] == area["total_px"]
    assert area["ignore_px"] > 0 and 0.0 < area["ignore_frac"] < 1.0
    assert report["stats"]["rejection_rate"] == {"frames_rejected": 0,
                                                 "frames_total": 2,
                                                 "rejections": 0}
    bad = at.verify_batch(at.inject(_batch(with_line=True), "tamper"))
    assert bad["stats"]["rejection_rate"]["frames_rejected"] == 2


def test_missing_truth_points_leave_projection_unknown_and_not_a_pass_claim():
    """没有真值点 = 未测；ok 只表示"没有拒绝码"，证据计数必须显示 0。"""
    batch = _batch(with_line=True, n_frames=1)
    batch["frames"][0]["truth_points"] = []
    report = at.verify_batch(batch)
    assert report["ok"] is True and not report["rejections"]
    assert report["stats"]["projection"]["checked"] == 0
    assert report["channels"]["LINE"]["status"] == "unknown"
    assert report["stats"]["evidence"]["line_truth_frames"] == 0


def test_expected_channels_gate_the_batch_result():
    batch = _batch(with_line=False)
    good = at.verify_batch(batch, expected={"LINE": "not_applicable",
                                            "ROAD": "measured"})
    assert good["ok"] is True and not good["stats"]["expected_mismatches"]
    bad = at.verify_batch(batch, expected={"channels": {"LINE": "measured"}})
    assert bad["ok"] is False
    assert bad["stats"]["expected_mismatches"][0]["channel"] == "LINE"


# ── 8. 凭证 ─────────────────────────────────────────────────────────────────
def _provenance(**report):
    return {
        "generator": {"name": "make_synthetic_batch", "version": "1",
                      "sha": "abc123abc123abc1"},
        "asset": {"map": "synthetic_grid", "segment": "probe_straight",
                  "sha": "asset001"},
        "run": {"id": "run-1", "scene_seed": 0, "game_version": "synthetic",
                "renderer": "raycast_v1"},
        "camera": {"name": "front_main", "calibration_sha": "cal001"},
        "labels": {"unknown_reason": ""},
        "report": dict({"verified": True}, **report),
    }


def test_engine_verified_is_written_only_when_report_passes_and_generator_claims_it(tmp_path):
    batch = _batch(with_line=True)
    ok_report = at.verify_batch(batch)
    bad_report = at.verify_batch(at.inject(batch, "tamper"))

    d_ok = tmp_path / "ok"
    blob = at.write_truth_credentials(d_ok, batch_report=ok_report,
                                      provenance=_provenance())
    assert blob["label_source"] == "engine_verified"
    assert blob["truth_contract"] == "v1"
    assert blob["truth_provenance"]["report"]["verified"] is True
    assert blob["truth_provenance"]["report"]["verifier_version"]
    assert blob["truth_provenance"]["labels"]["label_sha"]
    assert [f["label_sha"] for f in blob["frames"]] and \
        all(f["source_image_sha"] for f in blob["frames"])
    cred = read_dir_credentials(d_ok)
    assert cred["truth_verified"] is True
    assert cred["truth_provenance"]["report"]["verified"] is True

    d_bad = tmp_path / "bad"
    blob_bad = at.write_truth_credentials(d_bad, batch_report=bad_report,
                                          provenance=_provenance())
    assert blob_bad["label_source"] == ""
    assert blob_bad["truth_provenance"]["report"]["verified"] is False
    assert "ok is False" in blob_bad["why"]
    assert read_dir_credentials(d_bad)["truth_verified"] is False

    # 报告过了但生成器没有 verified 声明：仍不写 engine_verified
    d_claim = tmp_path / "noclaim"
    blob_claim = at.write_truth_credentials(
        d_claim, batch_report=ok_report,
        provenance=_provenance(verified=False))
    assert blob_claim["label_source"] == ""
    assert "did not claim" in blob_claim["why"]
    assert read_dir_credentials(d_claim)["truth_verified"] is False


def test_write_truth_credentials_merges_an_existing_sidecar(tmp_path):
    d = tmp_path / "merge"
    d.mkdir()
    (d / "annotation.json").write_text(json.dumps(
        {"label_source": "human_revision", "reviewer": "owner",
         "frames": [{"frame_id": "old_frame", "path": "front/old.npz"}]}),
        encoding="utf-8")
    report = at.verify_batch(_batch(with_line=True))
    blob = at.write_truth_credentials(d, batch_report=report,
                                      provenance=_provenance())
    assert blob["label_source"] == "engine_verified"
    assert blob["previous_label_source"] == "human_revision"
    assert blob["reviewer"] == "owner"
    ids = {f.get("frame_id") for f in blob["frames"]}
    assert "old_frame" in ids and any(str(i).startswith("front_main")
                                      for i in ids)
    assert json.loads((d / "annotation.json").read_text(encoding="utf-8"))[
        "truth_contract"] == "v1"


def test_truth_verified_is_false_for_old_and_forged_sidecars(tmp_path):
    old = {"label_source": "human_revision", "frames": [{"path": "a"}]}
    assert truth_verified(old) is False
    forged = {"label_source": "engine_verified", "truth_contract": "v1",
              "truth_provenance": {"generator": {"sha": "x"},
                                   "labels": {"label_sha": "y"},
                                   "report": {"verifier_version": "auto_truth/1"}}}
    assert truth_verified(forged) is False, "没有 report.verified 就不是 verified"

    def _variant(**report_keys):
        blob = json.loads(json.dumps(forged))
        blob["truth_provenance"]["report"]["verified"] = True
        blob["truth_provenance"]["report"].update(report_keys)
        return blob

    assert truth_verified(_variant(verifier_version="")) is False, "缺校验器版本"
    no_sha = _variant()
    no_sha["truth_provenance"]["generator"]["sha"] = ""
    assert truth_verified(no_sha) is False, "缺生成器 sha"
    no_label = _variant()
    no_label["truth_provenance"]["labels"]["label_sha"] = ""
    assert truth_verified(no_label) is False, "缺标签 sha"
    assert truth_verified(_variant()) is True
    wrong_contract = _variant()
    wrong_contract["truth_contract"] = "v2"
    assert truth_verified(wrong_contract) is False
    assert truth_verified({"truth_contract": "v1"}) is False

    d = tmp_path / "old_dir"
    d.mkdir()
    (d / "annotation.json").write_text(json.dumps(old), encoding="utf-8")
    cred = read_dir_credentials(d)
    assert cred["truth_provenance"] is None and cred["truth_verified"] is False
    assert cred["label_source"] == "human_revision" and cred["frames"] == 1


def test_the_written_report_hash_covers_the_batch_report(tmp_path):
    report = at.verify_batch(_batch(n_frames=1))
    blob = at.write_truth_credentials(tmp_path / "sha", batch_report=report,
                                      provenance=_provenance())
    sha = blob["truth_provenance"]["report"]["test_report_sha"]
    assert len(sha) == 16
    report2 = json.loads(json.dumps(report))
    blob2 = at.write_truth_credentials(tmp_path / "sha2", batch_report=report2,
                                       provenance=_provenance())
    assert blob2["truth_provenance"]["report"]["test_report_sha"] == sha


def test_an_empty_batch_is_not_ok_and_cannot_be_certified(tmp_path):
    report = at.verify_batch({})
    assert report["ok"] is False and report["stats"]["empty_batch"] is True
    assert report["stats"]["n_frames"] == 0
    blob = at.write_truth_credentials(tmp_path / "empty", batch_report=report,
                                      provenance=_provenance())
    assert blob["label_source"] == ""
    assert read_dir_credentials(tmp_path / "empty")["truth_verified"] is False


def test_rejection_codes_are_declared_constants():
    assert set(at.REJECTION_CODES) == {
        at.CAMERA_FLIP, at.RESOLUTION_MISMATCH, at.TIME_MISMATCH,
        at.ASYNC_STALE_FRAME, at.OCCLUSION_INSERT, at.LABEL_TAMPERED,
        at.BLACK_FRAME, at.PALETTE_CHANGED, at.UNKNOWN_CLASS,
        at.PROJECTION_MISMATCH}
    for report in (at.verify_batch(_batch(n_frames=1)),
                   at.verify_batch(at.inject(_batch(n_frames=1), "flip"))):
        for rejection in report["rejections"]:
            assert rejection["code"] in at.REJECTION_CODES
            assert "frame_id" in rejection and "detail" in rejection
