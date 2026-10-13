# 第二个 LINE=measured 独立池族结果（2026-10-13）

* anchor0：道路链 60m，不满足 2站最小间距约束，布站失败；原样保留；
* anchor1：known_line 站通过，3 帧导出 `engine_verified`，LINE/ROAD measured、
  SHOULDER/ROLE unknown；material_mix 站隔离（near_left coverage 0.4155、
  calibrated residual 2.43px、paint agreement 0.673）；
* 新独立池实际规模：**3 LINE=measured 帧**，不是预期的 12 帧大池；后续只作
  独立资格/训练接线回归，不作稳定学习收益结论。
