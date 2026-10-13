# 第二 LINE=measured 独立池小训练回归预注册修订（2026-10-13）

基于 `T16_LINE_MEASURED_LARGE2_RESULT_20261013.md`：预期大池未达成，但 3 帧
LINE=measured 新 anchor pool 已通过审计。保持前一轮协议：同初始化、5 seeds、60 steps、
同 8 个开发包、硬门；本轮仅验证新池的资格掩码和逐种子评测，不作收益结论。

输出 `logs/experiments/t16_line_measured_large2_smoke_20261013/seed{47..51}`；
记录 train_hist/checkpoint channel_qualification、line IoU、recall、precision、collapse、
UNKNOWN。精度无有效分母必须保持 UNKNOWN。
