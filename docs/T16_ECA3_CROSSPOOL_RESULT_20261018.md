# east_coast 首批 LINE=measured 配对训练结果（2026-10-18）

预注册：`docs/T16_ECA3_CROSSPOOL_PREREG_20261018.md`。
E3=6帧 east_coast（2 站 a0/a4、engine_verified、ECA3 冻结导出门准入）、
H=9帧 human；5 seeds `{47..51}`、60 steps、同初始化/8开发包/硬门；
10 作业 0 失败、UNKNOWN=0、unverified=0；跨池审计 105 帧 105 唯一摘要、
0 重复组（预注册内证据）。

## E3−H

| 指标 | mean±std | 正向 | verdict | collapse |
|---|---:|---:|---|---|
| line IoU | −0.2684±0.1617 | 0/5 | champion_better | 2 |
| recall | −0.2846±0.1645 | 0/5 | champion_better | 2 |
| precision | −0.3281±0.5059 | 3/5 | inconclusive | 2 |

逐种子 line-IoU 差值：−0.3510、−0.2883、−0.4841、−0.0978、−0.1210。
collapse=2 对（均 E3 臂：seed47 IoU 0.0013、seed49 三项全 0.0——60 步内
完全退化）；排除 collapse 对后 IoU mean −0.1690、recall −0.1853，**仍为负**。
precision 在非崩坏种子为 0.9518/0.9273/0.9677（预测像素极少的高精度假象），
崩坏种子 0.0641/0.0，故配对 inconclusive（95% CI 半宽 0.628）。
原始 paired verdict 未被任何排除诊断替换。

H 臂复现性：与 WCU6 配对（同种子同协议）逐种子 IoU 最大差 0.0190
（seed47 逐位一致；seed51 0.4096 vs 0.4286），在已记录 0.0356 噪声下限内。

## 结论边界

east 第三图的首个配对数据点与 west 六批中的多数方向一致：6 帧 engine 池
在同初始化/同步预算下**不敌 9 帧人工池**（IoU/recall 5/5 为负，两项
`champion_better`），排除 2 个崩坏对后仍为负。这再次说明准入成功≠学习
收益：east 池准入达标（2站/6帧）不转化为收益优势；小池（3–9 帧）内容、
种子与预算交互继续主导本研究，不建立来源排序。
不晋级模型、不改硬门、不消费最终集、不新增人工标注。
