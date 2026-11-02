# T16 线分割重训预注册：认证池首训（2026-11-02）

## 目的

用 utah 认证池（3 站点族/15 帧 engine_verified，批次 1+2）重训线分割模型，
同口径 holdout 评测 line_iou_near，对幸存基线 v13b（39 帧旧语料 holdout，
**line_iou_near=0.116**，模型本体已随外部删除损失）做指示性对照。15 帧小池
不构成"顶尖"主张——本轮是可复现的起点数据点。

## 数据与划分

- 池：`logs/experiments/t16_utah2_export_20261101/`（a3s0、a4s0 族）+
  `..._offset5/`（a0s0、a3s0、a4s0 族），每族 3 帧，全部 engine_verified
  （导出门准入，label=annotation 线类）。
- 线像素占比实测：mean 3.9%（min 0.24% / max 8.1%，旧语料 0.35%）。
- 划分：**留一族**——训练 4 族（12 帧；batch1-a3、batch1-a4、offset5-a0、
  offset5-a3），评测留出族 = offset5-a4（3 帧）。跨族天然隔离；族内帧级
  split-gap 0.5 s。

## 训练配置（预注册）

| 项 | 值 |
| --- | --- |
| 初始化 | random（无 --init；谱系显式记录 init_source=random） |
| epochs / batch / lr | 40 / 8 / 1e-3 |
| line-weight / Tversky / clDice | 2.0 / 1.0 / 1.0（微调线通道默认，thin-line 工具启用） |
| paint-source | engine_annotation（**无人工标注**） |
| 设备 | CUDA（torch cu128，实测可用） |
| 输出 | `logs/m5_seg/seg_model_v14_utah_line`（受 seg_model* 保护前缀） |

## 判据

- 主指标：留出族 `m5_eval_seg.py --json` 的 line_iou_near / miou / pixel_acc；
  与 0.116 的对照标注为"外部参照，非同集配对"。
- 训练日志必须记录每帧线占比、损失、内部 val（0.2）曲线；训练失败/退化为
  UNKNOWN 时原样记录，不美化。
- **不晋级**：晋级需池扩量与开发门复审；本轮只产出可复现基线。

## 纪束

不降门、不消费最终集、无人工标注、不晋级模型；模型与报告落盘后即刻归档
（evidence_archives）。
