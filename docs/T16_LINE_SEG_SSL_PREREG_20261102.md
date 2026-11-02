# T16 线专用自监督预注册：去噪自编码器编码器预训练（2026-11-02）

## 臂与理由

训练侧 8 个监督杠杆全部证伪后的剩余路径（CLOSEOUT 可续接路径 1）：
`scripts/m5_ssl_pretrain.py` 用**无标签重建**（输入加噪、目标原图，L1）
预训练 SegUNet 编码器（e1/e2/e3/mid），checkpoint 可被 `--init` 严格加载
（解码器/头随机）。fold-A 探针：SSL 20 epochs 于 fold-A 训练帧
（54 族/162 帧，无标签），随后同配置微调（lw2/40ep/random 解码器）。

## 判据

- fold-A 留出 line IoU 对照 **0.2334**（v26 lw2 基线）。
- 若提升：扩展到完整 4 折复核；若持平/下降：如实记录负结果，
  平台结论不变。
- 输入加噪 σ=0.05；SSL 输出 checkpoint 的 `train_args.init_source="random"`
  （链根），`ssl_pretrain` 字段记录预训练证据。

## 纪束

不降门、不消费最终集、无人工标注、不晋级；产物 `logs/m5_seg/ssl_pretrain_*.pt`
与 `seg_model_v28_*` 哈希归档。
