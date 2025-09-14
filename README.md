# VAE_models（scRNA 分析）

一个用于单细胞 RNA 测序（scRNA-seq）表征学习与批次效应整合的轻量项目。集成 Scanpy 基线流程、Harmony 批次校正，以及基于负二项分布（NB）的 VAE/CVAE 表征学习，并对不同方法的细胞分离效果、批次效应去除效果进行对比评估。

## 功能概览
- Scanpy 基线：Normalize → Log1p → HVG → Scale → PCA → 邻居图 → UMAP → Leiden。
- Harmony 整合（可选）：基于 PCA 的批次效应校正与可视化。
- VAE_NB：
  - 支持 Gaussian/NB 重构（默认 NB）。
  - NB 路径以原始 counts 为目标，解码输出与 size_factors 相乘（mu_eff = mu × sf）后对齐目标。
  - KL 调度与 free-bits 支持，数值更稳定。
- CVAE_NB：
  - 条件版 VAE，支持多个离散条件（Embedding）与连续条件；同样支持 Gaussian/NB 重构。
  - 可用于以“样本/批次”等为条件进行整合。
- 评估与输出：
  - 生成多种表示（Scanpy PCA / Harmony / VAE_NB / CVAE_NB）下的 UMAP 图。
  - 绘制训练损失曲线（Total / Recon / KL）与方法对比图。
  - 计算基于细胞类型的轮廓系数（silhouette score）并出图。
  - 导出集成结果 h5ad，便于后续分析。

## 快速开始
1) 安装依赖（示例）：
```
pip install -U torch scanpy numpy pandas scikit-learn scipy matplotlib
# 可选：提升 CSV 解析性能
pip install pyarrow
# 可选：Harmony 批次校正
pip install harmonypy
```

2) 运行完整流程（包含 Scanpy → Harmony → VAE_NB → CVAE_NB → 评估与导出）：
```
python scRNA_scanpy_VAE_CVAE_NB.py
```
- 默认数据：`data/GSE149614_HCC_scRNA_test_5N5T_counts.h5ad`
- 结果目录：`results/`
  - UMAP 可视化图片
  - 训练损失曲线与方法对比图
  - 轮廓系数对比图 `silhouette_score_comparison.png`
  - 集成结果 `integrated_analysis_results.h5ad`

## 常用配置
- 数据路径：`DATA_PATH`
- 训练参数：`epochs`、`batch_size`、`latent_dim`、`lr`
- KL 相关：`beta_final`、`kl_schedule`（cosine/cyclical/linear）与 `free_bits`
- CVAE 条件：在 `make_counts_training_tensors_v2` 中设置 `discrete_key`（支持单列或列表）与 `cont_keys`
- 重构分布：初始化模型时设置 `recon_dist="gaussian"|"nb"`


## 数据要求与注意事项
- 输入建议为包含原始计数的 h5ad：`adata.X` 为 counts；`adata.obs` 至少包含 `celltype`、`sample` 等元数据列（按需作为条件）。
- NB 路径下：训练目标是原始 counts；损失在 `mu × size_factors` 与 counts 之间计算。
- 资源设置：显存/内存不足时可调小 `batch_size`，或在张量构建时开启 HVG 子集（`use_hvgs=True` 并设置 `n_top_genes`）。