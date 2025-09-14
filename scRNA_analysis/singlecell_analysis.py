#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅保留：Scanpy 基本聚类（PCA/邻居/UMAP）与 Harmony 批次校正及对应可视化。
不包含任何 VAE/CVAE 训练与相关代码。
"""

import os
import numpy as np
import scanpy as sc
import scanpy.external as sce
import scipy.sparse as sp
import matplotlib.pyplot as plt
import argparse

# 基本路径（相对当前脚本）
BASE_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(PROJ_ROOT, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_comparison')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Scanpy 参数
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white')

# 输入数据路径（可按需修改）
IN_ADATA = os.path.join(DATA_DIR, 'GSE149614_HCC_scRNA.h5ad')

# 命令行参数：支持 --adata 与 --outdir 覆盖默认路径
parser = argparse.ArgumentParser(description="Scanpy 基本聚类与 Harmony 批次校正（UMAP 可视化）")
parser.add_argument("--adata", type=str, default=IN_ADATA, help="输入 .h5ad 文件路径")
parser.add_argument("--outdir", type=str, default=RESULTS_DIR, help="输出目录，用于保存 SVG 图")
# 新增：聚类算法与分辨率（默认与 Scanpy 常见实践一致：Leiden, 1.0）
parser.add_argument("--cluster", type=str, choices=["none", "leiden", "louvain"], default="leiden", help="邻居图后执行的聚类算法")
parser.add_argument("--resolution", type=float, default=1.0, help="聚类分辨率")
args = parser.parse_args()
IN_ADATA = args.adata
RESULTS_DIR = args.outdir
os.makedirs(RESULTS_DIR, exist_ok=True)

print('=' * 60)
print('Step 1: 读取 h5ad')
print('=' * 60)
adata = sc.read_h5ad(IN_ADATA)
print(f"adata 形状: {adata.shape}")
print(f"obs 列: {list(adata.obs.columns)}")

# 简单数据清理：去除潜在的 NaN/Inf
print('\n数据清理: 检查 NaN/Inf...')
try:
    X = adata.X
    if sp.issparse(X):
        # 稀疏矩阵直接检查很慢，这里转换为密集仅用于 NaN/Inf 检测与清理（如内存不足可跳过）
        X = X.toarray()
    X = np.asarray(X)
    nan_cnt = np.isnan(X).sum()
    inf_cnt = np.isinf(X).sum()
    if nan_cnt > 0 or inf_cnt > 0:
        print(f"发现 NaN={nan_cnt}, Inf={inf_cnt}，将置零。")
        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0
        adata.X = X
    else:
        print('未发现 NaN/Inf。')
except Exception as e:
    print(f"数据清理失败（忽略继续）: {e}")

print('\n' + '=' * 60)
print('Step 2: Scanpy 基线 PCA/邻居/UMAP + 聚类')
print('=' * 60)
adata_scanpy = adata.copy()

# 判断是否为原始 counts（粗略阈值）并执行标准化 + log1p
try:
    Xmax = (adata_scanpy.X.max() if not sp.issparse(adata_scanpy.X) else adata_scanpy.X.max())
    Xmax_val = float(Xmax) if hasattr(Xmax, 'tolist') else float(Xmax)
except Exception:
    Xmax_val = 1.0

if Xmax_val > 50:
    sc.pp.normalize_total(adata_scanpy, target_sum=1e4)
    sc.pp.log1p(adata_scanpy)
else:
    print('检测到数据似为已预处理表达，跳过 normalize/log1p')

try:
    sc.pp.scale(adata_scanpy, max_value=10)
except Exception as e:
    print(f"scale 失败（可能数据过大或已标准化）: {e}")

# PCA -> 邻居 -> UMAP
try:
    sc.tl.pca(adata_scanpy, svd_solver='arpack', n_comps=50)
except Exception as e:
    print(f"PCA 失败: {e}\n退化为直接基于原始表达建图（可能较慢）")

try:
    # 若 PCA 成功，则 adata_scanpy.obsm['X_pca'] 存在；否则使用默认表达
    if 'X_pca' in adata_scanpy.obsm_keys():
        sc.pp.neighbors(adata_scanpy, n_neighbors=15, n_pcs=40)
    else:
        sc.pp.neighbors(adata_scanpy, n_neighbors=15)
    sc.tl.umap(adata_scanpy)
    # 聚类（根据参数执行）
    if args.cluster == 'leiden':
        sc.tl.leiden(adata_scanpy, resolution=args.resolution)
    elif args.cluster == 'louvain':
        sc.tl.louvain(adata_scanpy, resolution=args.resolution)
    else:
        print('未执行聚类（--cluster none）')
except Exception as e:
    print(f"邻居/UMAP/聚类 失败: {e}")

# 保存 Scanpy 基线 UMAP 到原 adata 并可视化
if 'X_umap' in adata_scanpy.obsm_keys():
    adata.obsm['scanpy_umap'] = adata_scanpy.obsm['X_umap']
    try:
        # 分别按 sample / celltype / cluster 输出
        saved_any = False
        color_keys = []
        for key in ['sample', 'celltype', 'leiden', 'louvain']:
            if key in adata_scanpy.obs.columns:
                color_keys.append(key)
        if len(color_keys) > 0:
            plt.figure(figsize=(10, 6))
            sc.pl.embedding(adata_scanpy, basis='X_umap', color=color_keys, show=False, frameon=False)
            plt.tight_layout()
            fname = f"scanpy_umap_by_{'_'.join(color_keys)}.svg"
            plt.savefig(os.path.join(RESULTS_DIR, fname), bbox_inches='tight')
            plt.close()
            print(f"已保存: {os.path.join(RESULTS_DIR, fname)}")
            saved_any = True
        if not saved_any:
            # 若都没有可用的 obs 列，则输出不着色版本
            plt.figure(figsize=(10, 6))
            sc.pl.embedding(adata, basis='scanpy_umap', title='Scanpy UMAP', show=False, frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_umap.svg'), bbox_inches='tight')
            plt.close()
            print(f"已保存: {os.path.join(RESULTS_DIR, 'scanpy_umap.svg')}")
    except Exception as e:
        print(f"保存 scanpy_umap 失败: {e}")
else:
    print('未生成 Scanpy UMAP，跳过保存。')

print('\n' + '=' * 60)
print('Step 3: Harmony 批次校正 + UMAP')
print('=' * 60)
try:
    adata_harmony = adata_scanpy.copy()
    # 确保存在 PCA
    if 'X_pca' not in adata_harmony.obsm_keys():
        sc.tl.pca(adata_harmony, svd_solver='arpack', n_comps=50)
    # 基于 sample 进行 Harmony 校正
    if 'sample' not in adata_harmony.obs.columns:
        raise RuntimeError("未找到 obs['sample']，无法进行 Harmony 批次校正")
    sce.pp.harmony_integrate(adata_harmony, key='sample', basis='X_pca', adjusted_basis='X_pca_harmony')
    sc.pp.neighbors(adata_harmony, use_rep='X_pca_harmony', n_neighbors=15)
    sc.tl.umap(adata_harmony)
    # 聚类（可选，与上面基线一致）
    if args.cluster == 'leiden':
        sc.tl.leiden(adata_harmony, resolution=args.resolution)
    elif args.cluster == 'louvain':
        sc.tl.louvain(adata_harmony, resolution=args.resolution)

    adata.obsm['harmony_umap'] = adata_harmony.obsm['X_umap']
    try:
        # 分别按 sample / celltype / cluster 输出
        saved_any = False
        color_keys = []
        for key in ['sample', 'celltype', 'leiden', 'louvain']:
            if key in adata_harmony.obs.columns:
                color_keys.append(key)
        if len(color_keys) > 0:
            plt.figure(figsize=(10, 6))
            sc.pl.embedding(adata_harmony, basis='X_umap', color=color_keys, title=f'Harmony UMAP', show=False, frameon=False)
            plt.tight_layout()
            fname = f"harmony_umap_by_{'_'.join(color_keys)}.svg"
            plt.savefig(os.path.join(RESULTS_DIR, fname), bbox_inches='tight')
            plt.close()
            print(f"已保存: {os.path.join(RESULTS_DIR, fname)}")
            saved_any = True
        if not saved_any:
            # 若既无 sample 也无 celltype，则输出不着色版本
            plt.figure(figsize=(10, 6))
            sc.pl.embedding(adata_harmony, basis='X_umap', title='Harmony UMAP', show=False, frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'harmony_umap.svg'), bbox_inches='tight')
            plt.close()
            print(f"已保存: {os.path.join(RESULTS_DIR, 'harmony_umap.svg')}")
    except Exception as e:
        print(f"保存 harmony_umap 失败: {e}")
except Exception as e:
    print(f"Harmony 执行失败：{e}。如需使用，请先安装 harmonypy：pip install harmonypy")

print('\n流程完成。已生成 scanpy/harmony 的基线可视化与聚类（若启用）。')