# # 单细胞RNA测序数据分析：从Scanpy到VAE/CVAE完整流程
# 
# 本notebook展示了如何使用传统Scanpy方法以及深度学习VAE/CVAE方法进行单细胞数据分析和批次效应校正

# ## 1. 环境准备和数据加载

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 设置Scanpy参数
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# 添加项目路径
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# 导入项目模块
from data_processing import make_counts_training_tensors_v2
from VAE_NB.models import VAE
from VAE_NB.training import train_vae
from CVAE_NB.models import CVAE
from CVAE_NB.training import train as train_cvae

print("✅ 环境配置完成")

# 数据路径配置
DATA_PATH = 'data/GSE149614_HCC_scRNA_test_5N5T_counts.h5ad'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 加载原始计数数据
adata_counts = sc.read_h5ad(DATA_PATH)
print(f"📊 数据维度: {adata_counts.shape}")
print(f"📋 细胞类型: {adata_counts.obs['celltype'].value_counts()}")
print(f"🧪 样本信息: {adata_counts.obs['sample'].value_counts()}")

# ### 工具函数定义

def ensure_size_factors(adata):
    """确保数据包含size_factors信息"""
    if 'size_factors' in adata.obs:
        return
    X = adata.X
    if hasattr(X, 'sum'):
        totals = np.asarray(X.sum(axis=1)).reshape(-1) if not hasattr(X.sum(axis=1), 'A1') else X.sum(axis=1).A1
    else:
        totals = X.sum(axis=1)
    med = np.median(totals[totals>0]) if np.any(totals>0) else 1.0
    sf = totals/(med if med>0 else 1.0)
    sf = np.where(sf<=0, 1.0, sf).astype(np.float32)
    adata.obs['size_factors'] = sf

def standardize_latents(latents):
    """标准化潜在表示"""
    return (latents - latents.mean(axis=0)) / (latents.std(axis=0) + 1e-8)

def save_training_curves(loss_list, recon_list, kl_list, title, save_path=None):
    """绘制并保存训练损失曲线"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(loss_list) + 1)
    
    ax1.plot(epochs, loss_list, 'b-', linewidth=2)
    ax1.set_title(f'{title} - Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, recon_list, 'r-', linewidth=2)
    ax2.set_title(f'{title} - Reconstruction Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reconstruction Loss')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epochs, kl_list, 'g-', linewidth=2)
    ax3.set_title(f'{title} - KL Divergence')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('KL Divergence')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

print("🔧 工具函数定义完成")

# ## 2. Scanpy传统分析流程

def run_scanpy_pipeline(adata_counts, n_top_genes=2000):
    """运行标准的Scanpy单细胞分析流程"""
    print("🔬 开始Scanpy分析流程...")
    
    adata = adata_counts.copy()
    ensure_size_factors(adata)
    adata.raw = adata
    
    # 标准化和对数变换
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # 高变基因选择
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    print(f"📊 选择了 {adata.var['highly_variable'].sum()} 个高变基因")
    adata = adata[:, adata.var['highly_variable']].copy()
    
    # 标准化、PCA、邻居图、UMAP
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, metric='euclidean')
    sc.tl.umap(adata, min_dist=0.2)
    
    # Leiden聚类
    try:
        sc.tl.leiden(adata, resolution=1.0)
        print(f"🎯 Leiden聚类识别了 {len(adata.obs['leiden'].unique())} 个簇")
    except Exception as e:
        print(f"⚠️ Leiden聚类失败: {e}")
    
    print("✅ Scanpy分析流程完成")
    return adata

# 运行Scanpy分析
adata_scanpy = run_scanpy_pipeline(adata_counts)

# 可视化Scanpy结果
sc.pl.umap(adata_scanpy, color='celltype', save='_scanpy_celltype.png')
sc.pl.umap(adata_scanpy, color='sample', save='_scanpy_sample.png')

# ## 3. Harmony批次校正

def run_harmony_integration(adata, batch_key='sample'):
    """使用Harmony进行批次校正"""
    print("🎵 开始Harmony批次校正...")
    
    try:
        import scanpy.external as sce
        
        if batch_key not in adata.obs.columns:
            print(f"⚠️ 批次键 '{batch_key}' 不存在，跳过Harmony分析")
            return adata
        
        # Harmony整合
        sce.pp.harmony_integrate(adata, key=batch_key, basis='X_pca')
        
        # 基于Harmony结果重新计算UMAP
        sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=15, metric='euclidean')
        sc.tl.umap(adata, min_dist=0.2)
        adata.obsm['X_harmony_umap'] = adata.obsm['X_umap'].copy()
        
        # 恢复原始UMAP
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15, metric='euclidean')
        sc.tl.umap(adata, min_dist=0.2)
        
        print("✅ Harmony分析完成")
        
    except ImportError:
        print("⚠️ 无法导入Harmony，跳过Harmony分析")
    except Exception as e:
        print(f"⚠️ Harmony分析失败: {e}")
    
    return adata

# 运行Harmony分析
adata_scanpy = run_harmony_integration(adata_scanpy)

# 可视化Harmony结果
if 'X_harmony_umap' in adata_scanpy.obsm:
    sc.pl.embedding(adata_scanpy, basis='X_harmony_umap', color='celltype', save='_harmony_celltype.png')
    sc.pl.embedding(adata_scanpy, basis='X_harmony_umap', color='sample', save='_harmony_sample.png')

# ## 4. VAE_NB模型训练

def train_vae_model(adata_counts, epochs=50, batch_size=128, lr=1e-3, latent_dim=32):
    """训练VAE_NB模型"""
    print("🧠 开始训练VAE_NB模型...")
    
    # 准备训练数据
    X, y_disc, y_cont, sf, x_target, cond_dims, cont_dim = make_counts_training_tensors_v2(
        adata_counts, discrete_key='sample', cont_keys=None, use_hvgs=False, return_dims=True
    )
    
    print(f"📊 训练数据维度: {X.shape}")
    
    # 创建数据加载器
    dataset = TensorDataset(X, y_disc, y_cont, sf, x_target)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 创建VAE模型
    vae_model = VAE(
        input_dim=X.shape[1], latent_dim=latent_dim,
        encoder_layers=[1024, 512, 256], decoder_layers=[256, 512, 1024],
        activation='gelu', norm='layernorm', dropout=0.1, recon_dist='nb'
    )
    
    print(f"🏗️ 模型参数数量: {sum(p.numel() for p in vae_model.parameters()):,}")
    
    # 训练模型
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)
    loss_list, recon_list, kl_list = train_vae(
        vae_model, train_loader, optimizer, epochs, conditional=False, recon_dist='nb',
        beta_final=0.05, beta_warmup_epochs=epochs, free_bits=0.10, 
        kl_schedule='cosine', kl_period=epochs
    )
    
    # 提取潜在表示
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_model = vae_model.to(device)
    vae_model.eval()
    
    latents_list = []
    with torch.no_grad():
        for batch in eval_loader:
            x_batch = batch[0].to(device)
            mu, _ = vae_model.encode(x_batch)
            latents_list.append(mu.cpu().numpy())
    
    latents = np.vstack(latents_list)
    latents_std = standardize_latents(latents)
    
    print("✅ VAE_NB训练完成")
    
    # 保存训练曲线
    save_training_curves(loss_list, recon_list, kl_list, 'VAE_NB', 
                         save_path=f'{RESULTS_DIR}/vae_nb_training_curves.png')
    plt.show()
    
    return latents, latents_std, {'loss': loss_list, 'recon': recon_list, 'kl': kl_list}

# 训练VAE模型
vae_latents, vae_latents_std, vae_loss_history = train_vae_model(adata_counts, epochs=50)

# 将VAE结果添加到数据中并计算UMAP
adata_scanpy.obsm['X_vae_nb'] = vae_latents
adata_scanpy.obsm['X_vae_nb_std'] = vae_latents_std

sc.pp.neighbors(adata_scanpy, use_rep='X_vae_nb_std', n_neighbors=15, metric='euclidean')
sc.tl.umap(adata_scanpy, min_dist=0.2)
adata_scanpy.obsm['X_vae_umap'] = adata_scanpy.obsm['X_umap'].copy()

# 恢复原始邻居图
sc.pp.neighbors(adata_scanpy, use_rep='X_pca', n_neighbors=15, metric='euclidean')
sc.tl.umap(adata_scanpy, min_dist=0.2)

# 可视化VAE结果
sc.pl.embedding(adata_scanpy, basis='X_vae_umap', color='celltype', save='_vae_celltype.png')
sc.pl.embedding(adata_scanpy, basis='X_vae_umap', color='sample', save='_vae_sample.png')

# ## 5. CVAE_NB模型训练

def train_cvae_model(adata_counts, epochs=50, batch_size=128, lr=1e-3, latent_dim=32):
    """训练CVAE_NB模型（条件化批次信息）"""
    print("🧠 开始训练CVAE_NB模型...")
    
    # 准备训练数据
    X, y_disc, y_cont, sf, x_target, cond_dims, cont_dim = make_counts_training_tensors_v2(
        adata_counts, discrete_key='sample', cont_keys=None, use_hvgs=False, return_dims=True
    )
    
    print(f"📊 训练数据维度: {X.shape}")
    print(f"🏷️ 条件维度: {cond_dims}, {cont_dim}")
    
    # 创建数据加载器
    dataset = TensorDataset(X, y_disc, y_cont, sf, x_target)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 创建CVAE模型
    cvae_model = CVAE(
        input_dim=X.shape[1],
        cond_dim=cond_dims if isinstance(cond_dims, (list, tuple)) else int(cond_dims),
        cont_dim=int(cont_dim), latent_dim=latent_dim,
        encoder_layers=[1024, 512, 256], decoder_layers=[256, 512, 1024],
        activation='gelu', norm='layernorm', dropout=0.1, recon_dist='nb', cond_in_encoder=True
    )
    
    print(f"🏗️ 模型参数数量: {sum(p.numel() for p in cvae_model.parameters()):,}")
    
    # 训练模型
    optimizer = torch.optim.Adam(cvae_model.parameters(), lr=lr)
    loss_list, recon_list, kl_list = train_cvae(
        cvae_model, train_loader, optimizer, epochs, conditional=True, recon_dist='nb',
        beta_final=0.05, beta_warmup_epochs=epochs, free_bits=0.10, 
        kl_schedule='cosine', kl_period=epochs
    )
    
    # 提取潜在表示
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvae_model = cvae_model.to(device)
    cvae_model.eval()
    
    latents_list = []
    with torch.no_grad():
        for batch in eval_loader:
            x_batch = batch[0].to(device)
            y_disc_batch = batch[1].to(device)
            y_cont_batch = batch[2].to(device)
            mu, _ = cvae_model.encode(x_batch, y_disc_batch, y_cont_batch)
            latents_list.append(mu.cpu().numpy())
    
    latents = np.vstack(latents_list)
    latents_std = standardize_latents(latents)
    
    print("✅ CVAE_NB训练完成")
    
    # 保存训练曲线
    save_training_curves(loss_list, recon_list, kl_list, 'CVAE_NB',
                         save_path=f'{RESULTS_DIR}/cvae_nb_training_curves.png')
    plt.show()
    
    return latents, latents_std, {'loss': loss_list, 'recon': recon_list, 'kl': kl_list}

# 训练CVAE模型
cvae_latents, cvae_latents_std, cvae_loss_history = train_cvae_model(adata_counts, epochs=50)

# 将CVAE结果添加到数据中并计算UMAP
adata_scanpy.obsm['X_cvae_nb'] = cvae_latents
adata_scanpy.obsm['X_cvae_nb_std'] = cvae_latents_std

sc.pp.neighbors(adata_scanpy, use_rep='X_cvae_nb_std', n_neighbors=15, metric='euclidean')
sc.tl.umap(adata_scanpy, min_dist=0.2)
adata_scanpy.obsm['X_cvae_umap'] = adata_scanpy.obsm['X_umap'].copy()

# 恢复原始邻居图
sc.pp.neighbors(adata_scanpy, use_rep='X_pca', n_neighbors=15, metric='euclidean')
sc.tl.umap(adata_scanpy, min_dist=0.2)

# 可视化CVAE结果
sc.pl.embedding(adata_scanpy, basis='X_cvae_umap', color='celltype', save='_cvae_celltype.png')
sc.pl.embedding(adata_scanpy, basis='X_cvae_umap', color='sample', save='_cvae_sample.png')

# ## 6. 结果对比和可视化

# ### 训练损失对比
if 'vae_loss_history' in locals() and 'cvae_loss_history' in locals():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs_vae = range(1, len(vae_loss_history['loss']) + 1)
    epochs_cvae = range(1, len(cvae_loss_history['loss']) + 1)
    
    # 总损失对比
    axes[0].plot(epochs_vae, vae_loss_history['loss'], 'b-', linewidth=2, label='VAE_NB')
    axes[0].plot(epochs_cvae, cvae_loss_history['loss'], 'r-', linewidth=2, label='CVAE_NB')
    axes[0].set_title('Total Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 重构损失对比
    axes[1].plot(epochs_vae, vae_loss_history['recon'], 'b-', linewidth=2, label='VAE_NB')
    axes[1].plot(epochs_cvae, cvae_loss_history['recon'], 'r-', linewidth=2, label='CVAE_NB')
    axes[1].set_title('Reconstruction Loss Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL散度对比
    axes[2].plot(epochs_vae, vae_loss_history['kl'], 'b-', linewidth=2, label='VAE_NB')
    axes[2].plot(epochs_cvae, cvae_loss_history['kl'], 'r-', linewidth=2, label='CVAE_NB')
    axes[2].set_title('KL Divergence Comparison')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/training_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ### 轮廓系数评估
def calculate_silhouette_scores(adata, representations, labels_key='celltype'):
    """计算不同表示方法的轮廓系数"""
    scores = {}
    
    if labels_key not in adata.obs.columns:
        print(f"⚠️ 标签 '{labels_key}' 不存在")
        return scores
    
    labels = adata.obs[labels_key].values
    
    for rep_name in representations:
        if rep_name in adata.obsm:
            try:
                rep_data = adata.obsm[rep_name]
                score = silhouette_score(rep_data, labels, metric='euclidean')
                scores[rep_name] = score
                print(f"📊 {rep_name}: Silhouette Score = {score:.4f}")
            except Exception as e:
                print(f"⚠️ 计算 {rep_name} 的轮廓系数失败: {e}")
    
    return scores

# 计算轮廓系数
representation_methods = {
    'X_pca': 'Scanpy PCA',
    'X_vae_nb_std': 'VAE_NB',
    'X_cvae_nb_std': 'CVAE_NB'
}

if 'X_pca_harmony' in adata_scanpy.obsm:
    representation_methods['X_pca_harmony'] = 'Harmony'

print("📈 计算轮廓系数（基于细胞类型）:")
silhouette_scores = calculate_silhouette_scores(adata_scanpy, representation_methods.keys(), 'celltype')

# 可视化轮廓系数对比
if silhouette_scores:
    methods = [representation_methods[k] for k in silhouette_scores.keys()]
    scores = list(silhouette_scores.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(methods)])
    plt.title('Silhouette Score Comparison (Cell Type Separation)', fontsize=14)
    plt.ylabel('Silhouette Score')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/silhouette_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ### 保存最终结果
adata_scanpy.write_h5ad(f'{RESULTS_DIR}/integrated_analysis_results.h5ad')
print(f"💾 分析结果已保存到: {RESULTS_DIR}/integrated_analysis_results.h5ad")

# 生成分析报告
print("\n📋 分析总结:")
print("="*50)
print(f"📊 数据集: {adata_counts.shape[0]} 个细胞, {adata_counts.shape[1]} 个基因")
print(f"🧪 样本数: {len(adata_counts.obs['sample'].unique())}")
print(f"🏷️ 细胞类型数: {len(adata_counts.obs['celltype'].unique())}")
print("\n🔬 分析方法:")
for method_key, method_name in representation_methods.items():
    if method_key in adata_scanpy.obsm:
        print(f"  ✅ {method_name}")
    else:
        print(f"  ❌ {method_name} (未运行)")

print(f"\n📁 结果文件保存在: {RESULTS_DIR}/")
print("   - UMAP可视化图片")
print("   - 训练损失曲线")
print("   - 轮廓系数对比")
print("   - 完整分析结果 (.h5ad)")
print("="*50)