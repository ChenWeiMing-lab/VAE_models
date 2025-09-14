#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_processing import select_top_hvgs, make_counts_training_tensors_v2
from VAE_NB.models import VAE
from VAE_NB.training import train_vae
from CVAE_NB.models import CVAE
from CVAE_NB.training import train as train_cvae
from scRNA_analysis.cli import ensure_umap, standardize_latents

COUNTS_PATH = os.path.join(PROJ_ROOT, 'data', 'GSE149614_HCC_scRNA_test_5N5T_counts.h5ad')
OUT_H5AD   = os.path.join(PROJ_ROOT, 'data', 'GSE149614_HCC_scRNA_test_5N5T_with_embeddings.h5ad')
SCANPY_DIR = os.path.join(BASE_DIR, 'results_models', 'scanpy_baseline')
VAE_DIR    = os.path.join(BASE_DIR, 'results_models', 'VAE_NB')
CVAE_DIR   = os.path.join(BASE_DIR, 'results_models', 'CVAE_NB')
HARMONY_DIR= os.path.join(BASE_DIR, 'results_models', 'harmony')
os.makedirs(SCANPY_DIR, exist_ok=True)
os.makedirs(VAE_DIR, exist_ok=True)
os.makedirs(CVAE_DIR, exist_ok=True)
os.makedirs(HARMONY_DIR, exist_ok=True)


def _ensure_size_factors(ad):
    if 'size_factors' in ad.obs: return
    X = ad.X
    if hasattr(X, 'sum'):
        totals = np.asarray(X.sum(axis=1)).reshape(-1) if not hasattr(X.sum(axis=1), 'A1') else X.sum(axis=1).A1
    else:
        totals = X.sum(axis=1)
    med = np.median(totals[totals>0]) if np.any(totals>0) else 1.0
    sf = totals/(med if med>0 else 1.0)
    sf = np.where(sf<=0, 1.0, sf).astype(np.float32)
    ad.obs['size_factors'] = sf


def save_umap_set(ad, basis_key, out_dir):
    try:
        mapping = {
            'celltype': 'celltype_umap',
            'sample': 'sample_umap',
            'leiden': 'clusters_umap',
        }
        available = [k for k in mapping.keys() if k in ad.obs.columns]
        for key in available:
            sc.pl.embedding(ad, basis=basis_key, color=[key], show=False)
            plt.savefig(os.path.join(out_dir, f"{mapping[key]}.svg"), bbox_inches='tight')
            plt.close()
    except Exception:
        pass


def add_harmony_umap(ad, batch_key='sample', n_neighbors=15, min_dist=0.2, metric='euclidean'):
    try:
        # 优先使用传入的批次键；若不存在则尝试 'batch'
        if batch_key not in ad.obs.columns:
            batch_key = 'batch' if 'batch' in ad.obs.columns else None
        if batch_key is None:
            return
        import scanpy.external as sce
        # 基于 PCA 进行 Harmony 融合，生成 X_pca_harmony
        sce.pp.harmony_integrate(ad, key=batch_key, basis='X_pca')
        # 使用 Harmony 融合后的表示计算 UMAP（放入副本以避免覆盖原图）
        ad_tmp = ad.copy()
        ad_tmp.obsm['X_lat'] = ad.obsm.get('X_pca_harmony')
        if ad_tmp.obsm['X_lat'] is None:
            return
        ensure_umap(ad_tmp, 'X_lat', n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        ad.obsm['X_harmony_umap'] = ad_tmp.obsm['X_umap']
    except Exception:
        pass


def run_scanpy_pipeline(ad_counts):
    ad = ad_counts.copy()
    _ensure_size_factors(ad)
    ad.raw = ad
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    # 使用不依赖 scikit-misc 的 HVG 方法以避免额外依赖
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor='seurat')
    ad = ad[:, ad.var['highly_variable']].copy()
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, svd_solver='arpack')
    sc.pp.neighbors(ad, n_neighbors=15, metric='euclidean')
    sc.tl.umap(ad, min_dist=0.2)
    try:
        sc.tl.leiden(ad, resolution=1.0)
    except Exception:
        pass
    # 保存基线 UMAP 三张图
    save_umap_set(ad, 'X_umap', SCANPY_DIR)
    # 计算并添加 Harmony UMAP
    add_harmony_umap(ad, batch_key='sample')
    return ad


def train_vae_nb(ad_counts, epochs=100, batch_size=128, lr=1e-3, latent_dim=32):
    X, y_disc, y_cont, sf, x_tgt, cond_dims, cont_dim = make_counts_training_tensors_v2(
        ad_counts, discrete_key='sample', cont_keys=None, use_hvgs=False, return_dims=True)
    ds = TensorDataset(X, y_disc, y_cont, sf, x_tgt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    dle = DataLoader(ds, batch_size=batch_size, shuffle=False)
    vae = VAE(input_dim=X.shape[1], latent_dim=latent_dim,
              encoder_layers=[1024,512,256], decoder_layers=[256,512,1024],
              activation='gelu', norm='layernorm', dropout=0.1, recon_dist='nb')
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    train_vae(vae, dl, opt, epochs, conditional=False, recon_dist='nb',
              beta_final=0.05, beta_warmup_epochs=epochs, free_bits=0.10,
              kl_schedule='cosine', kl_period=epochs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device); vae.eval()
    lat = []
    with torch.no_grad():
        for b in dle:
            xb = b[0].to(device)
            mu,_ = vae.encode(xb)
            lat.append(mu.cpu().numpy())
    L = np.vstack(lat)
    Lz = standardize_latents(L)
    return L, Lz


def train_cvae_nb(ad_counts, epochs=100, batch_size=128, lr=1e-3, latent_dim=32):
    X, y_disc, y_cont, sf, x_tgt, cond_dims, cont_dim = make_counts_training_tensors_v2(
        ad_counts, discrete_key='sample', cont_keys=None, use_hvgs=False, return_dims=True)
    ds = TensorDataset(X, y_disc, y_cont, sf, x_tgt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    dle = DataLoader(ds, batch_size=batch_size, shuffle=False)
    cvae = CVAE(input_dim=X.shape[1], cond_dim=cond_dims if isinstance(cond_dims,(list,tuple)) else int(cond_dims),
                cont_dim=int(cont_dim), latent_dim=latent_dim,
                encoder_layers=[1024,512,256], decoder_layers=[256,512,1024],
                activation='gelu', norm='layernorm', dropout=0.1, recon_dist='nb', cond_in_encoder=True)
    opt = torch.optim.Adam(cvae.parameters(), lr=lr)
    train_cvae(cvae, dl, opt, epochs, conditional=True, recon_dist='nb',
               beta_final=0.05, beta_warmup_epochs=epochs, free_bits=0.10,
               kl_schedule='cosine', kl_period=epochs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvae = cvae.to(device); cvae.eval()
    lat = []
    with torch.no_grad():
        for b in dle:
            xb = b[0].to(device); yb_d = b[1].to(device); yb_c = b[2].to(device)
            mu,_ = cvae.encode(xb, yb_d, yb_c)
            lat.append(mu.cpu().numpy())
    L = np.vstack(lat)
    Lz = standardize_latents(L)
    return L, Lz


def main():
    print('Loading counts:', COUNTS_PATH)
    ad_counts = sc.read_h5ad(COUNTS_PATH)
    # Scanpy常规流程（基于 counts 的 log1p 正常化）
    ad_scanpy = run_scanpy_pipeline(ad_counts)

    # 训练 VAE_NB 与 CVAE_NB（100 epochs）
    print('Training VAE_NB (100 epochs) ...')
    L_vae, Lz_vae = train_vae_nb(ad_counts, epochs=100)
    print('Training CVAE_NB (100 epochs) ...')
    L_cvae, Lz_cvae = train_cvae_nb(ad_counts, epochs=100)

    # 汇总到一个 AnnData 并保存
    ad_out = ad_scanpy.copy()
    ad_out.obsm['X_vae_nb'] = L_vae
    ad_out.obsm['X_vae_nb_z'] = Lz_vae
    ad_out.obsm['X_cvae_nb'] = L_cvae
    ad_out.obsm['X_cvae_nb_z'] = Lz_cvae

    # 为模型潜变量各自计算 UMAP（基于 z-score 潜变量）
    try:
        ad_tmp = ad_out.copy(); ad_tmp.obsm['X_lat'] = ad_out.obsm['X_vae_nb_z']
        ensure_umap(ad_tmp, 'X_lat', n_neighbors=15, min_dist=0.2, metric='euclidean')
        ad_out.obsm['X_vae_nb_umap'] = ad_tmp.obsm['X_umap']
    except Exception:
        pass
    try:
        ad_tmp = ad_out.copy(); ad_tmp.obsm['X_lat'] = ad_out.obsm['X_cvae_nb_z']
        ensure_umap(ad_tmp, 'X_lat', n_neighbors=15, min_dist=0.2, metric='euclidean')
        ad_out.obsm['X_cvae_nb_umap'] = ad_tmp.obsm['X_umap']
    except Exception:
        pass

    # 保存四类模型对应的三张图到 results_models 目录（含 Harmony）
    save_umap_set(ad_out, 'X_umap', SCANPY_DIR)
    save_umap_set(ad_out, 'X_vae_nb_umap', VAE_DIR)
    save_umap_set(ad_out, 'X_cvae_nb_umap', CVAE_DIR)
    if 'X_harmony_umap' in ad_out.obsm:
        save_umap_set(ad_out, 'X_harmony_umap', HARMONY_DIR)

    print('Saving to:', OUT_H5AD)
    ad_out.write_h5ad(OUT_H5AD)
    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Scanpy + (C)VAE pipelines or just save UMAPs from existing embeddings.')
    parser.add_argument('--save-only', action='store_true', help='仅从已存在的 with_embeddings.h5ad 读取并保存 UMAP 图，而不重新训练或计算')
    args = parser.parse_args()

    if args.save_only:
        if not os.path.exists(OUT_H5AD):
            raise FileNotFoundError(f'未找到 {OUT_H5AD}，请先运行完整流程生成含 embeddings 的 h5ad 文件。')
        ad = sc.read_h5ad(OUT_H5AD)
        # 逐一保存三类 UMAP（若存在）
        if 'X_umap' in ad.obsm:
            save_umap_set(ad, 'X_umap', SCANPY_DIR)
        if 'X_vae_nb_umap' in ad.obsm:
            save_umap_set(ad, 'X_vae_nb_umap', VAE_DIR)
        if 'X_cvae_nb_umap' in ad.obsm:
            save_umap_set(ad, 'X_cvae_nb_umap', CVAE_DIR)
        # Harmony：若已存在则直接保存；否则若可从 X_pca 计算，则临时计算后保存
        if 'X_harmony_umap' in ad.obsm:
            save_umap_set(ad, 'X_harmony_umap', HARMONY_DIR)
        else:
            if 'X_pca' in ad.obsm and ('sample' in ad.obs.columns or 'batch' in ad.obs.columns):
                add_harmony_umap(ad, batch_key='sample')
                if 'X_harmony_umap' in ad.obsm:
                    save_umap_set(ad, 'X_harmony_umap', HARMONY_DIR)
        print('已根据现有 embeddings 保存 UMAP 图至 results_models/ 对应目录。')
    else:
        main()