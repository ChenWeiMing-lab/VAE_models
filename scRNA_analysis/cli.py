#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
# scanpy optional import moved below
# import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import anndata as an

BASE_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import set_seed
from data_processing import select_top_hvgs, make_counts_training_tensors_v2, make_gaussian_training_tensors
from VAE_NB.models import VAE
from VAE_NB.training import train_vae
from CVAE_NB.models import CVAE
from CVAE_NB.training import train as train_cvae


def prepare_data(adata_path, counts_path, n_top=2000):
    # read with scanpy if available, otherwise anndata
    if sc is not None:
        adata = sc.read_h5ad(adata_path)
    else:
        adata = an.read_h5ad(adata_path)
    ad_counts = None

    if counts_path is not None and os.path.exists(counts_path):
        if sc is not None:
            ad_counts = sc.read_h5ad(counts_path)
        else:
            ad_counts = an.read_h5ad(counts_path)
        # 对齐 HVG：在两者共有基因中选 top n_top HVGs
        adata_hvg = select_top_hvgs(adata, n_top=n_top, inplace=False)
        counts_hvg = select_top_hvgs(ad_counts, n_top=n_top, inplace=False)
        hv_common = list(set(adata_hvg.var_names).intersection(set(counts_hvg.var_names)))
        hv_common.sort()
        adata = adata[:, hv_common].copy()
        ad_counts = ad_counts[:, hv_common].copy()
        # 估计 size_factors（如缺失）
        try:
            if 'size_factors' not in ad_counts.obs:
                if sp.issparse(ad_counts.X):
                    totals = np.asarray(ad_counts.X.sum(axis=1)).reshape(-1)
                else:
                    totals = ad_counts.X.sum(axis=1)
                    if hasattr(totals, 'A1'):
                        totals = totals.A1
                med = np.median(totals[totals > 0]) if np.any(totals > 0) else 1.0
                sf = totals / (med if med > 0 else 1.0)
                sf = np.where(sf <= 0, 1.0, sf).astype(np.float32)
                ad_counts.obs['size_factors'] = sf
        except Exception as e:
            print(f"[WARN] compute size_factors failed: {e}")
        print(f"Unified genes: {len(hv_common)} | adata: {adata.shape} | counts: {ad_counts.shape}")
    else:
        # fallback: HVG on adata only for gaussian
        adata_hvg = select_top_hvgs(adata, n_top=n_top, inplace=False)
        hv = list(adata_hvg.var_names)
        adata = adata[:, hv].copy()
        print(f"Counts not provided. Using adata HVGs only: {len(hv)} genes")

    return adata, ad_counts


# try optional scanpy (may conflict in some envs)
try:
    import scanpy as sc  # keep order: after matplotlib import
except Exception:
    sc = None

def ensure_umap(ad, rep_key, n_neighbors=15, min_dist=0.2, metric='cosine'):
    if sc is not None:
        sc.pp.neighbors(ad, use_rep=rep_key, n_neighbors=n_neighbors, metric=metric)
        sc.tl.umap(ad, min_dist=min_dist)
    else:
        # fallback using umap-learn directly on the representation
        try:
            import umap
            X = ad.obsm[rep_key]
            um = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist), metric=metric, random_state=0)
            ad.obsm['X_umap'] = um.fit_transform(X)
        except Exception as e:
            print(f"[WARN] UMAP fallback failed: {e}")


def standardize_latents(L):
    mean = L.mean(axis=0, keepdims=True)
    std = L.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (L - mean) / std


def build_save_dir(model_name):
    # 固定保存目录到 scRNA_analysis/VAE_NB 或 scRNA_analysis/CVAE_NB（无时间戳）
    name = model_name.upper()
    if name == 'VAE':
        save_dir = os.path.join(BASE_DIR, 'VAE_NB')
    elif name == 'CVAE':
        save_dir = os.path.join(BASE_DIR, 'CVAE_NB')
    else:
        save_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_training_curves(loss_list, recon_list, kl_list, save_dir, title_prefix, filename_prefix):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(loss_list, '-', linewidth=2, label=f'{title_prefix} total')
        ax.set_title(f'Training Loss - {title_prefix}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{filename_prefix}_training_losses.svg'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] save training losses failed: {e}")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(recon_list, '-', linewidth=2, label='Recon')
        ax.plot(kl_list, '--', linewidth=2, label='KL')
        ax.set_title(f'{title_prefix}: Recon vs KL')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{filename_prefix}_recon_kl.svg'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] save recon-kl failed: {e}")


def save_outputs(ad, latents, latents_z, save_dir, prefix):
    # save latents
    try:
        np.save(os.path.join(save_dir, 'latents.npy'), latents)
        np.save(os.path.join(save_dir, 'latents_z.npy'), latents_z)
    except Exception as e:
        print(f"[WARN] save latents failed: {e}")
    # save obs
    try:
        ad.obs.to_csv(os.path.join(save_dir, 'obs.csv'))
    except Exception as e:
        print(f"[WARN] save obs.csv failed: {e}")
    # plots
    if sc is not None:
        try:
            if 'sample' in ad.obs.columns:
                sc.pl.umap(ad, color=['sample'], show=False)
                plt.savefig(os.path.join(save_dir, f'{prefix}_umap_sample.svg'), bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"[WARN] save sample umap failed: {e}")
        try:
            if 'celltype' in ad.obs.columns:
                sc.pl.umap(ad, color=['celltype'], show=False)
                plt.savefig(os.path.join(save_dir, f'{prefix}_umap_celltype.svg'), bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"[WARN] save celltype umap failed: {e}")
        try:
            if 'size_factors' in ad.obs.columns:
                sc.pl.umap(ad, color=['size_factors'], show=False)
                plt.savefig(os.path.join(save_dir, f'{prefix}_umap_sizefactors.svg'), bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"[WARN] save sizefactors umap failed: {e}")
        try:
            for key in ['leiden', 'louvain']:
                if key in ad.obs.columns:
                    sc.pl.umap(ad, color=[key], show=False)
                    plt.savefig(os.path.join(save_dir, f'{prefix}_umap_{key}.svg'), bbox_inches='tight')
                    plt.close()
        except Exception as e:
            print(f"[WARN] save cluster umap failed: {e}")
    else:
        # fallback plotting from precomputed UMAP (X_umap)
        try:
            U = ad.obsm.get('X_umap', None)
            if U is not None and 'sample' in ad.obs.columns:
                fig, ax = plt.subplots(figsize=(5, 5))
                for s in ad.obs['sample'].unique():
                    idx = (ad.obs['sample'] == s).values
                    ax.scatter(U[idx, 0], U[idx, 1], s=4, alpha=0.7, label=str(s))
                ax.set_title(f"{prefix} UMAP by sample")
                ax.set_xticks([]); ax.set_yticks([])
                ax.legend(loc='best', markerscale=3, fontsize=6)
                fig.savefig(os.path.join(save_dir, f'{prefix}_umap_sample.svg'), bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"[WARN] fallback sample umap failed: {e}")
        try:
            U = ad.obsm.get('X_umap', None)
            if U is not None and 'celltype' in ad.obs.columns:
                fig, ax = plt.subplots(figsize=(5, 5))
                for ct in ad.obs['celltype'].unique():
                    idx = (ad.obs['celltype'] == ct).values
                    ax.scatter(U[idx, 0], U[idx, 1], s=4, alpha=0.7, label=str(ct))
                ax.set_title(f"{prefix} UMAP by celltype")
                ax.set_xticks([]); ax.set_yticks([])
                ax.legend(loc='best', markerscale=3, fontsize=6, ncol=1)
                fig.savefig(os.path.join(save_dir, f'{prefix}_umap_celltype.svg'), bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"[WARN] fallback celltype umap failed: {e}")


def maybe_silhouette(latents, labels, save_dir, metric='euclidean'):
    if labels is None:
        return None
    try:
        from sklearn.metrics import silhouette_score
        score = silhouette_score(latents, labels, metric=metric)
        with open(os.path.join(save_dir, 'silhouette_celltype.txt'), 'w') as f:
            f.write(str(score))
        print(f"Silhouette(celltype, {metric}): {score:.4f}")
        return score
    except Exception as e:
        print(f"[WARN] silhouette failed: {e}")
        return None


def run_vae(args):
    set_seed(args.seed)
    adata, ad_counts = prepare_data(args.adata, args.counts, n_top=args.hvg)

    recon = args.recon.lower()
    if recon not in ('nb', 'gaussian'):
        raise ValueError("recon must be 'nb' or 'gaussian'")

    # tensors
    if recon == 'nb':
        if ad_counts is None:
            raise RuntimeError('NB 重构需要 counts h5ad 原始计数数据')
        X, y_disc, y_cont, sf, x_target, cond_dims, cont_dim = make_counts_training_tensors_v2(
            ad_counts, discrete_key='sample', cont_keys=None, use_hvgs=False, return_dims=True
        )
        ad_vis = ad_counts.copy()
    else:
        source_ad = ad_counts if ad_counts is not None else adata
        X, y_disc, y_cont, sf, x_target, cond_dims, cont_dim = make_gaussian_training_tensors(
            source_ad, discrete_key='sample', cont_keys=None, use_hvgs=False, return_dims=True
        )
        ad_vis = source_ad.copy()

    dataset = TensorDataset(X, y_disc, y_cont, sf, x_target)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_eval = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # build hidden layers from args if provided
    if getattr(args, 'hidden_dim', None) is not None and getattr(args, 'n_layers', None) is not None:
        enc_layers = [int(args.hidden_dim)] * int(args.n_layers)
    else:
        enc_layers = [1024, 512, 256]
    dec_layers = list(reversed(enc_layers))

    # model
    vae = VAE(
        input_dim=X.shape[1],
        latent_dim=args.latent_dim,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        activation='gelu',
        norm='layernorm',
        dropout=0.1,
        recon_dist=recon
    )
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    # Resolve KL configs (support fixed beta and custom schedule)
    _kl_schedule = None if (args.kl_schedule is None or args.kl_schedule.lower() == 'none') else args.kl_schedule
    _beta_final = args.beta_final
    _beta_warmup = args.beta_warmup_epochs if args.beta_warmup_epochs is not None else args.epochs
    _kl_period = args.kl_period if args.kl_period is not None else args.epochs
    if args.fixed_beta is not None:
        _beta_final = float(args.fixed_beta)
        _kl_schedule = None
        _beta_warmup = 0

    loss_list, recon_list, kl_list = train_vae(
        vae, dataloader, opt, args.epochs,
        conditional=False,
        recon_dist=recon,
        beta_final=_beta_final,
        beta_warmup_epochs=_beta_warmup,
        free_bits=args.free_bits,
        kl_schedule=_kl_schedule,
        kl_period=_kl_period
    )

    # extract latents
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device)
    vae.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader_eval:
            xb = batch[0].to(device)
            mu, _ = vae.encode(xb)
            latents.append(mu.cpu().numpy())
    latents = np.vstack(latents)
    latents_z = standardize_latents(latents)

    # diagnostics
    try:
        latent_std = np.std(latents, axis=0)
        np.save(os.path.join(args.save_dir, 'diag_latent_std.npy'), latent_std)
        print('[Diag] latent std: mean=%.4f, median=%.4f, min=%.4f, max=%.4f' % (
            latent_std.mean(), np.median(latent_std), latent_std.min(), latent_std.max()))
    except Exception as e:
        print(f"[WARN] save latent std failed: {e}")

    # choose representation per args
    ad_vis.obsm['X_latents'] = latents
    ad_vis.obsm['X_latents_z'] = latents_z
    rep_key = 'X_latents_z' if args.standardize_latents else 'X_latents'

    # ensure sf for coloring
    try:
        if 'size_factors' not in ad_vis.obs:
            if sp.issparse(ad_vis.X):
                totals = np.asarray(ad_vis.X.sum(axis=1)).reshape(-1)
            else:
                totals = ad_vis.X.sum(axis=1)
                if hasattr(totals, 'A1'):
                    totals = totals.A1
            med = np.median(totals[totals > 0]) if np.any(totals > 0) else 1.0
            sf_vals = totals / (med if med > 0 else 1.0)
            sf_vals = np.where(sf_vals <= 0, 1.0, sf_vals).astype(np.float32)
            ad_vis.obs['size_factors'] = sf_vals
    except Exception:
        pass

    ensure_umap(ad_vis, rep_key, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.neighbor_metric)

    # clustering
    try:
        if args.cluster == 'leiden':
            sc.tl.leiden(ad_vis, resolution=args.resolution)
        elif args.cluster == 'louvain':
            sc.tl.louvain(ad_vis, resolution=args.resolution)
    except Exception as e:
        print(f"[WARN] clustering failed: {e}")

    # outputs
    prefix = f"VAE_{recon.upper()}"
    save_outputs(ad_vis, latents, latents_z, args.save_dir, prefix)
    save_training_curves(loss_list, recon_list, kl_list, args.save_dir, f'VAE ({recon})', f'vae_{recon}')
    # silhouette on the used representation
    L_used = latents_z if args.standardize_latents else latents
    maybe_silhouette(L_used, ad_vis.obs['celltype'] if 'celltype' in ad_vis.obs.columns else None, args.save_dir, metric=args.neighbor_metric)


def run_cvae(args):
    set_seed(args.seed)
    adata, ad_counts = prepare_data(args.adata, args.counts, n_top=args.hvg)

    recon = args.recon.lower()
    if recon not in ('nb', 'gaussian'):
        raise ValueError("recon must be 'nb' or 'gaussian'")

    # tensors
    if recon == 'nb':
        if ad_counts is None:
            raise RuntimeError('NB 重构需要 counts h5ad 原始计数数据')
        X, y_disc, y_cont, sf, x_target, cond_dims, cont_dim = make_counts_training_tensors_v2(
            ad_counts, discrete_key='sample', cont_keys=None, use_hvgs=False, return_dims=True
        )
        ad_vis = ad_counts.copy()
    else:
        source_ad = ad_counts if ad_counts is not None else adata
        X, y_disc, y_cont, sf, x_target, cond_dims, cont_dim = make_gaussian_training_tensors(
            source_ad, discrete_key='sample', cont_keys=None, use_hvgs=False, return_dims=True
        )
        ad_vis = source_ad.copy()

    dataset = TensorDataset(X, y_disc, y_cont, sf, x_target)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_eval = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # build hidden layers from args if provided
    if getattr(args, 'hidden_dim', None) is not None and getattr(args, 'n_layers', None) is not None:
        enc_layers = [int(args.hidden_dim)] * int(args.n_layers)
    else:
        enc_layers = [1024, 512, 256]
    dec_layers = list(reversed(enc_layers))

    cvae = CVAE(
        input_dim=X.shape[1],
        cond_dim=cond_dims if isinstance(cond_dims, (list, tuple)) else int(cond_dims),
        cont_dim=int(cont_dim),
        latent_dim=args.latent_dim,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        activation='gelu',
        norm='layernorm',
        dropout=0.1,
        recon_dist=recon,
        cond_in_encoder=True,
    )
    opt = torch.optim.Adam(cvae.parameters(), lr=args.lr)

    # Resolve KL configs (support fixed beta and custom schedule)
    _kl_schedule = None if (args.kl_schedule is None or args.kl_schedule.lower() == 'none') else args.kl_schedule
    _beta_final = args.beta_final
    _beta_warmup = args.beta_warmup_epochs if args.beta_warmup_epochs is not None else args.epochs
    _kl_period = args.kl_period if args.kl_period is not None else args.epochs
    if args.fixed_beta is not None:
        _beta_final = float(args.fixed_beta)
        _kl_schedule = None
        _beta_warmup = 0

    loss_list, recon_list, kl_list = train_cvae(
        cvae, dataloader, opt, args.epochs,
        conditional=True,
        recon_dist=recon,
        beta_final=_beta_final,
        beta_warmup_epochs=_beta_warmup,
        free_bits=args.free_bits,
        kl_schedule=_kl_schedule,
        kl_period=_kl_period,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvae = cvae.to(device)
    cvae.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader_eval:
            xb = batch[0].to(device)
            yb_disc = batch[1].to(device)
            yb_cont = batch[2].to(device)
            mu, _ = cvae.encode(xb, yb_disc, yb_cont)
            latents.append(mu.cpu().numpy())
    latents = np.vstack(latents)
    latents_z = standardize_latents(latents)

    try:
        latent_std = np.std(latents, axis=0)
        np.save(os.path.join(args.save_dir, 'diag_latent_std.npy'), latent_std)
        print('[Diag] latent std: mean=%.4f, median=%.4f, min=%.4f, max=%.4f' % (
            latent_std.mean(), np.median(latent_std), latent_std.min(), latent_std.max()))
    except Exception as e:
        print(f"[WARN] save latent std failed: {e}")

    ad_vis.obsm['X_latents'] = latents
    ad_vis.obsm['X_latents_z'] = latents_z
    rep_key = 'X_latents_z' if args.standardize_latents else 'X_latents'

    try:
        if 'size_factors' not in ad_vis.obs:
            if sp.issparse(ad_vis.X):
                totals = np.asarray(ad_vis.X.sum(axis=1)).reshape(-1)
            else:
                totals = ad_vis.X.sum(axis=1)
                if hasattr(totals, 'A1'):
                    totals = totals.A1
            med = np.median(totals[totals > 0]) if np.any(totals > 0) else 1.0
            sf_vals = totals / (med if med > 0 else 1.0)
            sf_vals = np.where(sf_vals <= 0, 1.0, sf_vals).astype(np.float32)
            ad_vis.obs['size_factors'] = sf_vals
    except Exception:
        pass

    ensure_umap(ad_vis, rep_key, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.neighbor_metric)

    # clustering
    try:
        if args.cluster == 'leiden':
            sc.tl.leiden(ad_vis, resolution=args.resolution)
        elif args.cluster == 'louvain':
            sc.tl.louvain(ad_vis, resolution=args.resolution)
    except Exception as e:
        print(f"[WARN] clustering failed: {e}")

    model_name = 'CVAE'
    prefix = f"{model_name}_{recon.upper()}"
    save_outputs(ad_vis, latents, latents_z, args.save_dir, prefix)
    save_training_curves(loss_list, recon_list, kl_list, args.save_dir, f'{model_name} ({recon})', f'{model_name.lower()}_{recon}')
    L_used = latents_z if args.standardize_latents else latents
    sil_score = maybe_silhouette(L_used, ad_vis.obs['celltype'] if 'celltype' in ad_vis.obs.columns else None, args.save_dir, metric=args.neighbor_metric)

    # append summary.csv for CVAE runs (results_4models/CVAE_NB)
    try:
        res_dir = os.path.join(BASE_DIR, 'results_4models', 'CVAE_NB')
        os.makedirs(res_dir, exist_ok=True)
        csv_path = os.path.join(res_dir, 'summary.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write('exp,epochs,metric,kl,latent_dim,silhouette\n')
        kl_label = (args.kl_schedule if (args.kl_schedule is not None and str(args.kl_schedule).lower() != 'none') else 'none')
        with open(csv_path, 'a') as f:
            f.write(f"CVAE_{args.latent_dim},{args.epochs},{args.neighbor_metric},{kl_label},{args.latent_dim},{sil_score if sil_score is not None else ''}\n")
    except Exception as e:
        print(f"[WARN] append summary failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Unified CLI for VAE / CVAE pipelines')
    # 固定默认数据路径到指定测试集
    parser.add_argument('--adata', type=str, default=os.path.join(PROJ_ROOT, 'data', 'GSE149614_HCC_scRNA_test_5N5T.h5ad'))
    parser.add_argument('--counts', type=str, default=os.path.join(PROJ_ROOT, 'data', 'GSE149614_HCC_scRNA_test_5N5T_counts.h5ad'))
    parser.add_argument('--hvg', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--min_dist', type=float, default=0.2)
    parser.add_argument('--beta_final', type=float, default=0.05)
    parser.add_argument('--free_bits', type=float, default=0.10)
    parser.add_argument('--lr', type=float, default=1e-3)
    # new: neighbor metric / clustering / resolution / standardize
    parser.add_argument('--neighbor_metric', type=str, choices=['euclidean', 'cosine'], default='euclidean', help='kNN metric on latent space')
    parser.add_argument('--cluster', type=str, choices=['none', 'leiden', 'louvain'], default='leiden', help='Clustering algorithm to run after neighbors/UMAP')
    parser.add_argument('--resolution', type=float, default=1.0, help='Clustering resolution')
    parser.add_argument('--standardize_latents', action='store_true', help='Z-score latents per-dimension before neighbors/UMAP')
    # KL scheduling / fixed beta options
    parser.add_argument('--kl_schedule', type=str, choices=['none', 'linear', 'cosine', 'cyclical'], default='cosine', help='KL coefficient scheduling strategy')
    parser.add_argument('--kl_period', type=int, default=None, help='Period (epochs) for KL scheduling; default=epochs')
    parser.add_argument('--beta_warmup_epochs', type=int, default=None, help='Linear warmup epochs to reach beta_final when schedule is None')
    parser.add_argument('--fixed_beta', type=float, default=None, help='If set, use constant beta throughout training (overrides schedule/warmup)')
    # new: architecture alignment knobs
    parser.add_argument('--hidden_dim', type=int, default=None, help='If set with --n_layers, overrides default encoder/decoder layer widths (uniform)')
    parser.add_argument('--n_layers', type=int, default=None, help='Number of hidden layers in encoder/decoder (uniform)')

    subparsers = parser.add_subparsers(dest='cmd', required=True)

    # VAE subcommand
    p_vae = subparsers.add_parser('vae', help='Run VAE pipeline')
    p_vae.add_argument('--recon', type=str, choices=['nb', 'gaussian'], default='nb')

    # CVAE subcommand
    p_cvae = subparsers.add_parser('cvae', help='Run CVAE pipeline')
    p_cvae.add_argument('--recon', type=str, choices=['nb', 'gaussian'], default='nb')

    return parser.parse_args()


def main():
    args = parse_args()

    # 固定保存目录：scRNA_analysis/VAE_NB 或 scRNA_analysis/CVAE_NB
    model_name = 'VAE' if args.cmd == 'vae' else 'CVAE'
    args.save_dir = build_save_dir(model_name)
    recon = args.recon if hasattr(args, 'recon') else 'nb'

    print('=' * 60)
    print(f'Running {model_name} with recon={recon} | metric={args.neighbor_metric} | cluster={args.cluster} (res={args.resolution}) | zscore={args.standardize_latents} | results -> {args.save_dir}')
    print('=' * 60)

    if args.cmd == 'vae':
        run_vae(args)
    elif args.cmd == 'cvae':
        run_cvae(args)
    else:
        raise ValueError('Unknown command')


if __name__ == '__main__':
    main()