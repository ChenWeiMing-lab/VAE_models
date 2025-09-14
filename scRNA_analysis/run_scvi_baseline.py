# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import anndata as ad

try:
    import scvi
except Exception as e:
    raise RuntimeError("scvi-tools is not installed. Please install scvi-tools before running this script.") from e

BASE = "/root/Project/VAE_models"
COUNTS = os.path.join(BASE, "data", "GSE149614_HCC_scRNA_test_5N5T_counts.h5ad")


def main():
    adata = ad.read_h5ad(COUNTS)
    # Setup anndata: use batch_key=sample if available
    batch_key = "sample" if "sample" in adata.obs.columns else None
    scvi.model.SCVI.setup_anndata(adata, layer=None, batch_key=batch_key)

    model = scvi.model.SCVI(adata, n_latent=10, n_hidden=128, n_layers=1, gene_likelihood="zinb")
    model.train(max_epochs=100, plan_kwargs={"lr": 1e-3})

    latent = model.get_latent_representation()
    adata.obsm["X_scvi"] = latent

    # Compute silhouette (euclidean) if labels available
    sil = None
    try:
        from sklearn.metrics import silhouette_score
        if "celltype" in adata.obs.columns:
            sil = float(silhouette_score(latent, adata.obs["celltype"], metric="euclidean"))
            print(f"SCVI silhouette: {sil:.4f}")
    except Exception as e:
        print(f"[WARN] silhouette failed: {e}")

    save_dir = os.path.join(BASE, "scRNA_analysis", "results_4models", "SCVI")
    os.makedirs(save_dir, exist_ok=True)

    # Save artifacts
    try:
        np.save(os.path.join(save_dir, "latents.npy"), latent)
    except Exception as e:
        print(f"[WARN] save latents failed: {e}")
    try:
        adata.obs.to_csv(os.path.join(save_dir, "obs.csv"))
    except Exception as e:
        print(f"[WARN] save obs.csv failed: {e}")

    # Optional: compute UMAP directly from latent
    try:
        import umap
        umap_embed = umap.UMAP(n_neighbors=15, min_dist=0.5, metric="euclidean", random_state=0).fit_transform(latent)
        # Save UMAP figure colored by sample/celltype if available
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")
            # by sample
            if "sample" in adata.obs.columns:
                fig, ax = plt.subplots(figsize=(5, 5))
                for s in adata.obs["sample"].unique():
                    idx = np.array(adata.obs["sample"] == s)
                    ax.scatter(umap_embed[idx, 0], umap_embed[idx, 1], s=4, alpha=0.7, label=str(s))
                ax.set_title("SCVI UMAP by sample")
                ax.set_xticks([]); ax.set_yticks([])
                ax.legend(loc="best", markerscale=3, fontsize=6)
                fig.savefig(os.path.join(save_dir, "SCVI_umap_sample.svg"), bbox_inches="tight")
                plt.close(fig)
            # by celltype
            if "celltype" in adata.obs.columns:
                fig, ax = plt.subplots(figsize=(5, 5))
                for ct in adata.obs["celltype"].unique():
                    idx = np.array(adata.obs["celltype"] == ct)
                    ax.scatter(umap_embed[idx, 0], umap_embed[idx, 1], s=4, alpha=0.7, label=str(ct))
                ax.set_title("SCVI UMAP by celltype")
                ax.set_xticks([]); ax.set_yticks([])
                ax.legend(loc="best", markerscale=3, fontsize=6, ncol=1)
                fig.savefig(os.path.join(save_dir, "SCVI_umap_celltype.svg"), bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            print(f"[WARN] matplotlib/plot failed: {e}")
    except Exception as e:
        print(f"[WARN] UMAP failed: {e}")

    if sil is not None:
        try:
            with open(os.path.join(save_dir, "silhouette_celltype.txt"), "w") as f:
                f.write(str(sil))
        except Exception as e:
            print(f"[WARN] save silhouette failed: {e}")

    # Append to CVAE summary.csv for side-by-side comparison
    csv = os.path.join(BASE, "scRNA_analysis", "results_4models", "CVAE_NB", "summary.csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    try:
        if not os.path.exists(csv):
            with open(csv, "w") as f:
                f.write("exp,epochs,metric,kl,latent_dim,silhouette\n")
        with open(csv, "a") as f:
            f.write(f"SCVI_10,100,euclidean,NA,10,{sil if sil is not None else ''}\n")
    except Exception as e:
        print(f"[WARN] append summary failed: {e}")

    print("SCVI baseline finished and results saved to:", save_dir)


if __name__ == "__main__":
    main()