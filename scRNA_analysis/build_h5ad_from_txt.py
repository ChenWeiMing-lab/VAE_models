#!/usr/bin/env python3
import os, gzip
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


def main():
    base_dir = "/root/Project/VAE_models/CVAE_MoG/data"
    counts_path = os.path.join(base_dir, "GSE149614_HCC.scRNAseq.S71915.count.txt.gz")
    meta_path = os.path.join(base_dir, "GSE149614_HCC.metadata.updated.txt.gz")
    out_path = os.path.join(base_dir, "GSE149614_HCC_scRNA_nonfiltered.h5ad")

    # Read metadata (cells as index)
    print(f"Reading metadata from: {meta_path}")
    obs = pd.read_csv(meta_path, sep="\t", index_col=0)

    # Read counts header to get cell barcodes (preserve file order)
    print(f"Reading counts header from: {counts_path}")
    with gzip.open(counts_path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    all_cells = header[1:]

    obs_index_set = set(obs.index)
    cells = [c for c in all_cells if c in obs_index_set]
    if not cells:
        raise ValueError("No overlapping cell barcodes between counts columns and metadata index.")
    n_cells = len(cells)
    print(f"Cells kept: {n_cells} / {len(all_cells)}")

    # Stream counts in chunks to build sparse matrix (cells x genes)
    chunksize = 1000  # genes per chunk to limit memory
    data, rows, cols, gene_names = [], [], [], []
    gene_offset = 0

    reader = pd.read_csv(counts_path, sep="\t", index_col=0, chunksize=chunksize)
    for i, df in enumerate(reader):
        sub = df.loc[:, cells]
        gene_names.extend(sub.index.tolist())
        arr = sub.to_numpy(dtype=np.float32, copy=False)  # shape: (g_chunk, n_cells)
        nz_g, nz_c = np.nonzero(arr)
        if nz_g.size:
            vals = arr[nz_g, nz_c]
            rows.extend(nz_c.tolist())
            cols.extend((gene_offset + nz_g).tolist())
            data.extend(vals.tolist())
        gene_offset += arr.shape[0]
        if (i + 1) % 100 == 0:
            print(f"Processed {gene_offset} genes...")
        del df, sub, arr

    n_genes = gene_offset
    print(f"Total genes: {n_genes}")

    X = sp.coo_matrix((np.array(data, dtype=np.float32),
                       (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
                      shape=(n_cells, n_genes), dtype=np.float32).tocsr()

    # Reorder obs to cells order
    obs = obs.loc[cells, :]
    var = pd.DataFrame(index=pd.Index(gene_names, name=None))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    print(f"AnnData shape (cells x genes): {adata.shape}")
    print(f"Writing h5ad to: {out_path}")
    adata.write_h5ad(out_path)
    print("Done.")


if __name__ == "__main__":
    main()