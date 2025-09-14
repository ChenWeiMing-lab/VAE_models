import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 可选依赖：scanpy用于单细胞数据处理
try:
    import scanpy as sc  # noqa: F401
    HAS_SCANPY = True
except Exception:
    HAS_SCANPY = False
    print("Warning: scanpy not installed. Single-cell data processing functions will not be available.")
    print("Install with: pip install scanpy")


import pandas as pd
import numpy as np
# 仅在可用时导入scanpy，否则定义占位符
if HAS_SCANPY:
    import scanpy as sc
else:
    sc = None
import gzip
from pathlib import Path
import scipy.sparse as sp
# 使用可选的 pyarrow 引擎以加速 CSV 解析（如可用）
try:
    import pyarrow  # noqa: F401
    _CSV_ENGINE = "pyarrow"
except Exception:
    _CSV_ENGINE = None

# 设置scanpy参数
# sc.settings.verbosity = 3  # 详细输出
# sc.settings.set_figure_params(dpi=80, facecolor='white')

def load_data_from_gz_files(data_dir="/root/Project/VAE_models/CVAE_MoG/data"):
    """
    从压缩文件中读取单细胞数据并创建AnnData对象
    """
    data_path = Path(data_dir)
    
    # 读取计数矩阵与元数据路径
    count_file = data_path / "GSE149614_HCC.scRNAseq.S71915.count.txt.gz"
    metadata_file = data_path / "GSE149614_HCC.metadata.updated.txt.gz"
    
    print("正在读取计数矩阵...")
    # 方案A：使用 pandas 直接读取 gzip，并尝试更紧凑的dtype与更快的解析引擎
    engine_args = {"engine": _CSV_ENGINE} if _CSV_ENGINE else {}
    try:
        count_data = pd.read_csv(count_file, sep='\t', index_col=0, compression='gzip', dtype=np.int32, low_memory=False, **engine_args)
    except Exception:
        # 回退：不强制dtype
        count_data = pd.read_csv(count_file, sep='\t', index_col=0, compression='gzip', low_memory=False, **engine_args)
    
    print("正在读取元数据...")
    metadata = pd.read_csv(metadata_file, sep='\t', index_col=0, compression='gzip', low_memory=False, **engine_args)
    
    # 元数据内存优化：object 列转为 category
    for col in metadata.select_dtypes(include='object').columns:
        metadata[col] = metadata[col].astype('category')
    
    # 尝试对计数列做降精度（若为浮点则降为float32，整数降为更小整数）
    for col in count_data.columns:
        if np.issubdtype(count_data[col].dtype, np.floating):
            count_data[col] = pd.to_numeric(count_data[col], downcast='float')
        elif np.issubdtype(count_data[col].dtype, np.integer):
            count_data[col] = pd.to_numeric(count_data[col], downcast='integer')
    
    # 转置计数矩阵（scanpy期望细胞×基因的格式）
    if count_data.shape[0] > count_data.shape[1]:
        print("转置计数矩阵为细胞×基因格式")
        count_data = count_data.T
    
    # 方案B：使用稀疏矩阵构建 AnnData，显著降低内存
    X_sparse = sp.csr_matrix(count_data.to_numpy(copy=False))
    adata = sc.AnnData(X=X_sparse, obs=metadata, var=pd.DataFrame(index=count_data.columns))
    adata.obs_names = count_data.index
    adata.var_names = count_data.columns
    
    return adata

def preprocess_data(adata):
    """
    使用scanpy进行数据预处理和标准化
    """
    print(f"原始数据形状: {adata.shape}")
    
    # 1. 基本质量控制指标
    # 计算每个细胞的基因数量
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # 线粒体基因
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # 2. 过滤低质量细胞和基因
    print("过滤低质量细胞和基因...")
    sc.pp.filter_cells(adata, min_genes=200) # 过滤表达基因数少于200的细胞
    sc.pp.filter_genes(adata, min_cells=3) # 过滤在少于3个细胞中表达的基因
    print(f"过滤后数据形状: {adata.shape}")
    
    # 3. 标准化和对数变换
    print("进行数据标准化...")
    # 保存原始计数数据
    adata.raw = adata
    
    # 标准化 和 对数变换
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata) 
    
    # 4. 寻找高变基因
    print("寻找高变基因...")
    sc.pp.highly_variable_genes(adata,n_top_genes=2000, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # 只保留高变基因用于下游分析
    adata.raw = adata  # 保存完整数据
    adata = adata[:, adata.var.highly_variable]
    
    # 5. 缩放数据（z-score标准化）
    print("进行z-score标准化...")
    sc.pp.scale(adata, max_value=10)
    
    print(f"最终处理后数据形状: {adata.shape}")
    return adata



def load_singlecell_data(adata, discrete_key="cell_type", cont_keys=["n_counts", "percent_mt"], return_dims: bool = False):
    """
    读取单细胞 h5ad 数据，并返回 PyTorch 张量
    
    参数:
        adata: 单细胞AnnData对象
        discrete_key (str | List[str]): 用作离散条件的列名（adata.obs 中）
        cont_keys (list[str]): 用作连续条件的列名（adata.obs 中）
        return_dims (bool): 若为 True，额外返回 (cond_dims, cont_dim)
    
    返回:
        若 return_dims 为 False: (X, y_discrete, y_cont)
        若 return_dims 为 True:  (X, y_discrete, y_cont, cond_dims, cont_dim)
    """
    if not HAS_SCANPY:
        raise ImportError("scanpy is required for single-cell data processing. Install with: pip install scanpy")
    # 输入数据 (cells × genes)
    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    # 安全标准化：避免零方差导致的 NaN/Inf
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 离散条件：支持 str 或 List[str]
    if isinstance(discrete_key, (list, tuple)):
        idx_list = []
        cond_dims = []
        for key in discrete_key:
            if key not in adata.obs:
                raise ValueError(f"{key} not found in adata.obs")
            le = LabelEncoder()
            idx = le.fit_transform(adata.obs[key].astype(str))
            idx_list.append(idx.reshape(-1, 1))
            cond_dims.append(int(len(le.classes_)))
        y_discrete_np = np.hstack(idx_list).astype(np.int64)  # [N, M]
    else:
        if discrete_key not in adata.obs:
            raise ValueError(f"{discrete_key} not found in adata.obs")
        le = LabelEncoder()
        y_discrete_np = le.fit_transform(adata.obs[discrete_key].astype(str)).astype(np.int64)  # [N]
        cond_dims = [int(len(le.classes_))]

    # 连续条件（保持二维，健壮处理）
    cont_list = []
    if cont_keys is not None:
        for key in cont_keys:
            if key not in adata.obs:
                # 忽略缺失列
                continue
            col = adata.obs[key].values.astype(np.float32).reshape(-1, 1)
            # 用列中位数填充 NaN；若全为 NaN 则填 0
            if np.isnan(col).any():
                med = np.nanmedian(col)
                if np.isnan(med):
                    med = 0.0
                col = np.where(np.isnan(col), med, col)
            cont_list.append(col)
    if len(cont_list) == 0:
        y_cont_np = np.zeros((adata.n_obs, 0), dtype=np.float32)
    else:
        y_cont_np = np.hstack(cont_list).astype(np.float32)
        # 对连续变量做安全标准化
        mean = np.nanmean(y_cont_np, axis=0, keepdims=True)
        std = np.nanstd(y_cont_np, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        y_cont_np = (y_cont_np - mean) / std
        y_cont_np = np.nan_to_num(y_cont_np, nan=0.0, posinf=0.0, neginf=0.0)
    cont_dim = int(y_cont_np.shape[1])

    # 转张量
    X = torch.tensor(X, dtype=torch.float32)
    y_discrete = torch.tensor(y_discrete_np, dtype=torch.long)  # [N] 或 [N, M]
    y_cont = torch.tensor(y_cont_np, dtype=torch.float32)       # [N, cont_dim]

    if return_dims:
        return X, y_discrete, y_cont, cond_dims, cont_dim
    else:
        return X, y_discrete, y_cont


def add_latent_to_adata(model, adata, discrete_key="cell_type", cont_keys=["n_counts", "percent_mt"], use_mu=True):
    """
    从 CVAE 提取潜在变量，并加入到 AnnData 中 (adata.obsm["X_latent"])
    
    参数:
        model: 训练好的 CVAE 模型
        adata: AnnData 对象
        discrete_key: 用作离散条件的 obs 列（str 或 List[str]）
        cont_keys: 用作连续条件的 obs 列（可为 None 或空列表）
        use_mu: True -> 用 encoder 的 μ; False -> 采样 z
    
    返回:
        adata: 更新后的AnnData对象
    """
    device = next(model.parameters()).device

    # 输入特征：与训练阶段一致的安全标准化
    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    # 离散条件：支持单变量或多变量
    if isinstance(discrete_key, (list, tuple)):
        idx_list = []
        for key in discrete_key:
            if key not in adata.obs:
                raise ValueError(f"{key} not found in adata.obs")
            le = LabelEncoder()
            idx = le.fit_transform(adata.obs[key].astype(str))
            idx_list.append(idx.reshape(-1, 1))
        y_discrete_np = np.hstack(idx_list).astype(np.int64)  # [N, M]
    else:
        if discrete_key not in adata.obs:
            raise ValueError(f"{discrete_key} not found in adata.obs")
        le = LabelEncoder()
        y_discrete_np = le.fit_transform(adata.obs[discrete_key].astype(str)).astype(np.int64)  # [N]

    y_discrete = torch.tensor(y_discrete_np, dtype=torch.long).to(device)

    # 连续条件：允许 None/空列表，忽略缺失列，填充 NaN 并安全标准化
    cont_list = []
    if cont_keys is not None:
        for key in cont_keys:
            if key not in adata.obs:
                continue
            col = adata.obs[key].values.astype(np.float32).reshape(-1, 1)
            if np.isnan(col).any():
                med = np.nanmedian(col)
                if np.isnan(med):
                    med = 0.0
                col = np.where(np.isnan(col), med, col)
            cont_list.append(col)
    if len(cont_list) == 0:
        y_cont_np = np.zeros((adata.n_obs, 0), dtype=np.float32)
    else:
        y_cont_np = np.hstack(cont_list).astype(np.float32)
        mean = np.nanmean(y_cont_np, axis=0, keepdims=True)
        std = np.nanstd(y_cont_np, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        y_cont_np = (y_cont_np - mean) / std
        y_cont_np = np.nan_to_num(y_cont_np, nan=0.0, posinf=0.0, neginf=0.0)
    y_cont = torch.tensor(y_cont_np, dtype=torch.float32).to(device)

    # 对齐离散与连续条件的形状到模型预期
    expected_m = len(getattr(model, 'cond_dims', [])) if hasattr(model, 'cond_dims') else (1 if y_discrete.dim() == 1 else y_discrete.shape[1])
    if expected_m <= 0:
        expected_m = 1
    if expected_m == 1:
        # 单变量：接受 [N] 或 [N,1]
        if y_discrete.dim() == 2 and y_discrete.size(1) == 1:
            y_discrete = y_discrete[:, 0]
        elif y_discrete.dim() == 2 and y_discrete.size(1) != 1:
            raise ValueError(f"模型期望 1 个离散变量，但收到形状 {tuple(y_discrete.shape)}。请传入与训练时一致的 discrete_key（单个列名）。")
    else:
        # 多变量：需要 [N, M]
        if y_discrete.dim() == 1:
            y_discrete = y_discrete.view(-1, 1)
        if y_discrete.size(1) != expected_m:
            raise ValueError(f"模型期望 {expected_m} 个离散变量，但收到 {y_discrete.size(1)} 个。请传入与训练时一致的 discrete_key 列表。")

    expected_c = int(getattr(model, 'cont_dim', y_cont.shape[1]))
    if expected_c <= 0:
        # 模型不使用连续条件：确保提供 [N, 0]
        if y_cont.dim() == 1:
            y_cont = y_cont.view(-1, 1)
        if y_cont.shape[1] != 0:
            y_cont = y_cont[:, :0]
    else:
        if y_cont.dim() == 1:
            y_cont = y_cont.view(-1, 1)
        if y_cont.shape[1] < expected_c:
            # 右侧用 0 补齐
            pad_cols = expected_c - y_cont.shape[1]
            pad = torch.zeros((y_cont.shape[0], pad_cols), dtype=y_cont.dtype, device=y_cont.device)
            y_cont = torch.cat([y_cont, pad], dim=1)
        elif y_cont.shape[1] > expected_c:
            # 截断多余列
            y_cont = y_cont[:, :expected_c]

    # 提取潜在变量
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X, y_discrete, y_cont)
        latent_t = mu if use_mu else model.reparameterize(mu, logvar)
        latent = latent_t.cpu().numpy()

    # 保存到 AnnData
    adata.obsm["X_latent"] = latent
    return adata


def load_h5ad_file(file_path, discrete_key="cell_type", cont_keys=["n_counts", "percent_mt"]):
    """
    从文件路径加载h5ad文件并处理数据
    
    参数:
        file_path (str): h5ad 文件路径
        discrete_key (str): 用作离散条件的列名
        cont_keys (list[str]): 用作连续条件的列名
    
    返回:
        X, y_discrete, y_cont: 处理后的数据张量
        adata: 原始AnnData对象
    """
    if not HAS_SCANPY:
        raise ImportError("scanpy is required for loading h5ad files. Install with: pip install scanpy")
    
    # 读取 h5ad 文件
    adata = sc.read_h5ad(file_path)
    
    # 处理数据
    X, y_discrete, y_cont = load_singlecell_data(adata, discrete_key, cont_keys)
    
    return X, y_discrete, y_cont, adata


def select_top_hvgs(adata, n_top: int = 2000, inplace: bool = False):
    """统一选择top高变基因，确保不同流程使用相同的基因集合。
    若inplace=False，返回筛选后的新adata；否则在原对象上标记并子集化。
    假设输入adata可能是原始counts或已log标准化数据。
    """
    ad = adata.copy() if not inplace else adata
    # 如果是原始counts，先进行normalize+log1p，仅用于HVG选择，不覆盖raw
    try:
        tmp = ad.copy()
        if sp.issparse(tmp.X):
            pass
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat_v3")
        hv_mask = tmp.var["highly_variable"].values
    except Exception as e:
        # 回退：避免对 seurat_v3 的依赖，并清理 inf/NaN
        tmp = ad.copy()
        # 将数据转换为稠密并移除 NaN/Inf
        try:
            X_arr = tmp.X.toarray() if sp.issparse(tmp.X) else np.asarray(tmp.X)
        except Exception:
            X_arr = np.asarray(tmp.X)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        tmp.X = X_arr
        # 尝试标准化与对数化（若失败则跳过）
        try:
            sc.pp.normalize_total(tmp, target_sum=1e4)
            sc.pp.log1p(tmp)
        except Exception:
            pass
        # 使用不需要 scikit-misc 的 flavor，若仍失败则采用基于方差的简化策略
        try:
            sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat")
        except Exception:
            try:
                sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="cell_ranger")
            except Exception:
                # 最终兜底：按方差选取 top n_top 基因
                X_dense = tmp.X.toarray() if sp.issparse(tmp.X) else np.asarray(tmp.X)
                X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=0.0, neginf=0.0)
                if X_dense.ndim != 2:
                    X_dense = X_dense.reshape((X_dense.shape[0], -1))
                # 方差按列（基因）计算
                vars_ = X_dense.var(axis=0)
                if n_top <= 0 or n_top >= vars_.shape[0]:
                    hv_mask = np.ones_like(vars_, dtype=bool)
                else:
                    idx = np.argpartition(-vars_, n_top - 1)[:n_top]
                    hv_mask = np.zeros_like(vars_, dtype=bool)
                    hv_mask[idx] = True
                if inplace:
                    ad._inplace_subset_var(hv_mask)
                    return ad
                else:
                    return ad[:, hv_mask].copy()
        hv_mask = tmp.var["highly_variable"].values
    if inplace:
        ad._inplace_subset_var(hv_mask)
        return ad
    else:
        return ad[:, hv_mask].copy()


def _compute_size_factors(adata, c: float = 1e4):
    """Compute and return scaled library sizes (library_size/c).
    - Ensure adata.obs['library_size'] exists as the per-cell total counts (float32)
    - Return sf = library_size/c (float32) and also write adata.obs['size_factors'] = sf for compatibility
    """
    # compute or read library_size
    if "library_size" in adata.obs:
        totals = adata.obs["library_size"].to_numpy()
    else:
        if sp.issparse(adata.X):
            totals = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        else:
            totals = adata.X.sum(axis=1).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=1)
        totals = np.asarray(totals).reshape(-1)
        adata.obs["library_size"] = totals.astype(np.float32)
    # scaled by constant c
    sf = (totals.astype(np.float32) / float(c)).astype(np.float32)
    sf = np.maximum(sf, 0.0).astype(np.float32)
    # write compatibility column
    adata.obs["size_factors"] = sf
    return sf.astype(np.float32)


def make_counts_training_tensors_v2(adata, discrete_key="sample", cont_keys=None, use_hvgs: bool = False, n_top_genes: int = 2000, return_dims: bool = False):
    """基于原始 counts 构造用于 NB/Gaussian 统一训练的五元组：
    返回：X_enc=log1p(counts), y_discrete, y_cont, sf=library_size/1e4, x_target=counts
    说明：
    - 编码器输入改为 log1p(normalize_total(counts, 1e4))（仅作用于 encoder 输入）
    - 重构目标仍为原始 counts
    - 训练中将解码器输出 mu 与 sf 相乘（mu_eff=mu*sf）后与 counts 对齐
    - cont_keys 会做标准化；discrete_key 用 LabelEncoder 编码
    """
    ad = adata.copy()
    # 可选 HVG 子集（通常外部已统一基因集合，这里默认 False）
    if use_hvgs:
        try:
            tmp = ad.copy()
            sc.pp.normalize_total(tmp, target_sum=1e4)
            sc.pp.log1p(tmp)
            sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat_v3")
            hv_mask = tmp.var["highly_variable"].values
        except Exception:
            sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes)
            hv_mask = ad.var["highly_variable"].values
        ad = ad[:, hv_mask].copy()

    # 先计算 size_factors（library_size/1e4），随后构造 encoder 输入
    size_factors = _compute_size_factors(ad)  # [N]

    # counts 与 log1p(normalize_total(counts, 1e4))
    X = ad.X
    X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X_arr = X_arr.astype(np.float32)
    sf_col = size_factors.reshape(-1, 1).astype(np.float32)
    sf_safe = np.where(sf_col <= 0.0, 1.0, sf_col)
    # 归一：counts_norm = counts / sf  （等价于 normalize_total target_sum=1e4）
    X_enc_arr = np.log1p(X_arr / sf_safe).astype(np.float32)

    # 条件变量
    if isinstance(discrete_key, (list, tuple)):
        idx_list = []
        cond_dims = []
        for key in discrete_key:
            if key not in ad.obs:
                raise ValueError(f"{key} not found in adata.obs")
            le = LabelEncoder()
            idx = le.fit_transform(ad.obs[key].astype(str))
            idx_list.append(idx.reshape(-1, 1))
            cond_dims.append(int(len(le.classes_)))
        y_discrete_np = np.hstack(idx_list).astype(np.int64)
    else:
        if discrete_key not in ad.obs:
            raise ValueError(f"{discrete_key} not found in adata.obs")
        le = LabelEncoder()
        y_discrete_np = le.fit_transform(ad.obs[discrete_key].astype(str)).astype(np.int64)
        cond_dims = [int(len(le.classes_))]

    cont_list = []
    if cont_keys is not None:
        for key in cont_keys:
            if key not in ad.obs:
                continue
            col = ad.obs[key].values.astype(np.float32).reshape(-1, 1)
            if np.isnan(col).any():
                med = np.nanmedian(col)
                if np.isnan(med):
                    med = 0.0
                col = np.where(np.isnan(col), med, col)
            cont_list.append(col)
    if len(cont_list) == 0:
        y_cont_np = np.zeros((ad.n_obs, 0), dtype=np.float32)
    else:
        y_cont_np = np.hstack(cont_list).astype(np.float32)
        mean = np.nanmean(y_cont_np, axis=0, keepdims=True)
        std = np.nanstd(y_cont_np, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        y_cont_np = (y_cont_np - mean) / std
        y_cont_np = np.nan_to_num(y_cont_np, nan=0.0, posinf=0.0, neginf=0.0)

    # 转张量（sf 已在上方计算）
    X_enc_t = torch.tensor(X_enc_arr, dtype=torch.float32)
    y_discrete_t = torch.tensor(y_discrete_np, dtype=torch.long)
    y_cont_t = torch.tensor(y_cont_np, dtype=torch.float32)
    sf_t = torch.tensor(size_factors.reshape(-1, 1), dtype=torch.float32)
    x_target_t = torch.tensor(X_arr, dtype=torch.float32)

    if return_dims:
        return X_enc_t, y_discrete_t, y_cont_t, sf_t, x_target_t, cond_dims, int(y_cont_t.shape[1])
    else:
        return X_enc_t, y_discrete_t, y_cont_t, sf_t, x_target_t


def make_gaussian_training_tensors(adata, discrete_key=["sample"], cont_keys=None, use_hvgs: bool = False, n_top_genes: int = 2000, return_dims: bool = False):
    """与 make_counts_training_tensors_v2 相同，但用于 Gaussian 路径：
    - X_enc=log1p(counts)
    - y_discrete/y_cont 同上
    - sf=library_size/1e4
    - x_target=counts（用于 MSE: MSE(mu*sf, counts)）
    """
    return make_counts_training_tensors_v2(
        adata,
        discrete_key=discrete_key,
        cont_keys=cont_keys,
        use_hvgs=use_hvgs,
        n_top_genes=n_top_genes,
        return_dims=return_dims,
    )

# ... existing code ...