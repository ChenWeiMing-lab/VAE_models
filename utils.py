import torch
import numpy as np
import random

# ====== 设置随机种子 ======
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ====== 数据生成函数（批次效应） ======
def generate_data(n_samples=2000, num_classes=3, feature_dim=200, cond_dim=3, cont_dim=2):
    data, labels, cont_attrs, cond_labels = [], [], [], []

    cond_offsets = np.linspace(-2.0, 2.0, cond_dim)
    base_per_cond = n_samples // cond_dim
    remainder = n_samples % cond_dim

    for c in range(cond_dim):
        n_cond_samples = base_per_cond + (1 if c < remainder else 0)
        samples_per_class = n_cond_samples // num_classes
        leftover = n_cond_samples % num_classes

        cond_offset = cond_offsets[c]
        for i in range(num_classes):
            n_samples_class = samples_per_class + (1 if i < leftover else 0)
            mean = np.zeros(feature_dim)
            mean[i * (feature_dim // num_classes):(i + 1) * (feature_dim // num_classes)] = 3.0
            cluster = np.random.randn(n_samples_class, feature_dim) + mean + cond_offset
            # 只在cont_dim > 0时生成连续特征
            if cont_dim > 0:
                cont_feature = np.random.randn(n_samples_class, cont_dim) + i * 1.0
                cont_attrs.append(cont_feature)
            else:
                # cont_dim=0时，创建空的占位符
                cont_attrs.append(np.zeros((n_samples_class, 1)))  # 占位符，后续会被正确处理

            data.append(cluster)
            labels.extend([i] * n_samples_class)
            cond_labels.extend([c] * n_samples_class)

    data = np.vstack(data)
    labels = np.array(labels)
    cond_labels = np.array(cond_labels)
    
    # 处理连续属性
    if cont_dim > 0:
        cont_attrs = np.vstack(cont_attrs)
    else:
        # cont_dim=0时，创建形状为(n_samples, 0)的空数组
        cont_attrs = np.empty((len(data), 0))

    idx = np.random.permutation(len(data))   # 用真实大小
    return (
        torch.tensor(data[idx], dtype=torch.float32),
        torch.tensor(labels[idx], dtype=torch.long),
        torch.tensor(cont_attrs[idx], dtype=torch.float32),
        torch.tensor(cond_labels[idx], dtype=torch.long)
    )

    
 