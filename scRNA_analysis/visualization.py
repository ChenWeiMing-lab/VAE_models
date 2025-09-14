import torch
import matplotlib.pyplot as plt
import numpy as np
import umap


# ====== 可视化函数 ======
def visualize_latent(model, data, real_labels, cont_attrs, cond_labels, conditional=False, title="VAE", save_path=None):
    """
    可视化潜在空间
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32)
        y_disc = torch.tensor(cond_labels, dtype=torch.long) if conditional else None
        y_cont = torch.tensor(cont_attrs, dtype=torch.float32) if conditional else None
        
        if conditional and (y_disc is not None or y_cont is not None):
            mu, _ = model.encode(
                data.to(device),
                y_disc.to(device) if y_disc is not None else None,
                y_cont.to(device) if y_cont is not None else None
            )
        else:
            mu, _ = model.encode(data.to(device))
    
    # UMAP 降维
    embedding = umap.UMAP(n_components=2).fit_transform(mu.cpu().numpy())  # , random_state=42
    
    # 绘图
    plt.figure(figsize=(5, 4.5))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=real_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'{title} - Latent Space Visualization')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # 保存图片
    if save_path is not None:
        save_path = f'{title.lower()}_latent_space.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()
        # plt.close()  # 关闭图形以释放内存

    return save_path


def compare_models_visualization(models_data, titles=None):
    """比较多个模型的潜在空间可视化
    
    Args:
        models_data: 包含(model, data, y_disc, y_cont, cond_labels, conditional)的列表
        titles: 模型标题列表
    """
    n_models = len(models_data)
    if titles is None:
        titles = [f"Model {i+1}" for i in range(n_models)]
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, (model, data, y_disc, y_cont, cond_labels, conditional) in enumerate(models_data):
        device = next(model.parameters()).device
        model.eval()
        
        with torch.no_grad():
            if conditional:
                mu, _ = model.encode(data.to(device), y_disc.to(device), y_cont.to(device))
            else:
                mu, _ = model.encode(data.to(device))
        
        embedding = umap.UMAP(n_components=2).fit_transform(mu.cpu().numpy())    # , random_state=42
        
        scatter = axes[i].scatter(
            embedding[:, 0], embedding[:, 1],
            c=y_disc.cpu().numpy(),
            cmap="tab10",
            alpha=0.7
        )
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("UMAP 1")
        axes[i].set_ylabel("UMAP 2")
    
    plt.tight_layout()
    plt.show()
