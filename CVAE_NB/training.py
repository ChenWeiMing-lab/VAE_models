import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math


# ====== MoG-KL: helpers ======
# log N(z | mu, diag(exp(logvar))) per-sample, summing over latent dim
def _log_normal_diag(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # z, mu, logvar: [..., D]
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    # 使用 torch.tensor 创建常量而不是Python float
    log_2pi = torch.tensor(2 * 3.14159265359, dtype=z.dtype, device=z.device).log()
    return -0.5 * (log_2pi * z.size(-1) + (logvar + (z - mu) ** 2 / torch.exp(logvar)).sum(dim=-1))


def _log_p_mog(z: torch.Tensor, mog_mu: torch.Tensor, mog_logvar: torch.Tensor, mog_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute log p_MoG(z) with logsumexp stability.
    z: [S, B, D] or [B, D]
    mog_mu: [K, D], mog_logvar: [K, D], mog_logits: [K]
    returns: [S, B] or [B]
    """
    if z.dim() == 2:
        z = z.unsqueeze(0)  # [1, B, D]
    S, B, D = z.shape
    K, Dm = mog_mu.shape
    assert Dm == D, f"latent dim mismatch: {D} vs {Dm}"

    # temperature for mixture weights
    if temperature is None:
        temperature = 1.0
    pi = torch.softmax(mog_logits / float(temperature), dim=0)  # [K]
    log_pi = torch.log(pi.clamp_min(1e-12))  # [K]

    # expand for broadcasting
    z_e = z.unsqueeze(2)            # [S, B, 1, D]
    mu_e = mog_mu.view(1, 1, K, D)  # [1, 1, K, D]
    lv_e = mog_logvar.view(1, 1, K, D)

    comp_logprob = _log_normal_diag(z_e, mu_e, lv_e)  # [S, B, K]
    comp_logprob = comp_logprob + log_pi.view(1, 1, K)
    log_p = torch.logsumexp(comp_logprob, dim=2)  # [S, B]
    return log_p  # [S, B]


def _component_log_probs(z: torch.Tensor, mog_mu: torch.Tensor, mog_logvar: torch.Tensor, mog_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return per-component log-probabilities log pi_k + log N(z|mu_k, Sigma_k).
    Args:
        z: [B, D]
        mog_mu: [K, D]
        mog_logvar: [K, D]
        mog_logits: [K]
    Returns:
        log_probs: [B, K]
    """
    if temperature is None:
        temperature = 1.0
    B, D = z.shape
    K, Dm = mog_mu.shape
    assert D == Dm, f"latent dim mismatch: {D} vs {Dm}"
    pi = torch.softmax(mog_logits / float(temperature), dim=0)  # [K]
    log_pi = torch.log(pi.clamp_min(1e-12)).view(1, K)  # [1, K]
    z_e = z.view(B, 1, D)              # [B, 1, D]
    mu_e = mog_mu.view(1, K, D)        # [1, K, D]
    lv_e = mog_logvar.view(1, K, D)    # [1, K, D]
    comp_ln = _log_normal_diag(z_e, mu_e, lv_e)  # [B, K]
    return comp_ln + log_pi            # [B, K]


def mog_kl_mc(mu: torch.Tensor, logvar: torch.Tensor,
              mog_mu: torch.Tensor, mog_logvar: torch.Tensor, mog_logits: torch.Tensor,
              num_samples: int = 1, temperature: float = 1.0) -> torch.Tensor:
    """Monte Carlo estimate of KL[q(z|x) || p_MoG(z)] = E_q[log q - log p]
    Returns scalar KL averaged over batch and samples.
    """
    B, D = mu.shape
    S = int(max(1, num_samples))
    # 解析项：E_q[log q(z|x)]（对角高斯）
    log_2pi = torch.tensor(2 * 3.14159265359, dtype=mu.dtype, device=mu.device).log()
    lv_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
    log_q_analytic = -0.5 * (D * log_2pi + (1.0 + lv_clamped).sum(dim=1))  # [B]

    # 仍用 MC 估计 E_q[log p_MoG(z)]
    std = torch.exp(0.5 * lv_clamped)
    eps = torch.randn(S, B, D, device=mu.device, dtype=mu.dtype)
    z = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # [S, B, D]
    log_p = _log_p_mog(z, mog_mu, mog_logvar, mog_logits, temperature=temperature)  # [S, B]
    log_p_mean = log_p.mean(dim=0)  # [B]

    # KL = E_q[log q] - E_q[log p]
    kl = (log_q_analytic - log_p_mean).mean()  # scalar
    return kl


# ====== VAE 损失函数（保留以兼容旧接口） ======
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ====== NB 重构的负对数似然损失（Negative Binomial NLL） ======
def nb_nll_loss(mu: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative binomial negative log-likelihood per-batch mean.
    Parameterization: x ~ NB(mean=mu, inverse-dispersion=theta)
    log p(x|mu,theta) = lgamma(x+theta) - lgamma(theta) - lgamma(x+1)
                        + theta*log(theta/(theta+mu)) + x*log(mu/(theta+mu))
    We return -mean(log p) over batch and features.
    Note: x expected as counts (non-negative, can be float), mu>0, theta>0.
    """
    mu = mu.clamp_min(eps)
    theta = theta.clamp_min(eps)
    t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
    t2 = theta * (torch.log(theta) - torch.log(theta + mu))
    t3 = x * (torch.log(mu) - torch.log(theta + mu))
    log_prob = t1 + t2 + t3
    return -log_prob.mean()


# ====== 条件 VAE 训练函数 ======

def train(model, dataloader, optimizer, epochs, conditional=True, grad_clip=5.0,
          beta_final: float = 1.0,
          beta_warmup_epochs: int = 0,
          recon_dist: str = "gaussian",
          free_bits: float | None = None,
          kl_schedule: str | None = None,
          kl_period: int | None = None):
    """通用训练函数，支持条件 VAE（不包含 MoG）。
    Returns:
        loss_list, recon_list, kl_list
    """
    assert conditional, "This train() is intended for conditional CVAE."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_list, recon_list, kl_list = [], [], []

    recon_dist = recon_dist.lower()

    # KL 系数调度器
    def _current_beta(epoch_idx: int) -> float:
        per = int(kl_period) if (kl_period is not None and kl_period > 0) else int(epochs)
        if kl_schedule is not None:
            t = ((epoch_idx % per) + 1) / float(per)
            if kl_schedule.lower() == 'cosine':
                return float(beta_final) * 0.5 * (1.0 - math.cos(math.pi * min(1.0, t)))
            elif kl_schedule.lower() == 'cyclical':
                return float(beta_final) * 0.5 * (1.0 - math.cos(math.pi * t))
            elif kl_schedule.lower() == 'linear':
                return float(min(1.0, (epoch_idx + 1) / float(per))) * float(beta_final)
        if beta_warmup_epochs and beta_warmup_epochs > 0:
            return float(min(1.0, (epoch_idx + 1) / float(beta_warmup_epochs))) * float(beta_final)
        return float(beta_final)

    for epoch in range(epochs):
        model.train()
        total_loss = total_recon = total_kl = 0.0
        num_batches = 0
        beta = _current_beta(epoch)

        for batch in dataloader:
            # 解析批次，统一支持五元组：(x_enc, y_disc, y_cont, sf, x_target)
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 5:
                    x_enc = batch[0].to(device)
                    y_disc = batch[1].to(device)
                    y_cont = batch[2].to(device)
                    sf = batch[3].to(device)
                    x_target = batch[4].to(device)
                elif len(batch) == 4:
                    x_enc = batch[0].to(device)
                    y_disc = batch[1].to(device)
                    y_cont = batch[2].to(device)
                    sf = batch[3].to(device)
                    x_target = x_enc
                else:  # len==3 常见于旧Gaussian路径
                    x_enc = batch[0].to(device)
                    y_disc = batch[1].to(device)
                    y_cont = batch[2].to(device)
                    sf = None
                    x_target = x_enc
            else:
                raise ValueError("Conditional model expects batch as a tuple/list.")

            dec_out, mu, logvar = model(x_enc, y_disc, y_cont)

            # reconstruction loss（统一 μ_eff = μ * sf，与 counts 目标对齐）
            if recon_dist == "gaussian":
                x_recon = dec_out
                mu_eff = x_recon if sf is None else (x_recon * sf)
                recon_loss = nn.MSELoss()(mu_eff, x_target)
            elif recon_dist == "nb":
                mu_nb, theta_nb = dec_out
                mu_eff = mu_nb if sf is None else (mu_nb * sf)
                recon_loss = nb_nll_loss(mu_eff, theta_nb, x_target)
            else:
                raise ValueError(f"Unsupported recon_dist: {recon_dist}")

            # 标准 N(0, I) 先验 KL；支持 free-bits
            kl_per_dim_batch = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
            if free_bits is not None and free_bits > 0:
                kl_dim = kl_per_dim_batch.mean(dim=0)  # [D]
                kl_dim = torch.clamp(kl_dim - float(free_bits), min=0.0)
                kl_component = kl_dim.mean()
            else:
                kl_component = kl_per_dim_batch.mean()

            loss = recon_loss + beta * kl_component

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_recon += float(recon_loss.item())
            total_kl += float(kl_component.item())
            num_batches += 1

        if num_batches > 0:
            loss_list.append(total_loss / num_batches)
            recon_list.append(total_recon / num_batches)
            kl_list.append(total_kl / num_batches)
        else:
            loss_list.append(float('nan'))
            recon_list.append(float('nan'))
            kl_list.append(float('nan'))

    return loss_list, recon_list, kl_list


def plot_training_loss(loss_list, title="Training Loss"):
    plt.figure(figsize=(4, 3.5))
    plt.plot(loss_list, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()