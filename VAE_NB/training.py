import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional


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


def train_vae(model,
              dataloader: DataLoader,
              optimizer,
              epochs: int,
              conditional: bool = False,
              grad_clip: Optional[float] = 5.0,
              beta_final: float = 1.0,
              beta_warmup_epochs: int = 0,
              recon_dist: str = "gaussian",
              free_bits: float | None = None,
              kl_schedule: str | None = None,
              kl_period: int | None = None):
    """VAE training loop with KL scheduling and optional free-bits.
    This function is specialized for non-conditional VAE (no MoG prior support).

    Returns: (loss_list, recon_list, kl_list)
    """
    if conditional:
        raise ValueError("train_vae is for non-conditional VAE only (conditional=False)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_list, recon_list, kl_list = [], [], []

    recon_dist = recon_dist.lower()

    # KL coefficient scheduler
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
            # Support batches shaped as (x, y_disc, y_cont, sf, x_target) or simpler
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 5:
                    x_enc = batch[0].to(device)
                    sf = batch[3].to(device)
                    x_target = batch[4].to(device)
                elif len(batch) == 4:
                    x_enc = batch[0].to(device)
                    sf = batch[3].to(device)
                    x_target = x_enc
                else:
                    x_enc = batch[0].to(device)
                    sf = None
                    x_target = x_enc
            else:
                x_enc = batch.to(device)
                sf = None
                x_target = x_enc

            dec_out, mu, logvar = model(x_enc)

            # Reconstruction loss
            if recon_dist == "gaussian":
                x_recon = dec_out  # non-negative mean
                mu_eff = x_recon if sf is None else (x_recon * sf)
                recon_loss = nn.MSELoss()(mu_eff, x_target)
            elif recon_dist == "nb":
                mu_nb, theta_nb = dec_out
                mu_eff = mu_nb if sf is None else (mu_nb * sf)
                recon_loss = nb_nll_loss(mu_eff, theta_nb, x_target)
            else:
                raise ValueError(f"Unsupported recon_dist: {recon_dist}")

            # Standard N(0,I) KL with optional free-bits
            kl_per_dim_batch = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
            if free_bits is not None and free_bits > 0:
                kl_dim = kl_per_dim_batch.mean(dim=0)
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

        epoch_loss = (total_loss / max(1, num_batches))
        epoch_recon = (total_recon / max(1, num_batches))
        epoch_kl = (total_kl / max(1, num_batches))
        loss_list.append(epoch_loss)
        recon_list.append(epoch_recon)
        kl_list.append(epoch_kl)

    return loss_list, recon_list, kl_list