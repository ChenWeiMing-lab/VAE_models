# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Sequence


class VAE(nn.Module):
    """标准 VAE，简化接口且支持多层 MLP 配置（与 CVAE_MoG 类似参数）
    支持两种重构分布：
      - gaussian: 非负均值回归（通过 softplus 约束）
      - nb: 负二项分布计数模型，采用每基因共享的逆离散度（theta）参数
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: Optional[Sequence[int]] = (128, 64),
        decoder_layers: Optional[Sequence[int]] = (128, 64),
        activation: str = "gelu",
        norm: Optional[str] = "layernorm",
        dropout: float = 0.1,
        recon_dist: str = "gaussian",
    ):
        super(VAE, self).__init__()

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_layers = list(encoder_layers) if (encoder_layers is not None and len(encoder_layers) > 0) else []
        self.decoder_layers = list(decoder_layers) if (decoder_layers is not None and len(decoder_layers) > 0) else []
        self.activation_name = activation.lower() if activation is not None else "gelu"
        self.norm_name = norm.lower() if norm is not None else "none"
        self.dropout_p = float(dropout) if dropout is not None else 0.0

        def act_layer(name: str):
            if name == "relu":
                return nn.ReLU()
            if name == "silu":
                return nn.SiLU()
            return nn.GELU()

        def norm_layer(name: str, dim: int):
            if name == "layernorm":
                return nn.LayerNorm(dim)
            if name == "batchnorm":
                return nn.BatchNorm1d(dim)
            return None

        def build_mlp(in_dim: int, hidden_dims: Sequence[int]):
            layers = []
            last = in_dim
            for h in hidden_dims:
                layers.append(nn.Linear(last, h))
                layers.append(act_layer(self.activation_name))
                nl = norm_layer(self.norm_name, h)
                if nl is not None:
                    layers.append(nl)
                if self.dropout_p and self.dropout_p > 0:
                    layers.append(nn.Dropout(self.dropout_p))
                last = h
            return nn.Sequential(*layers), last

        # 编码器
        if len(self.encoder_layers) == 0:
            raise ValueError("encoder_layers must be a non-empty sequence.")
        self.encoder_mlp, enc_out_dim = build_mlp(self.input_dim, self.encoder_layers)

        self.fc_mu = nn.Linear(enc_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, self.latent_dim)

        # 解码器
        if len(self.decoder_layers) == 0:
            raise ValueError("decoder_layers must be a non-empty sequence.")
        self.decoder_mlp, dec_out_dim = build_mlp(self.latent_dim, self.decoder_layers)
        self.recon_dist = recon_dist.lower()
        if self.recon_dist == "gaussian":
            self.decoder_head = nn.Linear(dec_out_dim, self.input_dim)
        elif self.recon_dist == "nb":
            # 使用基因共享的离散度参数以提升稳定性（每个基因一个 theta，所有细胞共享）
            self.decoder_mu = nn.Linear(dec_out_dim, self.input_dim)
            # 以 log_theta 形式参数化，保证正值：theta = softplus(log_theta) + 1e-4
            self.log_theta = nn.Parameter(torch.zeros(self.input_dim))
        else:
            raise ValueError(f"Unsupported recon_dist: {recon_dist}")

    def encode(self, x: torch.Tensor):
        h = self.encoder_mlp(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor):
        h = self.decoder_mlp(z)
        if self.recon_dist == "gaussian":
            return torch.nn.functional.softplus(self.decoder_head(h))
        else:
            # mu: 每细胞-每基因的均值（正值）；theta: 每基因共享的逆离散度（正值），广播到 batch
            mu = torch.exp(self.decoder_mu(h))
            theta = torch.nn.functional.softplus(self.log_theta) + 1e-4  # [G]
            theta = theta.unsqueeze(0).expand(mu.size(0), -1)  # [B, G]
            return mu, theta

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        dec_out = self.decode(z)
        return dec_out, mu, logvar