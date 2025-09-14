# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Union, Sequence, Optional


class CVAE(nn.Module):
    """条件变分自编码器，分类条件用 nn.Embedding，连续条件直接拼接
    支持单个或多个分类条件：
      - cond_dim: int 或 List[int]（每个分类变量的类别数）
      - embed_dim: int 或 List[int]（每个分类变量的嵌入维度）
    结构：
      - Encoder MLP: [128, 64]，每层 Linear → GELU → LayerNorm → Dropout(0.1)
      - Decoder MLP: [128, 64]，每层 Linear → GELU → LayerNorm → Dropout(0.1)
      - 输出头：mu/logvar（encoder 末端），以及 decoder 最后一层接 Linear → input_dim
    """
    def __init__(
        self,
        input_dim,
        cond_dim: Union[int, Sequence[int]],
        cont_dim,
        latent_dim,
        embed_dim: Union[int, Sequence[int]] = 16,
        encoder_layers: Optional[Sequence[int]] = (128, 64),
        decoder_layers: Optional[Sequence[int]] = (128, 64),
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.1,
        recon_dist: str = "gaussian",  # "gaussian" or "nb" (negative binomial)
        cond_in_encoder: bool = True,   # 新增：是否在编码器侧使用条件
    ):
        super(CVAE, self).__init__()
        # 归一化 cond_dims 与 embed_dims（保证兼容单变量与多变量）
        if isinstance(cond_dim, int):
            self.cond_dims = [int(cond_dim)]
        else:
            self.cond_dims = [int(c) for c in cond_dim]
        if isinstance(embed_dim, int):
            self.embed_dims = [int(embed_dim)] * len(self.cond_dims)
        else:
            assert len(embed_dim) == len(self.cond_dims), "embed_dim 列表长度必须与分类变量数一致"
            self.embed_dims = [int(e) for e in embed_dim]

        self.cont_dim = int(cont_dim)            # 连续条件维度
        self.embed_dim = sum(self.embed_dims)    # 总嵌入维度（用于构造网络）
        self.cond_in_encoder = bool(cond_in_encoder)

        # 多个分类变量的嵌入层
        self.label_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=e) for c, e in zip(self.cond_dims, self.embed_dims)
        ])
        
        # 计算总条件维度，支持cont_dim=0的情况
        total_cond_dim = self.embed_dim + (self.cont_dim if self.cont_dim > 0 else 0)

        # 保存配置
        self.encoder_layers = list(encoder_layers) if (encoder_layers is not None and len(encoder_layers) > 0) else []
        self.decoder_layers = list(decoder_layers) if (decoder_layers is not None and len(decoder_layers) > 0) else []
        self.activation_name = activation.lower()
        self.norm_name = norm.lower() if norm is not None else "none"
        self.dropout_p = float(dropout) if dropout is not None else 0.0

        # 构建激活/归一化工厂
        def act_layer(name: str):
            if name == "relu":
                return nn.ReLU()
            if name == "silu":
                return nn.SiLU()
            # 默认 gelu
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

        # 编码器 MLP
        if len(self.encoder_layers) == 0:
            raise ValueError("encoder_layers must be a non-empty sequence.")
        enc_in_dim = input_dim + (total_cond_dim if self.cond_in_encoder else 0)
        self.encoder_mlp, enc_out_dim = build_mlp(enc_in_dim, self.encoder_layers)

        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # 解码器 MLP
        if len(self.decoder_layers) == 0:
            raise ValueError("decoder_layers must be a non-empty sequence.")
        self.decoder_mlp, dec_out_dim = build_mlp(latent_dim + total_cond_dim, self.decoder_layers)
        # recon head(s)
        self.recon_dist = recon_dist.lower()
        if self.recon_dist == "gaussian":
            self.decoder_head = nn.Linear(dec_out_dim, input_dim)
        elif self.recon_dist == "nb":
            # NB parameters: mean (mu) and inverse-dispersion (theta > 0)
            self.decoder_mu = nn.Linear(dec_out_dim, input_dim)
            self.decoder_theta = nn.Linear(dec_out_dim, input_dim)
        else:
            raise ValueError(f"Unsupported recon_dist: {recon_dist}")

    def _embed_discrete(self, y_discrete: torch.Tensor) -> torch.Tensor:
        """根据分类变量数，生成拼接后的 embedding 向量。
           - 单分类变量：y_discrete 形状 [N] 或 [N, 1]
           - 多分类变量：y_discrete 形状 [N, M]，第 i 列对应第 i 个变量
        """
        if len(self.label_embeddings) == 1:
            # 兼容单变量：允许传 [N] 或 [N, 1]
            if y_discrete.dim() == 2:
                y_idx = y_discrete[:, 0]
            else:
                y_idx = y_discrete
            return self.label_embeddings[0](y_idx)  # [N, embed_dim]
        else:
            # 多变量必须是二维 [N, M]
            if y_discrete.dim() != 2 or y_discrete.size(1) != len(self.label_embeddings):
                raise ValueError(f"期望 y_discrete 形状为 [N, {len(self.label_embeddings)}]，实际为 {tuple(y_discrete.shape)}")
            embs = []
            for i, emb in enumerate(self.label_embeddings):
                embs.append(emb(y_discrete[:, i]))  # [N, embed_i]
            return torch.cat(embs, dim=-1)  # [N, sum(embed_dims)]

    def encode(self, x, y_discrete, y_cont):
        """编码器：结合输入数据和条件信息
        当 self.cond_in_encoder=False 时，编码器不使用条件，仅基于 x 估计 q(z|x)。"""
        if self.cond_in_encoder:
            y_emb = self._embed_discrete(y_discrete)   # [B, sum(embed_dims)]
            # 根据cont_dim是否为0来决定是否拼接连续条件
            if self.cont_dim > 0:
                cond = torch.cat([y_emb, y_cont], dim=-1)  # 拼接条件
            else:
                cond = y_emb  # 只使用离散条件
            h = self.encoder_mlp(torch.cat([x, cond], dim=-1))
        else:
            h = self.encoder_mlp(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # 数值稳定性：避免 exp(logvar) 溢出/下溢
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, y_discrete, y_cont):
        """解码器：结合潜在变量和条件信息
        Returns:
            - if recon_dist == 'gaussian': x_recon (非负 μ)
            - if recon_dist == 'nb': (mu, theta)
        """
        y_emb = self._embed_discrete(y_discrete)
        # 根据cont_dim是否为0来决定是否拼接连续条件
        if self.cont_dim > 0:
            cond = torch.cat([y_emb, y_cont], dim=-1)
        else:
            cond = y_emb  # 只使用离散条件
        h = self.decoder_mlp(torch.cat([z, cond], dim=-1))
        if self.recon_dist == "gaussian":
            # 输出非负 μ
            return torch.nn.functional.softplus(self.decoder_head(h))
        else:  # NB
            # 使用 softplus 更稳健地保证正值，避免 exp 带来的数值爆炸
            mu = torch.nn.functional.softplus(self.decoder_mu(h))
            theta = torch.nn.functional.softplus(self.decoder_theta(h)) + 1e-4  # >0
            return mu, theta

    def forward(self, x, y_discrete, y_cont):
        """前向传播"""
        mu, logvar = self.encode(x, y_discrete, y_cont)
        z = self.reparameterize(mu, logvar)
        dec_out = self.decode(z, y_discrete, y_cont)
        return dec_out, mu, logvar

    def sample_from_prior(self, n_samples, y_discrete, y_cont):
        """从标准正态先验 N(0, I) 采样并解码。"""
        device = next(self.parameters()).device
        z = torch.randn(int(n_samples), self.fc_mu.out_features, device=device)
        with torch.no_grad():
            dec_out = self.decode(z, y_discrete, y_cont)
            if self.recon_dist == "gaussian":
                x_gen = dec_out
            else:
                mu, theta = dec_out
                x_gen = mu
        return x_gen, z


