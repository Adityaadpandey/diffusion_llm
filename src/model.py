"""Bidirectional Transformer encoder for masked discrete diffusion.

The model is BERT-like (no causal mask) and timestep-conditioned via learned
embeddings added to token embeddings. It predicts p(x0 | xt, t) independently
at every position.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps, similar to positional encoding."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args:
            t: (B,) integer timesteps.
        Returns:
            (B, dim) float embeddings.
        """
        angles = t[:, None].float() * self.freqs[None, :]  # (B, half)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm transformer encoder layer with bidirectional attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h

        # Pre-LN feed-forward
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        return x


class DiffusionTransformer(nn.Module):
    """Timestep-conditioned bidirectional transformer for masked diffusion."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.time_embed = SinusoidalTimestepEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.output.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.attn.in_proj_weight)
            nn.init.zeros_(layer.attn.in_proj_bias)
            nn.init.xavier_uniform_(layer.attn.out_proj.weight)
            nn.init.zeros_(layer.attn.out_proj.bias)
            nn.init.xavier_uniform_(layer.ff[0].weight)
            nn.init.zeros_(layer.ff[0].bias)
            nn.init.xavier_uniform_(layer.ff[3].weight)
            nn.init.zeros_(layer.ff[3].bias)
        if not self.tie_weights:
            nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, L) token ids (may include [MASK]).
            t: (B,) integer timesteps in [1, T].
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = x.shape
        device = x.device

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        token_emb = self.token_embed(x)
        pos_emb = self.pos_embed(positions)
        time_emb = self.time_proj(self.time_embed(t)).unsqueeze(1)  # (B, 1, D)

        h = self.dropout(token_emb + pos_emb + time_emb)

        # Causal mask is intentionally None -> bidirectional attention.
        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)
        logits = self.output(h)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
