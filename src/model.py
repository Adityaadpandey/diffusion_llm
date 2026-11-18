"""Bidirectional Transformer encoder for masked discrete diffusion.

The model is BERT-like (no causal mask) and timestep-conditioned via
adaptive LayerNorm (adaLN-Zero, as in DiT). The timestep embedding produces
per-layer scale/shift/gate modulation parameters, which is a much stronger
conditioning signal than adding the timestep embedding once at the input.
It predicts p(x0 | xt, t) independently at every position.
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation: x * (1 + scale) + shift, broadcasting over length."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for a single head dimension."""

    def __init__(self, dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # interleaved pairs
        self.register_buffer("cos", emb.cos())  # (max_seq_len, dim)
        self.register_buffer("sin", emb.sin())

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to q and k of shape (B, H, L, D)."""
        L = q.shape[2]
        cos = self.cos[:L].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:L].unsqueeze(0).unsqueeze(0)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings (no causal mask)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        self.attn_dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).view(B, L, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, hd).transpose(1, 2)

        q, k = self.rotary(q, k)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm transformer encoder layer with adaLN-Zero conditioning.

    The LayerNorms are non-affine; their scale/shift come from the conditioning
    vector `c`. Each sublayer output is gated by a conditioning-derived gate that
    starts at zero, so the block is an identity at initialization (stable training).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.use_rope = use_rope
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        if use_rope:
            self.attn = RotarySelfAttention(d_model, num_heads, dropout, max_seq_len)
        else:
            self.attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        # adaLN-Zero: produce 6 modulation tensors from the conditioning vector.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, L, D) hidden states.
            c: (B, D) conditioning vector (timestep embedding).
        """
        (
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Modulated, gated self-attention.
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.use_rope:
            h = self.attn(h)
        else:
            h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        # Modulated, gated feed-forward.
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.ff(h)
        x = x + gate_mlp.unsqueeze(1) * h
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
        use_rope: bool = False,
        rotary_max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights
        self.use_rope = use_rope

        self.token_embed = nn.Embedding(vocab_size, d_model)
        # Learned absolute positions are only used in the non-RoPE path.
        self.pos_embed = None if use_rope else nn.Embedding(max_seq_len, d_model)
        self.time_embed = SinusoidalTimestepEmbedding(d_model)
        # MLP that turns the raw sinusoidal embedding into the conditioning vector.
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        # RoPE buffers are sized independently of training length so you can
        # generate sequences longer than max_seq_len at inference time.
        rope_len = rotary_max_seq_len if use_rope else max_seq_len
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, use_rope, rope_len)
            for _ in range(num_layers)
        ])

        # Final adaLN-modulated norm before projecting to vocab.
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.output.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, std=0.02)

        # Standard init for the time MLP.
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        for layer in self.layers:
            if isinstance(layer.attn, RotarySelfAttention):
                layer.attn.init_weights()
            else:
                nn.init.xavier_uniform_(layer.attn.in_proj_weight)
                nn.init.zeros_(layer.attn.in_proj_bias)
                nn.init.xavier_uniform_(layer.attn.out_proj.weight)
                nn.init.zeros_(layer.attn.out_proj.bias)
            nn.init.xavier_uniform_(layer.ff[0].weight)
            nn.init.zeros_(layer.ff[0].bias)
            nn.init.xavier_uniform_(layer.ff[3].weight)
            nn.init.zeros_(layer.ff[3].bias)
            # adaLN-Zero: zero the modulation output so each block starts as identity.
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaLN_modulation[-1].bias)

        # Zero the final modulation too.
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

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

        if self.use_rope:
            # RoPE injects position info inside attention, no learned embedding.
            h = self.token_embed(x)
        else:
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            h = self.token_embed(x) + self.pos_embed(positions)
        h = self.dropout(h)

        # Conditioning vector from the timestep, shared across all layers.
        c = self.time_mlp(self.time_embed(t))  # (B, D)

        # Causal mask is intentionally None -> bidirectional attention.
        for layer in self.layers:
            h = layer(h, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        logits = self.output(h)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Legacy architecture (pre-adaLN) — kept so older checkpoints still load.
# These checkpoints add the timestep embedding once at the input and use
# affine LayerNorms (time_proj, final_norm.weight, layers.N.norm{1,2}.weight).
# --------------------------------------------------------------------------- #

class LegacyTransformerEncoderLayer(nn.Module):
    """Original pre-LN encoder layer with affine norms (no adaLN)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        return x


class LegacyDiffusionTransformer(nn.Module):
    """Original timestep-conditioned transformer (timestep added at the input)."""

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
            LegacyTransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.output.weight = self.token_embed.weight

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        device = x.device
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        token_emb = self.token_embed(x)
        pos_emb = self.pos_embed(positions)
        time_emb = self.time_proj(self.time_embed(t)).unsqueeze(1)
        h = self.dropout(token_emb + pos_emb + time_emb)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.output(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def detect_arch(safetensors_path: str) -> str:
    """Return 'adaln' (new) or 'legacy' based on a checkpoint's tensor keys."""
    from safetensors import safe_open

    with safe_open(safetensors_path, framework="pt") as f:
        keys = set(f.keys())
    # The new model has time_mlp/adaLN params; the legacy one has time_proj.
    if any(k.startswith("time_mlp") or "adaLN" in k for k in keys):
        return "adaln"
    return "legacy"


def build_diffusion_model(model_config: dict, arch: str = "adaln") -> nn.Module:
    """Instantiate the right architecture given a model config dict."""
    cls = DiffusionTransformer if arch == "adaln" else LegacyDiffusionTransformer
    return cls(**model_config)
