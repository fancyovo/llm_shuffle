from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class TransformerConfig:
    vocab_size: int = 4096
    max_seq_len: int = 8192
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    d_ff: int = 1792
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_embeddings: bool = False
    bias: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (self.weight * x_norm).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)
        cos = self.cos[start_pos : start_pos + seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin[start_pos : start_pos + seq_len].to(dtype=x.dtype, device=x.device)
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        if cfg.d_model != cfg.n_heads * cfg.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.cfg.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.cfg.d_model)
        return self.resid_dropout(self.out(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=cfg.bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * up))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.drop(self.embed(input_ids))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


def build_model(model_cfg: dict) -> TransformerLM:
    cfg = TransformerConfig(**model_cfg)
    return TransformerLM(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_param_count(cfg: TransformerConfig) -> int:
    embed = cfg.vocab_size * cfg.d_model
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    attn = 4 * cfg.d_model * cfg.d_model
    ffn = 3 * cfg.d_model * cfg.d_ff
    norms = 2 * cfg.d_model
    per_layer = attn + ffn + norms
    final_norm = cfg.d_model
    return embed + head + cfg.n_layers * per_layer + final_norm
