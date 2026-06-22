"""Masked discrete diffusion: schedules, noising, loss, and sampling.

The forward process masks each token independently with probability beta(t).
The reverse process trains a model to predict p(x0 | xt, t) and then samples
x_{t-1} either via the analytic posterior or via confidence-based unmasking.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #

def linear_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
    """t / T."""
    return t.float() / T


def cosine_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
    """1 - cos((t/T) * pi/2)."""
    return 1.0 - torch.cos((t.float() / T) * (math.pi / 2.0))


def sqrt_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
    """sqrt(t / T)."""
    return (t.float() / T) ** 0.5


SCHEDULES = {
    "linear": linear_schedule,
    "cosine": cosine_schedule,
    "sqrt": sqrt_schedule,
}


def get_schedule(name: str) -> Callable[[torch.Tensor, int], torch.Tensor]:
    if name not in SCHEDULES:
        raise ValueError(f"Unknown schedule {name}. Choose from {list(SCHEDULES.keys())}.")
    return SCHEDULES[name]


# --------------------------------------------------------------------------- #
# Forward process / loss
# --------------------------------------------------------------------------- #

def sample_mask(beta: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Sample a boolean mask with probability beta (per sample).

    Args:
        beta: (B,) mask probabilities.
        shape: target shape, first dim must equal beta.shape[0].
    Returns:
        bool tensor of shape `shape`.
    """
    beta = beta.view(-1, *([1] * (len(shape) - 1)))
    return torch.bernoulli(beta.expand(shape)).bool()


def forward_noising(
    x0: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    schedule_fn: Callable[[torch.Tensor, int], torch.Tensor],
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply masked corruption to clean sequence x0.

    Returns:
        xt: corrupted sequence.
        mask: boolean mask of shape (B, L).
        beta_t: (B,) corruption probabilities.
    """
    beta_t = schedule_fn(t, T)
    mask = sample_mask(beta_t, x0.shape)
    xt = x0.clone()
    xt[mask] = mask_token_id
    return xt, mask, beta_t


def diffusion_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    schedule_fn: Callable[[torch.Tensor, int], torch.Tensor],
    T: int,
) -> tuple[torch.Tensor, dict]:
    """ELBO-based training loss: weighted CE on masked positions.

    Args:
        model: predicts logits (B, L, V).
        x0: clean token ids (B, L).
        t: timesteps (B,) in [1, T].
    Returns:
        loss scalar and info dict.
    """
    B, L = x0.shape
    xt, mask, beta_t = forward_noising(x0, t, mask_token_id, schedule_fn, T)

    logits = model(xt, t)  # (B, L, V)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        x0.reshape(-1),
        reduction="none",
    ).view(B, L)

    # Train on all positions. Unmasked positions are already x0, so this
    # also teaches the model to copy known tokens, which stabilizes early
    # training and improves generation quality.
    loss = ce.mean()

    num_masked = mask.float().sum(dim=1)
    info = {
        "num_masked": num_masked.mean().item(),
        "beta_mean": beta_t.mean().item(),
    }
    return loss, info


# --------------------------------------------------------------------------- #
# Reverse sampling
# --------------------------------------------------------------------------- #

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k along the last dimension."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = logits.topk(k, dim=-1)
    min_val = values[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_val, float("-inf"))


@torch.no_grad()
def reverse_step_random(
    model_probs: torch.Tensor,
    xt: torch.Tensor,
    t: int,
    t_prev: int,
    mask_token_id: int,
    schedule_fn: Callable[[torch.Tensor, int], torch.Tensor],
    T: int,
) -> torch.Tensor:
    """Analytic posterior sampling for masking diffusion.

    Unmasked positions stay unchanged. For masked positions, stay masked with
    probability beta(t_prev)/beta(t); otherwise sample from model_probs.
    """
    B, L, V = model_probs.shape
    device = xt.device

    beta_t = schedule_fn(torch.tensor([t], device=device, dtype=torch.float32), T).item()
    beta_t_prev = (
        schedule_fn(torch.tensor([t_prev], device=device, dtype=torch.float32), T).item()
        if t_prev > 0
        else 0.0
    )

    sampled = torch.multinomial(model_probs.view(B * L, V), num_samples=1).view(B, L)

    stay_masked_prob = beta_t_prev / beta_t if beta_t > 0 else 0.0
    stay = torch.rand(B, L, device=device) < stay_masked_prob

    x_prev = xt.clone()
    masked_pos = xt == mask_token_id
    x_prev[masked_pos] = torch.where(
        stay[masked_pos],
        torch.tensor(mask_token_id, device=device),
        sampled[masked_pos],
    )
    return x_prev


@torch.no_grad()
def reverse_step_confidence(
    model_probs: torch.Tensor,
    xt: torch.Tensor,
    unmask_fraction: float,
    mask_token_id: int,
) -> torch.Tensor:
    """Unmask the most confident `unmask_fraction` of currently masked tokens.

    Already-unmasked positions are left untouched.
    """
    B, L, V = model_probs.shape

    confidence, tokens = model_probs.max(dim=-1)  # (B, L)
    masked = xt == mask_token_id

    # Never touch unmasked positions.
    confidence = confidence.clone()
    confidence[~masked] = -float("inf")

    n_masked = masked.sum().item()
    n_unmask = max(1, int(n_masked * unmask_fraction))
    if n_unmask >= n_masked:
        n_unmask = n_masked

    x_prev = xt.clone().view(-1)
    if n_unmask > 0:
        flat_conf = confidence.view(-1)
        _, top_idx = flat_conf.topk(n_unmask, largest=True)
        flat_tokens = tokens.view(-1)
        x_prev[top_idx] = flat_tokens[top_idx]

    return x_prev.view(B, L)


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    seq_len: int,
    mask_token_id: int,
    T: int,
    inference_steps: int,
    schedule_name: str = "cosine",
    sampling_mode: Literal["confidence", "random"] = "confidence",
    temperature: float = 1.0,
    top_k: int = 0,
    device: torch.device | str = "cpu",
    prompt_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a sequence from fully masked (or partially prompted) state.

    Args:
        model: trained diffusion model.
        seq_len: target sequence length.
        mask_token_id: id of [MASK].
        T: total training timesteps.
        inference_steps: number of reverse steps (can be << T).
        schedule_name: corruption schedule name.
        sampling_mode: "confidence" or "random".
        temperature: sampling temperature.
        top_k: top-k filtering (0 disables).
        device: torch device.
        prompt_ids: optional (1, P) tensor of token ids to keep fixed at start.
    Returns:
        generated token ids (1, seq_len).
    """
    model.eval()
    schedule_fn = get_schedule(schedule_name)

    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    if prompt_ids is not None:
        prompt_ids = prompt_ids.to(device)
        p_len = prompt_ids.size(1)
        if p_len > seq_len:
            prompt_ids = prompt_ids[:, :seq_len]
            p_len = seq_len
        x[:, :p_len] = prompt_ids

    # Reverse schedule: e.g. [1000, 985, ..., 1]
    timesteps = torch.linspace(T, 1, inference_steps).long().tolist()

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device)
        logits = model(x, t_tensor)  # (1, L, V)
        logits = logits / temperature
        logits = top_k_filter(logits, top_k)
        probs = logits.softmax(dim=-1)

        if sampling_mode == "random":
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            x = reverse_step_random(probs, x, t, t_prev, mask_token_id, schedule_fn, T)
        else:
            # Confidence: unmask a fraction each step.
            unmask_fraction = 1.0 / inference_steps
            x = reverse_step_confidence(probs, x, unmask_fraction, mask_token_id)

    return x
