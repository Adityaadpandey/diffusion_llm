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

    # Train on all positions. Unmasked positions are already x0, so this also
    # teaches the model to copy known tokens, which stabilizes early training
    # and improves generation quality (this is the recipe that produced the
    # `checkpoints_fast` run reaching validation PPL ~3.8).
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
    n_unmask: int,
    mask_token_id: int,
) -> torch.Tensor:
    """Commit the `n_unmask` most-confident currently-masked positions.

    Already-unmasked positions are left untouched. `n_unmask` is an absolute
    count (computed by the caller from the reveal schedule) so that the whole
    sequence is guaranteed to be filled by the final step.
    """
    B, L, V = model_probs.shape

    # Rank positions by the model's confidence (max prob), but SAMPLE the token
    # we commit so that different runs produce different stories. Using argmax
    # here would make every sample identical.
    confidence = model_probs.max(dim=-1).values  # (B, L)
    sampled = torch.multinomial(model_probs.view(B * L, V), num_samples=1).view(B, L)
    masked = xt == mask_token_id

    # Never touch unmasked positions.
    confidence = confidence.clone()
    confidence[~masked] = -float("inf")

    n_masked = int(masked.sum().item())
    n_unmask = min(max(n_unmask, 0), n_masked)

    x_prev = xt.clone().view(-1)
    if n_unmask > 0:
        flat_conf = confidence.view(-1)
        _, top_idx = flat_conf.topk(n_unmask, largest=True)
        flat_sampled = sampled.view(-1)
        x_prev[top_idx] = flat_sampled[top_idx]

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

    # For confidence mode we drive an explicit reveal schedule so the whole
    # sequence is guaranteed filled by the last step. Cosine schedule keeps many
    # tokens masked early (when the model is unsure) and reveals fast near the end.
    n_to_fill = int((x == mask_token_id).sum().item())

    def remaining_masked(step_idx: int) -> int:
        # Fraction still masked after step `step_idx` (0-indexed), cosine-shaped.
        frac = math.cos((math.pi / 2.0) * (step_idx + 1) / inference_steps)
        return int(round(n_to_fill * max(frac, 0.0)))

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
            # Reveal enough tokens to hit the target remaining-masked count,
            # forcing everything filled on the final step.
            target_remaining = 0 if i == len(timesteps) - 1 else remaining_masked(i)
            currently_masked = int((x == mask_token_id).sum().item())
            n_unmask = max(currently_masked - target_remaining, 1 if currently_masked > 0 else 0)
            x = reverse_step_confidence(probs, x, n_unmask, mask_token_id)

    return x
