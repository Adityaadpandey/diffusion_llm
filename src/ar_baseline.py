"""Left-to-right (autoregressive) baseline with the same size and recipe.

It reuses the diffusion model's encoder block with a causal mask, so any gap
between this and the masked diffusion model comes from how the text is modelled
and not from a different network or optimiser. Run:

    python src/ar_baseline.py --config configs/tinystories_ar.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn, optim
from tqdm import tqdm

from data import build_dataset, get_dataloaders
from model import LegacyTransformerEncoderLayer
from train import get_device, get_linear_warmup_cosine_decay, set_seed


class ARTransformer(nn.Module):
    """GPT-style decoder: learned positions, pre-LN blocks, tied output layer."""

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_seq_len, dropout=0.1, tie_weights=True):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            LegacyTransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.output.weight = self.token_embed.weight
        # GPT-style small init. With the default N(0,1) embeddings and tied
        # weights the first logits are huge (loss ~40 instead of ~10), which
        # would handicap the baseline for no good reason.
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids (B, L) are the *inputs*; logits[:, i] predicts the token after ids[:, i]."""
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        h = self.dropout(self.token_embed(ids) + self.pos_embed(pos))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ids.device), 1)
        for layer in self.layers:
            h = layer(h, attn_mask=causal)
        return self.output(self.final_norm(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_ar(model_cfg: dict) -> ARTransformer:
    keys = ("vocab_size", "d_model", "num_layers", "num_heads", "d_ff",
            "max_seq_len", "dropout", "tie_weights")
    return ARTransformer(**{k: model_cfg[k] for k in keys if k in model_cfg})


def with_bos(x: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Shift right and put BOS first, so all L tokens get predicted (like the diffusion side)."""
    return torch.cat([torch.full_like(x[:, :1], bos_id), x[:, :-1]], dim=1)


def token_nll(model: nn.Module, x: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Per-token negative log-likelihood in nats, shape (B, L)."""
    logits = model(with_bos(x, bos_id)).float()
    return F.cross_entropy(logits.transpose(1, 2), x, reduction="none")


@torch.no_grad()
def val_nll(model: nn.Module, val_loader, bos_id: int, device) -> float:
    """Average NLL per token (nats) over the whole validation set."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x in val_loader:
        nll = token_nll(model, x.to(device), bos_id)
        total += nll.sum().item()
        count += nll.numel()
    model.train(was_training)
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    tcfg = config["training"]
    set_seed(tcfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, tokenizer = build_dataset(
        dataset_name=config["dataset"],
        tokenizer_name=config["tokenizer_name"],
        max_seq_len=config["max_seq_len"],
        val_split=config["val_split"],
        text_column=config["text_column"],
        num_proc=tcfg.get("tokenize_workers", 4),
        streaming=config.get("streaming", False),
        max_samples=config.get("max_samples", None),
    )
    config["model"]["vocab_size"] = len(tokenizer)
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.eos_token_id
    train_loader, val_loader = get_dataloaders(
        train_ds, val_ds, tcfg["batch_size"], tcfg.get("num_workers", 0)
    )

    model = build_ar(config["model"]).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    optimizer = optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                            betas=(0.9, 0.98), eps=1e-8, weight_decay=tcfg["weight_decay"])
    accum = tcfg["gradient_accumulation_steps"]
    scheduler = get_linear_warmup_cosine_decay(
        optimizer, tcfg["warmup_steps"] // accum,
        tcfg.get("lr_schedule_steps", tcfg["max_steps"]) // accum,
    )
    use_amp = device.type in ("mps", "cuda") and tcfg.get("mixed_precision") == "bf16"
    out_dir = Path(config["checkpoint_dir"])

    def save(step: int) -> None:
        from safetensors.torch import save_model
        d = out_dir / f"step_{step:07d}"
        d.mkdir(parents=True, exist_ok=True)
        save_model(model, d / "model.safetensors")
        (d / "config.yaml").write_text(yaml.safe_dump(config))
        tokenizer.save_pretrained(d)

    model.train()
    step, running = 0, 0.0
    train_iter = iter(train_loader)
    pbar = tqdm(total=tcfg["max_steps"], desc="AR training")
    while step < tcfg["max_steps"]:
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = token_nll(model, x, bos_id).mean() / accum
        loss.backward()
        running += loss.item()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=f"{running:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            running = 0.0
        step += 1
        pbar.update(1)
        if step % tcfg["eval_every"] == 0:
            nll = val_nll(model, val_loader, bos_id, device)
            print(f"\n[Step {step}] Val NLL: {nll:.4f} nats/token  (ppl {math.exp(nll):.2f})", flush=True)
        if step % tcfg["save_every"] == 0 or step == tcfg["max_steps"]:
            save(step)
    pbar.close()


if __name__ == "__main__":
    main()
