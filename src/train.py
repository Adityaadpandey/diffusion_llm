"""Training loop for masked discrete diffusion language model."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data import build_dataset, get_dataloaders
from diffusion import diffusion_loss, get_schedule, sample
from model import DiffusionTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: torch.device) -> DiffusionTransformer:
    mcfg = config["model"]
    model = DiffusionTransformer(
        vocab_size=mcfg["vocab_size"],
        d_model=mcfg["d_model"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        d_ff=mcfg["d_ff"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=mcfg.get("dropout", 0.1),
        tie_weights=mcfg.get("tie_weights", True),
    )
    model.to(device)
    return model


def get_linear_warmup_cosine_decay(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step == 0:
            return 0.0
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
    config: dict,
    tokenizer,
    checkpoint_dir: str,
    is_best: bool = False,
) -> None:
    from safetensors.torch import save_model

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    prefix = "best" if is_best else f"step_{step:07d}"
    out_dir = ckpt_dir / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    save_model(model, out_dir / "model.safetensors")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        out_dir / "training.pt",
    )
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    tokenizer.save_pretrained(out_dir)

    print(f"Saved checkpoint to {out_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    from safetensors.torch import load_model

    ckpt_path = Path(checkpoint_dir)
    load_model(model, ckpt_path / "model.safetensors", device=str(device))

    training_path = ckpt_path / "training.pt"
    step = 0
    if training_path.exists():
        training_state = torch.load(training_path, map_location=device)
        if optimizer is not None and "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None and "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])
        step = training_state.get("step", 0)
    return step


def evaluate(model: nn.Module, val_loader, config: dict, device: torch.device) -> float:
    model.eval()
    dcfg = config["diffusion"]
    T = dcfg["T"]
    mask_token_id = config.get("mask_token_id")
    schedule_fn = get_schedule(dcfg["schedule"])

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x0 in val_loader:
            x0 = x0.to(device)
            B = x0.size(0)
            t = torch.randint(1, T + 1, (B,), device=device)
            loss, info = diffusion_loss(model, x0, t, mask_token_id, schedule_fn, T)
            total_loss += loss.item() * info["num_masked"] * B
            total_tokens += info["num_masked"] * B

    model.train()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def generate_samples(
    model: nn.Module,
    tokenizer,
    config: dict,
    device: torch.device,
    num_samples: int = 3,
) -> list[str]:
    dcfg = config["diffusion"]
    seq_len = config["model"]["max_seq_len"]
    mask_token_id = config.get("mask_token_id")

    texts = []
    for _ in range(num_samples):
        out = sample(
            model,
            seq_len=seq_len,
            mask_token_id=mask_token_id,
            T=dcfg["T"],
            inference_steps=dcfg["inference_steps"],
            schedule_name=dcfg["schedule"],
            sampling_mode=dcfg["sampling_mode"],
            temperature=dcfg["temperature"],
            top_k=dcfg["top_k"],
            device=device,
        )
        texts.append(tokenizer.decode(out[0].tolist(), skip_special_tokens=True))
    return texts


def train(config: dict, resume_from: str | None = None) -> None:
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    train_ds, val_ds, tokenizer = build_dataset(
        dataset_name=config["dataset"],
        tokenizer_name=config["tokenizer_name"],
        max_seq_len=config["max_seq_len"],
        val_split=config["val_split"],
        text_column=config["text_column"],
        num_proc=config["training"].get("num_workers", 4),
        streaming=config.get("streaming", False),
        max_samples=config.get("max_samples", None),
    )
    mask_token_id = tokenizer.convert_tokens_to_ids(config["mask_token"])
    config["mask_token_id"] = mask_token_id

    train_loader, val_loader = get_dataloaders(
        train_ds,
        val_ds,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 0),
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")

    # Model
    model = build_model(config, device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    tcfg = config["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_linear_warmup_cosine_decay(
        optimizer,
        tcfg["warmup_steps"],
        tcfg["max_steps"],
    )

    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        print(f"Resumed from step {start_step}")

    # AMP: MPS supports autocast but not GradScaler; CPU training is fp32 only here.
    requested = tcfg.get("mixed_precision", "fp32")
    use_amp = requested in ("fp16", "bf16") and device.type in ("mps", "cuda")
    if requested == "fp16":
        amp_dtype = torch.float16
    elif requested == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    # Training state
    model.train()
    dcfg = config["diffusion"]
    T = dcfg["T"]
    schedule_fn = get_schedule(dcfg["schedule"])
    grad_accum = tcfg["gradient_accumulation_steps"]
    eval_every = tcfg["eval_every"]
    save_every = tcfg["save_every"]
    max_steps = tcfg["max_steps"]

    step = start_step
    accum_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(total=max_steps, initial=start_step, desc="Training")

    train_iter = iter(train_loader)
    while step < max_steps:
        try:
            x0 = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x0 = next(train_iter)

        x0 = x0.to(device)
        B = x0.size(0)
        t = torch.randint(1, T + 1, (B,), device=device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, info = diffusion_loss(model, x0, t, mask_token_id, schedule_fn, T)
            loss = loss / grad_accum

        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            lr = scheduler.get_last_lr()[0]
            pbar.update(grad_accum)
            pbar.set_postfix(
                loss=f"{accum_loss:.4f}",
                lr=f"{lr:.2e}",
                beta=f"{info['beta_mean']:.3f}",
                masked=f"{info['num_masked']:.1f}",
            )
            accum_loss = 0.0

            # Evaluation
            if (step + 1) % eval_every == 0 and val_loader is not None:
                ppl = evaluate(model, val_loader, config, device)
                print(f"\n[Step {step + 1}] Validation PPL: {ppl:.3f}")
                samples = generate_samples(model, tokenizer, config, device)
                for i, s in enumerate(samples):
                    print(f"  Sample {i + 1}: {s[:200]}{'...' if len(s) > 200 else ''}")
                model.train()

            # Checkpointing
            if (step + 1) % save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step + 1,
                    config,
                    tokenizer,
                    config["checkpoint_dir"],
                )

        step += 1

    # Final checkpoint
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        step,
        config,
        tokenizer,
        config["checkpoint_dir"],
    )
    pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train masked diffusion LM")
    parser.add_argument("--config", default="configs/tinystories_m4pro.yaml", help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Checkpoint directory to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
